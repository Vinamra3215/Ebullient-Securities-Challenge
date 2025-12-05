import os
import glob
import re
import shutil
import sys
import argparse
import pathlib
import pandas as pd
import numpy as np
import time
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit, int64, float64

# --- Configuration ---
TEMP_DIR = "/data/quant14/signals/VB2_temp_signal_processing_combined"
OUTPUT_FILE = "/data/quant14/signals/combined_trading_signals.csv"
REQUIRED_COLUMNS = ['Time', 'Price']

# Blacklist
BLACKLIST_DAYS = {104, 165, 110, 115, 6, 209, 135}

# Strategy Parameters (Files)
STRATEGY_PARAMS = {
    "pb9t1": "PB9_T1", 
}

# ==========================================
# STRATEGY 1 PARAMETERS (High Volatility)
# ==========================================
S1_SIGMA_THRESHOLD = 0.12
S1_MU_THRESHOLD = 100.0  # Logic looks for Mu > 100

# 1. Primary Entry (Short)
S1_ENTRY_RISE_THRESHOLD = 0.8

# 2. Primary Exit (Short)
S1_PRIMARY_STATIC_SL = 0.7
S1_PRIMARY_TP_ACTIVATION = 0.65
S1_PRIMARY_TRAIL_CALLBACK = 0.15

# 3. Reversal Trigger (Long)
S1_REVERSAL_DROP_TRIGGER = 0.2

# 4. Reversal Exit (Long)
S1_SECONDARY_STATIC_SL = 0.15
S1_SECONDARY_TP_ACTIVATION = 0.8
S1_SECONDARY_TRAIL_CALLBACK = 0.1

# ==========================================
# STRATEGY 2 PARAMETERS (Medium Volatility)
# ==========================================
S2_SIGMA_MIN = 0.09
S2_SIGMA_MAX = 0.12
S2_MU_THRESHOLD = 100.0  # Logic looks for Mu < 100

# Entry (Long)
S2_ENTRY_DROP_THRESHOLD = 0.9

# Exit (Long)
S2_STATIC_SL = 4.0
S2_TP_ACTIVATION = 1.0
S2_TRAIL_CALLBACK = 0.1

# Reversal (Short)
S2_REVERSAL_RISE_TRIGGER = 0.2

# Exit (Short)
S2_SHORT_STATIC_SL = 0.15
S2_SHORT_TP_ACTIVATION = 5.2
S2_SHORT_TRAIL_CALLBACK = 0.04

# ==========================================
# SHARED SETTINGS
# ==========================================
DECISION_WINDOW_SECONDS = 30 * 60
COOLDOWN_PERIOD_SECONDS = 5

# ==========================================
# STRATEGY 1 CORE (High Vol: Gap Long -> Main Short)
# ==========================================

@njit(cache=True, nogil=True)
def backtest_core_s1(
    time_sec, price, 
    day_open_price,
    start_index,      
    can_trade,   
    cooldown_seconds,
    entry_rise_threshold,
    # Explicitly passing S1 params to avoid scope issues in Numba
    primary_sl, primary_tp, primary_cb,
    reversal_drop,
    sec_sl, sec_tp, sec_cb
):
    """
    Strategy 1 Logic: Pre-Rise Long -> Main Short -> Reversal Long
    """
    n = len(price)
    signals = np.zeros(n, dtype=int64)
    positions = np.zeros(n, dtype=int64)
    
    # State Variables
    position = 0
    entry_price = 0.0
    
    # --- Pre-Rise (Gap Fill) State ---
    pre_rise_target = day_open_price + entry_rise_threshold
    pre_rise_long_active = False
    pending_short_flip = False
    flip_wait_start_time = 0

    # Execution Flags
    short_trade_executed = False  
    
    # Trailing State (Short - Primary)
    short_trailing_active = False
    short_trailing_valley_price = 0.0

    # Trailing State (Long - Reversal)
    long_trailing_active = False
    long_trailing_peak_price = 0.0
    
    # Reversal Trigger State
    waiting_for_long_reversal = False
    long_reversal_target_price = 0.0
    
    last_signal_time = -cooldown_seconds - 1
    
    for i in range(n):
        if i < start_index: continue

        curr_t = time_sec[i]
        curr_p = price[i]
        is_last_tick = (i == n - 1)
        
        signal = 0
        
        if is_last_tick and position != 0:
            signal = -position 
        else:
            # --- SPECIAL LOGIC: GAP FILL LONG ---
            if i == start_index and can_trade and position == 0:
                if curr_p < pre_rise_target:
                    signal = 1 # Enter Pre-Rise Long
                    pre_rise_long_active = True

            # --- A. PRE-RISE LONG LOGIC ---
            elif pre_rise_long_active:
                if position == 1:
                    if curr_p >= pre_rise_target:
                        signal = -1 # Close Long
                        pre_rise_long_active = False
                        pending_short_flip = True
                        flip_wait_start_time = curr_t
            
            # --- B. PENDING SHORT FLIP ---
            elif pending_short_flip:
                if (curr_t - flip_wait_start_time) >= cooldown_seconds:
                    signal = -1 # Enter Main Short
                    pending_short_flip = False
                    short_trade_executed = True 
                    waiting_for_long_reversal = False 

            # --- C. STANDARD STRATEGY LOGIC ---
            else:
                # --- PRIMARY POSITION (SHORT) ---
                if position == -1:
                    pnl = entry_price - curr_p
                    exit_signal = False
                    
                    if pnl <= -primary_sl:
                        exit_signal = True
                    else:
                        if not short_trailing_active:
                            if pnl >= primary_tp:
                                short_trailing_active = True
                                short_trailing_valley_price = curr_p
                        
                        if short_trailing_active:
                            if curr_p < short_trailing_valley_price:
                                short_trailing_valley_price = curr_p
                            elif curr_p >= (short_trailing_valley_price + primary_cb):
                                exit_signal = True
                    
                    if exit_signal:
                        signal = 1  # Buy to Close
                        waiting_for_long_reversal = True
                        long_reversal_target_price = curr_p - reversal_drop

                # --- SECONDARY POSITION (LONG - REVERSAL) ---
                elif position == 1:
                    pnl = curr_p - entry_price
                    exit_signal = False
                    
                    if pnl <= -sec_sl:
                        exit_signal = True
                    else:
                        if not long_trailing_active:
                            if pnl >= sec_tp:
                                long_trailing_active = True
                                long_trailing_peak_price = curr_p
                        
                        if long_trailing_active:
                            if curr_p > long_trailing_peak_price:
                                long_trailing_peak_price = curr_p
                            elif curr_p <= (long_trailing_peak_price - sec_cb):
                                exit_signal = True

                    if exit_signal:
                        signal = -1 # Sell to Close

                # --- FLAT (ENTRY) LOGIC ---
                elif position == 0:
                    if waiting_for_long_reversal:
                        if curr_p <= long_reversal_target_price:
                            signal = 1  # Enter Long (Reversal)
                            waiting_for_long_reversal = False 
                    
                    elif can_trade and not short_trade_executed:
                        cooldown_over = (curr_t - last_signal_time) >= cooldown_seconds
                        if cooldown_over:
                            if curr_p >= pre_rise_target:
                                signal = -1 # Enter Short
                                short_trade_executed = True
                                waiting_for_long_reversal = False 
        
        # Apply Signal
        if signal != 0:
            if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                signal = 0
            else:
                if position == 0:
                    entry_price = curr_p
                    short_trailing_active = False
                    short_trailing_valley_price = 0.0
                    long_trailing_active = False
                    long_trailing_peak_price = 0.0
                elif position != 0 and signal == -position:
                    entry_price = 0.0
                    short_trailing_active = False
                    short_trailing_valley_price = 0.0
                    long_trailing_active = False
                    long_trailing_peak_price = 0.0
                
                position += signal
                last_signal_time = curr_t
        
        signals[i] = signal
        positions[i] = position
        
    return signals, positions

# ==========================================
# STRATEGY 2 CORE (Med Vol: Gap Short -> Main Long)
# ==========================================

@njit(cache=True, nogil=True)
def backtest_core_s2(
    time_sec, price, 
    day_open_price,
    start_index,      
    can_trade,   
    cooldown_seconds,
    entry_drop_threshold,
    # Explicitly passing S2 params
    static_sl, tp_activation, trail_cb,
    reversal_rise,
    short_sl, short_tp, short_cb
):
    """
    Strategy 2 Logic: Pre-Drop Short -> Main Long -> Reversal Short
    """
    n = len(price)
    signals = np.zeros(n, dtype=int64)
    positions = np.zeros(n, dtype=int64)
    
    position = 0
    entry_price = 0.0
    
    pre_drop_target = day_open_price - entry_drop_threshold
    pre_drop_short_active = False
    pending_long_flip = False
    flip_wait_start_time = 0
    
    trailing_active = False
    trailing_peak_price = 0.0 

    short_trailing_active = False
    short_trailing_valley_price = 0.0
    
    waiting_for_short = False
    short_target_price = 0.0
    
    last_signal_time = -cooldown_seconds - 1
    
    for i in range(n):
        if i < start_index: continue

        curr_t = time_sec[i]
        curr_p = price[i]
        is_last_tick = (i == n - 1)
        
        signal = 0
        
        if is_last_tick and position != 0:
            signal = -position 
        else:
            # --- SPECIAL LOGIC: GAP FILL SHORT ---
            if i == start_index and can_trade and position == 0:
                if curr_p > pre_drop_target:
                    signal = -1 # Enter Pre-Drop Short
                    pre_drop_short_active = True
            
            # --- A. PRE-DROP SHORT LOGIC ---
            elif pre_drop_short_active:
                if position == -1:
                    if curr_p <= pre_drop_target:
                        signal = 1 # Close Short
                        pre_drop_short_active = False
                        pending_long_flip = True
                        flip_wait_start_time = curr_t
            
            # --- B. PENDING LONG FLIP ---
            elif pending_long_flip:
                if (curr_t - flip_wait_start_time) >= cooldown_seconds:
                    signal = 1 # Enter Long
                    pending_long_flip = False
                    waiting_for_short = False 
            
            # --- C. STANDARD STRATEGY LOGIC ---
            else:
                # --- LONG POSITION LOGIC ---
                if position == 1:
                    pnl = curr_p - entry_price
                    exit_signal = False
                    
                    if pnl <= -static_sl:
                        exit_signal = True
                    else:
                        if not trailing_active:
                            if pnl >= tp_activation:
                                trailing_active = True
                                trailing_peak_price = curr_p
                        
                        if trailing_active:
                            if curr_p > trailing_peak_price:
                                trailing_peak_price = curr_p
                            elif curr_p <= (trailing_peak_price - trail_cb):
                                exit_signal = True
                    
                    if exit_signal:
                        signal = -1  # Close Long
                        waiting_for_short = True
                        short_target_price = curr_p + reversal_rise

                # --- SHORT POSITION LOGIC (REVERSAL) ---
                elif position == -1:
                    pnl = entry_price - curr_p
                    
                    if pnl <= -short_sl:
                        signal = 1 # Buy to Close (SL Hit)
                    else:
                        if not short_trailing_active:
                            if pnl >= short_tp:
                                short_trailing_active = True
                                short_trailing_valley_price = curr_p 
                        
                        if short_trailing_active:
                            if curr_p < short_trailing_valley_price:
                                short_trailing_valley_price = curr_p
                            elif curr_p >= (short_trailing_valley_price + short_cb):
                                signal = 1 # Buy to Close (Trailing Hit)

                # --- FLAT (ENTRY) LOGIC ---
                elif position == 0:
                    if waiting_for_short:
                        if curr_p >= short_target_price:
                            signal = -1  # Enter Short
                            waiting_for_short = False 
                    
                    elif can_trade:
                        cooldown_over = (curr_t - last_signal_time) >= cooldown_seconds
                        if cooldown_over:
                            if curr_p <= pre_drop_target:
                                signal = 1 # Enter Long
                                waiting_for_short = False 
        
        # Apply Signal
        if signal != 0:
            if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                signal = 0
            else:
                if position == 0:
                    entry_price = curr_p
                    trailing_active = False
                    trailing_peak_price = 0.0
                    short_trailing_active = False
                    short_trailing_valley_price = 0.0
                elif position != 0 and signal == -position:
                    entry_price = 0.0
                    trailing_active = False
                    trailing_peak_price = 0.0
                    short_trailing_active = False
                    short_trailing_valley_price = 0.0
                
                position += signal
                last_signal_time = curr_t
        
        signals[i] = signal
        positions[i] = position
        
    return signals, positions

# --- Trade Reporting ---

def generate_trade_reports_csv(output_file):
    SAVE_DIR = "/home/raid/Quant14/VB_Feature_Analysis/Histogram/"
    SAVE_PATH = os.path.join(SAVE_DIR, "trade_reports_combined.csv")

    os.makedirs(SAVE_DIR, exist_ok=True)
    
    try:
        df = pd.read_csv(output_file)
    except FileNotFoundError:
        print("Error: File not found:", output_file)
        return

    if "Signal" not in df.columns or "Day" not in df.columns:
        print("Error: Columns missing.")
        return

    signals_df = df[df["Signal"] != 0].copy()
    if len(signals_df) == 0:
        print("No trades found.")
        return

    trades = []
    position = None
    entry = None

    for idx, row in signals_df.iterrows():
        signal = row["Signal"]
        if pd.isna(signal): continue

        if position is None:
            # Entry (1 for Long, -1 for Short)
            if signal in [1, -1]:
                position = signal
                entry = row
        else:
            # Exit
            if signal == -position:
                direction = "LONG" if position == 1 else "SHORT"
                pnl = (float(row["Price"]) - float(entry["Price"])) * position
                
                trades.append({
                    "day": entry["Day"],
                    "strategy": row.get("Strategy", "Unknown"),
                    "direction": direction,
                    "entry_time": entry["Time"],
                    "entry_price": float(entry["Price"]),
                    "exit_time": row["Time"],
                    "exit_price": float(row["Price"]),
                    "pnl": pnl
                })
                position = None
                entry = None

    if len(trades) == 0:
        print("No completed trades.")
        return

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(SAVE_PATH, index=False)
    print(f"✓ trade_reports.csv saved at: {SAVE_PATH}")

# --- Processing ---

def extract_day_num(filepath):
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1

def process_day(file_path: str, day_num: int, temp_dir: pathlib.Path, strategy_params) -> str:
    try:
        columns = REQUIRED_COLUMNS + list(strategy_params.values())
        
        df = pd.read_parquet(file_path, columns=columns)
        
        if df.empty:
            return None

        df = df.reset_index(drop=True)
        df = df.sort_values("Time").reset_index(drop=True)
        df["Time_sec"] = pd.to_timedelta(df["Time"].astype(str)).dt.total_seconds().astype(int)
        df["Day"] = day_num

        # Blacklist Logic
        if day_num in BLACKLIST_DAYS:
            df["Signal"] = 0
            df["Position"] = 0
            df["Strategy"] = "Blacklist"
            output_path = temp_dir / f"day{day_num}.csv"
            final_columns = REQUIRED_COLUMNS + ["Signal", "Position", "Day", "Strategy"]
            df[final_columns].to_csv(output_path, index=False)
            return str(output_path)

        # Valid Day Logic
        time_sec_arr = df["Time_sec"].values.astype(np.int64)
        price_arr = df["Price"].values.astype(np.float64)
        
        if len(price_arr) < 10:
            return None
            
        day_open_price = price_arr[0]
        
        # --- Volatility Gating & Strategy Selection ---
        mask_30min = time_sec_arr <= DECISION_WINDOW_SECONDS
        
        signals = np.zeros(len(price_arr), dtype=np.int64)
        positions = np.zeros(len(price_arr), dtype=np.int64)
        selected_strategy = "None"
        
        if np.any(mask_30min):
            prices_30min = price_arr[mask_30min]
            mu = np.mean(prices_30min)
            sigma = np.std(prices_30min)
            start_idx = np.searchsorted(time_sec_arr, DECISION_WINDOW_SECONDS, side='right')

            # -----------------------------------------------
            # STRATEGY SELECTION LOGIC
            # -----------------------------------------------
            
            # STRATEGY 1: Sigma > 0.12 (and Mu > 100)
            if sigma > S1_SIGMA_THRESHOLD and mu > S1_MU_THRESHOLD:
                selected_strategy = "S1_HighVol"
                signals, positions = backtest_core_s1(
                    time_sec_arr, price_arr, day_open_price, int(start_idx), 
                    True, COOLDOWN_PERIOD_SECONDS, S1_ENTRY_RISE_THRESHOLD,
                    S1_PRIMARY_STATIC_SL, S1_PRIMARY_TP_ACTIVATION, S1_PRIMARY_TRAIL_CALLBACK,
                    S1_REVERSAL_DROP_TRIGGER,
                    S1_SECONDARY_STATIC_SL, S1_SECONDARY_TP_ACTIVATION, S1_SECONDARY_TRAIL_CALLBACK
                )

            # STRATEGY 2: 0.09 <= Sigma <= 0.12 (and Mu < 100)
            elif (S2_SIGMA_MIN <= sigma <= S2_SIGMA_MAX) and mu < S2_MU_THRESHOLD:
                selected_strategy = "S2_MedVol"
                signals, positions = backtest_core_s2(
                    time_sec_arr, price_arr, day_open_price, int(start_idx), 
                    True, COOLDOWN_PERIOD_SECONDS, S2_ENTRY_DROP_THRESHOLD,
                    S2_STATIC_SL, S2_TP_ACTIVATION, S2_TRAIL_CALLBACK,
                    S2_REVERSAL_RISE_TRIGGER,
                    S2_SHORT_STATIC_SL, S2_SHORT_TP_ACTIVATION, S2_SHORT_TRAIL_CALLBACK
                )
        
        df["Signal"] = signals
        df["Position"] = positions
        df["Strategy"] = selected_strategy
        
        output_path = temp_dir / f"day{day_num}.csv"
        final_columns = REQUIRED_COLUMNS + ["Signal", "Position", "Day", "Strategy"]
        df[final_columns].to_csv(output_path, index=False)
        
        del df
        gc.collect()
        return str(output_path)

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main(directory: str, max_workers: int, strategy_params):
    if sys.platform != 'win32':
        try:
            mp.set_start_method('fork', force=True)
        except RuntimeError:
            pass

    start_time = time.time()
    temp_dir_path = pathlib.Path(TEMP_DIR)
    
    if temp_dir_path.exists():
        shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)

    files = sorted(glob.glob(os.path.join(directory, "day*.parquet")), key=extract_day_num)
    if not files:
        print("No parquet files found.")
        return

    processed_files = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_day, f, extract_day_num(f), temp_dir_path, strategy_params): f
            for f in files
        }
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res:
                    processed_files.append(res)
            except Exception as e:
                print(f"Error in worker: {e}")

    if not processed_files:
        print("No files processed.")
        return

    processed_sorted = sorted(processed_files, key=extract_day_num)
    print(f"Merging {len(processed_sorted)} files...")
    
    with open(OUTPUT_FILE, "wb") as out:
        for i, csv_file in enumerate(processed_sorted):
            with open(csv_file, "rb") as inp:
                if i == 0:
                    shutil.copyfileobj(inp, out)
                else:
                    inp.readline() 
                    shutil.copyfileobj(inp, out)

    print("Generating trade reports...")
    generate_trade_reports_csv(OUTPUT_FILE)

    shutil.rmtree(temp_dir_path)
    print(f"✅ Output saved: {OUTPUT_FILE}")
    print(f"⏱ Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str)
    parser.add_argument("--max-workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    main(args.directory, args.max_workers, STRATEGY_PARAMS)