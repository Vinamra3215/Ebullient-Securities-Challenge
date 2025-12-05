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
TEMP_DIR = "/data/quant14/signals/VB2_temp_signal_processing"
OUTPUT_FILE = "/data/quant14/signals/temp_trading_signals_EBY.csv"
REQUIRED_COLUMNS = ['Time', 'Price']

# Blacklist
BLACKLIST_DAYS = {104, 165, 110, 115, 6, 209, 135}

# Strategy Parameters
STRATEGY_PARAMS = {
    "pb9t1": "PB9_T1", 
}

# --- PARAMETERS ---
# Volatility Gating (First 30 Minutes)
DECISION_WINDOW_SECONDS = 30 * 60
SIGMA_THRESHOLD = 0.12  
MU_THRESHOLD = 100.0    

# Entry (Long)
ENTRY_DROP_THRESHOLD = 0.8  

# Exit (Long)
STATIC_SL = 4.0
TP_ACTIVATION = 1.0
TRAIL_CALLBACK = 0.1

# Reversal (Short)
REVERSAL_RISE_TRIGGER = 0.2   # Price must rise 0.2 above Long exit price to trigger Short

# Exit (Short)
SHORT_STATIC_SL = 0.15        # Static Stop Loss
SHORT_TP_ACTIVATION = 0.2     # NEW: Profit required to activate trailing
SHORT_TRAIL_CALLBACK = 0.04   # NEW: Trailing distance

COOLDOWN_PERIOD_SECONDS = 5

# --- Numba Optimized Trading Core ---

@njit(cache=True, nogil=True)
def backtest_core(
    time_sec, price, 
    day_open_price,
    start_index,      
    can_trade_long,   
    cooldown_seconds
):
    """
    Numba Optimized Core.
    Logic:
    1. Volatility Gate: Only trade if passed.
    2. Long Entry: Price <= Open - 0.8.
    3. Long Exit: SL 4.0, TP Act 1.0, Trail 0.1.
    4. Short Reversal: Upon Long Exit, if Price >= ExitPrice + 0.2 -> Short.
    5. Short Exit: 
       - Static SL 0.15.
       - TP Activation 0.2 -> Trail 0.05.
       - OR End of Day.
    """
    n = len(price)
    signals = np.zeros(n, dtype=int64)
    positions = np.zeros(n, dtype=int64)
    
    # State Variables
    position = 0
    entry_price = 0.0
    
    # Trailing Stop State (For Longs)
    trailing_active = False
    trailing_peak_price = 0.0 

    # Trailing Stop State (For Shorts) - NEW
    short_trailing_active = False
    short_trailing_valley_price = 0.0
    
    # Reversal State
    waiting_for_short = False
    short_target_price = 0.0
    
    last_signal_time = -cooldown_seconds - 1
    
    # Iterate through ticks
    for i in range(n):
        # 1. Skip data before decision window
        if i < start_index:
            continue

        curr_t = time_sec[i]
        curr_p = price[i]
        is_last_tick = (i == n - 1)
        
        signal = 0
        
        # 2. FORCED SQUARE OFF (End of Day)
        if is_last_tick and position != 0:
            signal = -position # Closes Long (-1) or Short (+1)
        
        # 3. Strategy Logic
        else:
            # --- LONG POSITION LOGIC ---
            if position == 1:
                pnl = curr_p - entry_price
                exit_signal = False
                
                # A. Static Stop Loss
                if pnl <= -STATIC_SL:
                    exit_signal = True
                
                # B. Trailing Logic
                else:
                    if not trailing_active:
                        if pnl >= TP_ACTIVATION:
                            trailing_active = True
                            trailing_peak_price = curr_p
                    
                    if trailing_active:
                        if curr_p > trailing_peak_price:
                            trailing_peak_price = curr_p
                        elif curr_p <= (trailing_peak_price - TRAIL_CALLBACK):
                            exit_signal = True
                
                if exit_signal:
                    signal = -1  # Close Long
                    # Setup Reversal Trigger
                    waiting_for_short = True
                    short_target_price = curr_p + REVERSAL_RISE_TRIGGER

            # --- SHORT POSITION LOGIC ---
            elif position == -1:
                # Calculate PnL for Short (Entry - Current)
                # Positive PnL means Price went DOWN (Profit). 
                pnl = entry_price - curr_p
                
                # A. Static Stop Loss
                if pnl <= -SHORT_STATIC_SL:
                    signal = 1 # Buy to Close (SL Hit)
                
                # B. Trailing Logic (NEW)
                else:
                    # Activation
                    if not short_trailing_active:
                        if pnl >= SHORT_TP_ACTIVATION:
                            short_trailing_active = True
                            short_trailing_valley_price = curr_p # Track the lowest price seen
                    
                    # Execution
                    if short_trailing_active:
                        # Update Valley (Lowest Price)
                        if curr_p < short_trailing_valley_price:
                            short_trailing_valley_price = curr_p
                        
                        # Check Callback (Price Rises from Valley)
                        elif curr_p >= (short_trailing_valley_price + SHORT_TRAIL_CALLBACK):
                            signal = 1 # Buy to Close (Trailing Hit)

            # --- FLAT (ENTRY) LOGIC ---
            elif position == 0:
                
                # A. Check Reversal Short Trigger First
                if waiting_for_short:
                    if curr_p >= short_target_price:
                        signal = -1  # Enter Short
                        waiting_for_short = False # Trigger consumed
                
                # B. Check Standard Long Entry
                if signal == 0 and can_trade_long:
                    cooldown_over = (curr_t - last_signal_time) >= cooldown_seconds
                    
                    if cooldown_over:
                        if curr_p <= (day_open_price - ENTRY_DROP_THRESHOLD):
                            signal = 1 # Enter Long
                            waiting_for_short = False 
        
        # 4. Apply Signal
        if signal != 0:
            # Avoid redundant signals (debounce)
            if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                signal = 0
            else:
                # State Update Logic
                if position == 0:
                    # Opening a new position
                    entry_price = curr_p
                    # Reset Long Trailing
                    trailing_active = False
                    trailing_peak_price = 0.0
                    # Reset Short Trailing
                    short_trailing_active = False
                    short_trailing_valley_price = 0.0
                
                elif position != 0 and signal == -position:
                    # Closing a position
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
    SAVE_PATH = os.path.join(SAVE_DIR, "trade_reports.csv")

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
            output_path = temp_dir / f"day{day_num}.csv"
            final_columns = REQUIRED_COLUMNS + ["Signal", "Position", "Day"]
            df[final_columns].to_csv(output_path, index=False)
            return str(output_path)

        # Valid Day Logic
        time_sec_arr = df["Time_sec"].values.astype(np.int64)
        price_arr = df["Price"].values.astype(np.float64)
        
        if len(price_arr) < 10:
            return None
            
        day_open_price = price_arr[0]
        
        # --- Volatility Gating ---
        mask_30min = time_sec_arr <= DECISION_WINDOW_SECONDS
        
        if not np.any(mask_30min):
             can_trade_long = False
             start_idx = 0
        else:
            prices_30min = price_arr[mask_30min]
            mu = np.mean(prices_30min)
            sigma = np.std(prices_30min)
            
            # Gating Condition: Sigma > Threshold AND Mu < Threshold
            is_volatile = sigma > SIGMA_THRESHOLD
            mu_condition = mu < MU_THRESHOLD
            
            can_trade_long = is_volatile and mu_condition
            start_idx = np.searchsorted(time_sec_arr, DECISION_WINDOW_SECONDS, side='right')

        # Call Numba Function
        signals, positions = backtest_core(
            time_sec_arr, 
            price_arr, 
            day_open_price,
            int(start_idx),
            bool(can_trade_long),
            COOLDOWN_PERIOD_SECONDS
        )

        df["Signal"] = signals
        df["Position"] = positions
        
        output_path = temp_dir / f"day{day_num}.csv"
        final_columns = REQUIRED_COLUMNS + ["Signal", "Position", "Day"]
        df[final_columns].to_csv(output_path, index=False)
        print(f"✅ Processed Day {day_num} | Volatile: {can_trade_long}")
        
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


# === Backtest Summary ===
# Initial Capital               : 100.00
# Final Capital                 : 103.24
# Total PnL                     : 3.24
# Total Transaction Cost         : 0.99
# Penalty Counts                : 0
# Final Returns                 : 3.24%
# CAGR                          : 2.9221%
# Annualized Returns            : 2.9266%
# Sharpe Ratio                  : 1.2193
# Calmar Ratio                  : 2.3392
# Maximum Drawdown              : -1.2511%
# No. of Days                   : 279
# Winning Days                  : 10
# Losing Days                   : 11
# Best Day                      : 192
# Worst Day                     : 277
# Best Day PnL                  : 1.21
# Worst Day PnL                 : -0.83
# Total Trades                  : 50
# Winning Trades                : 17
# Losing Trades                 : 33
# Win Rate (%)                  : 34.00%
# Average Winning Trade         : 0.5045
# Average Losing Trade          : -0.1108
# Average Hold Period (seconds) : 6,388.24
# Average Hold Period (minutes) : 106.47
# ========================