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
# Using a combined output folder
TEMP_DIR = "/data/quant14/signals/VB2_combined_processing_temp"
OUTPUT_FILE = "/data/quant14/signals/combined_mu_gated_signals.csv"
REQUIRED_COLUMNS = ['Time', 'Price']

# Blacklist (Merged from both)
BLACKLIST_DAYS = {104, 165, 110, 115, 6, 209, 135}

# Strategy Parameters Helper
STRATEGY_PARAMS_COLS = {
    "pb9t1": "PB9_T1", 
}

# --- GLOBAL SETTINGS ---
DECISION_WINDOW_SECONDS = 30 * 60
MU_SPLIT_THRESHOLD = 100.0  # The main decider: < 100 runs Strat 1, > 100 runs Strat 2
COOLDOWN_PERIOD_SECONDS = 5

# ==========================================
#      STRATEGY 1 PARAMETERS (LOW MU)
# ==========================================
# (Logic for Days where Mu < 100)

L_MU_REVERSAL_RISE_TRIGGER = 0.2

# Low Mu - Strat 1: High Volatility (Sigma > 0.12)
L_MU_STRAT_1_PARAMS = {
    "entry_drop": 0.8,
    "static_sl": 0.7,
    "tp_activation": 1.0,
    "trail_callback": 0.1,
    "short_static_sl": 999.0,
    "short_tp_activation": 999.0,
    "short_trail_callback": 0.04,
    "pre_drop_sl": 999.0  # Set high to effectively disable for Strat 1
}

# Low Mu - Strat 2: Medium Volatility (0.09 <= Sigma <= 0.12)
L_MU_STRAT_2_PARAMS = {
    "entry_drop": 0.9,
    "static_sl": 999.0,
    "tp_activation": 1.0,
    "trail_callback": 0.1,
    "short_static_sl": 0.15,
    "short_tp_activation": 999.0,
    "short_trail_callback": 0.04,
    "pre_drop_sl": 0.6    # Active for Strat 2
}

# ==========================================
#      STRATEGY 2 PARAMETERS (HIGH MU)
# ==========================================
# (Logic for Days where Mu > 100)

H_MU_SIGMA_HIGH_THRESHOLD = 0.12 
H_MU_SIGMA_MID_MIN = 0.09     
H_MU_SIGMA_MID_MAX = 0.12

# === High Mu - Strategy 1 (High Volatility - Short First) ===
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
# 5. Pre-Rise Long Stop Loss
S1_PRE_RISE_SL = 0.4

# === High Mu - Strategy 2 (Mid Volatility - Long First) ===
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
#       NUMBA CORE: LOW MU LOGIC
# ==========================================

@njit(cache=True, nogil=True)
def backtest_core_low_mu(
    time_sec, price, 
    day_open_price,
    start_index,        
    can_trade_long,     
    cooldown_seconds,
    reversal_rise_trigger,
    # Dynamic Parameters passed based on Sigma
    entry_drop_threshold,
    static_sl,
    tp_activation,
    trail_callback,
    short_static_sl,
    short_tp_activation,
    short_trail_callback,
    pre_drop_sl
):
    """
    Unified Numba Core for LOW MU days (< 100).
    Handles both HighVol and MedVol via passed parameters.
    """
    n = len(price)
    signals = np.zeros(n, dtype=int64)
    positions = np.zeros(n, dtype=int64)
    
    # State Variables
    position = 0
    entry_price = 0.0
    
    # Pre-Drop (Gap-Fill) Logic State
    pre_drop_target = day_open_price - entry_drop_threshold
    pre_drop_short_active = False
    pending_long_flip = False
    flip_wait_start_time = 0
    
    # Trailing Stop State (For Longs)
    trailing_active = False
    trailing_peak_price = 0.0 

    # Trailing Stop State (For Shorts)
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
            # --- SPECIAL LOGIC: 30 MIN TRIGGER (GAP FILL SHORT) ---
            if i == start_index and can_trade_long and position == 0:
                if curr_p > pre_drop_target:
                    signal = -1 # Enter Pre-Drop Short
                    pre_drop_short_active = True
            
            # --- A. PRE-DROP SHORT LOGIC ---
            elif pre_drop_short_active:
                if position == -1:
                    pnl_short = entry_price - curr_p
                    
                    # 1. Check STOP LOSS
                    if pnl_short <= -pre_drop_sl:
                        signal = 1 # Close Short (SL Hit)
                        pre_drop_short_active = False
                        # SL hit means we reset standard logic

                    # 2. Check Target (Open - Threshold)
                    elif curr_p <= pre_drop_target:
                        signal = 1 # Close Short (Target Hit)
                        pre_drop_short_active = False
                        # Prepare for Long Entry after cooldown
                        pending_long_flip = True
                        flip_wait_start_time = curr_t
            
            # --- B. PENDING LONG FLIP (Cooldown) ---
            elif pending_long_flip:
                if (curr_t - flip_wait_start_time) >= cooldown_seconds:
                    signal = 1 # Enter Long
                    pending_long_flip = False
                    waiting_for_short = False # Reset any reversal flags
            
            # --- C. STANDARD STRATEGY LOGIC ---
            else:
                # --- LONG POSITION LOGIC ---
                if position == 1:
                    pnl = curr_p - entry_price
                    exit_signal = False
                    
                    # Static Stop Loss
                    if pnl <= -static_sl:
                        exit_signal = True
                    
                    # Trailing Logic
                    else:
                        if not trailing_active:
                            if pnl >= tp_activation:
                                trailing_active = True
                                trailing_peak_price = curr_p
                        
                        if trailing_active:
                            if curr_p > trailing_peak_price:
                                trailing_peak_price = curr_p
                            elif curr_p <= (trailing_peak_price - trail_callback):
                                exit_signal = True
                    
                    if exit_signal:
                        signal = -1  # Close Long
                        # Setup Reversal Trigger
                        waiting_for_short = True
                        short_target_price = curr_p + reversal_rise_trigger

                # --- SHORT POSITION LOGIC (REVERSAL SHORTS) ---
                elif position == -1:
                    pnl = entry_price - curr_p
                    
                    # Static Stop Loss
                    if pnl <= -short_static_sl:
                        signal = 1 # Buy to Close (SL Hit)
                    
                    # Trailing Logic
                    else:
                        if not short_trailing_active:
                            if pnl >= short_tp_activation:
                                short_trailing_active = True
                                short_trailing_valley_price = curr_p 
                        
                        if short_trailing_active:
                            if curr_p < short_trailing_valley_price:
                                short_trailing_valley_price = curr_p
                            elif curr_p >= (short_trailing_valley_price + short_trail_callback):
                                signal = 1 # Buy to Close (Trailing Hit)

                # --- FLAT (ENTRY) LOGIC ---
                elif position == 0:
                    
                    # Check Reversal Short Trigger First
                    if waiting_for_short:
                        if curr_p >= short_target_price:
                            signal = -1  # Enter Short
                            waiting_for_short = False 
                    
                    # Check Standard Long Entry
                    elif can_trade_long:
                        cooldown_over = (curr_t - last_signal_time) >= cooldown_seconds
                        
                        if cooldown_over:
                            if curr_p <= pre_drop_target:
                                signal = 1 # Enter Long
                                waiting_for_short = False 
        
        # 4. Apply Signal
        if signal != 0:
            if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                signal = 0
            else:
                if position != 0 and signal == -position:
                    entry_price = 0.0
                    trailing_active = False
                    trailing_peak_price = 0.0
                    short_trailing_active = False
                    short_trailing_valley_price = 0.0
                elif position == 0:
                    entry_price = curr_p
                    trailing_active = False
                    trailing_peak_price = 0.0
                    short_trailing_active = False
                    short_trailing_valley_price = 0.0
                
                position += signal
                last_signal_time = curr_t
        
        signals[i] = signal
        positions[i] = position
        
    return signals, positions

# ==========================================
#       NUMBA CORE: HIGH MU LOGIC
# ==========================================

@njit(cache=True, nogil=True)
def backtest_high_mu_s1(
    time_sec, price, 
    day_open_price,
    start_index,       
    can_trade_short,    
    cooldown_seconds,
    entry_rise_threshold,
    pre_rise_sl
):
    """
    High Mu Strategy 1: High Volatility (sigma > 0.12)
    Short-first strategy with gap-fill long
    """
    n = len(price)
    signals = np.zeros(n, dtype=int64)
    positions = np.zeros(n, dtype=int64)
    
    position = 0
    entry_price = 0.0
    
    # Pre-Rise (Gap Fill) State
    pre_rise_target = day_open_price + entry_rise_threshold
    pre_rise_long_active = False
    pending_short_flip = False
    flip_wait_start_time = 0

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
        if i < start_index:
            continue

        curr_t = time_sec[i]
        curr_p = price[i]
        is_last_tick = (i == n - 1)
        
        signal = 0
        
        # FORCED SQUARE OFF (End of Day)
        if is_last_tick and position != 0:
            signal = -position 
        
        else:
            # 30 MIN TRIGGER (GAP FILL LONG)
            if i == start_index and can_trade_short and position == 0:
                if curr_p < pre_rise_target:
                    signal = 1
                    pre_rise_long_active = True

            # PRE-RISE LONG LOGIC
            elif pre_rise_long_active:
                if position == 1:
                    # Stop Loss Check
                    if (curr_p - entry_price) <= -pre_rise_sl:
                        signal = -1
                        pre_rise_long_active = False
                    
                    # Target Hit
                    elif curr_p >= pre_rise_target:
                        signal = -1
                        pre_rise_long_active = False
                        pending_short_flip = True
                        flip_wait_start_time = curr_t
            
            # PENDING SHORT FLIP (Cooldown)
            elif pending_short_flip:
                if (curr_t - flip_wait_start_time) >= cooldown_seconds:
                    signal = -1
                    pending_short_flip = False
                    short_trade_executed = True
                    waiting_for_long_reversal = False

            else:
                # PRIMARY POSITION (SHORT)
                if position == -1:
                    pnl = entry_price - curr_p
                    exit_signal = False
                    
                    # Static Stop Loss
                    if pnl <= -S1_PRIMARY_STATIC_SL:
                        exit_signal = True
                    
                    # Trailing Logic
                    else:
                        if not short_trailing_active:
                            if pnl >= S1_PRIMARY_TP_ACTIVATION:
                                short_trailing_active = True
                                short_trailing_valley_price = curr_p
                        
                        if short_trailing_active:
                            if curr_p < short_trailing_valley_price:
                                short_trailing_valley_price = curr_p
                            elif curr_p >= (short_trailing_valley_price + S1_PRIMARY_TRAIL_CALLBACK):
                                exit_signal = True
                    
                    if exit_signal:
                        signal = 1
                        waiting_for_long_reversal = True
                        long_reversal_target_price = curr_p - S1_REVERSAL_DROP_TRIGGER

                # SECONDARY POSITION (LONG - REVERSAL)
                elif position == 1:
                    pnl = curr_p - entry_price
                    exit_signal = False
                    
                    # Static Stop Loss
                    if pnl <= -S1_SECONDARY_STATIC_SL:
                        exit_signal = True
                    
                    # Trailing Logic
                    else:
                        if not long_trailing_active:
                            if pnl >= S1_SECONDARY_TP_ACTIVATION:
                                long_trailing_active = True
                                long_trailing_peak_price = curr_p
                        
                        if long_trailing_active:
                            if curr_p > long_trailing_peak_price:
                                long_trailing_peak_price = curr_p
                            elif curr_p <= (long_trailing_peak_price - S1_SECONDARY_TRAIL_CALLBACK):
                                exit_signal = True

                    if exit_signal:
                        signal = -1

                # FLAT (ENTRY) LOGIC
                elif position == 0:
                    
                    # Check Reversal Long Trigger
                    if waiting_for_long_reversal:
                        if curr_p <= long_reversal_target_price:
                            signal = 1
                            waiting_for_long_reversal = False 
                    
                    # Check Primary Short Entry
                    elif can_trade_short and not short_trade_executed:
                        cooldown_over = (curr_t - last_signal_time) >= cooldown_seconds
                        
                        if cooldown_over:
                            if curr_p >= pre_rise_target:
                                signal = -1
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

@njit(cache=True, nogil=True)
def backtest_high_mu_s2(
    time_sec, price, 
    day_open_price,
    start_index,       
    can_trade_long,    
    cooldown_seconds,
    entry_drop_threshold 
):
    """
    High Mu Strategy 2: Mid Volatility
    Long-first strategy with gap-fill short
    """
    n = len(price)
    signals = np.zeros(n, dtype=int64)
    positions = np.zeros(n, dtype=int64)
    
    position = 0
    entry_price = 0.0
    
    # Pre-Drop (Gap-Fill) Logic State
    pre_drop_target = day_open_price - entry_drop_threshold
    pre_drop_short_active = False
    pending_long_flip = False
    flip_wait_start_time = 0
    
    # Trailing Stop State (For Longs)
    trailing_active = False
    trailing_peak_price = 0.0 

    # Trailing Stop State (For Shorts)
    short_trailing_active = False
    short_trailing_valley_price = 0.0
    
    # Reversal State
    waiting_for_short = False
    short_target_price = 0.0
    
    last_signal_time = -cooldown_seconds - 1
    
    for i in range(n):
        if i < start_index:
            continue

        curr_t = time_sec[i]
        curr_p = price[i]
        is_last_tick = (i == n - 1)
        
        signal = 0
        
        # FORCED SQUARE OFF (End of Day)
        if is_last_tick and position != 0:
            signal = -position
        
        else:
            # 30 MIN TRIGGER (GAP FILL SHORT)
            if i == start_index and can_trade_long and position == 0:
                if curr_p > pre_drop_target:
                    signal = -1
                    pre_drop_short_active = True
            
            # PRE-DROP SHORT LOGIC
            elif pre_drop_short_active:
                if position == -1:
                    if curr_p <= pre_drop_target:
                        signal = 1
                        pre_drop_short_active = False
                        pending_long_flip = True
                        flip_wait_start_time = curr_t
            
            # PENDING LONG FLIP (Cooldown)
            elif pending_long_flip:
                if (curr_t - flip_wait_start_time) >= cooldown_seconds:
                    signal = 1
                    pending_long_flip = False
                    waiting_for_short = False
            
            else:
                # LONG POSITION LOGIC
                if position == 1:
                    pnl = curr_p - entry_price
                    exit_signal = False
                    
                    # Static Stop Loss
                    if pnl <= -S2_STATIC_SL:
                        exit_signal = True
                    
                    # Trailing Logic
                    else:
                        if not trailing_active:
                            if pnl >= S2_TP_ACTIVATION:
                                trailing_active = True
                                trailing_peak_price = curr_p
                        
                        if trailing_active:
                            if curr_p > trailing_peak_price:
                                trailing_peak_price = curr_p
                            elif curr_p <= (trailing_peak_price - S2_TRAIL_CALLBACK):
                                exit_signal = True
                    
                    if exit_signal:
                        signal = -1
                        waiting_for_short = True
                        short_target_price = curr_p + S2_REVERSAL_RISE_TRIGGER

                # SHORT POSITION LOGIC (REVERSAL SHORTS)
                elif position == -1:
                    pnl = entry_price - curr_p
                    
                    # Static Stop Loss
                    if pnl <= -S2_SHORT_STATIC_SL:
                        signal = 1
                    
                    # Trailing Logic
                    else:
                        if not short_trailing_active:
                            if pnl >= S2_SHORT_TP_ACTIVATION:
                                short_trailing_active = True
                                short_trailing_valley_price = curr_p 
                        
                        if short_trailing_active:
                            if curr_p < short_trailing_valley_price:
                                short_trailing_valley_price = curr_p
                            elif curr_p >= (short_trailing_valley_price + S2_SHORT_TRAIL_CALLBACK):
                                signal = 1

                # FLAT (ENTRY) LOGIC
                elif position == 0:
                    
                    # Check Reversal Short Trigger
                    if waiting_for_short:
                        if curr_p >= short_target_price:
                            signal = -1
                            waiting_for_short = False 
                    
                    # Check Standard Long Entry
                    elif can_trade_long:
                        cooldown_over = (curr_t - last_signal_time) >= cooldown_seconds
                        
                        if cooldown_over:
                            if curr_p <= pre_drop_target:
                                signal = 1
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
            # Entry
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
                    "strategy_type": row.get("Strat_Type", "Unknown"), 
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

def process_day(file_path: str, day_num: int, temp_dir: pathlib.Path, strategy_params_cols) -> str:
    try:
        columns = REQUIRED_COLUMNS + list(strategy_params_cols.values())
        
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
            df["Strat_Type"] = 0
            output_path = temp_dir / f"day{day_num}.csv"
            final_columns = REQUIRED_COLUMNS + ["Signal", "Position", "Day", "Strat_Type"]
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
        
        can_trade = False
        start_idx = 0
        strat_type_label = "None"
        
        # Default empty arrays
        signals = np.zeros(len(price_arr), dtype=np.int64)
        positions = np.zeros(len(price_arr), dtype=np.int64)

        if np.any(mask_30min):
            prices_30min = price_arr[mask_30min]
            mu = np.mean(prices_30min)
            sigma = np.std(prices_30min)
            
            # ===============================================
            #              MAIN DECISION TREE
            # ===============================================
            
            # --- BRANCH 1: LOW MU (< 100) -> Runs Old "Strategy 1" Logic ---
            if mu < MU_SPLIT_THRESHOLD:
                can_trade_low_mu = False
                active_params = None
                
                # Check Internal Sigma Logic for Low Mu
                if 0.09 <= sigma <= 0.12:
                    can_trade_low_mu = True
                    active_params = L_MU_STRAT_2_PARAMS
                    strat_type_label = "LowMu_Strat2"
                elif sigma > 0.12:
                    can_trade_low_mu = True
                    active_params = L_MU_STRAT_1_PARAMS
                    strat_type_label = "LowMu_Strat1"
                
                if can_trade_low_mu:
                    start_idx = np.searchsorted(time_sec_arr, DECISION_WINDOW_SECONDS, side='right')
                    
                    signals, positions = backtest_core_low_mu(
                        time_sec_arr, 
                        price_arr, 
                        day_open_price,
                        int(start_idx),
                        True,
                        COOLDOWN_PERIOD_SECONDS,
                        L_MU_REVERSAL_RISE_TRIGGER,
                        float(active_params["entry_drop"]),
                        float(active_params["static_sl"]),
                        float(active_params["tp_activation"]),
                        float(active_params["trail_callback"]),
                        float(active_params["short_static_sl"]),
                        float(active_params["short_tp_activation"]),
                        float(active_params["short_trail_callback"]),
                        float(active_params["pre_drop_sl"])
                    )

            # --- BRANCH 2: HIGH MU (> 100) -> Runs Old "Strategy 2" Logic ---
            elif mu > MU_SPLIT_THRESHOLD:
                can_trade_high_mu = False
                h_mu_strat_mode = 0 # 1 or 2
                
                # Check Internal Sigma Logic for High Mu (Based on Strat 2 Code)
                if sigma > H_MU_SIGMA_HIGH_THRESHOLD:
                    # Strategy 1 High Vol
                    # Note: Strat 2 code checked mu > threshold here, which is already true
                    can_trade_high_mu = True
                    h_mu_strat_mode = 1
                    strat_type_label = "HighMu_Strat1"
                
                elif H_MU_SIGMA_MID_MIN <= sigma <= H_MU_SIGMA_MID_MAX:
                    # Strategy 2 Mid Vol
                    # Note: Original Strat 2 code checked mu < threshold here.
                    # Since we are in the Mu > 100 block, this branch effectively stays disabled
                    # unless strict parameter modification is allowed.
                    # Per instructions "dont change any conditions", we keep logic.
                    # If the original logic required Mu < 100, this won't run here.
                    if mu < MU_SPLIT_THRESHOLD: 
                         can_trade_high_mu = True
                         h_mu_strat_mode = 2
                         strat_type_label = "HighMu_Strat2"
                    else:
                        can_trade_high_mu = False
                
                if can_trade_high_mu:
                    start_idx = np.searchsorted(time_sec_arr, DECISION_WINDOW_SECONDS, side='right')
                    
                    if h_mu_strat_mode == 1:
                        signals, positions = backtest_high_mu_s1(
                            time_sec_arr, 
                            price_arr, 
                            day_open_price,
                            int(start_idx),
                            True,
                            COOLDOWN_PERIOD_SECONDS,
                            S1_ENTRY_RISE_THRESHOLD,
                            S1_PRE_RISE_SL
                        )
                    elif h_mu_strat_mode == 2:
                        signals, positions = backtest_high_mu_s2(
                            time_sec_arr, 
                            price_arr, 
                            day_open_price,
                            int(start_idx),
                            True,
                            COOLDOWN_PERIOD_SECONDS,
                            S2_ENTRY_DROP_THRESHOLD
                        )

        df["Signal"] = signals
        df["Position"] = positions
        df["Strat_Type"] = strat_type_label
        
        output_path = temp_dir / f"day{day_num}.csv"
        final_columns = REQUIRED_COLUMNS + ["Signal", "Position", "Day", "Strat_Type"]
        df[final_columns].to_csv(output_path, index=False)
        
        del df
        gc.collect()
        return str(output_path)

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main(directory: str, max_workers: int, strategy_params_cols):
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
            executor.submit(process_day, f, extract_day_num(f), temp_dir_path, strategy_params_cols): f
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

    main(args.directory, args.max_workers, STRATEGY_PARAMS_COLS)
