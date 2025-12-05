import os
import glob
import re
import shutil
import sys
import argparse
import pathlib
import time
import gc
import multiprocessing as mp
import pandas as pd
import numpy as np
from numba import njit, float64, int64

# --- Configuration ---
TEMP_DIR = "/data/quant14/signals/VB2_temp_signal_processing"
OUTPUT_FILE = "/data/quant14/signals/temp_trading_signals_EBX.csv"
COOLDOWN_PERIOD_SECONDS = 5
MIN_TRADE_DURATION_SECONDS = 5
REQUIRED_COLUMNS = ['Time', 'Price']

STRATEGY_PARAMS = {
    "pb9t1": "PB9_T1",
}

# -------------------------------------------------------------------------
# NUMBA OPTIMIZED FUNCTIONS (CPU SPEEDUP)
# -------------------------------------------------------------------------

@njit(float64[:](float64[:], float64[:], int64))
def calculate_kama_numba(price, sc, valid_start_idx):
    """
    Numba optimized recursive KAMA calculation.
    """
    n = len(price)
    kama = np.full(n, np.nan)
    
    # Initialize at the first valid index
    if valid_start_idx < n:
        kama[valid_start_idx] = price[valid_start_idx]
        
        for i in range(valid_start_idx + 1, n):
            # Check for NaNs in inputs to avoid propagation errors
            if np.isnan(price[i-1]) or np.isnan(sc[i-1]) or np.isnan(kama[i-1]):
                kama[i] = kama[i-1]
            else:
                kama[i] = kama[i-1] + sc[i-1] * (price[i-1] - kama[i-1])
                
    return kama

@njit(parallel=False)
def backtest_core_numba(
    time_sec, prices, 
    atr_high, std_high, 
    hma_9, ema_130, sma_240, 
    kama_slope, 
    atr_threshold, std_threshold, kama_slope_entry,
    cooldown_seconds, min_duration_seconds
):
    """
    Core trading logic compiled to machine code for extreme CPU speed.
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int64)
    positions = np.zeros(n, dtype=np.int64)
    
    position = 0
    entry_price = 0.0
    entry_time = 0
    trailing_active = False
    trailing_price = 0.0
    last_signal_time = -cooldown_seconds
    
    TAKE_PROFIT = 0.35
    TRAIL_STOP = 0.15

    for i in range(n):
        current_time = time_sec[i]
        price = prices[i]
        is_last_tick = (i == n - 1)
        signal = 0
        
        # 1. Generate Logic Signal
        if is_last_tick and position != 0:
            signal = -position
        else:
            cooldown_over = (current_time - last_signal_time) >= cooldown_seconds
            
            desired_signal = 0
            # Entry Logic
            if cooldown_over:
                volatility_confirmed = (atr_high[i] > atr_threshold) and (std_high[i] > std_threshold)
                
                if position == 0 and volatility_confirmed:
                    trend_long = (hma_9[i] > ema_130[i]) and (ema_130[i] > sma_240[i])
                    trend_short = (hma_9[i] < ema_130[i]) and (ema_130[i] < sma_240[i])
                    
                    if trend_long and kama_slope[i] > kama_slope_entry:
                        desired_signal = 1
                    elif trend_short and kama_slope[i] < -kama_slope_entry:
                        desired_signal = -1

            # Manage Signal vs Position
            if position == 0 and desired_signal != 0:
                signal = desired_signal
            elif position != 0:
                # Exit / Trailing Logic
                price_diff = 0.0
                if position == 1:
                    price_diff = price - entry_price
                else:
                    price_diff = entry_price - price
                
                # Activate Trailing
                if not trailing_active and price_diff >= TAKE_PROFIT:
                    trailing_active = True
                    trailing_price = price
                
                if trailing_active:
                    if position == 1:
                        if price > trailing_price:
                            trailing_price = price
                        elif price <= trailing_price - TRAIL_STOP:
                            signal = -1
                            trailing_active = False
                    elif position == -1:
                        if price < trailing_price:
                            trailing_price = price
                        elif price >= trailing_price + TRAIL_STOP:
                            signal = 1
                            trailing_active = False
                
                # Reverse Logic (if strategy allows flipping)
                if signal == 0 and desired_signal != 0 and desired_signal == -position:
                    signal = -position

            # Minimum Duration Check
            if signal != 0 and position != 0:
                time_in_trade = current_time - entry_time
                if time_in_trade < min_duration_seconds:
                    signal = 0
        
        # 2. Update State
        if signal != 0:
            # Prevent redundant signals
            if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                signal = 0
            else:
                # Entering new trade
                if position == 0 and signal != 0:
                    entry_price = price
                    entry_time = current_time
                    trailing_active = False
                    trailing_price = 0.0
                
                # Closing trade
                if position != 0 and signal == -position:
                    entry_price = 0.0
                    entry_time = 0
                    trailing_price = 0.0
                    trailing_active = False
                
                position += signal
                last_signal_time = current_time
                
                # Flipping (Short to Long or Long to Short)
                if position != 0 and entry_time == 0:
                    entry_time = current_time
                    entry_price = price

                # Reset if flat
                if position == 0:
                    entry_time = 0
                    entry_price = 0.0
                    trailing_price = 0.0
                    trailing_active = False

        signals[i] = signal
        positions[i] = position

    return signals, positions

# -------------------------------------------------------------------------
# DATA PROCESSING
# -------------------------------------------------------------------------

def extract_day_num(filepath):
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1

def extract_day_num_csv(filepath):
    match = re.search(r'day(\d+)\.csv', str(filepath)) # Fixed regex for CSV in merge step
    return int(match.group(1)) if match else -1

def prepare_derived_features_cpu(df: pd.DataFrame, strategy_params: dict) -> pd.DataFrame:
    col_name = strategy_params["pb9t1"]
    if col_name not in df.columns:
        raise KeyError(f"Required feature {col_name} not found.")

    price = df[col_name].to_numpy(dtype=float)
    window = 30

    # Pandas/Numpy for vectorized operations
    kama_signal = np.abs(pd.Series(price).diff(window))
    kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window, min_periods=1).sum()
    er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()

    sc_fast = 2 / (60 + 1)
    sc_slow = 2 / (300 + 1)
    sc = ((er * (sc_fast - sc_slow)) + sc_slow) ** 2

    # --- Numba Optimization for Recursive KAMA ---
    valid_start = np.where(~np.isnan(price))[0]
    if len(valid_start) == 0:
        raise ValueError("No valid price values found.")
    
    # Call Numba function
    kama = calculate_kama_numba(price, sc, valid_start[0])

    df["KAMA"] = kama
    df["KAMA_Slope"] = df["KAMA"].diff(2).shift(1)
    
    # Calculate indicators using Pandas (fast enough for vector ops)
    tr = df[col_name].diff().abs()
    atr = tr.ewm(span=30, adjust=False).mean()
    df["ATR"] = atr.shift(1)
    df["ATR_High"] = df["ATR"].expanding(min_periods=1).max()

    std = df[col_name].rolling(window=120, min_periods=1).std()
    df["STD"] = std.shift(1)
    df["STD_High"] = df["STD"].expanding(min_periods=1).max()

    def wma(series, period):
        return series.rolling(period).apply(
            lambda x: np.dot(x, np.arange(1, period + 1)) / np.arange(1, period + 1).sum(), 
            raw=True
        )

    # Standard Rolling features
    wma9 = wma(df["Price"], 9)
    wma9_half = wma(df["Price"], 9//2)
    df["HMA_9"] = (2 * wma9_half - wma9).rolling(3).mean().shift(1)
    df["EMA_130"] = df["Price"].ewm(span=130, adjust=False).mean().shift(1)
    df["SMA_240"] = df["Price"].rolling(240).mean().shift(1)

    return df

def process_day(args):
    """
    Worker function to process a single day.
    Unpacks args to be compatible with mp.Pool.map/imap.
    """
    file_path, day_num, temp_dir, strategy_params = args
    
    try:
        columns = REQUIRED_COLUMNS + list(strategy_params.values())
        
        # CPU Processing: Read with Pandas
        df = pd.read_parquet(file_path, columns=columns)
        
        df = df.reset_index(drop=True)
        df = df.sort_values("Time").reset_index(drop=True)
        df["Time_sec"] = pd.to_timedelta(df["Time"].astype(str)).dt.total_seconds().astype(int)

        # 1. Feature Engineering
        df = prepare_derived_features_cpu(df, strategy_params)

        # 2. Prepare Data for Numba (Handle NaNs)
        # Numba needs clean float arrays. We fill NaNs with 0 for logic comparisons
        # assuming the logic handles 0s correctly (e.g., price > 0).
        time_sec = df["Time_sec"].values.astype(np.int64)
        prices = df["Price"].values.astype(np.float64)
        atr_high = df["ATR_High"].fillna(0).values.astype(np.float64)
        std_high = df["STD_High"].fillna(0).values.astype(np.float64)
        hma_9 = df["HMA_9"].fillna(0).values.astype(np.float64)
        ema_130 = df["EMA_130"].fillna(0).values.astype(np.float64)
        sma_240 = df["SMA_240"].fillna(0).values.astype(np.float64)
        kama_slope = df["KAMA_Slope"].fillna(0).values.astype(np.float64)

        # Constants
        ATR_THRESHOLD = 0.0
        STD_THRESHOLD = 0.08
        KAMA_SLOPE_ENTRY = 0.0008

        # 3. Run Numba Backtest
        signals, positions = backtest_core_numba(
            time_sec, prices, 
            atr_high, std_high, 
            hma_9, ema_130, sma_240, 
            kama_slope,
            ATR_THRESHOLD, STD_THRESHOLD, KAMA_SLOPE_ENTRY,
            COOLDOWN_PERIOD_SECONDS, MIN_TRADE_DURATION_SECONDS
        )

        df["Signal"] = signals
        df["Position"] = positions

        # 4. Save Output
        output_path = temp_dir / f"day{day_num}.csv"
        final_columns = REQUIRED_COLUMNS + list(strategy_params.values()) + [
            "Signal", "Position", "KAMA", "KAMA_Slope", "ATR", "STD"
        ]

        df[final_columns].to_csv(output_path, index=False)
        
        # Cleanup
        del df
        del signals
        del positions
        gc.collect()
        
        return str(output_path)

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main(directory: str, max_workers: int, strategy_params):
    start_time = time.time()
    temp_dir_path = pathlib.Path(TEMP_DIR)
    
    if temp_dir_path.exists():
        shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)

    print(f"Created temp directory: {temp_dir_path}")
    print(f"Using CPU with {max_workers} workers.")

    # 1. Get all files
    all_files = sorted(glob.glob(os.path.join(directory, "day*.parquet")), key=extract_day_num)
    
    if not all_files:
        print("No parquet files found.")
        return

    # 2. Apply Blacklist (Day 60 to 79)
    files = []
    skipped_count = 0
    for f in all_files:
        day_num = extract_day_num(f)
        if 60 <= day_num <= 79:
            skipped_count += 1
            continue
        files.append(f)

    print(f"Found {len(all_files)} files. Blacklisted {skipped_count} files (Day 60-79). Processing {len(files)} files.")

    if not files:
        print("No files left to process after blacklisting.")
        return

    # Prepare arguments for multiprocessing
    task_args = [
        (f, extract_day_num(f), temp_dir_path, strategy_params) 
        for f in files
    ]

    processed_files = []
    
    # Using multiprocessing.Pool
    with mp.Pool(processes=max_workers) as pool:
        for result in pool.imap_unordered(process_day, task_args):
            if result:
                processed_files.append(result)
                print(f"✅ Processed file: {pathlib.Path(result).name}")

    if not processed_files:
        print("No files processed.")
        return

    print("Merging files...")
    # Note: merge extracts day num from CSV filenames now
    processed_sorted = sorted(processed_files, key=extract_day_num_csv)
    
    with open(OUTPUT_FILE, "wb") as out:
        for i, csv_file in enumerate(processed_sorted):
            with open(csv_file, "rb") as inp:
                if i == 0:
                    shutil.copyfileobj(inp, out)
                else:
                    inp.readline() # Skip header
                    shutil.copyfileobj(inp, out)

    shutil.rmtree(temp_dir_path)
    print(f"✅ Output saved: {OUTPUT_FILE}")
    print(f"⏱ Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str)
    parser.add_argument("--max-workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    # Numba compilation happens on first call, so we might see a slight delay on the very first file.
    main(args.directory, args.max_workers, STRATEGY_PARAMS)