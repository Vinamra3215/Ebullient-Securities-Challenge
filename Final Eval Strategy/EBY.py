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
# NUMBA OPTIMIZED FUNCTIONS
# -------------------------------------------------------------------------

@njit(cache=True)
def calc_ekf_numba(prices):
    """
    Lightweight 1D Extended Kalman Filter implemented in Numba.
    Replaces filterpy for massive speedup.
    """
    n = len(prices)
    trend = np.zeros(n)
    
    # Initial State
    x = prices[0]
    P = 1.0
    Q = 0.05
    R = 0.2
    
    for i in range(n):
        # Predict
        # x = x (F=1)
        P = P + Q
        
        # Update
        z = np.log(prices[i])
        H = 1.0 / x
        y = z - np.log(x)
        S = H * P * H + R
        K = P * H / S
        x = x + K * y
        P = (1 - K * H) * P
        
        trend[i] = x
        
    return trend

@njit(cache=True)
def calculate_recursive_indicators_numba(price, sc, valid_start_idx):
    """
    Calculates recursive KAMA/EAMA logic.
    """
    n = len(price)
    out = np.full(n, np.nan)
    
    if valid_start_idx < n:
        out[valid_start_idx] = price[valid_start_idx]
        for i in range(valid_start_idx + 1, n):
            if np.isnan(price[i-1]) or np.isnan(sc[i-1]) or np.isnan(out[i-1]):
                out[i] = out[i-1]
            else:
                out[i] = out[i-1] + sc[i-1] * (price[i-1] - out[i-1])
    return out

@njit(cache=True)
def run_strategy_numba(
    time_sec, prices, 
    atr_high, std_high, 
    hma_9, ema_130, sma_240, 
    kama_slope, 
    cooldown_seconds, min_trade_duration
):
    """
    Core trading loop compiled to machine code.
    """
    n = len(prices)
    signals = np.zeros(n, dtype=int64)
    positions = np.zeros(n, dtype=int64)
    
    # Constants
    ATR_THRESHOLD = 0.0
    STD_THRESHOLD = 0.1
    KAMA_SLOPE_ENTRY = 0.0008
    TAKE_PROFIT = 0.35
    TRAIL_STOP = 0.15
    
    # State
    position = 0
    entry_price = 0.0
    entry_time = 0
    trailing_active = False
    trailing_price = 0.0
    last_signal_time = -cooldown_seconds - 1

    for i in range(n):
        current_time = time_sec[i]
        price = prices[i]
        is_last_tick = (i == n - 1)
        
        signal = 0
        
        # 1. Force Close at End
        if is_last_tick and position != 0:
            signal = -position
        else:
            # 2. Generate Base Signal
            cooldown_over = (current_time - last_signal_time) >= cooldown_seconds
            desired_signal = 0
            
            if cooldown_over:
                # Volatility Check
                if atr_high[i] > ATR_THRESHOLD and std_high[i] > STD_THRESHOLD:
                    if position == 0:
                        trend_long = (hma_9[i] > ema_130[i]) and (ema_130[i] > sma_240[i])
                        trend_short = (hma_9[i] < ema_130[i]) and (ema_130[i] < sma_240[i])
                        
                        if trend_long and kama_slope[i] > KAMA_SLOPE_ENTRY:
                            desired_signal = 1
                        elif trend_short and kama_slope[i] < -KAMA_SLOPE_ENTRY:
                            desired_signal = -1

            # 3. Manage Position Logic
            if position == 0 and desired_signal != 0:
                signal = desired_signal
            elif position != 0:
                # Calculate Price Diff
                if position == 1:
                    price_diff = price - entry_price
                else:
                    price_diff = entry_price - price
                
                # Activate Trailing
                if not trailing_active and price_diff >= TAKE_PROFIT:
                    trailing_active = True
                    trailing_price = price
                
                # Check Trailing Stop
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
                
                # Reverse Logic
                if signal == 0 and desired_signal != 0 and desired_signal == -position:
                    signal = -position

            # 4. Minimum Duration Check
            if signal != 0 and position != 0:
                time_in_trade = current_time - entry_time
                if time_in_trade < min_trade_duration:
                    signal = 0

        # 5. Apply Signal & Update State
        if signal != 0:
            if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                signal = 0
            else:
                # Update Entry/Exit Metadata
                if position == 0 and signal != 0:
                    entry_price = price
                    entry_time = current_time
                    trailing_active = False
                    trailing_price = 0.0
                
                if position != 0 and signal == -position:
                    entry_price = 0.0
                    entry_time = 0
                    trailing_price = 0.0
                    trailing_active = False

                position += signal
                last_signal_time = current_time
                
                # Handle Flip (Short -> Long or Long -> Short) directly
                if position != 0 and entry_time == 0:
                    entry_time = current_time
                    entry_price = price

                if position == 0:
                    entry_time = 0
                    entry_price = 0.0
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

def prepare_derived_features_cpu(df: pd.DataFrame, strategy_params: dict) -> pd.DataFrame:
    col = strategy_params["pb9t1"]
    if col not in df.columns:
        raise KeyError(f"Required feature {col} not found.")

    price = df[col].to_numpy(dtype=float)
    
    # --- KAMA (Vectorized Parts) ---
    window = 30
    kama_signal = np.abs(pd.Series(price).diff(window))
    kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window, min_periods=1).sum()
    er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()
    
    sc_fast = 2 / (60 + 1)
    sc_slow = 2 / (300 + 1)
    sc = ((er * (sc_fast - sc_slow)) + sc_slow) ** 2

    # --- KAMA (Numba Part) ---
    valid_start = np.where(~np.isnan(price))[0]
    start_idx = valid_start[0] if len(valid_start) > 0 else 0
    df["KAMA"] = calculate_recursive_indicators_numba(price, sc, start_idx)
    
    df["KAMA_Slope"] = df["KAMA"].diff(2).shift(1)
    df["KAMA_Slope_abs"] = df["KAMA_Slope"].abs()

    # --- ATR & STD ---
    tr = df[col].diff().abs()
    atr = tr.ewm(span=30, adjust=False).mean()
    df["ATR"] = atr.shift(1)
    df["ATR_High"] = df["ATR"].expanding(min_periods=1).max()

    std = df[col].rolling(window=120, min_periods=1).std()
    df["STD"] = std.shift(1)
    df["STD_High"] = df["STD"].expanding(min_periods=1).max()

    # --- EAMA ---
    ss = df["Price"].ewm(span=30, adjust=False).mean()
    eama_period = 15
    eama_fast = 30
    eama_slow = 120

    direction_ss = ss.diff(eama_period).abs()
    volatility_ss = ss.diff().abs().rolling(eama_period).sum()
    er_ss = (direction_ss / volatility_ss).fillna(0)

    fast_sc = 2/(eama_fast+1)
    slow_sc = 2/(eama_slow+1)
    sc_ss = ((er_ss*(fast_sc - slow_sc)) + slow_sc)**2
    
    # Reuse the Numba function for EAMA recursion
    # Note: eama starts at ss[0], so we pass ss values and sc_ss
    ss_vals = ss.to_numpy()
    sc_ss_vals = sc_ss.to_numpy()
    df["EAMA"] = calculate_recursive_indicators_numba(ss_vals, sc_ss_vals, 0)
    
    df["EAMA_Slope"] = df["EAMA"].diff().fillna(0)
    df["EAMA_Slope_MA"] = df["EAMA"].rolling(5).mean().fillna(0)

    # --- Extended Kalman Filter (Numba) ---
    # Replaces the slow loop + filterpy
    df["EKFTrend"] = pd.Series(calc_ekf_numba(price)).shift(1).bfill()

    # --- Moving Averages ---
    def wma(series, period):
        return series.rolling(period).apply(
            lambda x: np.dot(x, np.arange(1, period + 1)) / np.arange(1, period + 1).sum(), 
            raw=True
        )

    wma9 = wma(df["Price"], 9)
    wma9_half = wma(df["Price"], 9//2)
    df["HMA_9"] = (2 * wma9_half - wma9).rolling(3).mean().shift(1)
    df["EMA_130"] = df["Price"].ewm(span=120, adjust=False).mean().shift(1)
    df["SMA_240"] = df["Price"].rolling(220).mean().shift(1)

    return df

def process_day(args):
    """
    Worker function for multiprocessing.
    """
    file_path, day_num, temp_dir, strategy_params = args
    
    try:
        columns = REQUIRED_COLUMNS + list(strategy_params.values())
        
        # CPU Load
        df = pd.read_parquet(file_path, columns=columns)
        
        if df.empty:
            return None

        df = df.reset_index(drop=True)
        df = df.sort_values("Time").reset_index(drop=True)
        df["Time_sec"] = pd.to_timedelta(df["Time"].astype(str)).dt.total_seconds().astype(int)

        # 1. Feature Engineering (CPU + Numba)
        df = prepare_derived_features_cpu(df, strategy_params)

        # 2. Heiken-Ashi (Pandas Vectorized)
        tmp = df[["Time", "Price"]].copy()
        tmp["Time_dt"] = pd.to_timedelta(tmp["Time"].astype(str))
        tmp = tmp.set_index("Time_dt")
        ohlc = tmp["Price"].resample("5s").ohlc().dropna()
        
        if len(ohlc) > 0:
            ha_close = (ohlc["open"] + ohlc["high"] + ohlc["low"] + ohlc["close"]) / 4
            ha_open = np.zeros(len(ohlc))
            ha_open[0] = (ohlc["open"].iloc[0] + ohlc["close"].iloc[0]) / 2
            
            # Simple loop for HA Open (fast enough in python for small resampled data, or use numba)
            # Keeping python for simplicity as it's resampled (small size)
            for i in range(1, len(ohlc)):
                ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2
                
            ha_is_red = ha_close < ha_open
            ha_is_green = ha_close > ha_open
            
            # Reindex back to original timeframe
            # Note: We use the index (Time) to align
            full_idx = df["Time"].astype(str).apply(pd.to_timedelta)
            
            # Create Series for reindexing
            s_red = pd.Series(ha_is_red.values, index=ohlc.index)
            s_green = pd.Series(ha_is_green.values, index=ohlc.index)
            
            df["HA_red"] = s_red.reindex(full_idx, method="ffill").fillna(False).astype(int).values
            df["HA_green"] = s_green.reindex(full_idx, method="ffill").fillna(False).astype(int).values
        else:
            df["HA_red"] = 0
            df["HA_green"] = 0

        # 3. Trading Strategy (Numba Optimized)
        # Prepare numpy arrays for Numba
        time_sec = df["Time_sec"].values.astype(np.int64)
        prices = df["Price"].values.astype(np.float64)
        atr_high = df["ATR_High"].fillna(0).values.astype(np.float64)
        std_high = df["STD_High"].fillna(0).values.astype(np.float64)
        hma_9 = df["HMA_9"].fillna(0).values.astype(np.float64)
        ema_130 = df["EMA_130"].fillna(0).values.astype(np.float64)
        sma_240 = df["SMA_240"].fillna(0).values.astype(np.float64)
        kama_slope = df["KAMA_Slope"].fillna(0).values.astype(np.float64)
        
        signals, positions = run_strategy_numba(
            time_sec, prices, atr_high, std_high, 
            hma_9, ema_130, sma_240, kama_slope,
            COOLDOWN_PERIOD_SECONDS, MIN_TRADE_DURATION_SECONDS
        )

        df["Signal"] = signals
        df["Position"] = positions

        # 4. Output
        output_path = temp_dir / f"day{day_num}.csv"
        final_columns = REQUIRED_COLUMNS + list(strategy_params.values()) + [
            "Signal", "Position", "KAMA", "KAMA_Slope", "ATR", "STD", "HA_red", "HA_green"
        ]
        
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
    start_time = time.time()
    temp_dir_path = pathlib.Path(TEMP_DIR)
    
    if temp_dir_path.exists():
        shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)

    print(f"Created temp directory: {temp_dir_path}")

    files = sorted(glob.glob(os.path.join(directory, "day*.parquet")), key=extract_day_num)
    if not files:
        print("No parquet files found.")
        return

    # Prepare args for Pool
    task_args = [(f, extract_day_num(f), temp_dir_path, strategy_params) for f in files]
    processed_files = []

    print(f"Starting multiprocessing with {max_workers} workers...")
    
    # Use multiprocessing.Pool
    with mp.Pool(processes=max_workers) as pool:
        # imap_unordered is generally faster for pipeline processing
        for res in pool.imap_unordered(process_day, task_args):
            if res:
                processed_files.append(res)
                print(f"✅ Processed: {os.path.basename(res)}")

    if not processed_files:
        print("No files processed.")
        return

    print("Merging files...")
    processed_sorted = sorted(processed_files, key=extract_day_num)
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

    # Numba JIT warmup (optional, but good practice to avoid first-task latency)
    print("Compiling Numba functions...")
    # Dummy call to trigger compilation
    _ = calc_ekf_numba(np.array([100.0, 101.0], dtype=np.float64))
    
    main(args.directory, args.max_workers, STRATEGY_PARAMS)