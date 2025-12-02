import os
import glob
import re
import shutil
import sys
import argparse
import pathlib
import pandas as pd
import numpy as np
import cudf  # Replaced dask_cudf
import time
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from filterpy.kalman import ExtendedKalmanFilter

# --- Configuration ---
TEMP_DIR = "/data/quant14/signals/VB2_temp_signal_processing"
OUTPUT_FILE = "/data/quant14/signals/temp_trading_signals_EBX.csv"
COOLDOWN_PERIOD_SECONDS = 5
MIN_TRADE_DURATION_SECONDS = 5
REQUIRED_COLUMNS = ['Time', 'Price']

STRATEGY_PARAMS = {
    "pb9t1": "PB9_T1",
}

def extract_day_num(filepath):
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1

def prepare_derived_features(df: pd.DataFrame, strategy_params: dict) -> pd.DataFrame:
    if strategy_params["pb9t1"] not in df.columns:
        raise KeyError(f"Required feature {strategy_params['pb9t1']} not found.")

    price = df[strategy_params["pb9t1"]].to_numpy(dtype=float)
    window = 30

    kama_signal = np.abs(pd.Series(price).diff(window))
    kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window, min_periods=1).sum()
    er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()

    sc_fast = 2 / (60 + 1)
    sc_slow = 2 / (300 + 1)
    sc = ((er * (sc_fast - sc_slow)) + sc_slow) ** 2

    kama = np.full(len(price), np.nan)
    valid_start = np.where(~np.isnan(price))[0]
    if len(valid_start) == 0:
        raise ValueError("No valid price values found.")
    start = valid_start[0]
    kama[start] = price[start]

    for i in range(start + 1, len(price)):
        if np.isnan(price[i - 1]) or np.isnan(sc[i - 1]) or np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1]
        else:
            kama[i] = kama[i - 1] + sc[i - 1] * (price[i - 1] - kama[i - 1])

    df["KAMA"] = kama
    df["KAMA_Slope"] = df["KAMA"].diff(2).shift(1)
    df["KAMA_Slope_abs"] = df["KAMA_Slope"].abs()

    tr = df[strategy_params["pb9t1"]].diff().abs()
    atr = tr.ewm(span=30, adjust=False).mean()
    df["ATR"] = atr.shift(1)
    df["ATR_High"] = df["ATR"].expanding(min_periods=1).max()

    std = df[strategy_params["pb9t1"]].rolling(window=120, min_periods=1).std()
    df["STD"] = std.shift(1)
    df["STD_High"] = df["STD"].expanding(min_periods=1).max()

    def wma(arr, period):
        weights = np.arange(1, period + 1)
        return pd.Series(arr).rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    wma9 = wma(df["Price"], 9)
    wma9_half = wma(df["Price"], 9//2)
    df["HMA_9"] = (2 * wma9_half - wma9).rolling(3).mean().shift(1)
    df["EMA_130"] = df["Price"].ewm(span=130, adjust=False).mean().shift(1)
    df["SMA_240"] = df["Price"].rolling(240).mean().shift(1)

    return df

def generate_signal(row, position, cooldown_over, strategy_params):
    signal = 0
    ATR_THRESHOLD = 0.0
    STD_THRESHOLD = 0.08
    KAMA_SLOPE_ENTRY = 0.0008

    if cooldown_over:
        volatility_confirmed = (row["ATR_High"] > ATR_THRESHOLD and row["STD_High"] > STD_THRESHOLD)
        
        if position == 0 and volatility_confirmed:
            trend_long  = (row["HMA_9"] > row["EMA_130"] > row["SMA_240"])
            trend_short = (row["HMA_9"] < row["EMA_130"] < row["SMA_240"])

            if trend_long and row["KAMA_Slope"] > KAMA_SLOPE_ENTRY:
                signal = 1
            elif trend_short and row["KAMA_Slope"] < -KAMA_SLOPE_ENTRY:
                signal = -1

    return signal

def process_day(file_path: str, day_num: int, temp_dir: pathlib.Path, strategy_params) -> str:
    try:
        columns = REQUIRED_COLUMNS + list(strategy_params.values())
        
        # Use cudf directly to avoid Dask scheduler overhead
        gdf = cudf.read_parquet(file_path, columns=columns)
        df = gdf.to_pandas()
        
        # Immediate cleanup of GPU dataframe
        del gdf

        df = df.reset_index(drop=True)
        df = df.sort_values("Time").reset_index(drop=True)
        df["Time_sec"] = pd.to_timedelta(df["Time"].astype(str)).dt.total_seconds().astype(int)

        df = prepare_derived_features(df, strategy_params)

        # Trading Loop
        position = 0
        entry_price = None
        entry_time_sec = None
        trailing_active = False
        trailing_price = None
        last_signal_time = -COOLDOWN_PERIOD_SECONDS
        signals = [0] * len(df)
        positions = [0] * len(df)

        TAKE_PROFIT = 0.35
        TRAIL_STOP = 0.15

        for i in range(len(df)):
            row = df.iloc[i]
            current_time = int(row["Time_sec"])
            price = row["Price"]
            is_last_tick = (i == len(df) - 1)
            signal = 0

            if is_last_tick and position != 0:
                signal = -position
            else:
                cooldown_over = (current_time - last_signal_time) >= COOLDOWN_PERIOD_SECONDS
                desired_signal = generate_signal(row, position, cooldown_over, strategy_params)

                if position == 0 and desired_signal != 0:
                    signal = desired_signal
                elif position != 0:
                    price_diff = price - entry_price if position == 1 else entry_price - price

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
                    
                    if signal == 0 and desired_signal != 0 and desired_signal == -position:
                        signal = -position

                if signal != 0 and position != 0:
                    time_in_trade = current_time - (entry_time_sec if entry_time_sec is not None else -999999)
                    if time_in_trade < MIN_TRADE_DURATION_SECONDS:
                        signal = 0

            if signal != 0:
                if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                    signal = 0
                else:
                    if position == 0 and signal != 0:
                        entry_price = price
                        entry_time_sec = int(current_time)
                        trailing_active = False
                        trailing_price = None

                    if position != 0 and signal == -position:
                        entry_price = None
                        entry_time_sec = None
                        trailing_price = None
                        trailing_active = False

                    position += signal
                    last_signal_time = int(current_time)

                    if position != 0 and entry_time_sec is None:
                        entry_time_sec = int(current_time)
                        entry_price = price

                    if position == 0:
                        entry_time_sec = None
                        entry_price = None
                        trailing_price = None
                        trailing_active = False

            signals[i] = signal
            positions[i] = position

        df["Signal"] = signals
        df["Position"] = positions

        output_path = temp_dir / f"day{day_num}.csv"
        final_columns = REQUIRED_COLUMNS + list(strategy_params.values()) + [
            "Signal", "Position", "KAMA", "KAMA_Slope", "ATR", "STD"
        ]

        df[final_columns].to_csv(output_path, index=False)
        print(f"✅ Processed Day {day_num}")
        
        # Explicit GC
        del df
        gc.collect()
        
        return str(output_path)

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

def main(directory: str, max_workers: int, strategy_params):
    # Enforce spawn to prevent CUDA context corruption in children
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

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
                print(f"Error: {e}")

    if not processed_files:
        print("No files processed.")
        return

    processed_sorted = sorted(processed_files, key=extract_day_num)
    with open(OUTPUT_FILE, "wb") as out:
        for i, csv_file in enumerate(processed_sorted):
            with open(csv_file, "rb") as inp:
                if i == 0:
                    shutil.copyfileobj(inp, out)
                else:
                    inp.readline()
                    shutil.copyfileobj(inp, out)

    shutil.rmtree(temp_dir_path)
    print(f"✅ Output saved: {OUTPUT_FILE}")
    print(f"⏱ Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str)
    parser.add_argument("--max-workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    main(args.directory, args.max_workers, STRATEGY_PARAMS)