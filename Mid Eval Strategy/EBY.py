import os
import glob
import re
import shutil
import sys
import argparse
import pathlib
import pandas as pd
import numpy as np
import dask_cudf
import cudf
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Configuration ---
TEMP_DIR = "/data/quant14/signals/VB2_temp_signal_processing"
OUTPUT_FILE = "/data/quant14/signals/temp_trading_signals_EBX.csv"
COOLDOWN_PERIOD_SECONDS = 15
REQUIRED_COLUMNS = ['Time', 'Price']

# --- Strategy Config Example ---
STRATEGY_PARAMS = {
    "pb9t1": "PB9_T1",
}

# -------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------
def extract_day_num(filepath):
    """Extracts the day number 'n' from a filepath like '.../day{n}.parquet'."""
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1


# -------------------------------------------------------------------
# Rolling Feature Preparation (No Forward Bias)
# -------------------------------------------------------------------
def prepare_derived_features(df: pd.DataFrame, strategy_params: dict) -> pd.DataFrame:
    """
    Compute features WITHOUT forward bias.
    All indicators use .shift(1) to ensure we only use past data.
    """
    
    # --- KAMA Calculation ---
    price = df["PB9_T1"].to_numpy(dtype=float)
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
    
    # KAMA Slope - using shift to avoid forward bias
    df["KAMA_Slope"] = df["KAMA"].diff(2).shift(1)
    df["KAMA_Slope_abs"] = df["KAMA_Slope"].abs()

    # --- ATR Calculation ---
    tr = df['PB9_T1'].diff().abs()
    atr = tr.ewm(span=30, adjust=False).mean()
    df["ATR"] = atr.shift(1)  # Shift to avoid forward bias
    df["ATR_High"] = df["ATR"].expanding(min_periods=1).max()

    # --- Standard Deviation ---
    std = df["PB9_T1"].rolling(window=20, min_periods=1).std()
    df["STD"] = std.shift(1)  # Shift to avoid forward bias
    df["STD_High"] = df["STD"].expanding(min_periods=1).max()
    
    return df


# -------------------------------------------------------------------
# Strategy (No Forward Bias)
# -------------------------------------------------------------------
def generate_signal(row, position, cooldown_over, strategy_params):
    signal = 0
    
    # --- Thresholds ---
    ATR_THRESHOLD = 0.01
    STD_THRESHOLD = 0.06
    KAMA_SLOPE_ENTRY = 0.0007   # strong trend needed for entry

    if cooldown_over:
        # First check: ATR and STD thresholds must be met
        volatility_confirmed = (row["ATR_High"] > ATR_THRESHOLD and 
                               row["STD_High"] > STD_THRESHOLD)
        
        # --- Entry conditions (only when flat) ---
        if position == 0 and volatility_confirmed:
            # Generate signal based on KAMA slope direction
            if row["KAMA_Slope"] > KAMA_SLOPE_ENTRY:
                signal = 1   # Go long on strong upward slope
            elif row["KAMA_Slope"] < -KAMA_SLOPE_ENTRY:
                signal = -1  # Go short on strong downward slope
        
        # if position == 1 and volatility_confirmed:
        #     # Generate signal based on KAMA slope direction
        #     if row["KAMA_Slope"] <= 0.0:
        #         signal = -1   # Go long on strong upward slope
        # if position == -1 and volatility_confirmed:
        #     if row["KAMA_Slope"] >= 0.0:
        #         signal = 1  # Go short on strong downward slope

    return signal


# -------------------------------------------------------------------
# Core Processing
# -------------------------------------------------------------------
def process_day(file_path: str, day_num: int, temp_dir: pathlib.Path, strategy_params) -> str:
    """Processes a single day's parquet file and applies strategy logic."""
    try:
        columns = REQUIRED_COLUMNS + list(strategy_params.values())

        # --- Read Data ---
        ddf = dask_cudf.read_parquet(file_path, columns=columns)
        gdf = ddf.compute()
        df = gdf.to_pandas()

        if df.empty:
            print(f"Warning: Day {day_num} file is empty. Skipping.")
            return None

        # --- Prepare Data ---
        df = df.reset_index(drop=True)
        df['Time_sec'] = pd.to_timedelta(df['Time'].astype(str)).dt.total_seconds().astype(int)
        df = prepare_derived_features(df, strategy_params)

        # --- Initialize State ---
        position = 0
        entry_price = None
        last_signal_time = -COOLDOWN_PERIOD_SECONDS
        signals = [0] * len(df)
        positions = [0] * len(df)

        # --- Process Row by Row ---
        for i in range(len(df)):
            row = df.iloc[i]
            current_time = row['Time_sec']
            price = row['Price']
            is_last_tick = (i == len(df) - 1)

            signal = 0

            # --- EOD Square Off ---
            if is_last_tick and position != 0:
                signal = -position

            else:
                cooldown_over = (current_time - last_signal_time) >= COOLDOWN_PERIOD_SECONDS
                signal = generate_signal(row, position, cooldown_over, strategy_params)

                # --- Entry Logic ---
                if position == 0 and signal != 0:
                    entry_price = price  # record entry price

                # --- Exit Logic (Price-based confirmation) ---
                elif cooldown_over and position != 0:
                    if position == 1:
                        price_diff = price - entry_price
                        if price_diff >= 0.2:
                            signal = -1
                    elif position == -1:
                        price_diff = entry_price - price
                        if price_diff >= 0.2:
                            signal = 1

            # --- Apply Signal ---
            if signal != 0:
                if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                    signal = 0  # redundant signal
                else:
                    position += signal
                    last_signal_time = current_time
                    if position == 0:  # position closed
                        entry_price = None

            signals[i] = signal
            positions[i] = position

        # --- Store Results ---
        df['Signal'] = signals
        df['Position'] = positions

        output_path = temp_dir / f"day{day_num}.csv"
        final_columns = REQUIRED_COLUMNS + list(strategy_params.values()) + [
            'Signal', 'Position', 'KAMA', 'KAMA_Slope', 'KAMA_Slope_abs', 
            'ATR', 'ATR_High', 'STD', 'STD_High'
        ]
        df[final_columns].to_csv(output_path, index=False)

        return str(output_path)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main(directory: str, max_workers: int, strategy_params):
    start_time = time.time()

    temp_dir_path = pathlib.Path(TEMP_DIR)
    if temp_dir_path.exists():
        print(f"Removing existing temp directory: {temp_dir_path}")
        shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)

    print(f"Created temp directory: {temp_dir_path}")

    # --- Find Input Files ---
    files_pattern = os.path.join(directory, "day*.parquet")
    all_files = glob.glob(files_pattern)
    sorted_files = sorted(all_files, key=extract_day_num)

    if not sorted_files:
        print(f"No parquet files found in {directory}")
        shutil.rmtree(temp_dir_path)
        return

    print(f"Found {len(sorted_files)} day files.")

    processed_files = []
    try:
        # --- Parallel Execution ---
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_day, f, extract_day_num(f), temp_dir_path, strategy_params): f
                for f in sorted_files
            }

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    if result:
                        processed_files.append(result)
                        print(f"Processed: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"Error in {file_path}: {e}")

        # --- Combine Results ---
        if not processed_files:
            print("No files processed successfully.")
            return

        sorted_csvs = sorted(processed_files, key=lambda p: extract_day_num(p))
        with open(OUTPUT_FILE, 'wb') as outfile:
            for i, csv_file in enumerate(sorted_csvs):
                with open(csv_file, 'rb') as infile:
                    if i == 0:
                        shutil.copyfileobj(infile, outfile)
                    else:
                        infile.readline()
                        shutil.copyfileobj(infile, outfile)

        print(f"Created combined output: {OUTPUT_FILE}")

    finally:
        if temp_dir_path.exists():
            shutil.rmtree(temp_dir_path)

    print(f"Total time: {time.time() - start_time:.2f}s")


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process daily trading parquet files into signals.")
    parser.add_argument("directory", type=str, help="Directory containing 'day{n}.parquet' files.")
    parser.add_argument("--max-workers", type=int, default=os.cpu_count(), help="Parallel workers.")
    args = parser.parse_args()

    main(args.directory, args.max_workers, STRATEGY_PARAMS)