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
COOLDOWN_PERIOD_SECONDS = 5
MIN_TRADE_DURATION_SECONDS = 5  # <-- time-based minimum trade duration (method 1)
REQUIRED_COLUMNS = ['Time', 'Price']

# --- Strategy Config ---
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
    """Compute features WITHOUT forward bias. Assumes df rows are sorted by time."""
    # Ensure PB9_T1 exists
    if strategy_params["pb9t1"] not in df.columns:
        raise KeyError(f"Required feature {strategy_params['pb9t1']} not found in dataframe.")

    # --- KAMA Calculation ---
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
    # use lagged slope to avoid forward bias
    df["KAMA_Slope"] = df["KAMA"].diff(2).shift(1)
    df["KAMA_Slope_abs"] = df["KAMA_Slope"].abs()

    # --- ATR Calculation ---
    tr = df[strategy_params["pb9t1"]].diff().abs()
    atr = tr.ewm(span=30, adjust=False).mean()
    df["ATR"] = atr.shift(1)
    df["ATR_High"] = df["ATR"].expanding(min_periods=1).max()

    # --- STD Calculation ---
    std = df[strategy_params["pb9t1"]].rolling(window=120, min_periods=1).std()
    df["STD"] = std.shift(1)
    df["STD_High"] = df["STD"].expanding(min_periods=1).max()

    # --- HMA(9) + EMA(120) + SMA(180) for Trend Filter ---

    # --- EAMA (Adaptive EMA on Supersmoother) from your Feature Script ---
    ss = df["Price"].ewm(span=30, adjust=False).mean()  # light smoothing first

    eama_period = 15
    eama_fast = 30
    eama_slow = 120

    direction_ss = ss.diff(eama_period).abs()
    volatility_ss = ss.diff().abs().rolling(eama_period).sum()
    er_ss = (direction_ss / volatility_ss).fillna(0)

    fast_sc = 2/(eama_fast+1)
    slow_sc = 2/(eama_slow+1)
    sc_ss = ((er_ss*(fast_sc - slow_sc)) + slow_sc)**2

    eama = np.zeros(len(ss))
    eama[0] = ss.iloc[0]

    for i in range(1, len(ss)):
        eama[i] = eama[i-1] + sc_ss.iloc[i] * (ss.iloc[i] - eama[i-1])

    df["EAMA"] = eama
    df["EAMA_Slope"] = pd.Series(eama).diff().fillna(0)
    df["EAMA_Slope_MA"] = pd.Series(eama).rolling(5).mean().fillna(0)

    # --- EKFTrend (Extended Kalman Filter Trend) ---
    from filterpy.kalman import ExtendedKalmanFilter

    prices_np = df["Price"].values.astype(float)
    ekf = ExtendedKalmanFilter(dim_x=1, dim_z=1)
    ekf.x = np.array([[prices_np[0]]])
    ekf.F = np.array([[1.0]])
    ekf.Q = np.array([[0.05]])
    ekf.R = np.array([[0.2]])

    def h(x): 
        return np.log(x)

    def H_jac(x): 
        return np.array([[1.0 / x[0][0]]])

    ekf_vals = []
    for p in prices_np:
        ekf.predict()
        ekf.update(np.array([[np.log(p)]]), HJacobian=H_jac, Hx=h)
        ekf_vals.append(ekf.x.item())

    df["EKFTrend"] = pd.Series(ekf_vals).shift(1).bfill()


    def wma(arr, period):
        weights = np.arange(1, period + 1)
        return pd.Series(arr).rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    # HMA(9)
    wma9 = wma(df["Price"], 9)
    wma9_half = wma(df["Price"], 9//2)
    df["HMA_9"] = (2 * wma9_half - wma9).rolling(3).mean().shift(1)

    # EMA(120)
    df["EMA_120"] = df["Price"].ewm(span=130, adjust=False).mean().shift(1)

    # SMA(200)
    df["SMA_200"] = df["Price"].rolling(200).mean().shift(1)

    return df


# -------------------------------------------------------------------
# Signal Generation Logic
# -------------------------------------------------------------------
def generate_signal(row, position, cooldown_over, strategy_params):
    """
    Returns desired signal:
      0 = no action
      1 = open/close long (positive = open long or close short)
     -1 = open/close short (negative = open short or close long)
    Note: This function does not consider min-duration — gating is done in the main loop.
    """
    signal = 0

    ATR_THRESHOLD = 0.0
    STD_THRESHOLD = 0.08
    KAMA_SLOPE_ENTRY = 0.0008

    if cooldown_over:
        # Volatility confirmation
        volatility_confirmed = (row["ATR_High"] > ATR_THRESHOLD and
                                row["STD_High"] > STD_THRESHOLD)

        # Entry logic
        if position == 0 and volatility_confirmed:
            # --- Trend Alignment Check (HMA9 > EMA120 > SMA180) ---
            trend_long  = (row["HMA_9"] > row["EMA_120"] > row["SMA_200"])
            trend_short = (row["HMA_9"] < row["EMA_120"] < row["SMA_200"])

            # --- Combined KAMA + Trend Entry Logic ---
            if trend_long and row["KAMA_Slope"] > KAMA_SLOPE_ENTRY:
                signal = 1   # Open Long

            elif trend_short and row["KAMA_Slope"] < -KAMA_SLOPE_ENTRY:
                signal = -1  # Open Short

    return signal


# -------------------------------------------------------------------
# Per-Day Processing
# -------------------------------------------------------------------
def process_day(file_path: str, day_num: int, temp_dir: pathlib.Path, strategy_params) -> str:
    """Process a single day's parquet file and generate signals."""
    try:
        columns = REQUIRED_COLUMNS + list(strategy_params.values())
        ddf = dask_cudf.read_parquet(file_path, columns=columns)
        gdf = ddf.compute()
        df = gdf.to_pandas()

        if df.empty:
            print(f"⚠ Day {day_num} empty, skipping.")
            return None

        # Reset and ensure time ordering BEFORE feature calculation
        df = df.reset_index(drop=True)
        # Sort by Time to ensure rolling/EMA are causal and in correct chronological order
        df = df.sort_values("Time").reset_index(drop=True)

        # convert 'Time' to seconds from midnight (int)
        df["Time_sec"] = pd.to_timedelta(df["Time"].astype(str)).dt.total_seconds().astype(int)

        # compute derived features (assumes df sorted)
        df = prepare_derived_features(df, strategy_params)

        # ===== HEIKEN-ASHI (5s) PREPARATION =====
        # We'll create a copy for resampling so we don't lose row order/index in the main df.
        tmp = df[["Time", "Price"]].copy()
        tmp["Time_dt"] = pd.to_timedelta(tmp["Time"].astype(str))
        tmp = tmp.set_index("Time_dt")

        # Build normal OHLC 5s candles from second-based series
        ohlc = tmp["Price"].resample("5s").ohlc().dropna()

        # Build Heikin-Ashi from the OHLC
        ha = pd.DataFrame(index=ohlc.index)
        ha["close"] = (ohlc["open"] + ohlc["high"] + ohlc["low"] + ohlc["close"]) / 4

        ha["open"] = np.nan
        if len(ha) > 0:
            ha.iloc[0, ha.columns.get_loc("open")] = (ohlc["open"].iloc[0] + ohlc["close"].iloc[0]) / 2
            for i in range(1, len(ha)):
                ha.iloc[i, ha.columns.get_loc("open")] = (ha["open"].iloc[i-1] + ha["close"].iloc[i-1]) / 2

        # highs/lows of HA are the max/min of HA open/close and bar high/low
        ha["high"] = pd.concat([ha[["open", "close"]], ohlc["high"]], axis=1).max(axis=1)
        ha["low"]  = pd.concat([ha[["open", "close"]], ohlc["low"]], axis=1).min(axis=1)

        # Determine HA candle color
        ha["is_red"] = ha["close"] < ha["open"]
        ha["is_green"] = ha["close"] > ha["open"]

        # Forward-fill HA colors back to each second row of the main df.
        # Create array of timedeltas for each row in df
        df["Time_dt"] = pd.to_timedelta(df["Time"].astype(str))
        # Reindex HA series to the timestamps in df by forward filling (so each second gets the latest HA candle)
        # Use .reindex with the df Time_dt values (works because both are TimedeltaIndex-like)
        ha_is_red_reindexed = ha["is_red"].reindex(df["Time_dt"].values, method="ffill").astype(bool).values
        ha_is_green_reindexed = ha["is_green"].reindex(df["Time_dt"].values, method="ffill").astype(bool).values

        df["HA_red"] = ha_is_red_reindexed.astype(int)
        df["HA_green"] = ha_is_green_reindexed.astype(int)

        # drop temporary Time_dt column in tmp (we keep df["Time_dt"] for debugging if needed)
        # (We keep df in chronological order)

        # --- Prepare trading loop state ---
        position = 0
        entry_price = None
        entry_time_sec = None  # entry timestamp in seconds
        trailing_active = False
        trailing_price = None
        last_signal_time = -COOLDOWN_PERIOD_SECONDS
        signals = [0] * len(df)
        positions = [0] * len(df)

        # Parameters
        TAKE_PROFIT = 0.35   # activate trailing after this
        TRAIL_STOP = 0.15   # trail amount

        # HA counters for consecutive opposite candles
        red_count = 0
        green_count = 0

        for i in range(len(df)):
            row = df.iloc[i]
            current_time = int(row["Time_sec"])
            price = row["Price"]
            is_last_tick = (i == len(df) - 1)
            signal = 0

            # --- EOD Square-Off (forced) ---
            if is_last_tick and position != 0:
                # Force exit regardless of min-duration (must square off daily)
                signal = -position

            else:
                cooldown_over = (current_time - last_signal_time) >= COOLDOWN_PERIOD_SECONDS
                desired_signal = generate_signal(row, position, cooldown_over, strategy_params)

                # ENTRY handling
                if position == 0 and desired_signal != 0:
                    # Open position immediately (cooldown already checked by generate_signal)
                    signal = desired_signal

                # EXIT / TRAILING handling when in a position
                elif position != 0:
                    # calculate P&L distance in price units (since strategy uses prices)
                    price_diff = price - entry_price if position == 1 else entry_price - price

                    # Activate trailing stop after TAKE_PROFIT reached
                    if not trailing_active and price_diff >= TAKE_PROFIT:
                        trailing_active = True
                        trailing_price = price

                    # Manage trailing stop: this may produce an exit signal
                    if trailing_active:
                        if position == 1:
                            if price > trailing_price:
                                trailing_price = price  # move trail up
                            elif price <= trailing_price - TRAIL_STOP:
                                signal = -1  # exit long
                                trailing_active = False
                        elif position == -1:
                            if price < trailing_price:
                                trailing_price = price  # move trail down
                            elif price >= trailing_price + TRAIL_STOP:
                                signal = 1  # exit short
                                trailing_active = False

                    # # ===== EAMA + EKFTrend EXIT RULE =====
                    # # Trend confirmation exit
                    # if position == 1:  # LONG exit condition
                    #     if row["EKFTrend"] <= row["EAMA"]:
                    #         signal = -1   # exit long
                    #         trailing_active = False
                    #         red_count = 0
                    #         green_count = 0

                    # elif position == -1:  # SHORT exit condition
                    #     if row["EKFTrend"] >= row["EAMA"]:
                    #         signal = 1    # exit short
                    #         trailing_active = False
                    #         red_count = 0
                    #         green_count = 0


                    # ===== HEIKEN-ASHI EXIT RULE =====
                    # Count consecutive HA candles of opposite color (relative to position)
                    # Note: we examine the HA flag aligned to the current second (df["HA_red"/"HA_green"])
                    # if position == 1:          # LONG → check 4 red candles
                    #     if df["HA_red"].iloc[i] == 1:
                    #         red_count += 1
                    #     else:
                    #         red_count = 0

                    #     if red_count >= 10:
                    #         signal = -1  # exit long by HA rule
                    #         red_count = 0

                    # elif position == -1:       # SHORT → check 4 green candles
                    #     if df["HA_green"].iloc[i] == 1:
                    #         green_count += 1
                    #     else:
                    #         green_count = 0

                    #     if green_count >= 10:
                    #         signal = 1   # exit short by HA rule
                    #         green_count = 0

                    # Also honor opposite entry signal to flip position if cooldown allows
                    # Only allow flip if no HA exit already triggered (i.e., signal still 0)
                    if signal == 0 and desired_signal != 0 and desired_signal == -position:
                        # attempt to exit to flip
                        signal = -position

                # If not in position and no desired_entry, signal remains 0

                # --- Minimum duration gating (time-based) ---
                # If the computed signal would close a position, check min duration satisfied.
                if signal != 0 and position != 0:
                    # This is an exit (or flip). Check time since entry.
                    time_in_trade = current_time - (entry_time_sec if entry_time_sec is not None else -999999)
                    if time_in_trade < MIN_TRADE_DURATION_SECONDS:
                        # Block the exit until minimum duration met (except forced EOD handled above)
                        signal = 0

            # --- Apply Signal (after gating) ---
            if signal != 0:
                # Prevent duplicate same-side entries (e.g., repeated long entries)
                if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                    signal = 0
                else:
                    # Apply the signal
                    # If signal opens a new position from flat:
                    if position == 0 and signal != 0:
                        entry_price = price
                        entry_time_sec = int(current_time)
                        trailing_active = False
                        trailing_price = None

                    # If signal closes or flips:
                    if position != 0 and signal == -position:
                        # closing existing position
                        entry_price = None
                        entry_time_sec = None
                        trailing_price = None
                        trailing_active = False
                        # IMPORTANT: reset HA counters on close to avoid stale counts affecting next trade
                        red_count = 0
                        green_count = 0

                    # If flipping (e.g., long->short in single step) we treat as close then open:
                    # current code changes position by `position += signal`, so flipping happens automatically.
                    position += signal
                    last_signal_time = int(current_time)

                    # If after applying signal we're in a new non-zero position, set entry time if missing
                    if position != 0 and entry_time_sec is None:
                        entry_time_sec = int(current_time)
                        entry_price = price

                    # If we returned to flat, reset entry tracking
                    if position == 0:
                        entry_time_sec = None
                        entry_price = None
                        trailing_price = None
                        trailing_active = False

            signals[i] = signal
            positions[i] = position

        # --- Store Results ---
        df["Signal"] = signals
        df["Position"] = positions

        output_path = temp_dir / f"day{day_num}.csv"
        final_columns = REQUIRED_COLUMNS + list(strategy_params.values()) + [
            "Signal", "Position", "KAMA", "KAMA_Slope", "ATR", "STD", "HA_red", "HA_green"
        ]
        df[final_columns].to_csv(output_path, index=False)
        print(f"✅ Processed Day {day_num}")
        return str(output_path)

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KAMA Slope + Volatility Strategy with Heiken-Ashi exit")
    parser.add_argument("directory", type=str, help="Directory containing 'day{n}.parquet' files.")
    parser.add_argument("--max-workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    main(args.directory, args.max_workers, STRATEGY_PARAMS)
