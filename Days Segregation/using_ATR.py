import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import jit

# =====================================================================
# CONFIGURATION
# =====================================================================

CONFIG = {
    'DATA_DIR': '/data/quant14/EBX',
    'NUM_DAYS': 510,
    'PRICE_COLUMN': 'Price',
    'VOLATILITY_FEATURE': 'PB9_T1',
    'TIME_COLUMN': 'Time',          # must exist in parquet
    'PRICE_JUMP_THRESHOLD': 0.3,    # price jump detection
    'MIN_TRADE_DURATION': 15,       # seconds
    'TRADE_COOLDOWN': 15,           # seconds
    'ATR_WINDOW': 600,
    'ATR_THRESHOLD': 0.01,
    'MAX_WORKERS': 25,              # number of CPU workers
}

# =====================================================================
# NUMBA UTILITIES
# =====================================================================

@jit(nopython=True, fastmath=True)
def find_price_jump_pairs_time(prices, times, jump_threshold, min_duration, cooldown):
    """
    Detect price jumps >= jump_threshold with:
    - trade duration >= min_duration seconds
    - cooldown of 'cooldown' seconds after trade execution
    """
    n = len(prices)
    pairs = []
    i = 0
    while i < n - 1:
        start_price = prices[i]
        start_time = times[i]
        found = False

        for j in range(i + 1, n):
            if abs(prices[j] - start_price) >= jump_threshold:
                trade_duration = times[j] - start_time
                if trade_duration >= min_duration:
                    pairs.append((i, j))
                    cooldown_target = times[j] + cooldown
                    k = j + 1
                    while k < n and times[k] < cooldown_target:
                        k += 1
                    i = k
                    found = True
                    break
        if not found:
            i += 1
    return pairs


# =====================================================================
# PROCESS SINGLE DAY
# =====================================================================

def process_single_day(day_num, config):
    try:
        file_path = Path(config['DATA_DIR']) / f"day{day_num}.parquet"
        if not file_path.exists():
            return {'day': day_num, 'success': False, 'reason': 'file_missing'}

        df = pd.read_parquet(file_path)
        price_col = config['PRICE_COLUMN']
        vol_col = config['VOLATILITY_FEATURE']
        time_col = config['TIME_COLUMN']

        # --- validation ---
        for col in [price_col, vol_col, time_col]:
            if col not in df.columns:
                return {'day': day_num, 'success': False, 'reason': f'missing_{col}'}

        df = df[[price_col, vol_col, time_col]].dropna().copy()
        if len(df) < config['ATR_WINDOW']:
            return {'day': day_num, 'success': False, 'reason': 'insufficient_data'}

        # --- convert time to seconds ---
        df[time_col] = pd.to_timedelta(df[time_col]).dt.total_seconds()

        prices = df[price_col].astype(float).values
        pb9_t1 = df[vol_col].astype(float).values
        times = df[time_col].astype(float).values

        # --- compute ATR proxy ---
        df['atr_proxy'] = (
            pd.Series(pb9_t1)
            .diff()
            .abs()
            .rolling(config['ATR_WINDOW'], min_periods=1)
            .mean()
            .fillna(0)
        )
        atr_values = df['atr_proxy'].values

        # --- find first ATR threshold cross ---
        atr_hit_index = np.argmax(atr_values >= config['ATR_THRESHOLD'])
        if atr_values[atr_hit_index] < config['ATR_THRESHOLD']:
            atr_hit_index = None

        # --- detect price jumps (time-aware) ---
        pairs = find_price_jump_pairs_time(
            prices,
            times,
            config['PRICE_JUMP_THRESHOLD'],
            config['MIN_TRADE_DURATION'],
            config['TRADE_COOLDOWN']
        )

        # --- classify before/after ATR regime ---
        missed, considered = 0, 0
        for (i, j) in pairs:
            if atr_hit_index is None or j < atr_hit_index:
                missed += 1
            else:
                considered += 1

        return {
            'day': day_num,
            'success': True,
            'atr_hit_index': int(atr_hit_index) if atr_hit_index is not None else -1,
            'atr_hit_value': float(atr_values[atr_hit_index]) if atr_hit_index is not None else 0.0,
            'considerable_pairs': considered,
            'missed_pairs': missed,
            'total_pairs': len(pairs),
            'price_min': float(prices.min()),
            'price_max': float(prices.max()),
        }

    except Exception as e:
        return {'day': day_num, 'success': False, 'error': str(e)}

# =====================================================================
# SUMMARY REPORT
# =====================================================================

def save_summary_to_file(all_results, config, output_path='ATR.txt'):
    valid = [r for r in all_results if r.get('success', False)]
    total_missed = sum(r['missed_pairs'] for r in valid)
    total_considered = sum(r['considerable_pairs'] for r in valid)
    total_pairs = sum(r['total_pairs'] for r in valid)
    days_with_hit = sum(1 for r in valid if r['atr_hit_index'] >= 0)
    days_without_hit = len(valid) - days_with_hit

    with open(output_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("PRICE FLUCTUATION (≥ 0.3) — TIME-FILTERED TRADES (≥ 15s duration + 15s cooldown)\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"ATR Window: {config['ATR_WINDOW']} | ATR Threshold: {config['ATR_THRESHOLD']}\n")
        f.write(f"Min Trade Duration: {config['MIN_TRADE_DURATION']}s | Cooldown: {config['TRADE_COOLDOWN']}s\n")
        f.write(f"Days Processed: {len(valid)} / {config['NUM_DAYS']}\n\n")

        # Section A: Active ATR regime days
        f.write("=" * 100 + "\n")
        f.write("SECTION A — Days where ATR ≥ Threshold (Active Regime)\n")
        f.write("=" * 100 + "\n")
        f.write("Day | ATR @Index | ATR Value | Considered | Missed | Total | Price Min | Price Max\n")
        f.write("----|-------------|------------|-------------|--------|--------|-----------|----------\n")
        for r in valid:
            if r['atr_hit_index'] >= 0:
                f.write(f"{r['day']:3d} | {r['atr_hit_index']:11d} | {r['atr_hit_value']:10.5f} | "
                        f"{r['considerable_pairs']:11d} | {r['missed_pairs']:6d} | {r['total_pairs']:6d} | "
                        f"{r['price_min']:9.2f} | {r['price_max']:9.2f}\n")

        f.write(f"\nDays with ATR Cross: {days_with_hit}\n")
        f.write(f"Total Considered Pairs: {total_considered}\n")
        f.write(f"Average Considered/Day: {total_considered / max(days_with_hit, 1):.2f}\n")
        f.write("\n" + "=" * 100 + "\n\n")

        # Section B: Non-active ATR days
        f.write("=" * 100 + "\n")
        f.write("SECTION B — Days where ATR < Threshold (No Active Regime)\n")
        f.write("=" * 100 + "\n")
        f.write("Day | ATR @Index | ATR Value | Considered | Missed | Total | Price Min | Price Max\n")
        f.write("----|-------------|------------|-------------|--------|--------|-----------|----------\n")
        for r in valid:
            if r['atr_hit_index'] == -1:
                f.write(f"{r['day']:3d} | {'N/A':>11} | {r['atr_hit_value']:10.5f} | "
                        f"{r['considerable_pairs']:11d} | {r['missed_pairs']:6d} | {r['total_pairs']:6d} | "
                        f"{r['price_min']:9.2f} | {r['price_max']:9.2f}\n")

        f.write(f"\nDays without ATR Cross: {days_without_hit}\n")
        f.write(f"Total Missed Pairs: {total_missed}\n")
        f.write(f"Average Missed/Day: {total_missed / max(days_without_hit, 1):.2f}\n")

        # Summary
        f.write("\n" + "=" * 100 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 100 + "\n")
        f.write(f"Total Pairs: {total_pairs}\n")
        f.write(f"Total Missed: {total_missed}\n")
        f.write(f"Total Considered: {total_considered}\n")
        f.write(f"Average Pairs/Day: {total_pairs / max(len(valid), 1):.2f}\n")
        f.write("=" * 100 + "\nEND OF REPORT\n")

    print(f"\n✓ Segmented Summary saved to: {output_path}")

# =====================================================================
# MAIN (PARALLEL)
# =====================================================================

def main():
    config = CONFIG
    print("=" * 100)
    print("TIME-AWARE PRICE FLUCTUATION ANALYSIS (≥ 0.3) — ATR FILTERED (PARALLEL 25 CPUs)")
    print("=" * 100)
    print(f"Data Directory: {config['DATA_DIR']}")
    print(f"ATR Window: {config['ATR_WINDOW']} | ATR Threshold: {config['ATR_THRESHOLD']}")
    print(f"Price Jump Threshold: {config['PRICE_JUMP_THRESHOLD']}")
    print(f"Trade Duration ≥ {config['MIN_TRADE_DURATION']}s | Cooldown = {config['TRADE_COOLDOWN']}s")
    print("=" * 100)

    all_results = []

    with ProcessPoolExecutor(max_workers=config['MAX_WORKERS']) as executor:
        futures = {executor.submit(process_single_day, day, config): day for day in range(config['NUM_DAYS'])}
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                res = future.result()
                if res['success']:
                    all_results.append(res)
            except Exception as e:
                print(f"Error on day {futures[future]}: {e}")
            if i % 25 == 0:
                print(f"  Processed {i}/{config['NUM_DAYS']} days...")

    print(f"\n✓ Completed {len(all_results)} days.")
    save_summary_to_file(all_results, config)
    print("\n✓ Analysis Complete!")

# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()
