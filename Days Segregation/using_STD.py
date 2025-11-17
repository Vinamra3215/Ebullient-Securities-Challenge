import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import jit

# ====================================================================
# CONFIGURATION
# ====================================================================

CONFIG = {
    'DATA_DIR': '/data/quant14/EBX',
    'NUM_DAYS': 510,
    'PRICE_COLUMN': 'Price',
    'PB9_COLUMN': 'PB9_T1',
    'TIME_COLUMN': 'Time',
    'PRICE_JUMP_THRESHOLD': 0.3,
    'MIN_TRADE_DURATION': 15,     # minimum 15 sec trade duration
    'TRADE_COOLDOWN': 15,         # cooldown 15 sec after trade
    'SMA_WINDOW': 20,            # SMA window
    'VOL_THRESHOLD': 0.06,        # volatility threshold
    'MAX_WORKERS': 25,            # parallel processing
}

# ====================================================================
# NUMBA UTILITIES
# ====================================================================

@jit(nopython=True, fastmath=True)
def find_price_jump_pairs_time(prices, times, jump_threshold, min_duration, cooldown):
    """
    Detect price jumps ≥ jump_threshold with:
    - duration >= min_duration seconds
    - cooldown period of 'cooldown' seconds after trade
    """
    n = len(prices)
    pairs = []
    last_trade_end_time = -1e9  # cooldown tracking

    for i in range(n - 1):
        start_price = prices[i]
        start_time = times[i]

        # Skip if within cooldown period
        if start_time < last_trade_end_time:
            continue

        for j in range(i + 1, n):
            price_diff = abs(prices[j] - start_price)
            time_diff = times[j] - start_time

            if price_diff >= jump_threshold and time_diff >= min_duration:
                pairs.append((i, j))
                last_trade_end_time = times[j] + cooldown  # enforce cooldown
                break

    return pairs


@jit(nopython=True, fastmath=True)
def calculate_vol_strength_numba(series, window):
    """
    Compute rolling volatility strength = (rolling std / SMA) * 100
    """
    n = len(series)
    vol_strength = np.zeros(n)

    for i in range(window - 1, n):
        w = series[i - window + 1:i + 1]
        sma = np.mean(w)
        std = np.std(w)
        if sma != 0:
            vol_strength[i] = (std / sma) * 100
    return vol_strength


# ====================================================================
# PROCESS SINGLE DAY
# ====================================================================

def process_single_day(day_num, config):
    try:
        file_path = Path(config['DATA_DIR']) / f"day{day_num}.parquet"
        if not file_path.exists():
            return {'day': day_num, 'success': False, 'reason': 'file_missing'}

        df = pd.read_parquet(file_path)
        cols = [config['PB9_COLUMN'], config['PRICE_COLUMN'], config['TIME_COLUMN']]
        for c in cols:
            if c not in df.columns:
                return {'day': day_num, 'success': False, 'reason': f'missing_{c}'}

        df = df[cols].dropna().copy()
        if len(df) < config['SMA_WINDOW']:
            return {'day': day_num, 'success': False, 'reason': 'insufficient_data'}

        # Convert time to seconds for Numba
        df[config['TIME_COLUMN']] = pd.to_timedelta(df[config['TIME_COLUMN']]).dt.total_seconds()

        prices = df[config['PRICE_COLUMN']].values.astype(np.float64)
        pb9_values = df[config['PB9_COLUMN']].values.astype(np.float64)
        times = df[config['TIME_COLUMN']].values.astype(np.float64)

        # Compute volatility strength
        vol_strength = calculate_vol_strength_numba(pb9_values, config['SMA_WINDOW'])
        vol_max = float(np.nanmax(vol_strength))
        threshold = config['VOL_THRESHOLD']

        # Find threshold crossing
        hit_index = np.argmax(vol_strength >= threshold)
        if vol_strength[hit_index] < threshold:
            hit_index = None

        # Detect time-filtered price jump pairs
        pairs = find_price_jump_pairs_time(
            prices, times,
            config['PRICE_JUMP_THRESHOLD'],
            config['MIN_TRADE_DURATION'],
            config['TRADE_COOLDOWN']
        )

        # Total fluctuation count (regardless of threshold)
        total_pairs = len(pairs)

        # Classify based on vol threshold
        missed, considered = 0, 0
        for (i, j) in pairs:
            if hit_index is not None and j >= hit_index:
                considered += 1
            else:
                missed += 1

        return {
            'day': day_num,
            'success': True,
            'vol_hit_index': int(hit_index) if hit_index is not None else -1,
            'vol_hit_value': float(vol_strength[hit_index]) if hit_index is not None else 0.0,
            'vol_max': vol_max,
            'considerable_pairs': considered,
            'missed_pairs': missed,
            'total_pairs': total_pairs,
            'price_min': float(np.min(prices)),
            'price_max': float(np.max(prices)),
        }

    except Exception as e:
        return {'day': day_num, 'success': False, 'error': str(e)}


# ====================================================================
# SUMMARY REPORT
# ====================================================================

def save_summary_to_file(all_results, config, output_path='STD.txt'):
    valid = [r for r in all_results if r.get('success', False)]
    if not valid:
        print("\n⚠️ No valid days processed — skipping report.")
        return

    hit = [r for r in valid if r['vol_hit_index'] >= 0]
    no_hit = [r for r in valid if r['vol_hit_index'] == -1]

    total_missed = sum(r['missed_pairs'] for r in valid)
    total_considered = sum(r['considerable_pairs'] for r in valid)
    total_pairs = sum(r['total_pairs'] for r in valid)

    with open(output_path, 'w') as f:
        f.write("=" * 110 + "\n")
        f.write("PB9_T1 — TIME FILTERED VOLATILITY (SMA-based Rolling Std) — ΔPrice ≥ 0.3\n")
        f.write("=" * 110 + "\n\n")
        f.write(f"Min Trade Duration: {config['MIN_TRADE_DURATION']} sec | Cooldown: {config['TRADE_COOLDOWN']} sec\n")
        f.write(f"Data Directory: {config['DATA_DIR']}\n")
        f.write(f"SMA Window: {config['SMA_WINDOW']}\n")
        f.write(f"Volatility Threshold: {config['VOL_THRESHOLD']}\n")
        f.write(f"Days Processed: {len(valid)} / {config['NUM_DAYS']}\n\n")

        # Section A — Days with threshold hit
        f.write("=" * 110 + "\n")
        f.write("SECTION A — Days where VOL_STRENGTH ≥ 0.06\n")
        f.write("=" * 110 + "\n")
        f.write("Day | Vol Hit @Index | Vol Value | Considered | Missed | Total | PB9_T1 Min | PB9_T1 Max\n")
        f.write("----|----------------|------------|-------------|--------|--------|-------------|-------------\n")
        for r in hit:
            f.write(f"{r['day']:3d} | {r['vol_hit_index']:14d} | {r['vol_hit_value']:10.3f} | "
                    f"{r['considerable_pairs']:11d} | {r['missed_pairs']:6d} | {r['total_pairs']:6d} | "
                    f"{r['price_min']:11.2f} | {r['price_max']:11.2f}\n")

        # Section B — No threshold hit
        f.write("\n" + "=" * 110 + "\n")
        f.write("SECTION B — Days where VOL_STRENGTH never reached 0.06\n")
        f.write("=" * 110 + "\n")
        f.write("Day | Vol Max | Missed | Total | PB9_T1 Min | PB9_T1 Max\n")
        f.write("----|----------|--------|--------|-------------|-------------\n")
        for r in no_hit:
            f.write(f"{r['day']:3d} | {r['vol_max']:8.3f} | {r['missed_pairs']:6d} | {r['total_pairs']:6d} | "
                    f"{r['price_min']:11.2f} | {r['price_max']:11.2f}\n")

        # Overall
        f.write("\n" + "=" * 110 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 110 + "\n")

        # Compute total considerable days & ratio
        total_considerable_days = sum(1 for r in valid if r['considerable_pairs'] > 0)
        ratio_considered = (
            total_considered / total_considerable_days
            if total_considerable_days > 0 else 0
        )

        f.write(f"Total Pairs Detected: {total_pairs}\n")
        f.write(f"Total Considered (After Vol Threshold): {total_considered}\n")
        f.write(f"Total Missed (Before Vol Threshold): {total_missed}\n")
        f.write(f"Total Considerable Days: {total_considerable_days}\n")
        f.write(f"Ratio (Considerable Pairs / Considerable Days): {ratio_considered:.3f}\n")
        f.write(f"Average Total Pairs/Day: {total_pairs / max(len(valid), 1):.2f}\n")
        f.write("=" * 110 + "\nEND OF REPORT\n" + "=" * 110 + "\n")


    print(f"\n✓ Summary saved to: {output_path}")


# ====================================================================
# MAIN
# ====================================================================

def main():
    config = CONFIG
    print("=" * 100)
    print("PB9_T1 TIME-FILTERED VOLATILITY ANALYSIS — ΔPrice ≥ 0.3")
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

    if not all_results:
        print("\n⚠️ No successful results. Please check file paths or data format.")
        return

    print(f"\n✓ Successfully processed {len(all_results)} valid days.")
    save_summary_to_file(all_results, config)
    print("\n✓ PB9_T1 Time-Filtered Volatility Analysis Complete!")


# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == '__main__':
    main()
