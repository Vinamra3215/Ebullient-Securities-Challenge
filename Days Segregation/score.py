import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import jit

# =====================================================================
# CONFIGURATION
# =====================================================================

CONFIG = {
    'DATA_DIR': '/data/quant14/EBY',
    'NUM_DAYS': 279,
    'PRICE_COLUMN': 'Price',
    'VOLATILITY_FEATURE': 'PB9_T1',
    'PRICE_JUMP_THRESHOLD': 0.3,
    'ATR_WINDOW': 600,
    'ATR_THRESHOLD': 0.01,
    'SMA_WINDOW': 600,
    'VOL_THRESHOLD': 0.06,
    'PRICE_RANGE_THRESHOLD': 0.7,
    'CONSIDERABLE_TRADE_LIMIT': 8,
    'MAX_WORKERS': 25,
}

# =====================================================================
# NUMBA-ACCELERATED UTILITIES
# =====================================================================

@jit(nopython=True, fastmath=True)
def find_price_jump_pairs(prices, jump_threshold):
    n = len(prices)
    pairs = []
    i = 0
    while i < n:
        start = prices[i]
        for j in range(i + 1, n):
            if abs(prices[j] - start) >= jump_threshold:
                pairs.append((i, j))
                i = j + 1
                break
        else:
            i += 1
    return pairs

@jit(nopython=True, fastmath=True)
def calculate_vol_strength(series, window):
    n = len(series)
    vol_strength = np.zeros(n)
    for i in range(window - 1, n):
        window_slice = series[i - window + 1:i + 1]
        sma = np.mean(window_slice)
        std = np.std(window_slice)
        if sma != 0:
            vol_strength[i] = (std / sma) * 100
    return vol_strength

# =====================================================================
# DAILY PROCESSING
# =====================================================================

def process_single_day(day_num, config):
    try:
        file_path = Path(config['DATA_DIR']) / f"day{day_num}.parquet"
        if not file_path.exists():
            return {'day': day_num, 'success': False, 'reason': 'file_missing'}

        df = pd.read_parquet(file_path)
        price_col, vol_col = config['PRICE_COLUMN'], config['VOLATILITY_FEATURE']

        if price_col not in df.columns or vol_col not in df.columns:
            return {'day': day_num, 'success': False, 'reason': 'missing_columns'}

        df = df[[price_col, vol_col]].dropna().copy()
        if len(df) < config['ATR_WINDOW']:
            return {'day': day_num, 'success': False, 'reason': 'insufficient_data'}

        prices = df[price_col].astype(float).values
        pb9_t1 = df[vol_col].astype(float).values

        # --- ATR proxy ---
        df['atr_proxy'] = (
            pd.Series(pb9_t1)
            .diff()
            .abs()
            .rolling(config['ATR_WINDOW'], min_periods=1)
            .mean()
            .fillna(0)
        )
        atr_values = df['atr_proxy'].values
        atr_hit_index = np.argmax(atr_values >= config['ATR_THRESHOLD'])
        if atr_values[atr_hit_index] < config['ATR_THRESHOLD']:
            atr_hit_index = None

        # --- Volatility Strength ---
        vol_strength = calculate_vol_strength(pb9_t1, config['SMA_WINDOW'])
        vol_hit_index = np.argmax(vol_strength >= config['VOL_THRESHOLD'])
        if vol_strength[vol_hit_index] < config['VOL_THRESHOLD']:
            vol_hit_index = None

        # --- Price jumps ---
        pairs = find_price_jump_pairs(prices, config['PRICE_JUMP_THRESHOLD'])
        missed, considered = 0, 0
        for (i, j) in pairs:
            if atr_hit_index is None or j < atr_hit_index:
                missed += 1
            else:
                considered += 1

        # --- Scores ---
        price_range = float(prices.max() - prices.min())
        score_a = 1 if (atr_hit_index is not None and considered > config['CONSIDERABLE_TRADE_LIMIT']) else 0
        score_b = 1 if (vol_hit_index is not None and considered > config['CONSIDERABLE_TRADE_LIMIT']) else 0
        score_c = 1 if price_range > config['PRICE_RANGE_THRESHOLD'] else 0
        score_d = 2 if (
            atr_hit_index is not None and
            vol_hit_index is not None and
            considered > config['CONSIDERABLE_TRADE_LIMIT']
        ) else 0  # ← updated constraint here

        return {
            'day': day_num,
            'success': True,
            'atr_hit_index': int(atr_hit_index) if atr_hit_index is not None else -1,
            'atr_hit_value': float(atr_values[atr_hit_index]) if atr_hit_index is not None else 0.0,
            'vol_hit_index': int(vol_hit_index) if vol_hit_index is not None else -1,
            'vol_hit_value': float(vol_strength[vol_hit_index]) if vol_hit_index is not None else 0.0,
            'price_range': price_range,
            'considerable_pairs': considered,
            'missed_pairs': missed,
            'total_pairs': len(pairs),
            'price_min': float(prices.min()),
            'price_max': float(prices.max()),
            'score_a': score_a,
            'score_b': score_b,
            'score_c': score_c,
            'score_d': score_d
        }

    except Exception as e:
        return {'day': day_num, 'success': False, 'error': str(e)}

# =====================================================================
# SUMMARY REPORT
# =====================================================================

def save_summary_to_file(all_results, config, output_path='multi_factor_regime_summary_parallel.txt'):
    valid = [r for r in all_results if r.get('success', False)]
    total_score_a = sum(r['score_a'] for r in valid)
    total_score_b = sum(r['score_b'] for r in valid)
    total_score_c = sum(r['score_c'] for r in valid)
    total_score_d = sum(r['score_d'] for r in valid)

    with open(output_path, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("MULTI-FACTOR MARKET REGIME SCORING SUMMARY (PARALLEL 25-CPU)\n")
        f.write("=" * 120 + "\n\n")
        f.write(f"ATR Window: {config['ATR_WINDOW']} | SMA Window: {config['SMA_WINDOW']}\n")
        f.write(f"ATR Threshold: {config['ATR_THRESHOLD']} | Vol Threshold: {config['VOL_THRESHOLD']}\n")
        f.write(f"Price Range Threshold: {config['PRICE_RANGE_THRESHOLD']}\n")
        f.write(f"Days Processed: {len(valid)} / {config['NUM_DAYS']}\n\n")

        # --- Section A
        f.write("=" * 120 + "\n")
        f.write("SECTION A — ATR > 0.01 and Considered Trades > 8 → Score +1\n")
        f.write("=" * 120 + "\n")
        for r in valid:
            if r['score_a'] == 1:
                f.write(f"Day {r['day']:3d} | ATR {r['atr_hit_value']:.5f} | Trades {r['considerable_pairs']:2d} | Score_A {r['score_a']}\n")
        f.write(f"\nTOTAL SCORE (A): {total_score_a}\n\n")

        # --- Section B
        f.write("=" * 120 + "\n")
        f.write("SECTION B — Volatility > 0.06 and Considered Trades > 8 → Score +1\n")
        f.write("=" * 120 + "\n")
        for r in valid:
            if r['score_b'] == 1:
                f.write(f"Day {r['day']:3d} | Vol {r['vol_hit_value']:.5f} | Trades {r['considerable_pairs']:2d} | Score_B {r['score_b']}\n")
        f.write(f"\nTOTAL SCORE (B): {total_score_b}\n\n")

        # --- Section C
        f.write("=" * 120 + "\n")
        f.write("SECTION C — Absolute Price Range > 0.7 → Score +1\n")
        f.write("=" * 120 + "\n")
        for r in valid:
            if r['score_c'] == 1:
                f.write(f"Day {r['day']:3d} | Range {r['price_range']:.3f} | Score_C {r['score_c']}\n")
        f.write(f"\nTOTAL SCORE (C): {total_score_c}\n\n")

        # --- Section D
        f.write("=" * 120 + "\n")
        f.write("SECTION D — ATR > 0.01 and Vol > 0.06 and Trades > 8 → Score +2\n")
        f.write("=" * 120 + "\n")
        for r in valid:
            if r['score_d'] == 2:
                f.write(f"Day {r['day']:3d} | ATR {r['atr_hit_value']:.5f} | Vol {r['vol_hit_value']:.5f} | Trades {r['considerable_pairs']:2d} | Score_D {r['score_d']}\n")
        f.write(f"\nTOTAL SCORE (D): {total_score_d}\n\n")

        # --- Grand Summary
        f.write("=" * 120 + "\n")
        f.write("OVERALL TOTAL SCORE SUMMARY\n")
        f.write("=" * 120 + "\n")
        f.write(f"Section A: {total_score_a}\n")
        f.write(f"Section B: {total_score_b}\n")
        f.write(f"Section C: {total_score_c}\n")
        f.write(f"Section D: {total_score_d}\n")
        f.write(f"\nGRAND TOTAL SCORE: {total_score_a + total_score_b + total_score_c + total_score_d}\n")
        f.write("=" * 120 + "\nEND OF REPORT\n")

    print(f"\n✓ Multi-Factor Parallel Summary saved to: {output_path}")

# =====================================================================
# MAIN
# =====================================================================

def main():
    config = CONFIG
    print("=" * 100)
    print("MULTI-FACTOR MARKET REGIME SCORING — ATR + VOL + RANGE + PARALLEL CPU")
    print("=" * 100)
    print(f"Using up to {config['MAX_WORKERS']} CPU cores\n")

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

    save_summary_to_file(all_results, config)
    print("\n✓ Multi-Factor ATR/Vol/Range Parallel Analysis Complete!")

# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == '__main__':
    main()
