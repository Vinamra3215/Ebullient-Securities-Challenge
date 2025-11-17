import numpy as np
import pandas as pd
from pathlib import Path
from numba import jit

# ====================================================================
# CONFIGURATION
# ====================================================================

CONFIG = {
    'DATA_DIR': '/data/quant14/EBY',
    'NUM_DAYS': 279,
    'PB9_COLUMN': 'PB9_T1',
    'PRICE_JUMP_THRESHOLD': 0.3,
    'ADX_WINDOW': 20,
    'ADX_THRESHOLD': 30,  # typical ADX trend threshold
}

# ====================================================================
# NUMBA UTILITIES
# ====================================================================

@jit(nopython=True, fastmath=True)
def find_price_jump_pairs(prices, jump_threshold):
    """Identify pairs of indices where abs price change >= threshold."""
    n = len(prices)
    pairs = []
    i = 0
    while i < n:
        start_price = prices[i]
        for j in range(i + 1, n):
            if abs(prices[j] - start_price) >= jump_threshold:
                pairs.append((i, j))
                i = j + 1
                break
        else:
            i += 1
    return pairs


@jit(nopython=True, fastmath=True)
def calculate_normalized_adx(series, window):
    """
    True ADX-like computation from single PB9_T1 series.
    Uses directional movement (up/down) logic with normalization to [0, 100].
    """
    n = len(series)
    adx = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        up_move = series[i] - series[i - 1]
        down_move = series[i - 1] - series[i]

        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0

        # true range approximation (since only one series)
        tr[i] = abs(up_move)

    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)

    for i in range(window, n):
        avg_tr = np.mean(tr[i - window + 1:i + 1])
        avg_plus_dm = np.mean(plus_dm[i - window + 1:i + 1])
        avg_minus_dm = np.mean(minus_dm[i - window + 1:i + 1])

        if avg_tr == 0:
            continue

        plus_di[i] = 100 * (avg_plus_dm / avg_tr)
        minus_di[i] = 100 * (avg_minus_dm / avg_tr)

        diff = abs(plus_di[i] - minus_di[i])
        summ = plus_di[i] + minus_di[i]
        dx[i] = 100 * (diff / summ) if summ != 0 else 0

        # rolling mean of DX over window for ADX
        adx[i] = np.mean(dx[i - window + 1:i + 1])

    # normalize ADX to 0–100 range for consistency
    min_val, max_val = np.nanmin(adx), np.nanmax(adx)
    if max_val > min_val:
        adx = 100 * (adx - min_val) / (max_val - min_val)

    return adx


# ====================================================================
# PROCESS SINGLE DAY
# ====================================================================

def process_single_day(day_num, config):
    try:
        file_path = Path(config['DATA_DIR']) / f'day{day_num}.parquet'
        if not file_path.exists():
            return {'day': day_num, 'success': False, 'reason': 'file_missing'}

        df = pd.read_parquet(file_path)
        col = config['PB9_COLUMN']

        if col not in df.columns:
            return {'day': day_num, 'success': False, 'reason': f'missing_{col}'}

        df = df[[col]].dropna().copy()
        if len(df) < config['ADX_WINDOW']:
            return {'day': day_num, 'success': False, 'reason': 'insufficient_data'}

        pb9_values = df[col].values.astype(np.float64)

        # Compute ADX using normalized method
        adx = calculate_normalized_adx(pb9_values, config['ADX_WINDOW'])
        adx_max = float(np.nanmax(adx))
        threshold = config['ADX_THRESHOLD']

        # Find first index where ADX crosses threshold
        adx_hit_index = np.argmax(adx >= threshold)
        if adx[adx_hit_index] < threshold:
            adx_hit_index = None

        # Find price fluctuation pairs
        pairs = find_price_jump_pairs(pb9_values, config['PRICE_JUMP_THRESHOLD'])

        missed = 0
        considered = 0
        for (i, j) in pairs:
            if adx_hit_index is None or j < (adx_hit_index or 0):
                missed += 1
            else:
                considered += 1

        return {
            'day': day_num,
            'success': True,
            'adx_hit_index': int(adx_hit_index) if adx_hit_index is not None else -1,
            'adx_hit_value': float(adx[adx_hit_index]) if adx_hit_index is not None else 0.0,
            'adx_max': adx_max,
            'considerable_pairs': considered,
            'missed_pairs': missed,
            'total_pairs': len(pairs),
            'price_min': float(np.min(pb9_values)),
            'price_max': float(np.max(pb9_values)),
        }

    except Exception as e:
        return {'day': day_num, 'success': False, 'error': str(e)}


# ====================================================================
# SUMMARY REPORT
# ====================================================================

def save_summary_to_file(all_results, config, output_path='pb9_t1_adx_true_normalized.txt'):
    valid = [r for r in all_results if r.get('success', False)]
    no_hit = [r for r in valid if r['adx_hit_index'] == -1]
    hit = [r for r in valid if r['adx_hit_index'] >= 0]

    total_missed = sum(r['missed_pairs'] for r in valid)
    total_considered = sum(r['considerable_pairs'] for r in valid)
    total_pairs = sum(r['total_pairs'] for r in valid)

    with open(output_path, 'w') as f:
        f.write("=" * 110 + "\n")
        f.write("PB9_T1 — TRUE NORMALIZED ADX — FIXED THRESHOLD ANALYSIS\n")
        f.write("=" * 110 + "\n\n")
        f.write(f"Data Directory: {config['DATA_DIR']}\n")
        f.write(f"ADX Window: {config['ADX_WINDOW']}\n")
        f.write(f"Fixed ADX Threshold: {config['ADX_THRESHOLD']}\n")
        f.write(f"Days Processed: {len(valid)} / {config['NUM_DAYS']}\n\n")

        f.write("=" * 110 + "\n")
        f.write("SECTION A — Days where ADX reached ≥ Threshold\n")
        f.write("=" * 110 + "\n")
        f.write("Day | ADX Hit @Index | ADX Value | Considered | Missed | Total | PB9_T1 Min | PB9_T1 Max\n")
        f.write("----|----------------|------------|-------------|--------|--------|-------------|-------------\n")
        for r in hit:
            f.write(f"{r['day']:3d} | {r['adx_hit_index']:14d} | {r['adx_hit_value']:10.3f} | "
                    f"{r['considerable_pairs']:11d} | {r['missed_pairs']:6d} | {r['total_pairs']:6d} | "
                    f"{r['price_min']:11.2f} | {r['price_max']:11.2f}\n")

        f.write(f"\nDays with ADX ≥ Threshold: {len(hit)}\n")
        f.write(f"Average Missed (with hit): {np.mean([r['missed_pairs'] for r in hit]) if hit else 0:.2f}\n")

        f.write("\n" + "=" * 110 + "\n")
        f.write("SECTION B — Days where ADX never reached threshold\n")
        f.write("=" * 110 + "\n")
        f.write("Day | ADX Max | Missed | Total | PB9_T1 Min | PB9_T1 Max\n")
        f.write("----|----------|--------|--------|-------------|-------------\n")
        for r in no_hit:
            f.write(f"{r['day']:3d} | {r['adx_max']:8.3f} | {r['missed_pairs']:6d} | {r['total_pairs']:6d} | "
                    f"{r['price_min']:11.2f} | {r['price_max']:11.2f}\n")

        f.write(f"\nDays without ADX ≥ Threshold: {len(no_hit)}\n")
        f.write(f"Average ADX Max (No Hit): {np.mean([r['adx_max'] for r in no_hit]) if no_hit else 0:.2f}\n")

        f.write("\n" + "=" * 110 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 110 + "\n")
        f.write(f"Total Missed: {total_missed}\n")
        f.write(f"Total Considered: {total_considered}\n")
        f.write(f"Total Pairs: {total_pairs}\n")
        f.write(f"Average Total Pairs/Day: {total_pairs / len(valid):.2f}\n")
        f.write("=" * 110 + "\nEND OF REPORT\n" + "=" * 110 + "\n")

    print(f"\n✓ Normalized ADX summary saved to: {output_path}")


# ====================================================================
# MAIN
# ====================================================================

def main():
    config = CONFIG
    print("=" * 100)
    print("PB9_T1 — TRUE NORMALIZED ADX — FLUCTUATION ≥ 0.3 DETECTION")
    print("=" * 100)
    print(f"Data Directory: {config['DATA_DIR']}")
    print(f"ADX Window: {config['ADX_WINDOW']}")
    print("=" * 100)

    all_results = []
    for day in range(config['NUM_DAYS']):
        res = process_single_day(day, config)
        if res['success']:
            all_results.append(res)
        if (day + 1) % 25 == 0:
            print(f"  Processed {day + 1}/{config['NUM_DAYS']} days...")

    print(f"\n✓ Successfully processed {len(all_results)} days.")
    save_summary_to_file(all_results, config)
    print("\n✓ PB9_T1 Normalized ADX Analysis Complete!")

# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == '__main__':
    main()
