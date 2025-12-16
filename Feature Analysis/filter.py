import numpy as np
import pandas as pd
from pathlib import Path
from numba import jit

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'DATA_DIR': '/data/quant14/EBX',
    'NUM_DAYS': 510,
    'PRICE_COLUMN': 'Price',
    'VOLATILITY_FEATURE': 'PB9_T1',
    'PRICE_JUMP_THRESHOLD': 0.2, # This is the main threshold (0.20)
    'TIMESTAMP_COLUMN_NAMES': ['time', 'timestamp', 'datetime', 'date'],
    
    # ATR Parameters
    'ATR_WINDOW': 20,
    'ATR_STAGNANT_THRESHOLD': 0.01,
    
    # CUSUM Parameters
    'CUSUM_DRIFT': 0.005,
    'CUSUM_DECAY': 0.7,
    'CUSUM_EWM_SPAN': 180,
    'CUSUM_TREND_STD_MULT': 0.5,
    'CUSUM_CHOPPY_STD_MULT': 0.5,
    
    # Momentum Parameters
    'MOMENTUM_WINDOW': 10,
    'MOMENTUM_THRESHOLD': 0.0,
    'MOMENTUM_AVG_WINDOW': 50,
    
    # Trading Rules
    'START_DELAY_SECONDS': 600.0,
    'MIN_HOLD_SECONDS': 15.0,
    'MIN_COOLDOWN_SECONDS': 15.0,
}

# ============================================================================
# ATR CALCULATION
# ============================================================================

@jit(nopython=True, fastmath=True)
def calculate_true_range_from_pb9(pb9_t1_values):
    """
    Calculate True Range using PB9_T1 (previous second price)
    TR = |PB9_T1[i] - PB9_T1[i-1]|
    """
    n = len(pb9_t1_values)
    true_range = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        true_range[i] = abs(pb9_t1_values[i] - pb9_t1_values[i-1])
    
    return true_range


@jit(nopython=True, fastmath=True)
def rolling_mean_numba(arr, window):
    """Calculate rolling mean with Numba acceleration"""
    n = len(arr)
    result = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        start_idx = max(0, i - window + 1)
        result[i] = np.mean(arr[start_idx:i+1])
    
    return result


# ============================================================================
# CUSUM CALCULATION (EXACT COPY FROM YOUR CODE)
# ============================================================================

@jit(nopython=True, fastmath=True)
def calculate_directional_cusum(changes, drift=0.02, decay=0.9):
    """
    Calculate directional CUSUM on a pre-calculated 'changes' series.
    The opposing CUSUM now DECAYS instead of instantly resetting.
    """
    n = len(changes)
    
    # Initialize CUSUM tracking
    cusum_up = 0.0
    cusum_down = 0.0
    
    cusum_up_values = np.zeros(n, dtype=np.float64)
    cusum_down_values = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        change = changes[i] # Use the pre-calculated change
        
        # Upward move
        if change > drift:
            cusum_up += change
            cusum_down = max(0.0, cusum_down * decay) # Opposing side now decays
        
        # Downward move
        elif change < -drift:
            cusum_down += abs(change)
            cusum_up = max(0.0, cusum_up * decay) # Opposing side now decays
        
        # Small move (noise/consolidation) - both decay
        else:
            cusum_up = max(0.0, cusum_up * decay)
            cusum_down = max(0.0, cusum_down * decay)
        
        # Store values
        cusum_up_values[i] = cusum_up
        cusum_down_values[i] = cusum_down
    
    return cusum_up_values, cusum_down_values


def classify_regime_by_difference(cusum_up, cusum_down, 
                                  choppy_up_thresh, choppy_down_thresh,
                                  trending_up_thresh, trending_down_thresh):
    """
    Classifies every time step based on the CUSUM difference.
    (This function is from your CUSUM code)
    """
    n = len(cusum_up)
    cusum_diff = cusum_up - cusum_down
    
    # Default to 'Transition'
    regimes = np.full(n, 'Transition', dtype='<U15')
    
    # Prioritized Classification:
    
    # 1. Trending (Highest priority)
    is_trending_up = cusum_diff > trending_up_thresh
    is_trending_down = cusum_diff < trending_down_thresh
    
    regimes[is_trending_up] = 'Trending Up'
    regimes[is_trending_down] = 'Trending Down'

    # 2. Choppy (Second priority)
    is_choppy = (cusum_diff <= choppy_up_thresh) & (cusum_diff >= choppy_down_thresh)
    is_not_trending = ~is_trending_up & ~is_trending_down
    
    regimes[is_choppy & is_not_trending] = 'Choppy'

    # Convert to integer representation for Numba
    # 1 = Trend Up, -1 = Trend Down, 0 = Choppy/Transition
    regime_map = {'Trending Up': 1, 'Trending Down': -1, 'Choppy': 0, 'Transition': 0}
    regime_int = np.array([regime_map[r] for r in regimes], dtype=np.int8)

    return regime_int, cusum_diff


# ============================================================================
# SIGNAL GENERATION (EXACT COPY FROM YOUR CODE)
# ============================================================================

@jit(nopython=True, fastmath=True)
def generate_clean_signals(timestamps, prices, 
                           is_stagnant,        # Master Filter (ATR)
                           cusum_regime_int,   # Entry Filter (CUSUM)
                           momentum,           # Entry & Exit Filter (Momentum)
                           momentum_avg,       # Exit Filter (Momentum Average)
                           start_delay_seconds=1200.0,
                           min_hold_seconds=15.0,
                           min_cooldown_seconds=15.0,
                           momentum_threshold=0.0):
    """
    Generate trading signals based on the 3-stage hierarchical filter:
    1. MASTER FILTER: ATR (is_stagnant) - If true, force exit and block entry
    2. ENTRY:         CUSUM (cusum_regime_int) + Momentum (momentum)
    3. EXIT:          Momentum crosses its rolling average OR ATR Stagnation
    """
    n = len(prices)
    
    # Find the first timestamp after start_delay_seconds
    start_timestamp = timestamps[0] + start_delay_seconds
    
    clean_signals = np.zeros(n, dtype=np.int8)
    
    current_position = 0  # 0 = flat, 1 = long, -1 = short
    entry_time = 0.0 
    last_exit_time = 0.0
    
    for i in range(n):
        current_time = timestamps[i]
        
        # Skip signal generation before start_delay_seconds
        if current_time < start_timestamp:
            clean_signals[i] = 0
            continue
            
        # 1. GET ALL CURRENT-STATE VALUES
        is_stagnant_now = is_stagnant[i]
        cusum_signal = cusum_regime_int[i]
        momentum_val = momentum[i]
        momentum_avg_val = momentum_avg[i] # New value

        # 2. --- MASTER FILTER: ATR STAGNATION (EXIT) ---
        # If the market is stagnant, we force exit and do not allow entries.
        if is_stagnant_now:
            if current_position != 0: # We are in a position
                if current_time - entry_time >= min_hold_seconds:
                    if current_position == 1:
                        clean_signals[i] = -1 # Exit long
                    else:
                        clean_signals[i] = 1  # Exit short
                    
                    current_position = 0
                    last_exit_time = current_time
                    entry_time = 0.0
            
            # Whether we just exited or were already flat, do nothing else.
            continue # Skip to the next bar

        # 3. --- MARKET IS VOLATILE ---
        # If we are here, is_stagnant_now is False (market is volatile).
        
        # --- Entry Signal (CUSUM + Momentum) ---
        # "Keep the entry condition the same as before"
        raw_entry_signal = 0
        if cusum_signal == 1:
            if momentum_val > momentum_threshold:
                raw_entry_signal = 1
        elif cusum_signal == -1:
            if momentum_val < -momentum_threshold:
                raw_entry_signal = -1
        
        # --- Exit Signal (Momentum vs. its Average) ---
        # "exit when... momentum falling below its rolling average"
        exit_long_signal = momentum_val < momentum_avg_val 
        exit_short_signal = momentum_val > momentum_avg_val
        
        # 4. STATE MACHINE LOGIC
        
        if current_position == 0:
            # --- Handle Entries ---
            cooldown_elapsed = (last_exit_time == 0.0) or (current_time - last_exit_time >= min_cooldown_seconds)
            
            if raw_entry_signal != 0 and cooldown_elapsed:
                current_position = raw_entry_signal
                entry_time = current_time
                clean_signals[i] = raw_entry_signal
        
        elif current_position == 1:
            # --- Handle Long Exit (based on new Momentum logic) ---
            if exit_long_signal:
                if current_time - entry_time >= min_hold_seconds:
                    clean_signals[i] = -1  # Exit long
                    current_position = 0
                    last_exit_time = current_time
                    entry_time = 0.0
        
        elif current_position == -1:
            # --- Handle Short Exit (based on new Momentum logic) ---
            if exit_short_signal:
                if current_time - entry_time >= min_hold_seconds:
                    clean_signals[i] = 1   # Exit short
                    current_position = 0
                    last_exit_time = current_time
                    entry_time = 0.0
    
    # EOD Square-off: Force close any open position at the last bar
    if current_position != 0:
        if current_position == 1:
            clean_signals[n-1] = -1
        else:
            clean_signals[n-1] = 1
    
    return timestamps, prices, clean_signals


# ============================================================================
# PRICE JUMP PAIR DETECTION
# ============================================================================

@jit(nopython=True, fastmath=True)
def find_price_jump_pairs(prices, jump_threshold):
    """
    Find pairs of points where price moves >= jump_threshold
    
    Args:
        prices: Array of prices
        jump_threshold: Minimum price difference (e.g., 0.2)
    
    Returns:
        pairs: List of (start_idx, end_idx) tuples
    """
    n = len(prices)
    pairs = []
    
    i = 0
    while i < n:
        start_price = prices[i]
        
        # Find first next point with >= jump_threshold difference
        for j in range(i + 1, n):
            price_diff = abs(prices[j] - start_price)
            
            if price_diff >= jump_threshold:
                # Found a pair!
                pairs.append((i, j))
                # Continue from next point after j
                i = j + 1
                break
        else:
            # No pair found from this starting point, move to next
            i += 1
    
    return pairs


# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_single_day(day_num, config):
    """
    Process single day: calculate ATR, CUSUM, Momentum, and find price jump pairs
    
    Returns:
        dict with day analysis results
    """
    try:
        file_path = Path(config['DATA_DIR']) / f'day{day_num}.parquet'
        if not file_path.exists():
            return {'day': day_num, 'success': False}

        # Load data
        df = pd.read_parquet(file_path)

        # Check required columns
        price_col = config['PRICE_COLUMN']
        volatility_feature = config['VOLATILITY_FEATURE']
        
        if price_col not in df.columns or volatility_feature not in df.columns:
            return {'day': day_num, 'success': False}

        # Handle timestamps
        timestamp_col = next((c for c in df.columns if c.lower() in config['TIMESTAMP_COLUMN_NAMES']), None)
        
        if timestamp_col:
            original_timestamps = df[timestamp_col].values
            try:
                dt_series = pd.to_datetime(df[timestamp_col], format='mixed', errors='coerce')
                df['timestamp_seconds'] = (
                    dt_series.dt.hour * 3600 + 
                    dt_series.dt.minute * 60 + 
                    dt_series.dt.second + 
                    dt_series.dt.microsecond / 1e6
                )
                if df['timestamp_seconds'].isna().any():
                    raise ValueError("Timestamp conversion failed")
            except:
                df['timestamp_seconds'] = np.arange(len(df), dtype=np.float64)
        else:
            original_timestamps = np.arange(len(df))
            df['timestamp_seconds'] = np.arange(len(df), dtype=np.float64)

        # Select required columns
        all_cols = [price_col, volatility_feature, 'timestamp_seconds']
        df_clean = df[all_cols].dropna().copy()

        if len(df_clean) < max(config['CUSUM_EWM_SPAN'], config['ATR_WINDOW'], 
                               config['MOMENTUM_WINDOW'], config['MOMENTUM_AVG_WINDOW']):
            return {'day': day_num, 'success': False}

        # Get arrays
        prices = df_clean[price_col].values.astype(np.float64)
        pb9_t1_values = df_clean[volatility_feature].values.astype(np.float64)
        timestamps = df_clean['timestamp_seconds'].values.astype(np.float64)
        
        # Calculate min and max prices for the day
        price_min = np.min(prices)
        price_max = np.max(prices)
        
        # --- STRATEGY CALCULATIONS (EXACT COPY FROM YOUR CODE) ---
        
        # 1. ATR Proxy (Stagnation Filter)
        df_clean['atr_proxy'] = pd.Series(pb9_t1_values).diff().abs().rolling(
            window=config['ATR_WINDOW']).mean().fillna(method='bfill').values
        is_stagnant = (df_clean['atr_proxy'] < config['ATR_STAGNANT_THRESHOLD']).values
        
        # Calculate ATR max
        atr_max = np.max(df_clean['atr_proxy'].values)
        atr_always_below = (atr_max <= config['ATR_STAGNANT_THRESHOLD'])
        
        # 2. Momentum (Entry & Exit Filter)
        df_clean['momentum'] = pd.Series(pb9_t1_values).diff(config['MOMENTUM_WINDOW']).fillna(method='bfill').values
        
        # 3. Momentum Average (Exit Filter)
        df_clean['momentum_avg'] = pd.Series(df_clean['momentum']).rolling(
            window=config['MOMENTUM_AVG_WINDOW']).mean().fillna(method='bfill').values

        # 4. CUSUM Input (Entry Filter)
        df_clean['ci'] = pd.Series(pb9_t1_values).diff().fillna(method='bfill').values
        
        # 5. CUSUM Calculation
        changes_array = df_clean['ci'].values.astype(np.float64)
        cusum_up, cusum_down = calculate_directional_cusum(
            changes_array, config['CUSUM_DRIFT'], config['CUSUM_DECAY']
        )
        
        # 6. CUSUM Thresholds & Regime Classification
        cusum_diff_series = pd.Series(cusum_up - cusum_down)
        ewm_center = cusum_diff_series.ewm(span=config['CUSUM_EWM_SPAN']).mean()
        ewm_std = cusum_diff_series.ewm(span=config['CUSUM_EWM_SPAN']).std().fillna(method='bfill')
        
        trending_up_thresh = (ewm_center + (config['CUSUM_TREND_STD_MULT'] * ewm_std)).values
        trending_down_thresh = (ewm_center - (config['CUSUM_TREND_STD_MULT'] * ewm_std)).values
        choppy_up_thresh = (ewm_center + (config['CUSUM_CHOPPY_STD_MULT'] * ewm_std)).values
        choppy_down_thresh = (ewm_center - (config['CUSUM_CHOPPY_STD_MULT'] * ewm_std)).values
        
        cusum_regime_int, cusum_diff = classify_regime_by_difference(
            cusum_up, cusum_down, 
            choppy_up_thresh, choppy_down_thresh,
            trending_up_thresh, trending_down_thresh
        )

        # 7. Get remaining arrays for signal generator
        momentum = df_clean['momentum'].values.astype(np.float64)
        momentum_avg = df_clean['momentum_avg'].values.astype(np.float64)
        
        # 8. Generate final signals using YOUR EXACT LOGIC
        signal_times, signal_prices, trade_signals = generate_clean_signals(
            timestamps, prices, is_stagnant, cusum_regime_int, momentum,
            momentum_avg,
            config['START_DELAY_SECONDS'],
            config['MIN_HOLD_SECONDS'],
            config['MIN_COOLDOWN_SECONDS'],
            config['MOMENTUM_THRESHOLD']
        )
        
        # Check if strategy generated any signals
        has_strategy_signals = np.any(trade_signals != 0)
        
        # Count number of trades (entry signals)
        num_entry_signals = np.sum(np.abs(trade_signals) == 1)
        num_trades = num_entry_signals // 2
        
        # Find price jump pairs
        # *** MODIFICATION START ***
        pairs_020 = find_price_jump_pairs(prices, config['PRICE_JUMP_THRESHOLD']) # Original (0.20)
        pairs_025 = find_price_jump_pairs(prices, 0.25) # New (0.25)
        pairs_030 = find_price_jump_pairs(prices, 0.30) # New (0.30)
        
        # Calculate average pair times
        durations_020 = [timestamps[j] - timestamps[i] for i, j in pairs_020]
        durations_025 = [timestamps[j] - timestamps[i] for i, j in pairs_025]
        durations_030 = [timestamps[j] - timestamps[i] for i, j in pairs_030]

        avg_time_020 = np.mean(durations_020) if durations_020 else 0.0
        avg_time_025 = np.mean(durations_025) if durations_025 else 0.0
        avg_time_030 = np.mean(durations_030) if durations_030 else 0.0
        
        return {
            'day': day_num,
            'success': True,
            'atr_always_below_threshold': atr_always_below,
            'atr_max': atr_max,
            'has_strategy_signals': has_strategy_signals,
            'num_pairs': len(pairs_020),     # Original threshold
            'num_pairs_025': len(pairs_025), # New threshold
            'num_pairs_030': len(pairs_030), # New threshold
            'avg_pair_time_020': avg_time_020, # <<< MODIFIED
            'avg_pair_time_025': avg_time_025, # <<< MODIFIED
            'avg_pair_time_030': avg_time_030, # <<< MODIFIED
            'num_trades': num_trades,
            'price_min': price_min,
            'price_max': price_max,
        }
        # *** MODIFICATION END ***
        
    except Exception as e:
        return {'day': day_num, 'success': False, 'error': str(e)}


# ============================================================================
# SAVE SUMMARY TO FILE
# ============================================================================

def save_summary_to_file(all_results, config, output_path='price_summary_EBY.txt'):
    """
    Save analysis summary to a text file with 3 categories
    """
    # Separate into 3 categories
    category_a = [r for r in all_results if r['atr_always_below_threshold']]
    category_b = [r for r in all_results if not r['atr_always_below_threshold'] and not r['has_strategy_signals']]
    category_c = [r for r in all_results if not r['atr_always_below_threshold'] and r['has_strategy_signals']]
    
    with open(output_path, 'w') as f:
        f.write("="*90 + "\n")
        f.write("PRICE JUMP PAIR ANALYSIS - ATR + CUSUM + MOMENTUM STRATEGY\n")
        f.write("="*90 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Data Directory: {config['DATA_DIR']}\n")
        f.write(f"  Number of Days Processed: {config['NUM_DAYS']}\n")
        f.write(f"  Price Column: {config['PRICE_COLUMN']}\n")
        f.write(f"  Volatility Feature: {config['VOLATILITY_FEATURE']}\n")
        f.write(f"  Price Jump Threshold (Main): ±{config['PRICE_JUMP_THRESHOLD']}\n")
        f.write(f"  Price Jump Threshold (Addl): ±0.25, ±0.30\n\n") # Added
        
        f.write("Strategy Parameters:\n")
        f.write(f"  ATR Window: {config['ATR_WINDOW']} bars\n")
        f.write(f"  ATR Stagnant Threshold: {config['ATR_STAGNANT_THRESHOLD']}\n")
        f.write(f"  CUSUM Drift: {config['CUSUM_DRIFT']}\n")
        f.write(f"  CUSUM Decay: {config['CUSUM_DECAY']}\n")
        f.write(f"  CUSUM EWM Span: {config['CUSUM_EWM_SPAN']}\n")
        f.write(f"  Momentum Window: {config['MOMENTUM_WINDOW']}\n")
        f.write(f"  Momentum Avg Window: {config['MOMENTUM_AVG_WINDOW']}\n")
        f.write(f"  Momentum Threshold: {config['MOMENTUM_THRESHOLD']}\n\n")
        
        f.write("Trading Rules:\n")
        f.write(f"  Start Delay: {config['START_DELAY_SECONDS']}s ({config['START_DELAY_SECONDS']/60:.0f} min)\n")
        f.write(f"  Min Hold: {config['MIN_HOLD_SECONDS']}s\n")
        f.write(f"  Min Cooldown: {config['MIN_COOLDOWN_SECONDS']}s\n")
        f.write(f"  EOD Square-off: Enabled\n")
        f.write("="*90 + "\n\n")
        
        # *** MODIFICATION START ***
        
        # Category A
        f.write("CATEGORY A: ATR < 0.01 (Always Stagnant)\n")
        f.write("="*90 + "\n")
        f.write(f"Number of days: {len(category_a)}\n\n")
        
        if len(category_a) > 0:
            # <<< MODIFIED Header
            f.write("Day | P(0.20) | P(0.25) | P(0.30) | T(0.20) | T(0.25) | T(0.30) | Trades | ATR Max  | Price Min | Price Max\n")
            f.write("----|---------|---------|---------|--------|--------|--------|--------|----------|-----------|----------\n")
            for result in category_a:
                # <<< MODIFIED Row
                f.write(f"{result['day']:3d} | {result['num_pairs']:7d} | {result['num_pairs_025']:7d} | {result['num_pairs_030']:7d} | "
                        f"{result['avg_pair_time_020']:6.1f} | {result['avg_pair_time_025']:6.1f} | {result['avg_pair_time_030']:6.1f} | "
                        f"{result['num_trades']:6d} | {result['atr_max']:8.6f} | {result['price_min']:9.2f} | {result['price_max']:9.2f}\n")
            
            total_pairs_a = sum(r['num_pairs'] for r in category_a)
            total_pairs_a_025 = sum(r['num_pairs_025'] for r in category_a)
            total_pairs_a_030 = sum(r['num_pairs_030'] for r in category_a)
            total_trades_a = sum(r['num_trades'] for r in category_a)
            
            # <<< MODIFIED: Calculate total weighted time
            total_time_a_020 = sum(r['avg_pair_time_020'] * r['num_pairs'] for r in category_a)
            total_time_a_025 = sum(r['avg_pair_time_025'] * r['num_pairs_025'] for r in category_a)
            total_time_a_030 = sum(r['avg_pair_time_030'] * r['num_pairs_030'] for r in category_a)

            # <<< MODIFIED: Calculate overall average time (handle division by zero)
            avg_time_a_020 = (total_time_a_020 / total_pairs_a) if total_pairs_a > 0 else 0.0
            avg_time_a_025 = (total_time_a_025 / total_pairs_a_025) if total_pairs_a_025 > 0 else 0.0
            avg_time_a_030 = (total_time_a_030 / total_pairs_a_030) if total_pairs_a_030 > 0 else 0.0
            
            avg_pairs_a = total_pairs_a / len(category_a)
            avg_pairs_a_025 = total_pairs_a_025 / len(category_a)
            avg_pairs_a_030 = total_pairs_a_030 / len(category_a)
            avg_trades_a = total_trades_a / len(category_a)
            
            # <<< MODIFIED Summary Lines
            f.write(f"\nTotal pairs (0.20): {total_pairs_a} | Avg (per day): {avg_pairs_a:.2f} | Avg Time (s): {avg_time_a_020:.1f}\n")
            f.write(f"Total pairs (0.25): {total_pairs_a_025} | Avg (per day): {avg_pairs_a_025:.2f} | Avg Time (s): {avg_time_a_025:.1f}\n")
            f.write(f"Total pairs (0.30): {total_pairs_a_030} | Avg (per day): {avg_pairs_a_030:.2f} | Avg Time (s): {avg_time_a_030:.1f}\n")
            f.write(f"Total trades: {total_trades_a} | Avg: {avg_trades_a:.2f}\n")
        else:
            f.write("No days in this category.\n")
        
        f.write("\n" + "="*90 + "\n\n")
        
        # Category B
        f.write("CATEGORY B: ATR > 0.01 but Strategy gave NO signals\n")
        f.write("="*90 + "\n")
        f.write(f"Number of days: {len(category_b)}\n\n")
        
        if len(category_b) > 0:
            # <<< MODIFIED Header
            f.write("Day | P(0.20) | P(0.25) | P(0.30) | T(0.20) | T(0.25) | T(0.30) | Trades | ATR Max  | Price Min | Price Max\n")
            f.write("----|---------|---------|---------|--------|--------|--------|--------|----------|-----------|----------\n")
            for result in category_b:
                # <<< MODIFIED Row
                f.write(f"{result['day']:3d} | {result['num_pairs']:7d} | {result['num_pairs_025']:7d} | {result['num_pairs_030']:7d} | "
                        f"{result['avg_pair_time_020']:6.1f} | {result['avg_pair_time_025']:6.1f} | {result['avg_pair_time_030']:6.1f} | "
                        f"{result['num_trades']:6d} | {result['atr_max']:8.6f} | {result['price_min']:9.2f} | {result['price_max']:9.2f}\n")
            
            total_pairs_b = sum(r['num_pairs'] for r in category_b)
            total_pairs_b_025 = sum(r['num_pairs_025'] for r in category_b)
            total_pairs_b_030 = sum(r['num_pairs_030'] for r in category_b)
            total_trades_b = sum(r['num_trades'] for r in category_b)
            
            # <<< MODIFIED: Calculate total weighted time
            total_time_b_020 = sum(r['avg_pair_time_020'] * r['num_pairs'] for r in category_b)
            total_time_b_025 = sum(r['avg_pair_time_025'] * r['num_pairs_025'] for r in category_b)
            total_time_b_030 = sum(r['avg_pair_time_030'] * r['num_pairs_030'] for r in category_b)

            # <<< MODIFIED: Calculate overall average time (handle division by zero)
            avg_time_b_020 = (total_time_b_020 / total_pairs_b) if total_pairs_b > 0 else 0.0
            avg_time_b_025 = (total_time_b_025 / total_pairs_b_025) if total_pairs_b_025 > 0 else 0.0
            avg_time_b_030 = (total_time_b_030 / total_pairs_b_030) if total_pairs_b_030 > 0 else 0.0
            
            avg_pairs_b = total_pairs_b / len(category_b)
            avg_pairs_b_025 = total_pairs_b_025 / len(category_b)
            avg_pairs_b_030 = total_pairs_b_030 / len(category_b)
            avg_trades_b = total_trades_b / len(category_b)

            # <<< MODIFIED Summary Lines
            f.write(f"\nTotal pairs (0.20): {total_pairs_b} | Avg (per day): {avg_pairs_b:.2f} | Avg Time (s): {avg_time_b_020:.1f}\n")
            f.write(f"Total pairs (0.25): {total_pairs_b_025} | Avg (per day): {avg_pairs_b_025:.2f} | Avg Time (s): {avg_time_b_025:.1f}\n")
            f.write(f"Total pairs (0.30): {total_pairs_b_030} | Avg (per day): {avg_pairs_b_030:.2f} | Avg Time (s): {avg_time_b_030:.1f}\n")
            f.write(f"Total trades: {total_trades_b} | Avg: {avg_trades_b:.2f}\n")
        else:
            f.write("No days in this category.\n")
        
        f.write("\n" + "="*90 + "\n\n")
        
        # Category C
        f.write("CATEGORY C: ATR > 0.01 and Strategy gave signals\n")
        f.write("="*90 + "\n")
        f.write(f"Number of days: {len(category_c)}\n\n")
        
        if len(category_c) > 0:
            # <<< MODIFIED Header
            f.write("Day | P(0.20) | P(0.25) | P(0.30) | T(0.20) | T(0.25) | T(0.30) | Trades | ATR Max  | Price Min | Price Max\n")
            f.write("----|---------|---------|---------|--------|--------|--------|--------|----------|-----------|----------\n")
            for result in category_c:
                # <<< MODIFIED Row
                f.write(f"{result['day']:3d} | {result['num_pairs']:7d} | {result['num_pairs_025']:7d} | {result['num_pairs_030']:7d} | "
                        f"{result['avg_pair_time_020']:6.1f} | {result['avg_pair_time_025']:6.1f} | {result['avg_pair_time_030']:6.1f} | "
                        f"{result['num_trades']:6d} | {result['atr_max']:8.6f} | {result['price_min']:9.2f} | {result['price_max']:9.2f}\n")
            
            total_pairs_c = sum(r['num_pairs'] for r in category_c)
            total_pairs_c_025 = sum(r['num_pairs_025'] for r in category_c)
            total_pairs_c_030 = sum(r['num_pairs_030'] for r in category_c)
            total_trades_c = sum(r['num_trades'] for r in category_c)
            
            # <<< MODIFIED: Calculate total weighted time
            total_time_c_020 = sum(r['avg_pair_time_020'] * r['num_pairs'] for r in category_c)
            total_time_c_025 = sum(r['avg_pair_time_025'] * r['num_pairs_025'] for r in category_c)
            total_time_c_030 = sum(r['avg_pair_time_030'] * r['num_pairs_030'] for r in category_c)

            # <<< MODIFIED: Calculate overall average time (handle division by zero)
            avg_time_c_020 = (total_time_c_020 / total_pairs_c) if total_pairs_c > 0 else 0.0
            avg_time_c_025 = (total_time_c_025 / total_pairs_c_025) if total_pairs_c_025 > 0 else 0.0
            avg_time_c_030 = (total_time_c_030 / total_pairs_c_030) if total_pairs_c_030 > 0 else 0.0
            
            avg_pairs_c = total_pairs_c / len(category_c)
            avg_pairs_c_025 = total_pairs_c_025 / len(category_c)
            avg_pairs_c_030 = total_pairs_c_030 / len(category_c)
            avg_trades_c = total_trades_c / len(category_c)
            
            # <<< MODIFIED Summary Lines
            f.write(f"\nTotal pairs (0.20): {total_pairs_c} | Avg (per day): {avg_pairs_c:.2f} | Avg Time (s): {avg_time_c_020:.1f}\n")
            f.write(f"Total pairs (0.25): {total_pairs_c_025} | Avg (per day): {avg_pairs_c_025:.2f} | Avg Time (s): {avg_time_c_025:.1f}\n")
            f.write(f"Total pairs (0.30): {total_pairs_c_030} | Avg (per day): {avg_pairs_c_030:.2f} | Avg Time (s): {avg_time_c_030:.1f}\n")
            f.write(f"Total trades: {total_trades_c} | Avg: {avg_trades_c:.2f}\n")
        else:
            f.write("No days in this category.\n")
        
        f.write("\n" + "="*90 + "\n\n")
        
        # Overall Summary
        f.write("OVERALL SUMMARY\n")
        f.write("="*90 + "\n")
        f.write(f"Total valid days processed: {len(all_results)}\n")
        f.write(f"  Category A (ATR < 0.01): {len(category_a)} days\n")
        f.write(f"  Category B (ATR > 0.01, no signals): {len(category_b)} days\n")
        f.write(f"  Category C (ATR > 0.01, signals): {len(category_c)} days\n")
        f.write(f"  Failed/Invalid days: {config['NUM_DAYS'] - len(all_results)} days\n\n")
        
        total_pairs = sum(r['num_pairs'] for r in all_results)
        total_pairs_025 = sum(r['num_pairs_025'] for r in all_results)
        total_pairs_030 = sum(r['num_pairs_030'] for r in all_results)
        total_trades = sum(r['num_trades'] for r in all_results)
        
        # <<< MODIFIED: Calculate total weighted time
        total_time_020 = sum(r['avg_pair_time_020'] * r['num_pairs'] for r in all_results)
        total_time_025 = sum(r['avg_pair_time_025'] * r['num_pairs_025'] for r in all_results)
        total_time_030 = sum(r['avg_pair_time_030'] * r['num_pairs_030'] for r in all_results)

        # <<< MODIFIED: Calculate overall average time (handle division by zero)
        avg_time_020 = (total_time_020 / total_pairs) if total_pairs > 0 else 0.0
        avg_time_025 = (total_time_025 / total_pairs_025) if total_pairs_025 > 0 else 0.0
        avg_time_030 = (total_time_030 / total_pairs_030) if total_pairs_030 > 0 else 0.0
        
        f.write(f"Total price jump pairs (0.20): {total_pairs}\n")
        f.write(f"Total price jump pairs (0.25): {total_pairs_025}\n")
        f.write(f"Total price jump pairs (0.30): {total_pairs_030}\n")
        f.write(f"Total strategy trades across all days: {total_trades}\n")
        
        if len(all_results) > 0:
            f.write(f"\nAverage pairs per day (0.20): {total_pairs/len(all_results):.2f}\n")
            f.write(f"Average pairs per day (0.25): {total_pairs_025/len(all_results):.2f}\n")
            f.write(f"Average pairs per day (0.30): {total_pairs_030/len(all_results):.2f}\n")
            f.write(f"Average trades per day (all days): {total_trades/len(all_results):.2f}\n")
            
            # <<< MODIFIED: Added overall average time
            f.write(f"\nOverall avg pair time (0.20): {avg_time_020:.1f}s\n")
            f.write(f"Overall avg pair time (0.25): {avg_time_025:.1f}s\n")
            f.write(f"Overall avg pair time (0.30): {avg_time_030:.1f}s\n")
        
        # *** MODIFICATION END ***
        
        f.write("\n" + "="*90 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*90 + "\n")
    
    print(f"\n✓ Summary saved to: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    config = CONFIG
    
    print("="*80)
    print("PRICE JUMP PAIR ANALYSIS - ATR + CUSUM + MOMENTUM STRATEGY")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data Directory: {config['DATA_DIR']}")
    print(f"  Number of Days: {config['NUM_DAYS']}")
    print(f"  Price Jump Threshold: ±{config['PRICE_JUMP_THRESHOLD']} (plus ±0.25, ±0.30)") # Modified
    print(f"\n  Strategy: ATR Master Filter + CUSUM + Momentum")
    print(f"  ATR Threshold: {config['ATR_STAGNANT_THRESHOLD']}")
    print(f"  Start Delay: {config['START_DELAY_SECONDS']}s")
    print("="*80)
    
    # Process all days
    print(f"\nProcessing {config['NUM_DAYS']} days...")
    all_results = []
    
    for day_num in range(config['NUM_DAYS']):
        result = process_single_day(day_num, config)
        if result['success']:
            all_results.append(result)
        
        if (day_num + 1) % 50 == 0:
            print(f"  Processed {day_num + 1}/{config['NUM_DAYS']} days...")
    
    print(f"✓ Successfully processed {len(all_results)}/{config['NUM_DAYS']} days\n")
    
    # Separate into 3 categories
    category_a = [r for r in all_results if r['atr_always_below_threshold']]
    category_b = [r for r in all_results if not r['atr_always_below_threshold'] and not r['has_strategy_signals']]
    category_c = [r for r in all_results if not r['atr_always_below_threshold'] and r['has_strategy_signals']]
    
    # *** MODIFICATION START ***
    
    print("="*80)
    print("CATEGORY A: ATR < 0.01 (Always Stagnant)")
    print("="*80)
    print(f"Number of days: {len(category_a)}")
    if len(category_a) > 0:
        total_pairs_a = sum(r['num_pairs'] for r in category_a)
        total_pairs_a_025 = sum(r['num_pairs_025'] for r in category_a)
        total_pairs_a_030 = sum(r['num_pairs_030'] for r in category_a)
        total_trades_a = sum(r['num_trades'] for r in category_a)
        
        # <<< MODIFIED: Calculate total weighted time
        total_time_a_020 = sum(r['avg_pair_time_020'] * r['num_pairs'] for r in category_a)
        total_time_a_025 = sum(r['avg_pair_time_025'] * r['num_pairs_025'] for r in category_a)
        total_time_a_030 = sum(r['avg_pair_time_030'] * r['num_pairs_030'] for r in category_a)

        # <<< MODIFIED: Calculate overall average time (handle division by zero)
        avg_time_a_020 = (total_time_a_020 / total_pairs_a) if total_pairs_a > 0 else 0.0
        avg_time_a_025 = (total_time_a_025 / total_pairs_a_025) if total_pairs_a_025 > 0 else 0.0
        avg_time_a_030 = (total_time_a_030 / total_pairs_a_030) if total_pairs_a_030 > 0 else 0.0

        # <<< MODIFIED Print Lines
        print(f"Total pairs (0.20): {total_pairs_a} | Avg (per day): {total_pairs_a/len(category_a):.2f} | Avg Time (s): {avg_time_a_020:.1f}")
        print(f"Total pairs (0.25): {total_pairs_a_025} | Avg (per day): {total_pairs_a_025/len(category_a):.2f} | Avg Time (s): {avg_time_a_025:.1f}")
        print(f"Total pairs (0.30): {total_pairs_a_030} | Avg (per day): {total_pairs_a_030/len(category_a):.2f} | Avg Time (s): {avg_time_a_030:.1f}")
        print(f"Total trades: {total_trades_a} | Avg: {total_trades_a/len(category_a):.2f}")
    
    print("\n" + "="*80)
    print("CATEGORY B: ATR > 0.01 but Strategy gave NO signals")
    print("="*80)
    print(f"Number of days: {len(category_b)}")
    if len(category_b) > 0:
        total_pairs_b = sum(r['num_pairs'] for r in category_b)
        total_pairs_b_025 = sum(r['num_pairs_025'] for r in category_b)
        total_pairs_b_030 = sum(r['num_pairs_030'] for r in category_b)
        total_trades_b = sum(r['num_trades'] for r in category_b)

        # <<< MODIFIED: Calculate total weighted time
        total_time_b_020 = sum(r['avg_pair_time_020'] * r['num_pairs'] for r in category_b)
        total_time_b_025 = sum(r['avg_pair_time_025'] * r['num_pairs_025'] for r in category_b)
        total_time_b_030 = sum(r['avg_pair_time_030'] * r['num_pairs_030'] for r in category_b)

        # <<< MODIFIED: Calculate overall average time (handle division by zero)
        avg_time_b_020 = (total_time_b_020 / total_pairs_b) if total_pairs_b > 0 else 0.0
        avg_time_b_025 = (total_time_b_025 / total_pairs_b_025) if total_pairs_b_025 > 0 else 0.0
        avg_time_b_030 = (total_time_b_030 / total_pairs_b_030) if total_pairs_b_030 > 0 else 0.0

        # <<< MODIFIED Print Lines
        print(f"Total pairs (0.20): {total_pairs_b} | Avg (per day): {total_pairs_b/len(category_b):.2f} | Avg Time (s): {avg_time_b_020:.1f}")
        print(f"Total pairs (0.25): {total_pairs_b_025} | Avg (per day): {total_pairs_b_025/len(category_b):.2f} | Avg Time (s): {avg_time_b_025:.1f}")
        print(f"Total pairs (0.30): {total_pairs_b_030} | Avg (per day): {total_pairs_b_030/len(category_b):.2f} | Avg Time (s): {avg_time_b_030:.1f}")
        print(f"Total trades: {total_trades_b} | Avg: {total_trades_b/len(category_b):.2f}")
    
    print("\n" + "="*80)
    print("CATEGORY C: ATR > 0.01 and Strategy gave signals")
    print("="*80)
    print(f"Number of days: {len(category_c)}")
    if len(category_c) > 0:
        total_pairs_c = sum(r['num_pairs'] for r in category_c)
        total_pairs_c_025 = sum(r['num_pairs_025'] for r in category_c)
        total_pairs_c_030 = sum(r['num_pairs_030'] for r in category_c)
        total_trades_c = sum(r['num_trades'] for r in category_c)
        
        # <<< MODIFIED: Calculate total weighted time
        total_time_c_020 = sum(r['avg_pair_time_020'] * r['num_pairs'] for r in category_c)
        total_time_c_025 = sum(r['avg_pair_time_025'] * r['num_pairs_025'] for r in category_c)
        total_time_c_030 = sum(r['avg_pair_time_030'] * r['num_pairs_030'] for r in category_c)

        # <<< MODIFIED: Calculate overall average time (handle division by zero)
        avg_time_c_020 = (total_time_c_020 / total_pairs_c) if total_pairs_c > 0 else 0.0
        avg_time_c_025 = (total_time_c_025 / total_pairs_c_025) if total_pairs_c_025 > 0 else 0.0
        avg_time_c_030 = (total_time_c_030 / total_pairs_c_030) if total_pairs_c_030 > 0 else 0.0
        
        # <<< MODIFIED Print Lines
        print(f"Total pairs (0.20): {total_pairs_c} | Avg (per day): {total_pairs_c/len(category_c):.2f} | Avg Time (s): {avg_time_c_020:.1f}")
        print(f"Total pairs (0.25): {total_pairs_c_025} | Avg (per day): {total_pairs_c_025/len(category_c):.2f} | Avg Time (s): {avg_time_c_025:.1f}")
        print(f"Total pairs (0.30): {total_pairs_c_030} | Avg (per day): {total_pairs_c_030/len(category_c):.2f} | Avg Time (s): {avg_time_c_030:.1f}")
        print(f"Total trades: {total_trades_c} | Avg: {total_trades_c/len(category_c):.2f}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total valid days: {len(all_results)}")
    print(f"  Category A (ATR < 0.01): {len(category_a)} days")
    print(f"  Category B (ATR > 0.01, no signals): {len(category_b)} days")
    print(f"  Category C (ATR > 0.01, signals): {len(category_c)} days")
    
    total_pairs = sum(r['num_pairs'] for r in all_results)
    total_pairs_025 = sum(r['num_pairs_025'] for r in all_results)
    total_pairs_030 = sum(r['num_pairs_030'] for r in all_results)
    total_trades = sum(r['num_trades'] for r in all_results)

    # <<< MODIFIED: Calculate total weighted time
    total_time_020 = sum(r['avg_pair_time_020'] * r['num_pairs'] for r in all_results)
    total_time_025 = sum(r['avg_pair_time_025'] * r['num_pairs_025'] for r in all_results)
    total_time_030 = sum(r['avg_pair_time_030'] * r['num_pairs_030'] for r in all_results)

    # <<< MODIFIED: Calculate overall average time (handle division by zero)
    avg_time_020 = (total_time_020 / total_pairs) if total_pairs > 0 else 0.0
    avg_time_025 = (total_time_025 / total_pairs_025) if total_pairs_025 > 0 else 0.0
    avg_time_030 = (total_time_030 / total_pairs_030) if total_pairs_030 > 0 else 0.0

    # <<< MODIFIED Print Lines
    print(f"\nTotal pairs (0.20): {total_pairs} | Avg (per day): {total_pairs/len(all_results):.2f} | Overall Avg Time (s): {avg_time_020:.1f}")
    print(f"Total pairs (0.25): {total_pairs_025} | Avg (per day): {total_pairs_025/len(all_results):.2f} | Overall Avg Time (s): {avg_time_025:.1f}")
    print(f"Total pairs (0.30): {total_pairs_030} | Avg (per day): {total_pairs_030/len(all_results):.2f} | Overall Avg Time (s): {avg_time_030:.1f}")
    print(f"Total strategy trades: {total_trades} | Avg (per day): {total_trades/len(all_results):.2f}")
    
    # *** MODIFICATION END ***
    
    # Save summary to file (current directory)
    save_summary_to_file(all_results, config)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETED!")
    print("="*80)
    
    return all_results
    
# Add this new function for easy integration with plotting
def get_pairs_for_day(day_num):
    """
    Convenience function to get pairs for a specific day
    Returns: (success, pairs, dataframe)
    """
    try:
        file_path = Path(CONFIG['DATA_DIR']) / f'day{day_num}.parquet'
        if not file_path.exists():
            return False, [], None

        # Load data
        df = pd.read_parquet(file_path)

        # Check required columns
        price_col = CONFIG['PRICE_COLUMN']
        volatility_feature = CONFIG['VOLATILITY_FEATURE']
        
        if price_col not in df.columns or volatility_feature not in df.columns:
            return False, [], None

        # Get price array
        prices = df[price_col].dropna().values.astype(np.float64)
        
        # Find price jump pairs using the main threshold
        current_threshold = 0.3 # User's hardcoded value
        pairs = find_price_jump_pairs(prices, current_threshold) # <<< MODIFIED (used variable)
        
        print(f"Day {day_num}: Found {len(pairs)} pairs with threshold {current_threshold}") # <<< MODIFIED (fixed bug)
        
        return True, pairs, df
        
    except Exception as e:
        print(f"Error processing day {day_num}: {e}")
        return False, [], None

if __name__ == '__main__':
    results = main()