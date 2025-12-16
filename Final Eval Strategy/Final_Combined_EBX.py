import os
import glob
import re
import shutil
import pathlib
import pandas as pd
import numpy as np
import multiprocessing
import gc
import math
import argparse
from pykalman import KalmanFilter

# Numba check
try:
    from numba import jit, njit, int64, float64
    NUMBA_AVAILABLE = True
    print("Numba: Enabled")
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba: Not Found")
    def jit(nopython=True, fastmath=True):
        def decorator(func):
            return func
        return decorator
    njit = jit 

# CONFIG

CONFIG = {
    # Data
    'DATA_DIR': '/data/quant14/EBX',
    'NUM_DAYS': 510,
    'OUTPUT_FILE_DIR': '/data/quant14/signals/combined_signals_EBX2.csv',
    'TEMP_DIR': '/data/quant14/signals/temp_manager_v7_swap_p2p3',
    
    # Manager
    'UNIVERSAL_COOLDOWN': 5.0,
    'MAX_WORKERS': 64,
    
    # Columns
    'PRICE_COLUMN': 'Price',
    'TIME_COLUMN': 'Time',
    
    # P1: Kalman
    'P1_FEATURE_COLUMN': 'PB9_T1',
    'P1_KF_FAST_TRANSITION_COV': 0.24,
    'P1_KF_FAST_OBSERVATION_COV': 0.7,
    'P1_KF_FAST_INITIAL_STATE_COV': 100,
    'P1_TAKE_PROFIT': 0.42,
    'P1_TP1_MULTIPLIER': 1.0,
    'P1_TP2_MULTIPLIER': 1.75,
    'P1_TP3_MULTIPLIER': 2.0,
    'P1_TP1_POS': 0.7,
    'P1_TP2_POS': 0.5,
    'P1_STOP_LOSS': 0.5,
    'P1_KAMA_WINDOW': 30,
    'P1_KAMA_FAST_PERIOD': 60,
    'P1_KAMA_SLOW_PERIOD': 300,
    'P1_KAMA_SLOPE_DIFF': 2,
    'P1_KAMA_SLOPE_ENTRY_THRESHOLD': 0.001,
    'P1_KAMA_SLOPE_EXIT_THRESHOLD': 0.001,
    'P1_KAMA_SUM_WINDOW': 5,
    'P1_UCM_ALPHA': 0.8,
    'P1_UCM_BETA': 0.08,
    'P1_ATR_SPAN': 30,
    'P1_ATR_HIGH_THRESHOLD': 0.01,
    'P1_MIN_HOLD_SECONDS': 5.0,
    'P1_START_TIME': 300,

    # P2: Hawkes (Config)
    'P2_SUPER_SMOOTHER_PERIOD': 25,
    'P2_EAMA_PERIOD': 15,
    'P2_EAMA_FAST_PERIOD': 25,
    'P2_EAMA_SLOW_PERIOD': 100,
    'P2_HAWKES_LOOKBACK': 1800,
    'P2_HAWKES_THRESHOLD_ABS': 0.00005,
    'P2_HAWKES_ALPHA': 1.2,
    'P2_HAWKES_BETA': 1.8,
    'P2_HAWKES_SIGNAL_THRESHOLD': 0.08,
    'P2_HARD_WALL_UPPER': 101.06,
    'P2_HARD_WALL_LOWER': 98.94,
    'P2_WALL_MIN_PERIODS': 60,
    'P2_WALL_STD_MULT': 2.0,
    'P2_ATR_LOOKBACK': 28,
    'P2_ATR_MIN_TRADABLE': 0.015,
    'P2_TRAIL_STOP_ABS': 0.005,
    'P2_TRADE_COST_ABS': 0.005,

    # P3: Mu Split (Config)
    'P3_DECISION_WINDOW_SECONDS': 1800, 
    'P3_MU_THRESHOLD_SPLIT': 100.0,
    'P3_COOLDOWN_PERIOD_SECONDS': 5,
    
    # Low Mu
    'P3_LOW_MU_REVERSAL_RISE': 0.2,
    
    # Low Mu Strat A
    'P3_LM_A_ENTRY_DROP': 0.8,
    'P3_LM_A_PRE_DROP_SL': 0.4,
    'P3_LM_A_STATIC_SL': 0.7,
    'P3_LM_A_TP_ACT': 1.0,
    'P3_LM_A_TRAIL_CB': 0.1,
    'P3_LM_A_SHORT_SL': 4.4,
    'P3_LM_A_SHORT_TP_ACT': 6.2,
    'P3_LM_A_SHORT_TRAIL_CB': 0.04,

    # Low Mu Strat B
    'P3_LM_B_ENTRY_DROP': 0.5,
    'P3_LM_B_PRE_DROP_SL': 0.4,
    'P3_LM_B_STATIC_SL': 0.6,
    'P3_LM_B_TP_ACT': 1.2,
    'P3_LM_B_TRAIL_CB': 0.1,
    'P3_LM_B_SHORT_SL': 0.1,
    'P3_LM_B_SHORT_TP_ACT': 999.0,
    'P3_LM_B_SHORT_TRAIL_CB': 0.04,

    # High Mu
    'P3_HM_SIGMA_THRESH': 0.12,
    'P3_HM_ENTRY_RISE': 0.9,
    'P3_HM_PRE_RISE_SL': 0.9,
    'P3_HM_PRIM_SHORT_SL': 0.6,
    'P3_HM_PRIM_TP_ACT': 1.0,
    'P3_HM_PRIM_TRAIL_CB': 0.1,
    'P3_HM_REV_DROP_TRIG': 0.2,
    'P3_HM_SEC_LONG_SL': 0.15,
    'P3_HM_SEC_TP_ACT': 0.8,
    'P3_HM_SEC_TRAIL_CB': 0.1,

    # P4: JMA (Config)
    'P4_JMA_LENGTH': 1800,
    'P4_SUPER_SMOOTHER_PERIOD': 30,
    'P4_EAMA_ER_PERIOD': 15,
    'P4_EAMA_FAST': 60,
    'P4_EAMA_SLOW': 180,
    'P4_USE_DUAL_ACTIVATION': True,
    'P4_ATR_WINDOW': 15,
    'P4_ATR_THRESHOLD': 0.05,
    'P4_STDDEV_WINDOW': 20,
    'P4_STDDEV_THRESHOLD': 0.15,
    'P4_TP_POINTS': 1.0,
    'P4_TRAIL_POINTS': 0.15,
    'P4_MIN_SIGNAL_GAP': 240, 
    'P4_FLIP_DELAY': 240, 
    'P4_START_TICK': 1800,
}

# HELPERS

@jit(nopython=True, fastmath=True)
def compute_atr_fast(prices, period):
    n = len(prices)
    atr = np.zeros(n)
    if n < 2: return atr
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = abs(prices[i] - prices[i - 1])
    for i in range(n):
        if i < period - 1:
            w = i + 1; s = 0
        else:
            w = period; s = i - period + 1
        a = 0.0
        for j in range(s, i + 1): a += tr[j]
        atr[i] = a / w
    return atr

# HAWKES HELPERS

@jit(nopython=True, fastmath=True)
def calculate_super_smoother_numba(prices, period):
    n = len(prices)
    ss = np.zeros(n, dtype=np.float64)
    if n == 0: return ss
    pi = 3.14159265358979323846
    a1 = np.exp(-1.414 * pi / period)
    b1 = 2.0 * a1 * np.cos(1.414 * pi / period)
    c1 = 1.0 - b1 + a1 * a1
    c2 = b1; c3 = -a1 * a1
    ss[0] = prices[0]
    if n > 1: ss[1] = prices[1]
    for i in range(2, n):
        ss[i] = c1 * prices[i] + c2 * ss[i-1] + c3 * ss[i-2]
    return ss

@jit(nopython=True, fastmath=True)
def calculate_eama_numba(ss_values, eama_period, eama_fast, eama_slow):
    n = len(ss_values)
    eama = np.zeros(n, dtype=np.float64)
    if n == 0: return eama
    eama[0] = ss_values[0]
    fast_sc = 2.0 / (eama_fast + 1.0)
    slow_sc = 2.0 / (eama_slow + 1.0)
    for i in range(1, n):
        if i >= eama_period:
            direction = abs(ss_values[i] - ss_values[i - eama_period])
            volatility = 0.0
            for j in range(eama_period):
                if i - j > 0: volatility += abs(ss_values[i - j] - ss_values[i - j - 1])
            er = direction / volatility if volatility > 1e-10 else 0.0
        else: er = 0.0
        sc = ((er * (fast_sc - slow_sc)) + slow_sc) ** 2
        eama[i] = eama[i-1] + sc * (ss_values[i] - eama[i-1])
    return eama

@jit(nopython=True, fastmath=True)
def calculate_expanding_walls_numba(prices, min_periods, std_mult, fallback_upper, fallback_lower):
    n = len(prices)
    upper_walls = np.zeros(n, dtype=np.float64)
    lower_walls = np.zeros(n, dtype=np.float64)
    sum_x = 0.0; sum_sq_x = 0.0; sum_std = 0.0; std_count = 0.0
    for i in range(n):
        val = prices[i]
        sum_x += val
        sum_sq_x += (val * val)
        count = i + 1.0
        if count >= min_periods:
            mean = sum_x / count
            variance = (sum_sq_x / count) - (mean * mean)
            if variance < 0: variance = 0.0
            std = np.sqrt(variance)
            sum_std += std
            std_count += 1.0
            upper_walls[i] = mean + (std)
            lower_walls[i] = mean - (std)
        else:
            upper_walls[i] = fallback_upper
            lower_walls[i] = fallback_lower
    return upper_walls, lower_walls

@jit(nopython=True, fastmath=True)
def calculate_recursive_hawkes_on_eama(eama_prices, alpha, beta, time_delta, threshold):
    N = len(eama_prices)
    H_buy = np.zeros(N, dtype=np.float64)
    H_sell = np.zeros(N, dtype=np.float64)
    decay_factor = np.exp(-beta * time_delta)
    for i in range(1, N):
        current_H_buy = H_buy[i-1] * decay_factor
        current_H_sell = H_sell[i-1] * decay_factor
        eama_change = eama_prices[i] - eama_prices[i-1]
        if eama_change > threshold: current_H_buy += alpha 
        elif eama_change < -threshold: current_H_sell += alpha
        H_buy[i] = current_H_buy
        H_sell[i] = current_H_sell
    return H_buy, H_sell

@jit(nopython=True, fastmath=True)
def generate_hawkes_signals_numba(raw_prices, h_pos, h_neg, atr, upper_walls, lower_walls, ss_slope, 
                                  trade_cost, cooldown, hawkes_lookback, hawkes_thresh, 
                                  atr_min_tradable, trail_stop_abs):
    N = len(raw_prices)
    signals = np.zeros(N, dtype=np.int32)
    position = 0
    entry_price = 0.0
    traded_today = False
    day_tradable = False
    tp_activated = False
    tp_anchor = 0.0
    last_trade_time = -cooldown 
    
    HARD_WALL_UPPER = 100.19
    HARD_WALL_LOWER = 99.55

    for i in range(N):
        p = raw_prices[i]
        curr_upper = HARD_WALL_UPPER
        curr_lower = HARD_WALL_LOWER
        
        in_cooldown = i - last_trade_time < cooldown
        if (not day_tradable) and (atr[i] > atr_min_tradable):
            day_tradable = True
        
        if i < hawkes_lookback: continue
        
        h_buy_i = h_pos[i]
        h_sell_i = h_neg[i]
        
        distance_lower = max(0.0, p - curr_lower)
        long_signal_score = h_sell_i * distance_lower
        
        distance_upper = max(0.0, curr_upper - p)
        short_signal_score = h_buy_i * distance_upper
        
        long_sig = long_signal_score > hawkes_thresh
        short_sig = short_signal_score > hawkes_thresh
        
        # Entry
        if position == 0 and (not traded_today) and day_tradable and (not in_cooldown):
            current_slope = ss_slope[i]
            if long_sig and current_slope > 0.01:
                signals[i] = 1 
                position = 1
                entry_price = p + trade_cost
                traded_today = False 
                tp_activated = False; tp_anchor = 0.0
                last_trade_time = i
            elif short_sig and current_slope < -0.01:
                signals[i] = -1 
                position = -1
                entry_price = p - trade_cost
                traded_today = False 
                tp_activated = False; tp_anchor = 0.0
                last_trade_time = i
        
        # Manage
        if position != 0 and entry_price > 0:
            exit_now = False
            exit_signal = -position
            
            if position == 1: abs_move = p - entry_price
            else: abs_move = entry_price - p
            
            if (not tp_activated) and abs_move >= 0.2:
                tp_activated = False
                tp_anchor = p
            
            if tp_activated:
                if position == 1:
                    if p > tp_anchor: tp_anchor = p
                    if p < tp_anchor - trail_stop_abs: exit_now = True
                else:
                    if p < tp_anchor: tp_anchor = p
                    if p > tp_anchor + trail_stop_abs: exit_now = True

            if i == N - 1 and position != 0:
                exit_now = True
                exit_signal = -position
            
            if exit_now and (not in_cooldown):
                signals[i] = exit_signal
                position = 0
                entry_price = 0.0
                last_trade_time = i
                
    return signals

# STRATEGY 1 (KALMAN)

def prepare_p1_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    # KF Filtering
    initial_state_mean = df[config['PRICE_COLUMN']].iloc[0]
    kf_fast = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        transition_covariance=config['P1_KF_FAST_TRANSITION_COV'],
        observation_covariance=config['P1_KF_FAST_OBSERVATION_COV'],
        initial_state_mean=initial_state_mean,
        initial_state_covariance=config['P1_KF_FAST_INITIAL_STATE_COV']
    )
    filtered_state_means, _ = kf_fast.filter(df[config['P1_FEATURE_COLUMN']].dropna().values)
    smoothed_prices = pd.Series(filtered_state_means.squeeze(), index=df[config['P1_FEATURE_COLUMN']].dropna().index)
    
    df["KFTrendFast"] = smoothed_prices.shift(1)
    df["Take_Profit"] = config['P1_TAKE_PROFIT']
    
    # KAMA
    price = df[config['P1_FEATURE_COLUMN']].to_numpy(dtype=float)
    window = config['P1_KAMA_WINDOW']
    
    kama_signal = np.abs(pd.Series(price).diff(window))
    kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window, min_periods=1).sum()
    er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()
    
    sc_fast = 2 / (config['P1_KAMA_FAST_PERIOD'] + 1)
    sc_slow = 2 / (config['P1_KAMA_SLOW_PERIOD'] + 1)
    sc = ((er * (sc_fast - sc_slow)) + sc_slow) ** 2
    
    kama = np.full(len(price), np.nan)
    valid_start = np.where(~np.isnan(price))[0]
    if len(valid_start) > 0:
        start = valid_start[0]
        kama[start] = price[start]
        for i in range(start + 1, len(price)):
            if np.isnan(price[i - 1]) or np.isnan(sc[i - 1]) or np.isnan(kama[i - 1]):
                kama[i] = kama[i - 1]
            else:
                kama[i] = kama[i - 1] + sc[i - 1] * (price[i - 1] - kama[i - 1])
            
    df["KAMA"] = kama
    df["KAMA_Slope"] = df["KAMA"].diff(config['P1_KAMA_SLOPE_DIFF'])
    df["KAMA_Slope_abs"] = df["KAMA_Slope"].abs()
    
    # UCM
    series = df[config['P1_FEATURE_COLUMN']]
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is not None:
        alpha = config['P1_UCM_ALPHA']
        beta = config['P1_UCM_BETA']
        n = len(series)
        mu = np.zeros(n); beta_slope = np.zeros(n); filtered = np.zeros(n)
        
        mu[first_valid_idx] = series.loc[first_valid_idx]
        beta_slope[first_valid_idx] = 0
        filtered[first_valid_idx] = mu[first_valid_idx] + beta_slope[first_valid_idx]
        
        for t in range(first_valid_idx + 1, n):
            if np.isnan(series.iloc[t]):
                mu[t] = mu[t-1]; beta_slope[t] = beta_slope[t-1]; filtered[t] = filtered[t-1]
                continue
            mu_pred = mu[t-1] + beta_slope[t-1]
            beta_pred = beta_slope[t-1]
            mu[t] = mu_pred + alpha * (series.iloc[t] - mu_pred)
            beta_slope[t] = beta_pred + beta * (series.iloc[t] - mu_pred)
            filtered[t] = mu[t] + beta_slope[t]
        
        df["UCM"] = filtered
        df.loc[0, "UCM"] = 100
    
    # ATR
    tr = df[config['PRICE_COLUMN']].diff().abs()
    atr = tr.ewm(span=config['P1_ATR_SPAN']).mean()
    df["ATR"] = atr
    df["ATR_High"] = df["ATR"].expanding(min_periods=1).max()
    return df

def generate_p1_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    # P1 Logic
    df = prepare_p1_features(df, config)
    position = 0.0
    entry_price = None
    last_signal_time = -config['P1_MIN_HOLD_SECONDS']
    signals = [0.0] * len(df)
    
    tp1, tp2, tp3 = 0.0, 0.0, 0.0
    tp_stage = 0
    apply_sl = False; apply_tp = True
    sl = config['P1_STOP_LOSS']
    
    for i in range(len(df)):
        row = df.iloc[i]
        current_time = row['Time_sec']
        price = row[config['PRICE_COLUMN']]
        is_last_tick = (i == len(df) - 1)
        signal = 0.0
        position = round(position, 2)
        
        if is_last_tick:
            if position != 0: signal = -position
        else:
            cooldown_over = (current_time - last_signal_time) >= config['P1_MIN_HOLD_SECONDS']
            if cooldown_over and position == 0:
                if (row["KFTrendFast"] <= row["UCM"] and 
                    row["KAMA_Slope_abs"] > config['P1_KAMA_SLOPE_ENTRY_THRESHOLD'] and 
                    row["ATR_High"] > config['P1_ATR_HIGH_THRESHOLD']):
                    signal = 1; apply_tp = True
                elif (row["KFTrendFast"] >= row["UCM"] and 
                      row["KAMA_Slope_abs"] > config['P1_KAMA_SLOPE_ENTRY_THRESHOLD'] and 
                      row["ATR_High"] > config['P1_ATR_HIGH_THRESHOLD']):
                    signal = -1; apply_tp = True
                if signal != 0:
                    entry_price = price
                    tp1 = row["Take_Profit"] * config['P1_TP1_MULTIPLIER']
                    tp2 = row["Take_Profit"] * config['P1_TP2_MULTIPLIER']
                    tp3 = row["Take_Profit"] * config['P1_TP3_MULTIPLIER']
                    tp_stage = 0; apply_sl = False; sl = 0.5
            elif cooldown_over and position != 0 and signal == 0:
                if apply_tp:
                    if position > 0:
                        price_diff = price - entry_price
                        tp1_pos, tp2_pos = config['P1_TP1_POS'], config['P1_TP2_POS']
                    else:
                        price_diff = entry_price - price
                        tp1_pos, tp2_pos = -config['P1_TP1_POS'], -config['P1_TP2_POS']
                    
                    if price_diff >= tp3 and tp_stage < 3:
                        tp_stage = 3; apply_sl = False
                        signal = -position
                    elif price_diff >= tp2 and tp_stage < 2:
                        tp_stage = 2; apply_sl = True
                        if row["KAMA_Slope_abs"] < config['P1_KAMA_SLOPE_EXIT_THRESHOLD']:
                            signal = -position + tp2_pos
                    elif price_diff >= tp1 and tp_stage < 1:
                        tp_stage = 1; apply_sl = True
                        if row["KAMA_Slope_abs"] < config['P1_KAMA_SLOPE_EXIT_THRESHOLD']:
                            signal = -position + tp1_pos
                    
                    if apply_sl:
                        if tp_stage == 1 and (price_diff - tp1 < -sl): signal = -position
                        elif tp_stage == 2 and (price_diff - tp2 < -sl): signal = -position

        if signal != 0:
            if (position + signal > 1): signal = 1 - position; position = 1
            elif (position + signal < -1): signal = -1 - position; position = -1
            else: position += signal; last_signal_time = current_time; 
            if round(position, 2) == 0: entry_price = None
        signals[i] = signal
        
    return np.array(signals, dtype=np.float64), df

# STRATEGY 3 (MU SPLIT)

@njit(cache=True, nogil=True)
def backtest_core_low_mu(
    time_sec, price, 
    day_open_price,
    start_index,      
    can_trade_long,     
    cooldown_seconds,
    entry_drop_threshold,
    pre_drop_short_sl,
    static_sl,
    tp_activation,
    trail_callback,
    reversal_rise_trigger,
    short_static_sl,
    short_tp_activation,
    short_trail_callback
):
    # Low Mu Logic
    n = len(price)
    signals = np.zeros(n, dtype=int64)
    
    position = 0
    entry_price = 0.0
    
    pre_drop_target = day_open_price - entry_drop_threshold
    pre_drop_short_active = False
    pending_long_flip = False
    flip_wait_start_time = 0
    
    trailing_active = False
    trailing_peak_price = 0.0 
    short_trailing_active = False
    short_trailing_valley_price = 0.0
    
    waiting_for_short = False
    short_target_price = 0.0
    
    last_signal_time = -cooldown_seconds - 1
    
    for i in range(n):
        if i < start_index: continue

        curr_t = time_sec[i]
        curr_p = price[i]
        is_last_tick = (i == n - 1)
        signal = 0
        
        if is_last_tick and position != 0:
            signal = -position 
        else:
            if i == start_index and can_trade_long and position == 0:
                if curr_p > pre_drop_target:
                    signal = -1 
                    pre_drop_short_active = True
            elif pre_drop_short_active:
                if position == -1:
                    short_pnl = entry_price - curr_p
                    if short_pnl <= -pre_drop_short_sl:
                        signal = 1; pre_drop_short_active = False
                    elif curr_p <= pre_drop_target:
                        signal = 1; pre_drop_short_active = False
                        pending_long_flip = True; flip_wait_start_time = curr_t
            elif pending_long_flip:
                if (curr_t - flip_wait_start_time) >= cooldown_seconds:
                    signal = 1; pending_long_flip = False; waiting_for_short = False 
            else:
                if position == 1:
                    pnl = curr_p - entry_price
                    exit_signal = False
                    if pnl <= -static_sl: exit_signal = True
                    else:
                        if not trailing_active:
                            if pnl >= tp_activation: trailing_active = True; trailing_peak_price = curr_p
                        if trailing_active:
                            if curr_p > trailing_peak_price: trailing_peak_price = curr_p
                            elif curr_p <= (trailing_peak_price - trail_callback): exit_signal = True
                    if exit_signal:
                        signal = -1; waiting_for_short = True
                        short_target_price = curr_p + reversal_rise_trigger
                elif position == -1:
                    pnl = entry_price - curr_p
                    if pnl <= -short_static_sl: signal = 1 
                    else:
                        if not short_trailing_active:
                            if pnl >= short_tp_activation: short_trailing_active = True; short_trailing_valley_price = curr_p 
                        if short_trailing_active:
                            if curr_p < short_trailing_valley_price: short_trailing_valley_price = curr_p
                            elif curr_p >= (short_trailing_valley_price + short_trail_callback): signal = 1 
                elif position == 0:
                    if waiting_for_short:
                        if curr_p >= short_target_price: signal = -1; waiting_for_short = False 
                    elif can_trade_long:
                        cooldown_over = (curr_t - last_signal_time) >= cooldown_seconds
                        if cooldown_over:
                            if curr_p <= pre_drop_target: signal = 1; waiting_for_short = False 
        if signal != 0:
            if (position == 1 and signal == 1) or (position == -1 and signal == -1): signal = 0
            else:
                if position == 0:
                    entry_price = curr_p
                    trailing_active = False; trailing_peak_price = 0.0
                    short_trailing_active = False; short_trailing_valley_price = 0.0
                elif position != 0 and signal == -position:
                    entry_price = 0.0
                    trailing_active = False; trailing_peak_price = 0.0
                    short_trailing_active = False; short_trailing_valley_price = 0.0
                position += signal
                last_signal_time = curr_t
        signals[i] = signal
    return signals

@njit(cache=True, nogil=True)
def backtest_core_high_mu(
    time_sec, price, 
    day_open_price,
    start_index,      
    can_trade_short,   
    cooldown_seconds,
    entry_rise_threshold,
    pre_rise_sl,
    hm_prim_sl, hm_prim_tp, hm_prim_trail,
    hm_rev_drop,
    hm_sec_sl, hm_sec_tp, hm_sec_trail
):
    # High Mu Logic
    n = len(price)
    signals = np.zeros(n, dtype=int64)
    
    position = 0
    entry_price = 0.0
    
    pre_rise_target = day_open_price + entry_rise_threshold
    pre_rise_long_active = False
    pending_short_flip = False
    flip_wait_start_time = 0
    short_trade_executed = False  
    
    short_trailing_active = False; short_trailing_valley_price = 0.0
    long_trailing_active = False; long_trailing_peak_price = 0.0
    waiting_for_long_reversal = False; long_reversal_target_price = 0.0
    
    last_signal_time = -cooldown_seconds - 1
    
    for i in range(n):
        if i < start_index: continue
        curr_t = time_sec[i]; curr_p = price[i]
        is_last_tick = (i == n - 1)
        signal = 0
        
        if is_last_tick and position != 0: signal = -position 
        else:
            if i == start_index and can_trade_short and position == 0:
                if curr_p < pre_rise_target:
                    signal = 1; pre_rise_long_active = True
            elif pre_rise_long_active:
                if position == 1:
                    if (curr_p - entry_price) <= -pre_rise_sl: signal = -1; pre_rise_long_active = False
                    elif curr_p >= pre_rise_target:
                        signal = -1; pre_rise_long_active = False
                        pending_short_flip = True; flip_wait_start_time = curr_t
            elif pending_short_flip:
                if (curr_t - flip_wait_start_time) >= cooldown_seconds:
                    signal = -1; pending_short_flip = False; short_trade_executed = True; waiting_for_long_reversal = False
            else:
                if position == -1:
                    pnl = entry_price - curr_p
                    exit_signal = False
                    if pnl <= -hm_prim_sl: exit_signal = True
                    else:
                        if not short_trailing_active:
                            if pnl >= hm_prim_tp: short_trailing_active = True; short_trailing_valley_price = curr_p
                        if short_trailing_active:
                            if curr_p < short_trailing_valley_price: short_trailing_valley_price = curr_p
                            elif curr_p >= (short_trailing_valley_price + hm_prim_trail): exit_signal = True
                    if exit_signal:
                        signal = 1; waiting_for_long_reversal = True
                        long_reversal_target_price = curr_p - hm_rev_drop
                elif position == 1:
                    pnl = curr_p - entry_price
                    exit_signal = False
                    if pnl <= -hm_sec_sl: exit_signal = True
                    else:
                        if not long_trailing_active:
                            if pnl >= hm_sec_tp: long_trailing_active = True; long_trailing_peak_price = curr_p
                        if long_trailing_active:
                            if curr_p > long_trailing_peak_price: long_trailing_peak_price = curr_p
                            elif curr_p <= (long_trailing_peak_price - hm_sec_trail): exit_signal = True
                    if exit_signal: signal = -1
                elif position == 0:
                    if waiting_for_long_reversal:
                        if curr_p <= long_reversal_target_price: signal = 1; waiting_for_long_reversal = False 
                    elif can_trade_short and not short_trade_executed:
                        cooldown_over = (curr_t - last_signal_time) >= cooldown_seconds
                        if cooldown_over:
                            if curr_p >= pre_rise_target:
                                signal = -1; short_trade_executed = True; waiting_for_long_reversal = False 
        if signal != 0:
            if (position == 1 and signal == 1) or (position == -1 and signal == -1): signal = 0
            else:
                if position == 0:
                    entry_price = curr_p
                    short_trailing_active = False; short_trailing_valley_price = 0.0
                    long_trailing_active = False; long_trailing_peak_price = 0.0
                elif position != 0 and signal == -position:
                    entry_price = 0.0
                    short_trailing_active = False; short_trailing_valley_price = 0.0
                    long_trailing_active = False; long_trailing_peak_price = 0.0
                position += signal
                last_signal_time = curr_t
        signals[i] = signal
    return signals

def generate_mu_split_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    # P3 Wrapper
    price_arr = df[config['PRICE_COLUMN']].astype(float).values
    time_sec_arr = df['Time_sec'].values.astype(np.int64)
    
    if len(price_arr) < 10: return np.zeros(len(price_arr), dtype=np.float64)
    
    day_open_price = price_arr[0]
    decision_window = config['P3_DECISION_WINDOW_SECONDS']
    mask_30min = time_sec_arr <= decision_window
    
    if not np.any(mask_30min): return np.zeros(len(price_arr), dtype=np.float64)
    
    prices_30min = price_arr[mask_30min]
    mu = np.mean(prices_30min)
    sigma = np.std(prices_30min)
    
    start_idx = np.searchsorted(time_sec_arr, decision_window, side='right')
    signals = np.zeros(len(price_arr), dtype=np.int64)
    
    if mu < config['P3_MU_THRESHOLD_SPLIT']:
        if sigma > 0.13:
            signals = backtest_core_low_mu(
                time_sec_arr, price_arr, day_open_price, int(start_idx), True,
                config['P3_COOLDOWN_PERIOD_SECONDS'],
                config['P3_LM_A_ENTRY_DROP'], config['P3_LM_A_PRE_DROP_SL'],
                config['P3_LM_A_STATIC_SL'], config['P3_LM_A_TP_ACT'], config['P3_LM_A_TRAIL_CB'],
                config['P3_LOW_MU_REVERSAL_RISE'],
                config['P3_LM_A_SHORT_SL'], config['P3_LM_A_SHORT_TP_ACT'], config['P3_LM_A_SHORT_TRAIL_CB']
            )
        elif 0.07 <= sigma <= 0.1:
             signals = backtest_core_low_mu(
                time_sec_arr, price_arr, day_open_price, int(start_idx), True,
                config['P3_COOLDOWN_PERIOD_SECONDS'],
                config['P3_LM_B_ENTRY_DROP'], config['P3_LM_B_PRE_DROP_SL'],
                config['P3_LM_B_STATIC_SL'], config['P3_LM_B_TP_ACT'], config['P3_LM_B_TRAIL_CB'],
                config['P3_LOW_MU_REVERSAL_RISE'],
                config['P3_LM_B_SHORT_SL'], config['P3_LM_B_SHORT_TP_ACT'], config['P3_LM_B_SHORT_TRAIL_CB']
            )
    else: 
        if sigma > config['P3_HM_SIGMA_THRESH']:
            signals = backtest_core_high_mu(
                time_sec_arr, price_arr, day_open_price, int(start_idx), True,
                config['P3_COOLDOWN_PERIOD_SECONDS'],
                config['P3_HM_ENTRY_RISE'], config['P3_HM_PRE_RISE_SL'],
                config['P3_HM_PRIM_SHORT_SL'], config['P3_HM_PRIM_TP_ACT'], config['P3_HM_PRIM_TRAIL_CB'],
                config['P3_HM_REV_DROP_TRIG'],
                config['P3_HM_SEC_LONG_SL'], config['P3_HM_SEC_TP_ACT'], config['P3_HM_SEC_TRAIL_CB']
            )

    return signals.astype(np.float64)

# STRATEGY 2 (HAWKES)

def generate_hawkes_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    # P2 Wrapper
    raw_price = df[config['PRICE_COLUMN']].astype(float).values
    raw_price = np.where(np.isnan(raw_price) | (np.abs(raw_price) < 1e-10), 1e-10, raw_price)

    super_smoother = calculate_super_smoother_numba(raw_price, config['P2_SUPER_SMOOTHER_PERIOD'])
    
    ss_slope = np.zeros_like(super_smoother)
    if len(super_smoother) > 1: ss_slope[1:] = super_smoother[1:] - super_smoother[:-1]

    eama_price = calculate_eama_numba(
        super_smoother, config['P2_EAMA_PERIOD'], config['P2_EAMA_FAST_PERIOD'], config['P2_EAMA_SLOW_PERIOD']
    )
    
    atr = compute_atr_fast(raw_price, config['P2_ATR_LOOKBACK'])
    
    u_walls, l_walls = calculate_expanding_walls_numba(
        raw_price, config['P2_WALL_MIN_PERIODS'], config['P2_WALL_STD_MULT'], 
        config['P2_HARD_WALL_UPPER'], config['P2_HARD_WALL_LOWER']
    )

    h_pos, h_neg = calculate_recursive_hawkes_on_eama(
        eama_price, config['P2_HAWKES_ALPHA'], config['P2_HAWKES_BETA'], 1.0, config['P2_HAWKES_THRESHOLD_ABS']
    )
    
    signals = generate_hawkes_signals_numba(
        raw_price, h_pos, h_neg, atr, u_walls, l_walls, ss_slope, 
        config['P2_TRADE_COST_ABS'], config['UNIVERSAL_COOLDOWN'], 
        config['P2_HAWKES_LOOKBACK'], config['P2_HAWKES_SIGNAL_THRESHOLD'],
        config['P2_ATR_MIN_TRADABLE'], config['P2_TRAIL_STOP_ABS']
    )
    
    return signals.astype(np.float64)

# STRATEGY 4 (JMA)

@jit(nopython=True, fastmath=True)
def calculate_jma_numba(prices, length):
    # JMA Calc
    n = len(prices)
    out = np.zeros(n, dtype=np.float64)
    if n == 0: return out
    
    vol = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        vol[i] = abs(prices[i] - prices[i-1])
    
    vol_smooth = np.zeros(n, dtype=np.float64)
    vol_mean = np.zeros(n, dtype=np.float64)
    vol_smooth[0] = vol[0]; vol_mean[0] = vol[0] if vol[0] > 0 else 1.0
    
    for i in range(1, min(length, n)):
        sum_vol = 0.0
        for j in range(i + 1): sum_vol += vol[j]
        vol_smooth[i] = sum_vol / (i + 1)
        vol_mean[i] = vol_smooth[i] if vol_smooth[i] > 0 else 1.0
    
    for i in range(length, n):
        sum_vol = 0.0
        for j in range(length): sum_vol += vol[i - j]
        vol_smooth[i] = sum_vol / length
        sum_mean = 0.0
        for j in range(length): sum_mean += vol_smooth[i - j]
        vol_mean[i] = sum_mean / length
        if vol_mean[i] == 0: vol_mean[i] = 1.0
    
    vol_norm = np.zeros(n, dtype=np.float64)
    alpha_adapt = np.zeros(n, dtype=np.float64)
    alpha = 2.0 / (length + 1.0)
    
    for i in range(n):
        vol_norm[i] = vol_smooth[i] / vol_mean[i]
        if vol_norm[i] < 0.1: vol_norm[i] = 0.1
        elif vol_norm[i] > 10.0: vol_norm[i] = 10.0
        alpha_adapt[i] = alpha * (1.0 + 0.5 * (vol_norm[i] - 1.0))
        if alpha_adapt[i] < 0.001: alpha_adapt[i] = 0.001
        elif alpha_adapt[i] > 0.999: alpha_adapt[i] = 0.999
    
    out[0] = prices[0]
    for i in range(1, n):
        out[i] = alpha_adapt[i] * prices[i] + (1.0 - alpha_adapt[i]) * out[i-1]
    return out

@jit(nopython=True, fastmath=True)
def calculate_atr_simple_numba(prices, window):
    n = len(prices)
    atr_values = np.full(n, np.nan, dtype=np.float64)
    if n < 2: return atr_values
    changes = np.zeros(n, dtype=np.float64)
    for i in range(1, n): changes[i] = abs(prices[i] - prices[i-1])
    for i in range(window, n):
        sum_changes = 0.0
        for j in range(window): sum_changes += changes[i - j]
        atr_values[i] = sum_changes / window
    return atr_values

@jit(nopython=True, fastmath=True)
def calculate_rolling_stddev_numba(prices, window):
    n = len(prices)
    stddev_values = np.full(n, np.nan, dtype=np.float64)
    if n < window: return stddev_values
    for i in range(window, n):
        sum_prices = 0.0
        for j in range(window): sum_prices += prices[i - j]
        mean = sum_prices / window
        sum_sq_diff = 0.0
        for j in range(window):
            diff = prices[i - j] - mean
            sum_sq_diff += diff * diff
        stddev_values[i] = math.sqrt(sum_sq_diff / window)
    return stddev_values

@jit(nopython=True, fastmath=True)
def detect_crossover(series1, series2, i):
    epsilon = 1e-10 
    if i < 1: return 0
    curr_diff = series1[i] - series2[i]
    prev_diff = series1[i-1] - series2[i-1]
    if prev_diff <= epsilon and curr_diff > epsilon: return 1   
    if prev_diff >= -epsilon and curr_diff < -epsilon: return -1  
    return 0

@jit(nopython=True, fastmath=True)
def generate_p4_core(timestamps, prices, jma, eama_slow, atr_values, stddev_values,
                     tp_points, trail_points, min_gap_seconds, flip_delay_seconds,
                     start_tick, use_dual, atr_thresh, stddev_thresh):
    # P4 Logic
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    eps = 1e-9 
    
    state = 0 
    tp_price = 0.0; trail_stop = 0.0; highest_price = 0.0; lowest_price = 0.0
    last_signal_time = -999999.0
    flip_wait_until = -999999.0
    
    day_activated = False
    pending_entry_type = 0 
    
    for i in range(n):
        if i < start_tick: continue
        
        curr_p = prices[i]; curr_t = timestamps[i]
        dt_last = curr_t - last_signal_time
        is_last = (i == n - 1)
        
        if use_dual and not day_activated:
            atr_ok = not np.isnan(atr_values[i]) and atr_values[i] >= atr_thresh
            std_ok = not np.isnan(stddev_values[i]) and stddev_values[i] >= stddev_thresh
            if atr_ok and std_ok:
                day_activated = True
                if pending_entry_type == 1:
                    signals[i] = 1.0; state = 1; tp_price = curr_p + tp_points
                    highest_price = curr_p; last_signal_time = curr_t; pending_entry_type = 0
                    continue
                elif pending_entry_type == -1:
                    signals[i] = -1.0; state = 3; tp_price = curr_p - tp_points
                    lowest_price = curr_p; last_signal_time = curr_t; pending_entry_type = 0
                    continue

        crossover = detect_crossover(jma, eama_slow, i)
        
        # Wait
        if state == 5: 
            if curr_t >= flip_wait_until and dt_last >= min_gap_seconds and not is_last:
                signals[i] = -1.0; state = 3; tp_price = curr_p - tp_points
                lowest_price = curr_p; last_signal_time = curr_t
            continue
        elif state == 6: 
            if curr_t >= flip_wait_until and dt_last >= min_gap_seconds and not is_last:
                signals[i] = 1.0; state = 1; tp_price = curr_p + tp_points
                highest_price = curr_p; last_signal_time = curr_t
            continue
            
        # Entry
        if state == 0:
            if crossover == 1 and dt_last >= min_gap_seconds and not is_last:
                if use_dual:
                    if day_activated:
                        signals[i] = 1.0; state = 1; tp_price = curr_p + tp_points
                        highest_price = curr_p; last_signal_time = curr_t
                    else: pending_entry_type = 1
                else:
                    signals[i] = 1.0; state = 1; tp_price = curr_p + tp_points
                    highest_price = curr_p; last_signal_time = curr_t
            elif crossover == -1 and dt_last >= min_gap_seconds and not is_last:
                if use_dual:
                    if day_activated:
                        signals[i] = -1.0; state = 3; tp_price = curr_p - tp_points
                        lowest_price = curr_p; last_signal_time = curr_t
                    else: pending_entry_type = -1
                else:
                    signals[i] = -1.0; state = 3; tp_price = curr_p - tp_points
                    lowest_price = curr_p; last_signal_time = curr_t

        # Long Manage
        elif state == 1 or state == 2:
            if is_last:
                signals[i] = -1.0; state = 0; last_signal_time = curr_t; continue
            
            if crossover == -1 and dt_last >= min_gap_seconds:
                signals[i] = -1.0; state = 5; flip_wait_until = curr_t + flip_delay_seconds
                last_signal_time = curr_t; continue
            
            if curr_p > highest_price: highest_price = curr_p
            if state == 2: trail_stop = curr_p - trail_points 
            
            if state == 1 and curr_p >= (tp_price - eps) and dt_last >= min_gap_seconds:
                if jma[i] > eama_slow[i]: state = 2; trail_stop = curr_p - trail_points
                else:
                    signals[i] = -1.0; state = 5; flip_wait_until = curr_t + flip_delay_seconds
                    last_signal_time = curr_t
            
            if state == 2 and curr_p <= (trail_stop + eps) and dt_last >= min_gap_seconds:
                signals[i] = -1.0; state = 5; flip_wait_until = curr_t + flip_delay_seconds
                last_signal_time = curr_t

        # Short Manage
        elif state == 3 or state == 4:
            if is_last:
                signals[i] = 1.0; state = 0; last_signal_time = curr_t; continue

            if crossover == 1 and dt_last >= min_gap_seconds:
                signals[i] = 1.0; state = 6; flip_wait_until = curr_t + flip_delay_seconds
                last_signal_time = curr_t; continue

            if curr_p < lowest_price: lowest_price = curr_p
            if state == 4: trail_stop = curr_p + trail_points
            
            if state == 3 and curr_p <= (tp_price + eps) and dt_last >= min_gap_seconds:
                if jma[i] < eama_slow[i]: state = 4; trail_stop = curr_p + trail_points
                else:
                    signals[i] = 1.0; state = 6; flip_wait_until = curr_t + flip_delay_seconds
                    last_signal_time = curr_t
            
            if state == 4 and curr_p >= (trail_stop - eps) and dt_last >= min_gap_seconds:
                signals[i] = 1.0; state = 6; flip_wait_until = curr_t + flip_delay_seconds
                last_signal_time = curr_t
                
    return signals

def generate_p4_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    # P4 Wrapper
    timestamps = df['Time_sec'].values.astype(np.float64)
    prices = df[config['PRICE_COLUMN']].values.astype(np.float64)
    prices = np.nan_to_num(prices, nan=1.0)
    
    atr = calculate_atr_simple_numba(prices, config['P4_ATR_WINDOW'])
    std = calculate_rolling_stddev_numba(prices, config['P4_STDDEV_WINDOW'])
    
    ss = calculate_super_smoother_numba(prices, config['P4_SUPER_SMOOTHER_PERIOD'])
    eama_slow = calculate_eama_numba(ss, config['P4_EAMA_ER_PERIOD'], 
                                     config['P4_EAMA_FAST'], config['P4_EAMA_SLOW'])
    jma = calculate_jma_numba(prices, config['P4_JMA_LENGTH'])
    
    signals = generate_p4_core(
        timestamps, prices, jma, eama_slow, atr, std,
        config['P4_TP_POINTS'], config['P4_TRAIL_POINTS'],
        config['P4_MIN_SIGNAL_GAP'], config['P4_FLIP_DELAY'],
        config['P4_START_TICK'], config['P4_USE_DUAL_ACTIVATION'],
        config['P4_ATR_THRESHOLD'], config['P4_STDDEV_THRESHOLD']
    )
    
    return signals.astype(np.float64)

# MANAGER

@jit(nopython=True, fastmath=True)
def apply_manager_logic(df_time_sec, p1_signals, p2_signals, p3_signals, p4_signals, config_cooldown):
    # Signal Manager
    n = len(df_time_sec)
    final_signals = np.zeros(n, dtype=np.float64)
    position = 0.0
    position_owner = 0
    last_signal_time = -config_cooldown
    
    priority_2_locked = False
    priority_3_locked = False
    priority_4_locked = False
    
    pending_reversal = False
    reversal_owner = 0
    reversal_target_signal = 0.0
    reversal_trigger_time = 0.0
    EPS = 1e-9
    
    for i in range(n):
        current_time = df_time_sec[i]
        p1_sig = p1_signals[i]
        p2_sig = p2_signals[i]
        p3_sig = p3_signals[i]
        p4_sig = p4_signals[i]
        
        is_last_bar = (i == n - 1)
        final_sig = 0.0
        
        if is_last_bar:
            if abs(position) > EPS:
                final_sig = -position
                position = 0.0
                position_owner = 0
            final_signals[i] = final_sig
            continue
            
        cooldown_ok = (current_time - last_signal_time) >= config_cooldown
        
        # Wait Phase
        if pending_reversal and abs(position) < EPS:
            if (current_time - reversal_trigger_time) >= config_cooldown:
                final_sig = reversal_target_signal 
                pending_reversal = False
                position_owner = reversal_owner
                last_signal_time = current_time
                reversal_owner = 0; reversal_target_signal = 0.0; reversal_trigger_time = 0.0
                
        # Entry Logic
        elif abs(position) < EPS and not pending_reversal:
            position = 0.0 
            if cooldown_ok:
                if abs(p1_sig) > 0.01:
                    final_sig = p1_sig; position_owner = 1
                    priority_2_locked = True; priority_3_locked = True; priority_4_locked = True; last_signal_time = current_time
                elif abs(p2_sig) > 0.01 and not priority_2_locked:
                    final_sig = p2_sig; position_owner = 2
                    priority_3_locked = True; priority_4_locked = True; last_signal_time = current_time
                elif abs(p3_sig) > 0.01 and not priority_3_locked:
                    final_sig = p3_sig; position_owner = 3; 
                    priority_4_locked = True; last_signal_time = current_time
                elif abs(p4_sig) > 0.01 and not priority_4_locked:
                    final_sig = p4_sig; position_owner = 4; last_signal_time = current_time
                    
        # Manage
        elif abs(position) > EPS and not pending_reversal:
            
            if position_owner == 1:
                if cooldown_ok and abs(p1_sig) > 0.01:
                    final_sig = p1_sig; last_signal_time = current_time

            elif position_owner == 2:
                if abs(p1_sig) > 0.01:
                    if (p1_sig > 0 and position > 0) or (p1_sig < 0 and position < 0):
                        position_owner = 1; priority_2_locked = True; priority_3_locked = True; priority_4_locked = True; last_signal_time = current_time
                    else:
                        final_sig = -position; pending_reversal = True; reversal_owner = 1
                        reversal_target_signal = p1_sig; reversal_trigger_time = current_time
                        priority_2_locked = True; priority_3_locked = True; priority_4_locked = True; last_signal_time = current_time
                elif cooldown_ok and abs(p2_sig) > 0.01:
                    final_sig = p2_sig; last_signal_time = current_time

            elif position_owner == 3:
                if abs(p1_sig) > 0.01:
                    if (p1_sig > 0 and position > 0) or (p1_sig < 0 and position < 0):
                        position_owner = 1; priority_2_locked = True; priority_3_locked = True; priority_4_locked = True; last_signal_time = current_time
                    else:
                        final_sig = -position; pending_reversal = True; reversal_owner = 1
                        reversal_target_signal = p1_sig; reversal_trigger_time = current_time
                        priority_2_locked = True; priority_3_locked = True; priority_4_locked = True; last_signal_time = current_time
                elif abs(p2_sig) > 0.01:
                    if (p2_sig > 0 and position > 0) or (p2_sig < 0 and position < 0):
                        position_owner = 2; priority_3_locked = True; priority_4_locked = True; last_signal_time = current_time
                    else:
                        final_sig = -position; pending_reversal = True; reversal_owner = 2
                        reversal_target_signal = p2_sig; reversal_trigger_time = current_time
                        priority_3_locked = True; priority_4_locked = True; last_signal_time = current_time
                elif cooldown_ok and abs(p3_sig) > 0.01:
                    final_sig = p3_sig; last_signal_time = current_time

            elif position_owner == 4:
                if abs(p1_sig) > 0.01:
                    if (p1_sig > 0 and position > 0) or (p1_sig < 0 and position < 0):
                        position_owner = 1; priority_2_locked = True; priority_3_locked = True; priority_4_locked = True; last_signal_time = current_time
                    else:
                        final_sig = -position; pending_reversal = True; reversal_owner = 1
                        reversal_target_signal = p1_sig; reversal_trigger_time = current_time
                        priority_2_locked = True; priority_3_locked = True; priority_4_locked = True; last_signal_time = current_time
                elif abs(p2_sig) > 0.01:
                    if (p2_sig > 0 and position > 0) or (p2_sig < 0 and position < 0):
                        position_owner = 2; priority_3_locked = True; priority_4_locked = True; last_signal_time = current_time
                    else:
                        final_sig = -position; pending_reversal = True; reversal_owner = 2
                        reversal_target_signal = p2_sig; reversal_trigger_time = current_time
                        priority_3_locked = True; priority_4_locked = True; last_signal_time = current_time
                elif abs(p3_sig) > 0.01:
                    if (p3_sig > 0 and position > 0) or (p3_sig < 0 and position < 0):
                        position_owner = 3; priority_4_locked = True; last_signal_time = current_time
                    else:
                        final_sig = -position; pending_reversal = True; reversal_owner = 3
                        reversal_target_signal = p3_sig; reversal_trigger_time = current_time
                        priority_4_locked = True; last_signal_time = current_time
                elif cooldown_ok and abs(p4_sig) > 0.01:
                    final_sig = p4_sig; last_signal_time = current_time

        if final_sig != 0:
            proposed_position = position + final_sig
            if proposed_position > 1.0 + EPS: final_sig = 1.0 - position
            elif proposed_position < -1.0 - EPS: final_sig = -1.0 - position
            position += final_sig
            if abs(position) < EPS: position = 0.0; position_owner = 0
        final_signals[i] = final_sig
        
    return final_signals

# FILE PROCESSING

def extract_day_num(filepath):
    match = re.search(r'day(\d+)\.(parquet|csv)', str(filepath))
    return int(match.group(1)) if match else -1

def extract_day_num_csv(filepath):
    match = re.search(r'day(\d+)\.csv', str(filepath))
    return int(match.group(1)) if match else -1

def process_single_day(file_path: str, day_num: int, temp_dir: pathlib.Path, config: dict) -> dict:
    # Process Day
    try:
        temp_dir_path = pathlib.Path(temp_dir)
        required_cols = [config['TIME_COLUMN'], config['PRICE_COLUMN'], config['P1_FEATURE_COLUMN']]
        required_cols = list(dict.fromkeys(required_cols))
        
        df = pd.read_parquet(file_path, columns=required_cols)
        if df.empty: return None
        df = df.reset_index(drop=True)
        
        if pd.api.types.is_numeric_dtype(df[config['TIME_COLUMN']]):
             df['Time_sec'] = df[config['TIME_COLUMN']].astype(int)
        else:
             df['Time_sec'] = pd.to_timedelta(df[config['TIME_COLUMN']].astype(str)).dt.total_seconds().astype(int)
        
        # Signals
        p1_signals, df_with_features = generate_p1_signals(df, config) 
        raw_hawkes_signals = generate_hawkes_signals(df, config)       
        raw_mu_split_signals = generate_mu_split_signals(df, config)   
        raw_jma_signals = generate_p4_signals(df, config)              
        
        # Remap
        p2_signals = raw_jma_signals       
        p3_signals = raw_hawkes_signals    
        p4_signals = raw_mu_split_signals  
        
        time_values = df['Time_sec'].values

        # Mean Crossover
        day_is_banned = False
        if len(df) > 300:
            early_prices = df[config['PRICE_COLUMN']].iloc[:301]
            expanding_mean = early_prices.expanding().mean()
            thresh = 100.09
            curr_means = expanding_mean.iloc[61:301].values 
            prev_means = expanding_mean.iloc[60:300].values 
            crossover_mask = (prev_means <= thresh) & (curr_means > thresh)
            if np.any(crossover_mask): day_is_banned = True
        
        # Manager
        if day_is_banned:
            final_signals = np.zeros(len(df), dtype=np.float64)
        else:
            final_signals = apply_manager_logic(
                time_values, p1_signals, p2_signals, p3_signals, p4_signals, config['UNIVERSAL_COOLDOWN']
            )
        
        # Stats
        p1_entries = np.sum(np.abs(p1_signals) > 0.5)
        p2_entries = np.sum(np.abs(p2_signals) > 0.5)
        p3_entries = np.sum(np.abs(p3_signals) > 0.5)
        p4_entries = np.sum(np.abs(p4_signals) > 0.5)
        final_entries = np.sum(np.abs(final_signals) > 0.5)
        
        # Output
        # Output
        output_df = pd.DataFrame({
            'Day': day_num,  # <--- ADD THIS LINE HERE
            config['TIME_COLUMN']: df[config['TIME_COLUMN']],
            config['PRICE_COLUMN']: df[config['PRICE_COLUMN']],
            'P1_Signal': p1_signals,
            'P2_Signal_JMA': p2_signals,
            'P3_Signal_Hawkes': p3_signals,
            'P4_Signal_Mu': p4_signals,
            'Signal': final_signals
        })
        
        output_path = temp_dir_path / f"day{day_num}.csv"
        output_df.to_csv(output_path, index=False)
        
        del df, df_with_features, p1_signals, p2_signals, p3_signals, p4_signals, final_signals
        return {
            'path': str(output_path), 'day_num': day_num,
            'p1_entries': p1_entries, 'p2_entries': p2_entries,
            'p3_entries': p3_entries, 'p4_entries': p4_entries,
            'final_entries': final_entries
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None
    finally:
        gc.collect()

def process_wrapper(args):
    return process_single_day(*args)

# MAIN

def generate_trade_reports_csv(output_file):
    SAVE_DIR = "/home/raid/Quant14/VB_Feature_Analysis/Histogram/"
    SAVE_PATH = os.path.join(SAVE_DIR, "trade_reports.csv")

    os.makedirs(SAVE_DIR, exist_ok=True)

    signals_file = output_file

    try:
        df = pd.read_csv(signals_file)
    except FileNotFoundError:
        print("Error: Combined signals file not found:", signals_file)
        return

    # Filter for non-zero signals
    signals_df = df[df["Signal"] != 0].copy()
    
    if len(signals_df) == 0:
        print("No trades found in final signal file.")
        return

    trades = []
    position = None # Tracks current position size (e.g., 1.0 or -1.0)
    entry = None    # Tracks the row data for the entry

    for idx, row in signals_df.iterrows():
        signal = row["Signal"]
        
        if pd.isna(signal) or abs(signal) < 1e-9:
            continue

        # --- LOGIC FOR OPENING NEW TRADES ---
        if position is None:
            position = signal
            entry = row
            
        # --- LOGIC FOR MANAGING EXISTING TRADES ---
        else:
            # Check if signal is opposite to current position (Close or Flip)
            # e.g., Pos is 1.0 (Long), Signal is -1.0 (Close) OR -2.0 (Flip)
            if (position > 0 and signal < 0) or (position < 0 and signal > 0):
                
                # 1. RECORD THE CLOSED TRADE
                # We calculate PnL based on the direction we held
                direction_mult = 1 if position > 0 else -1
                pnl = (float(row["Price"]) - float(entry["Price"])) * direction_mult
                
                trades.append({
                    "day": entry["Day"],
                    "direction": "LONG" if position > 0 else "SHORT",
                    "entry_time": entry["Time"],
                    "entry_price": float(entry["Price"]),
                    "exit_time": row["Time"],
                    "exit_price": float(row["Price"]),
                    "pnl": pnl
                })

                # 2. CHECK FOR FLIP (Remaining Position)
                # If Pos was 1 and Signal was -2, new net position is -1
                net_position = position + signal
                
                if abs(net_position) > 1e-9:
                    # We have flipped to a new position
                    position = net_position
                    entry = row # The exit of the old trade is the entry of the new one
                else:
                    # We are flat (Pos 1, Signal -1)
                    position = None
                    entry = None
            
            else:
                # Signal is in the same direction (Pyramiding/Adding)
                # Just update the position size
                position += signal

    if len(trades) == 0:
        print("No completed trades found.")
        return

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(SAVE_PATH, index=False)

    print(f"✓ trade_reports.csv saved at: {SAVE_PATH}")
    print(f"  Total Trades Generated: {len(trades)}")

def main(config):
    parser = argparse.ArgumentParser(description='EBY Manager')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--max-workers', type=int, default=64)
    
    args = parser.parse_args()
    
    CONFIG['DATA_DIR'] = args.data_dir
    CONFIG['MAX_WORKERS'] = int(args.max_workers)
    
    print("STRATEGY MANAGER STARTED")
    print(f"Workers: {config['MAX_WORKERS']} | Cooldown: {config['UNIVERSAL_COOLDOWN']}s")
    
    temp_dir_path = pathlib.Path(config['TEMP_DIR'])
    if temp_dir_path.exists(): shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)
    
    data_dir = config['DATA_DIR']
    files_pattern = os.path.join(data_dir, "day*.parquet")
    all_files = glob.glob(files_pattern)
    filtered_files = [f for f in all_files if extract_day_num(f) not in range(60,80)]
    sorted_files = sorted(all_files, key=extract_day_num)
    
    if not sorted_files:
        print("Error: No data files found.")
        return
    
    processed_files = []
    total_p1 = 0; total_p2 = 0; total_p3 = 0; total_p4 = 0; total_final = 0
    
    tasks = [(f, extract_day_num(f), str(temp_dir_path), config) for f in sorted_files]
    
    try:
        with multiprocessing.Pool(processes=config['MAX_WORKERS']) as pool:
            for result in pool.imap_unordered(process_wrapper, tasks):
                try:
                    if result:
                        processed_files.append(result['path'])
                        total_p1 += result['p1_entries']
                        total_p2 += result['p2_entries']
                        total_p3 += result['p3_entries']
                        total_p4 += result['p4_entries']
                        total_final += result['final_entries']
                        print(f"Day{result['day_num']:3d} | P1:{result['p1_entries']:3d} P2:{result['p2_entries']:3d} P3:{result['p3_entries']:3d} P4:{result['p4_entries']:3d} Final:{result['final_entries']:3d}")
                except Exception as e:
                    print(f"Error: {e}")
        
        if not processed_files: return
        
        # Combine
        sorted_csvs = sorted(processed_files, key=lambda p: extract_day_num_csv(p))
        output_path = pathlib.Path(config['OUTPUT_FILE_DIR'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as outfile:
            for i, csv_file in enumerate(sorted_csvs):
                with open(csv_file, 'rb') as infile:
                    if i == 0: shutil.copyfileobj(infile, outfile)
                    else: infile.readline(); shutil.copyfileobj(infile, outfile)
        
        final_df = pd.read_csv(output_path)
        total_bars = len(final_df)
        entries = (final_df['Signal'].abs() > 0).sum()
        
        print(f"Total Entries: {entries:,}")
        print(f"Breakdown: P1:{total_p1} P2:{total_p2} P3:{total_p3} P4:{total_p4}")
        print(f"Saved: {output_path}")

        # GENERATE REPORTS
        print("Generating trade reports...")
        generate_trade_reports_csv(output_path)
        
    finally:
        if temp_dir_path.exists(): shutil.rmtree(temp_dir_path)

if __name__ == "__main__":
    main(CONFIG)