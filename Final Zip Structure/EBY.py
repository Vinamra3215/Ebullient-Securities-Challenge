import os
import glob
import re
import shutil
import sys
import pathlib
import pandas as pd
import numpy as np
import multiprocessing
import gc
import time
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
    'DATA_DIR': './EBY/', 
    'NUM_DAYS': 279,
    'OUTPUT_FILE_DIR': '/data/quant14/signals/combined_signals_EBY_copy.csv',
    'TEMP_DIR': '/data/quant14/signals/EBY_temp_signals',
    
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
    'P1_TAKE_PROFIT': 0.4,
    'P1_TP1_MULTIPLIER': 1.0,
    'P1_TP2_MULTIPLIER': 1.25,
    'P1_TP3_MULTIPLIER': 1.75,
    'P1_TP1_POS': 0.7,
    'P1_TP2_POS': 0.5,
    'P1_STOP_LOSS': 0.5,
    'P1_KAMA_WINDOW': 30,
    'P1_KAMA_FAST_PERIOD': 60,
    'P1_KAMA_SLOW_PERIOD': 300,
    'P1_KAMA_SLOPE_DIFF': 2,
    'P1_KAMA_SLOPE_ENTRY_THRESHOLD': 0.0007,
    'P1_KAMA_SLOPE_EXIT_THRESHOLD': 0.001,
    'P1_KAMA_SUM_WINDOW': 5,
    'P1_KAMA_SUM_LONG_THRESHOLD': 0.005,
    'P1_KAMA_SUM_SHORT_THRESHOLD': 0.005,
    'P1_UCM_ALPHA': 0.1,
    'P1_UCM_BETA': 0.02,
    'P1_ATR_SPAN': 30,
    'P1_ATR_HIGH_LONG_THRESHOLD': 0.025,
    'P1_ATR_HIGH_SHORT_THRESHOLD': 0.025,
    'P1_MIN_HOLD_SECONDS': 5.0,
    'P1_START_TIME': 300,
    
    # P2: Mu-Gated
    'P2_DECISION_WINDOW_SECONDS': 30 * 60,
    'P2_MU_THRESHOLD': 100.0,
    'P2_REVERSAL_RISE_TRIGGER': 0.2,
    
    # P2 Sub-Strats
    'P2_STRAT_1_PARAMS': {
        "entry_drop": 0.8, "static_sl": 0.7, "tp_activation": 1.0, "trail_callback": 0.1,
        "short_static_sl": 999.0, "short_tp_activation": 999.0, "short_trail_callback": 0.04,
        "pre_drop_sl": 999.0 
    },
    'P2_STRAT_2_PARAMS': {
        "entry_drop": 0.9, "static_sl": 999.0, "tp_activation": 1.0, "trail_callback": 0.1,
        "short_static_sl": 0.15, "short_tp_activation": 999.0, "short_trail_callback": 0.04,
        "pre_drop_sl": 0.6
    },

    # P3: Dual Mu/Sigma
    'P3_DECISION_WINDOW_SECONDS': 30 * 60,
    'P3_MU_THRESHOLD': 100.0,
    'P3_SIGMA_HIGH_THRESHOLD': 0.12,
    'P3_SIGMA_MID_MIN': 0.09,
    'P3_SIGMA_MID_MAX': 0.12,
    
    # P3 Strat 1
    'P3_S1_ENTRY_RISE_THRESHOLD': 0.8,
    'P3_S1_PRIMARY_STATIC_SL': 0.7,
    'P3_S1_PRIMARY_TP_ACTIVATION': 0.65,
    'P3_S1_PRIMARY_TRAIL_CALLBACK': 0.15,
    'P3_S1_REVERSAL_DROP_TRIGGER': 0.2,
    'P3_S1_SECONDARY_STATIC_SL': 0.15,
    'P3_S1_SECONDARY_TP_ACTIVATION': 0.8,
    'P3_S1_SECONDARY_TRAIL_CALLBACK': 0.1,
    'P3_S1_PRE_RISE_SL': 0.4,
    
    # P3 Strat 2
    'P3_S2_ENTRY_DROP_THRESHOLD': 0.9,
    'P3_S2_STATIC_SL': 999.0,
    'P3_S2_TP_ACTIVATION': 1.0,
    'P3_S2_TRAIL_CALLBACK': 0.1,
    'P3_S2_REVERSAL_RISE_TRIGGER': 0.2,
    'P3_S2_SHORT_STATIC_SL': 0.15,
    'P3_S2_SHORT_TP_ACTIVATION': 999.0,
    'P3_S2_SHORT_TRAIL_CALLBACK': 0.04,

    # P4: Hawkes
    'P4_FEATURE_COLUMN': 'PB9_T1', 

    # P5: Mean-Sigma
    'P5_DECISION_WINDOW': 30 * 60, 
    'P5_MEAN_THRESHOLD': 100.0,
    'P5_SIGMA_MIN': 0.03,
    'P5_SIGMA_MAX': 0.06,
    'P5_TP_LONG': 0.85,
    'P5_TP_SHORT': 0.85,
    'P5_TRAILING_SL_INTERVAL': 1.5 * 60,
}

# HELPERS

@jit(nopython=True, fastmath=True)
def calculate_ehlers_super_smoother_numba(prices, period):
    n = len(prices)
    ss = np.zeros(n, dtype=np.float64)
    pi = math.pi
    a1 = np.exp(-1.414 * pi / period)
    b1 = 2 * a1 * np.cos(1.414 * pi / period)
    c1 = 1 - b1 + a1 * a1
    c2 = b1
    c3 = -a1 * a1
    if n > 0: ss[0] = prices[0]
    if n > 1: ss[1] = prices[1]
    for i in range(2, n):
        ss[i] = c1 * prices[i] + c2 * ss[i-1] + c3 * ss[i-2]
    return ss

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
            w = i + 1
            s = 0
        else:
            w = period
            s = i - period + 1
        a = 0.0
        for j in range(s, i + 1):
            a += tr[j]
        atr[i] = a / w
    return atr

# STRATEGY 1 (KF)

def prepare_p1_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    # KF Filtering
    kf_fast = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        transition_covariance=config['P1_KF_FAST_TRANSITION_COV'],
        observation_covariance=config['P1_KF_FAST_OBSERVATION_COV'],
        initial_state_mean=df[config['PRICE_COLUMN']].iloc[0],
        initial_state_covariance=config['P1_KF_FAST_INITIAL_STATE_COV']
    )
    filtered_state_means, _ = kf_fast.filter(df[config['P1_FEATURE_COLUMN']].dropna().values)
    smoothed_prices = pd.Series(filtered_state_means.squeeze(), index=df[config['P1_FEATURE_COLUMN']].dropna().index)
    df["KFTrendFast"] = smoothed_prices.shift(1)
    df["KFTrendFast_Lagged"] = df["KFTrendFast"].shift(1)
    df["Take_Profit"] = config['P1_TAKE_PROFIT']
    
    # KAMA Calculation
    price = df[config['P1_FEATURE_COLUMN']].to_numpy(dtype=float)
    price[0:60] = np.nan
    window = config['P1_KAMA_WINDOW']
    kama_signal = np.abs(pd.Series(price).diff(window))
    kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window).sum()
    er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()
    sc_fast = 2 / (config['P1_KAMA_FAST_PERIOD'] + 1)
    sc_slow = 2 / (config['P1_KAMA_SLOW_PERIOD'] + 1)
    sc = ((er * (sc_fast - sc_slow)) + sc_slow) ** 2
    kama = np.full(len(price), np.nan)
    valid_start = np.where(~np.isnan(price))[0]
    if len(valid_start) == 0: raise ValueError("No valid price values.")
    start = valid_start[0]
    kama[start] = price[start]
    for i in range(start + 1, len(price)):
        if np.isnan(price[i - 1]) or np.isnan(sc[i - 1]) or np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1]
        else:
            kama[i] = kama[i - 1] + sc[i - 1] * (price[i - 1] - kama[i - 1])
    
    # Feature Set
    df["KAMA"] = kama
    df["KAMA_Slope"] = df["KAMA"].diff(config['P1_KAMA_SLOPE_DIFF'])
    df["KAMA_Slope_abs"] = df["KAMA_Slope"].abs()
    df["KAMA_Slope_abs_Lagged_2"] = df["KAMA_Slope_abs"].shift(2)
    df["KAMA_Sum"] = df["KAMA_Slope_abs"].rolling(window=config['P1_KAMA_SUM_WINDOW'], min_periods=1).sum()
    
    # UCM Calculation
    series = df[config['P1_FEATURE_COLUMN']]
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is None: raise ValueError(f"{config['P1_FEATURE_COLUMN']} empty")
    alpha = config['P1_UCM_ALPHA']
    beta = config['P1_UCM_BETA']
    n = len(series)
    mu = np.zeros(n)
    beta_slope = np.zeros(n)
    filtered = np.zeros(n)
    mu[first_valid_idx] = series.loc[first_valid_idx]
    beta_slope[first_valid_idx] = 0
    filtered[first_valid_idx] = mu[first_valid_idx] + beta_slope[first_valid_idx]
    for t in range(first_valid_idx + 1, n):
        if np.isnan(series.iloc[t]):
            mu[t] = mu[t-1]
            beta_slope[t] = beta_slope[t-1]
            filtered[t] = filtered[t-1]
            continue
        mu_pred = mu[t-1] + beta_slope[t-1]
        beta_pred = beta_slope[t-1]
        mu[t] = mu_pred + alpha * (series.iloc[t] - mu_pred)
        beta_slope[t] = beta_pred + beta * (series.iloc[t] - mu_pred)
        filtered[t] = mu[t] + beta_slope[t]
    filtered[0:60] = np.nan
    df["UCM"] = filtered
    df["UCM_Lagged"] = df["UCM"].shift(1)
    
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
    tp1 = 0.0; tp2 = 0.0; tp3 = 0.0
    tp_stage = 0
    apply_tp = False; apply_sl = False
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
            if cooldown_over and position == 0 and current_time >= config['P1_START_TIME']:
                # Short Condition
                if (row["KFTrendFast"] >= row["UCM"] and 
                    row["KAMA_Slope_abs_Lagged_2"] > config['P1_KAMA_SLOPE_ENTRY_THRESHOLD'] and 
                    row["KAMA_Sum"] > config['P1_KAMA_SUM_SHORT_THRESHOLD'] and 
                    row["ATR_High"] > config['P1_ATR_HIGH_SHORT_THRESHOLD']):
                    signal = -1
                    apply_tp = True
                # Long Condition
                elif (row["KFTrendFast"] <= row["UCM"] and 
                      row["KAMA_Slope_abs_Lagged_2"] > config['P1_KAMA_SLOPE_ENTRY_THRESHOLD'] and 
                      row["KAMA_Sum"] > config['P1_KAMA_SUM_LONG_THRESHOLD'] and 
                      row["ATR_High"] > config['P1_ATR_HIGH_LONG_THRESHOLD']):
                    signal = 1
                    apply_tp = True
                
                if signal != 0:
                    entry_price = price
                    tp1 = row["Take_Profit"] * config['P1_TP1_MULTIPLIER']
                    tp2 = row["Take_Profit"] * config['P1_TP2_MULTIPLIER']
                    tp3 = row["Take_Profit"] * config['P1_TP3_MULTIPLIER']
                    tp_stage = 0
                    apply_sl = False
                    sl = 0.5
            
            elif cooldown_over and position != 0 and signal == 0:
                if apply_tp:
                    price_diff = price - entry_price if position > 0 else entry_price - price
                    tp1_pos = config['P1_TP1_POS'] if position > 0 else -config['P1_TP1_POS']
                    tp2_pos = config['P1_TP2_POS'] if position > 0 else -config['P1_TP2_POS']

                    if price_diff >= tp3 and tp_stage < 4:
                        apply_sl = False
                        signal = -position
                    elif price_diff >= tp3 and tp_stage < 3:
                        tp_stage = 3
                        apply_sl = True
                        if row["KAMA_Slope_abs"] < config['P1_KAMA_SLOPE_EXIT_THRESHOLD']:
                            signal = -position
                    elif price_diff >= tp2 and tp_stage < 2:
                        tp_stage = 2
                        apply_sl = True
                        if row["KAMA_Slope_abs"] < config['P1_KAMA_SLOPE_EXIT_THRESHOLD']:
                            signal = -position + tp2_pos
                    elif price_diff >= tp1 and tp_stage < 1:
                        tp_stage = 1
                        apply_sl = True
                        if row["KAMA_Slope_abs"] < config['P1_KAMA_SLOPE_EXIT_THRESHOLD']:
                            signal = -position + tp1_pos
                    
                    if apply_sl:
                        thresh = tp1 if tp_stage == 1 else (tp2 if tp_stage == 2 else tp3)
                        if price_diff - thresh < -sl:
                            signal = -position

        if signal != 0:
            if (position + signal > 1):
                signal = 1 - position
                position = 1
            elif (position + signal < -1):
                signal = -1 - position
                position = -1
            else:
                position += signal
                last_signal_time = current_time
                if round(position, 2) == 0: entry_price = None
        signals[i] = signal
    return np.array(signals, dtype=np.float64), df

# STRATEGY 2 (MU-GATED)

@njit(cache=True, nogil=True)
def backtest_core(
    time_sec, price, day_open_price, start_index, can_trade_long,      
    cooldown_seconds, reversal_rise_trigger, entry_drop_threshold,
    static_sl, tp_activation, trail_callback,
    short_static_sl, short_tp_activation, short_trail_callback, pre_drop_sl
):
    n = len(price)
    signals = np.zeros(n, dtype=int64)
    positions = np.zeros(n, dtype=int64)
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
                    pnl_short = entry_price - curr_p
                    if pnl_short <= -pre_drop_sl:
                        signal = 1
                        pre_drop_short_active = False
                    elif curr_p <= pre_drop_target:
                        signal = 1
                        pre_drop_short_active = False
                        pending_long_flip = True
                        flip_wait_start_time = curr_t
            elif pending_long_flip:
                if (curr_t - flip_wait_start_time) >= cooldown_seconds:
                    signal = 1
                    pending_long_flip = False
                    waiting_for_short = False
            else:
                if position == 1:
                    pnl = curr_p - entry_price
                    exit_signal = False
                    if pnl <= -static_sl: exit_signal = True
                    else:
                        if not trailing_active:
                            if pnl >= tp_activation:
                                trailing_active = True
                                trailing_peak_price = curr_p
                        if trailing_active:
                            if curr_p > trailing_peak_price: trailing_peak_price = curr_p
                            elif curr_p <= (trailing_peak_price - trail_callback): exit_signal = True
                    if exit_signal:
                        signal = -1
                        waiting_for_short = True
                        short_target_price = curr_p + reversal_rise_trigger
                elif position == -1:
                    pnl = entry_price - curr_p
                    if pnl <= -short_static_sl: signal = 1
                    else:
                        if not short_trailing_active:
                            if pnl >= short_tp_activation:
                                short_trailing_active = True
                                short_trailing_valley_price = curr_p 
                        if short_trailing_active:
                            if curr_p < short_trailing_valley_price: short_trailing_valley_price = curr_p
                            elif curr_p >= (short_trailing_valley_price + short_trail_callback): signal = 1
                elif position == 0:
                    if waiting_for_short:
                        if curr_p >= short_target_price:
                            signal = -1
                            waiting_for_short = False 
                    elif can_trade_long:
                        if (curr_t - last_signal_time) >= cooldown_seconds:
                            if curr_p <= pre_drop_target:
                                signal = 1
                                waiting_for_short = False 
        if signal != 0:
            if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                signal = 0
            else:
                if position != 0 and signal == -position:
                    entry_price = 0.0
                    trailing_active = False
                    trailing_peak_price = 0.0
                    short_trailing_active = False
                    short_trailing_valley_price = 0.0
                elif position == 0:
                    entry_price = curr_p
                    trailing_active = False
                    trailing_peak_price = 0.0
                    short_trailing_active = False
                    short_trailing_valley_price = 0.0
                position += signal
                last_signal_time = curr_t
        signals[i] = signal
        positions[i] = position
    return signals, positions

def generate_p2_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    # P2 Wrapper
    prices = df[config['PRICE_COLUMN']].values.astype(np.float64)
    timestamps = df['Time_sec'].values.astype(np.int64)
    prices = np.where(np.isnan(prices) | (np.abs(prices) < 1e-10), 1e-10, prices)
    if len(prices) < 10: return np.zeros(len(df), dtype=np.float64)

    day_open_price = prices[0]
    decision_window = config['P2_DECISION_WINDOW_SECONDS']
    mask_30min = timestamps <= decision_window
    can_trade_long = False
    active_params = None
    start_idx = 0
    
    # Check Conditions
    if np.any(mask_30min):
        prices_30min = prices[mask_30min]
        mu = np.mean(prices_30min)
        sigma = np.std(prices_30min)
        
        if mu < config['P2_MU_THRESHOLD']:
            if 0.09 <= sigma <= 0.12:
                can_trade_long = True
                active_params = config['P2_STRAT_2_PARAMS']
            elif sigma > 0.12:
                can_trade_long = True
                active_params = config['P2_STRAT_1_PARAMS']
    
    if not can_trade_long or active_params is None:
        return np.zeros(len(df), dtype=np.float64)

    start_idx = np.searchsorted(timestamps, decision_window, side='right')
    signals, _ = backtest_core(
        timestamps, prices, day_open_price, int(start_idx), bool(can_trade_long), 
        config['UNIVERSAL_COOLDOWN'], config['P2_REVERSAL_RISE_TRIGGER'],
        float(active_params["entry_drop"]), float(active_params["static_sl"]),
        float(active_params["tp_activation"]), float(active_params["trail_callback"]),
        float(active_params["short_static_sl"]), float(active_params["short_tp_activation"]),
        float(active_params["short_trail_callback"]), float(active_params["pre_drop_sl"])
    )
    return signals.astype(np.float64)

# STRATEGY 3 (DUAL MU/SIGMA)

@njit(cache=True, nogil=True)
def p3_backtest_strategy1(
    time_sec, price, day_open_price, start_index, can_trade_short,      
    cooldown_seconds, entry_rise_threshold, pre_rise_sl,
    primary_static_sl, primary_tp_activation, primary_trail_callback,
    reversal_drop_trigger, secondary_static_sl, secondary_tp_activation, secondary_trail_callback
):
    n = len(price)
    signals = np.zeros(n, dtype=int64)
    position = 0
    entry_price = 0.0
    pre_rise_target = day_open_price + entry_rise_threshold
    pre_rise_long_active = False
    pending_short_flip = False
    flip_wait_start_time = 0
    short_trade_executed = False
    short_trailing_active = False
    short_trailing_valley_price = 0.0
    long_trailing_active = False
    long_trailing_peak_price = 0.0
    waiting_for_long_reversal = False
    long_reversal_target_price = 0.0
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
            if i == start_index and can_trade_short and position == 0:
                if curr_p < pre_rise_target:
                    signal = 1
                    pre_rise_long_active = True
            elif pre_rise_long_active:
                if position == 1:
                    if (curr_p - entry_price) <= -pre_rise_sl:
                        signal = -1
                        pre_rise_long_active = False
                    elif curr_p >= pre_rise_target:
                        signal = -1
                        pre_rise_long_active = False
                        pending_short_flip = True
                        flip_wait_start_time = curr_t
            elif pending_short_flip:
                if (curr_t - flip_wait_start_time) >= cooldown_seconds:
                    signal = -1
                    pending_short_flip = False
                    short_trade_executed = True
                    waiting_for_long_reversal = False
            else:
                if position == -1:
                    pnl = entry_price - curr_p
                    exit_signal = False
                    if pnl <= -primary_static_sl: exit_signal = True
                    else:
                        if not short_trailing_active:
                            if pnl >= primary_tp_activation:
                                short_trailing_active = True
                                short_trailing_valley_price = curr_p
                        if short_trailing_active:
                            if curr_p < short_trailing_valley_price: short_trailing_valley_price = curr_p
                            elif curr_p >= (short_trailing_valley_price + primary_trail_callback): exit_signal = True
                    if exit_signal:
                        signal = 1
                        waiting_for_long_reversal = True
                        long_reversal_target_price = curr_p - reversal_drop_trigger
                elif position == 1:
                    pnl = curr_p - entry_price
                    exit_signal = False
                    if pnl <= -secondary_static_sl: exit_signal = True
                    else:
                        if not long_trailing_active:
                            if pnl >= secondary_tp_activation:
                                long_trailing_active = True
                                long_trailing_peak_price = curr_p
                        if long_trailing_active:
                            if curr_p > long_trailing_peak_price: long_trailing_peak_price = curr_p
                            elif curr_p <= (long_trailing_peak_price - secondary_trail_callback): exit_signal = True
                    if exit_signal:
                        signal = -1
                elif position == 0:
                    if waiting_for_long_reversal:
                        if curr_p <= long_reversal_target_price:
                            signal = 1
                            waiting_for_long_reversal = False 
                    elif can_trade_short and not short_trade_executed:
                        if (curr_t - last_signal_time) >= cooldown_seconds:
                            if curr_p >= pre_rise_target:
                                signal = -1
                                short_trade_executed = True
                                waiting_for_long_reversal = False 
        if signal != 0:
            if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                signal = 0
            else:
                if position == 0:
                    entry_price = curr_p
                    short_trailing_active = False
                    short_trailing_valley_price = 0.0
                    long_trailing_active = False
                    long_trailing_peak_price = 0.0
                elif position != 0 and signal == -position:
                    entry_price = 0.0
                    short_trailing_active = False
                    short_trailing_valley_price = 0.0
                    long_trailing_active = False
                    long_trailing_peak_price = 0.0
                position += signal
                last_signal_time = curr_t
        signals[i] = signal
    return signals

@njit(cache=True, nogil=True)
def p3_backtest_strategy2(
    time_sec, price, day_open_price, start_index, can_trade_long,      
    cooldown_seconds, entry_drop_threshold, static_sl, tp_activation, trail_callback,
    reversal_rise_trigger, short_static_sl, short_tp_activation, short_trail_callback
):
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
                    if curr_p <= pre_drop_target:
                        signal = 1
                        pre_drop_short_active = False
                        pending_long_flip = True
                        flip_wait_start_time = curr_t
            elif pending_long_flip:
                if (curr_t - flip_wait_start_time) >= cooldown_seconds:
                    signal = 1
                    pending_long_flip = False
                    waiting_for_short = False
            else:
                if position == 1:
                    pnl = curr_p - entry_price
                    exit_signal = False
                    if pnl <= -static_sl: exit_signal = True
                    else:
                        if not trailing_active:
                            if pnl >= tp_activation:
                                trailing_active = True
                                trailing_peak_price = curr_p
                        if trailing_active:
                            if curr_p > trailing_peak_price: trailing_peak_price = curr_p
                            elif curr_p <= (trailing_peak_price - trail_callback): exit_signal = True
                    if exit_signal:
                        signal = -1
                        waiting_for_short = True
                        short_target_price = curr_p + reversal_rise_trigger
                elif position == -1:
                    pnl = entry_price - curr_p
                    if pnl <= -short_static_sl: signal = 1
                    else:
                        if not short_trailing_active:
                            if pnl >= short_tp_activation:
                                short_trailing_active = True
                                short_trailing_valley_price = curr_p 
                        if short_trailing_active:
                            if curr_p < short_trailing_valley_price: short_trailing_valley_price = curr_p
                            elif curr_p >= (short_trailing_valley_price + short_trail_callback): signal = 1
                elif position == 0:
                    if waiting_for_short:
                        if curr_p >= short_target_price:
                            signal = -1
                            waiting_for_short = False 
                    elif can_trade_long:
                        if (curr_t - last_signal_time) >= cooldown_seconds:
                            if curr_p <= pre_drop_target:
                                signal = 1
                                waiting_for_short = False 
        if signal != 0:
            if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                signal = 0
            else:
                if position == 0:
                    entry_price = curr_p
                    trailing_active = False
                    trailing_peak_price = 0.0
                    short_trailing_active = False
                    short_trailing_valley_price = 0.0
                elif position != 0 and signal == -position:
                    entry_price = 0.0
                    trailing_active = False
                    trailing_peak_price = 0.0
                    short_trailing_active = False
                    short_trailing_valley_price = 0.0
                position += signal
                last_signal_time = curr_t
        signals[i] = signal
    return signals

def generate_p3_signals(df: pd.DataFrame, config: dict) -> tuple:
    # P3 Wrapper
    prices = df[config['PRICE_COLUMN']].values.astype(np.float64)
    timestamps = df['Time_sec'].values.astype(np.int64)
    prices = np.where(np.isnan(prices) | (np.abs(prices) < 1e-10), 1e-10, prices)
    if len(prices) < 10: return np.zeros(len(df), dtype=np.float64), df

    day_open_price = prices[0]
    decision_window = config['P3_DECISION_WINDOW_SECONDS']
    mask_30min = timestamps <= decision_window
    can_trade = False
    strategy_type = None
    start_idx = 0
    
    if np.any(mask_30min):
        prices_30min = prices[mask_30min]
        mu = np.mean(prices_30min)
        sigma = np.std(prices_30min)
        start_idx = np.searchsorted(timestamps, decision_window, side='right')
        
        # Select Strat
        if sigma > config['P3_SIGMA_HIGH_THRESHOLD'] and mu > config['P3_MU_THRESHOLD']:
            can_trade = True; strategy_type = 1
        elif (config['P3_SIGMA_MID_MIN'] <= sigma <= config['P3_SIGMA_MID_MAX']) and mu < config['P3_MU_THRESHOLD']:
            can_trade = True; strategy_type = 2

    if can_trade and strategy_type == 1:
        signals = p3_backtest_strategy1(
            timestamps, prices, day_open_price, int(start_idx), True,
            config['UNIVERSAL_COOLDOWN'], config['P3_S1_ENTRY_RISE_THRESHOLD'],
            config['P3_S1_PRE_RISE_SL'], config['P3_S1_PRIMARY_STATIC_SL'],
            config['P3_S1_PRIMARY_TP_ACTIVATION'], config['P3_S1_PRIMARY_TRAIL_CALLBACK'],
            config['P3_S1_REVERSAL_DROP_TRIGGER'], config['P3_S1_SECONDARY_STATIC_SL'],
            config['P3_S1_SECONDARY_TP_ACTIVATION'], config['P3_S1_SECONDARY_TRAIL_CALLBACK']
        )
    elif can_trade and strategy_type == 2:
        signals = p3_backtest_strategy2(
            timestamps, prices, day_open_price, int(start_idx), True,
            config['UNIVERSAL_COOLDOWN'], config['P3_S2_ENTRY_DROP_THRESHOLD'],
            config['P3_S2_STATIC_SL'], config['P3_S2_TP_ACTIVATION'],
            config['P3_S2_TRAIL_CALLBACK'], config['P3_S2_REVERSAL_RISE_TRIGGER'],
            config['P3_S2_SHORT_STATIC_SL'], config['P3_S2_SHORT_TP_ACTIVATION'],
            config['P3_S2_SHORT_TRAIL_CALLBACK']
        )
    else:
        signals = np.zeros(len(prices), dtype=np.int64)
    return signals.astype(np.float64), df

# STRATEGY 4 (HAWKES)

def prepare_p4_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    # Feature Gen
    price_vals = df["Price"].values
    n = len(price_vals)
    period = 25
    pi = math.pi
    a1 = math.exp(-1.414 * pi / period)
    b1 = 2 * a1 * math.cos(1.414 * pi / period)
    c1 = 1 - b1 + a1*a1
    ss = np.zeros(n)
    if n > 0: ss[0] = price_vals[0]
    if n > 1: ss[1] = price_vals[1]
    c2 = b1; c3 = -a1*a1
    for i in range(2, n):
        ss[i] = c1*price_vals[i] + c2*ss[i-1] + c3*ss[i-2]
    df["EhlersSuperSmoother"] = ss
    df["EhlersSuperSmoother_Slope"] = df["EhlersSuperSmoother"].diff()
    
    # AURA
    AURA_period = 10; AURA_fast = 20; AURA_slow = 100
    ss_series = pd.Series(ss) 
    direction_ss = ss_series.diff(AURA_period).abs() 
    volatility_ss = ss_series.diff(1).abs().rolling(AURA_period).sum() 
    er_ss = (direction_ss / volatility_ss).fillna(0).values 
    fast_sc = 2 / (AURA_fast + 1); slow_sc = 2 / (AURA_slow + 1) 
    sc_ss = ((er_ss * (fast_sc - slow_sc)) + slow_sc) ** 2 
    AURA = np.zeros(n) 
    if n > 0: AURA[0] = ss[0] 
    for i in range(1, n): 
        c = sc_ss[i] if not np.isnan(sc_ss[i]) else 0 
        AURA[i] = AURA[i-1] + c * (ss[i] - AURA[i-1]) 
    df["AURA"] = AURA 
    df["AURA_Slope"] = df["AURA"].diff().fillna(0)

    # AURA Slow
    AURA_period = 15; AURA_fast = 30; AURA_slow = 150
    ss_series = pd.Series(ss) 
    direction_ss = ss_series.diff(AURA_period).abs() 
    volatility_ss = ss_series.diff(1).abs().rolling(AURA_period).sum() 
    er_ss = (direction_ss / volatility_ss).fillna(0).values 
    fast_sc = 2 / (AURA_fast + 1); slow_sc = 2 / (AURA_slow + 1) 
    sc_ss = ((er_ss * (fast_sc - slow_sc)) + slow_sc) ** 2 
    AURA = np.zeros(n) 
    if n > 0: AURA[0] = ss[0] 
    for i in range(1, n): 
        c = sc_ss[i] if not np.isnan(sc_ss[i]) else 0 
        AURA[i] = AURA[i-1] + c * (ss[i] - AURA[i-1]) 
    df["AURA_Slow"] = AURA
    
    # Hawkes
    alpha = 1.2; beta = 1.8; threshold = 0.01; time_delta = 1
    decay = np.exp(-beta * time_delta)
    H_buy = np.zeros(n); H_sell = np.zeros(n)
    for i in range(1, n):
        H_buy[i] = H_buy[i-1] * decay
        H_sell[i] = H_sell[i-1] * decay
        if df["EhlersSuperSmoother_Slope"][i] > threshold: H_buy[i] += alpha
        if df["EhlersSuperSmoother_Slope"][i] < -threshold: H_sell[i] += alpha
    df["H_Buy"] = H_buy
    df["H_Sell"] = H_sell

    tr = df[config['P4_FEATURE_COLUMN']].diff().abs()
    atr = tr.ewm(span=30).mean()
    df["ATR"] = atr
    df["ATR_High"] = df["ATR"].expanding(min_periods=1).max()
    return df

def generate_signal_p4_logic(row, position, cooldown_over):
    # Logic
    position = round(position, 2)
    signal = 0
    if cooldown_over:
        if position == 0:
            if row['H_Buy'] > 1 and row["AURA_Slope"] > 0.001 and row["ATR_High"] < 0.025: signal = 1
            elif row['H_Sell'] > 1 and row["AURA_Slope"] < -0.001 and row["ATR_High"] < 0.025: signal = -1
        elif position > 0:
            if row["H_Sell"] > 1 or row["AURA_Slow"] < row["AURA"]: signal = -position
        elif position < 0:
            if row["H_Buy"] > 1 or row["AURA_Slow"] > row["AURA"]: signal = -position
    return -1 * signal

def generate_p4_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    # P4 Loop
    df = prepare_p4_features(df, config)
    position = 0.0
    entry_price = None
    last_signal_time = -config['UNIVERSAL_COOLDOWN']
    signals = np.zeros(len(df), dtype=np.float64)
    for i in range(len(df)):
        position = round(position, 2)
        row = df.iloc[i]
        current_time = row['Time_sec']
        price = row['Price'] 
        is_last_tick = (i == len(df) - 1)
        signal = 0.0
        if is_last_tick and position != 0:
            signal = -position
        else:
            cooldown_over = (current_time - last_signal_time) >= config['UNIVERSAL_COOLDOWN']
            signal = generate_signal_p4_logic(row, position, cooldown_over)
            if cooldown_over and position == 0 and signal != 0:
                entry_price = price
            elif cooldown_over and position != 0 and signal != 0:
                if position > 0: price_diff = price - entry_price
                elif position < 0: price_diff = entry_price - price
                if price_diff > 0: signal = 0.0
        if signal != 0:
            if (position + signal > 1):
                signal = 1 - position
                position = 1.0
            elif (position + signal < -1):
                signal = -1 - position
                position = -1.0
            else:
                position += signal
                last_signal_time = current_time
                if round(position, 2) == 0: entry_price = None
        signals[i] = signal
    return signals

# STRATEGY 5 (MEAN-SIGMA)

@njit(cache=True, nogil=True)
def p5_mean_sigma_backtest(
    time_sec, price, decision_window, direction, tp_long_amount, tp_short_amount, trailing_sl_interval
):
    n = len(price)
    signals = np.zeros(n, dtype=float64)
    if direction == 0: return signals
        
    position_size = 0.0
    entry_price = 0.0
    tp_hit = False
    checkpoint_time = 0.0
    checkpoint_price = 0.0
    tp_price_target = 0.0
    start_idx = 0
    
    for i in range(n):
        if time_sec[i] > decision_window:
            start_idx = i
            break
            
    if start_idx == 0 and time_sec[0] <= decision_window:
        if time_sec[n-1] <= decision_window: return signals
            
    # Entry
    signals[start_idx] = direction * 1.0
    position_size = 1.0
    entry_price = price[start_idx]
    
    if direction == 1: tp_price_target = entry_price + tp_long_amount
    else: tp_price_target = entry_price - tp_short_amount
        
    for i in range(start_idx + 1, n):
        curr_p = price[i]
        curr_t = time_sec[i]
        
        # EOD
        if i == n - 1:
            if position_size > 0:
                signals[i] = -position_size * direction
            break
            
        if not tp_hit:
            hit_now = False
            if direction == 1:
                if curr_p >= tp_price_target: hit_now = True
            else:
                if curr_p <= tp_price_target: hit_now = True
            
            if hit_now:
                signals[i] = -0.5 * direction
                position_size = 0.5
                tp_hit = True
                checkpoint_time = curr_t + trailing_sl_interval
                checkpoint_price = -1.0 
        else:
            if curr_t >= checkpoint_time:
                if checkpoint_price < 0.0:
                    checkpoint_price = curr_p
                    checkpoint_time = curr_t + trailing_sl_interval
                else:
                    should_exit = False
                    if direction == 1: 
                        if curr_p <= checkpoint_price: should_exit = True
                    else: 
                        if curr_p >= checkpoint_price: should_exit = True
                    
                    if should_exit:
                        signals[i] = -position_size * direction
                        position_size = 0.0
                        break 
                    else:
                        checkpoint_price = curr_p
                        checkpoint_time = curr_t + trailing_sl_interval
    return signals

def generate_p5_mean_sigma_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    # P5 Wrapper
    prices = df[config['PRICE_COLUMN']].values.astype(np.float64)
    timestamps = df['Time_sec'].values.astype(np.int64)
    if len(prices) < 10: return np.zeros(len(df), dtype=np.float64)
    
    decision_window = config['P5_DECISION_WINDOW']
    mask_30min = timestamps <= decision_window
    direction = 0
    
    if np.any(mask_30min):
        prices_30min = prices[mask_30min]
        mean_price = np.mean(prices_30min)
        sigma_price = np.std(prices_30min)
        mean_threshold = config['P5_MEAN_THRESHOLD']
        sigma_min = config['P5_SIGMA_MIN']
        sigma_max = config['P5_SIGMA_MAX']
        
        if sigma_min < sigma_price < sigma_max:
            if mean_price > mean_threshold: direction = 1
            elif mean_price < mean_threshold: direction = -1
    
    if direction == 0: return np.zeros(len(df), dtype=np.float64)
        
    signals = p5_mean_sigma_backtest(
        timestamps, prices, decision_window, direction,
        float(config['P5_TP_LONG']), float(config['P5_TP_SHORT']),
        float(config['P5_TRAILING_SL_INTERVAL'])
    )
    return signals

# MANAGER

@jit(nopython=True, fastmath=True)
def apply_manager_logic(df_time_sec, p1_signals, p2_signals, p3_signals, p4_signals, p5_signals, config_cooldown):
    # Signal Manager
    n = len(df_time_sec)
    final_signals = np.zeros(n, dtype=np.float64)
    position = 0.0
    position_owner = 0
    last_signal_time = -config_cooldown
    
    priority_2_locked = False
    priority_3_locked = False
    priority_4_locked = False
    priority_5_locked = False
    
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
        p5_sig = p5_signals[i]
        
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
        
        if pending_reversal and abs(position) < EPS:
            if (current_time - reversal_trigger_time) >= config_cooldown:
                final_sig = reversal_target_signal 
                pending_reversal = False
                position_owner = reversal_owner
                last_signal_time = current_time
                reversal_owner = 0
                reversal_target_signal = 0.0
                reversal_trigger_time = 0.0
                
        elif abs(position) < EPS and not pending_reversal:
            position = 0.0 
            if cooldown_ok:
                if abs(p1_sig) > 0.01:
                    final_sig = p1_sig
                    position_owner = 1
                    priority_2_locked = True
                    priority_3_locked = True
                    priority_4_locked = True
                    priority_5_locked = True
                    last_signal_time = current_time
                elif abs(p2_sig) > 0.01 and not priority_2_locked:
                    final_sig = p2_sig
                    position_owner = 2
                    priority_3_locked = True
                    priority_4_locked = True
                    priority_5_locked = True
                    last_signal_time = current_time
                elif abs(p3_sig) > 0.01 and not priority_3_locked:
                    final_sig = p3_sig
                    position_owner = 3
                    priority_4_locked = True
                    priority_5_locked = True
                    last_signal_time = current_time
                elif abs(p4_sig) > 0.01 and not priority_4_locked:
                    final_sig = p4_sig
                    position_owner = 4
                    priority_5_locked = True
                    last_signal_time = current_time
                elif abs(p5_sig) > 0.01 and not priority_5_locked:
                    final_sig = p5_sig
                    position_owner = 5
                    last_signal_time = current_time
                    
        elif abs(position) > EPS and not pending_reversal:
            if position_owner == 1:
                if cooldown_ok and abs(p1_sig) > 0.01:
                    final_sig = p1_sig
                    last_signal_time = current_time
            elif position_owner == 2:
                if abs(p1_sig) > 0.01:
                    final_sig = -position
                    pending_reversal = True
                    reversal_owner = 1
                    reversal_target_signal = p1_sig
                    reversal_trigger_time = current_time
                    priority_2_locked = True
                    priority_3_locked = True
                    priority_4_locked = True
                    priority_5_locked = True
                    last_signal_time = current_time
                elif cooldown_ok and abs(p2_sig) > 0.01:
                    final_sig = p2_sig
                    last_signal_time = current_time
            elif position_owner == 3:
                if abs(p1_sig) > 0.01:
                    final_sig = -position
                    pending_reversal = True
                    reversal_owner = 1
                    reversal_target_signal = p1_sig
                    reversal_trigger_time = current_time
                    priority_2_locked = True
                    priority_3_locked = True
                    priority_4_locked = True
                    priority_5_locked = True
                    last_signal_time = current_time
                elif abs(p2_sig) > 0.01:
                    final_sig = -position
                    pending_reversal = True
                    reversal_owner = 2
                    reversal_target_signal = p2_sig
                    reversal_trigger_time = current_time
                    priority_3_locked = True
                    priority_4_locked = True
                    priority_5_locked = True
                    last_signal_time = current_time
                elif cooldown_ok and abs(p3_sig) > 0.01:
                    final_sig = p3_sig
                    last_signal_time = current_time
            elif position_owner == 4:
                if abs(p1_sig) > 0.01:
                    final_sig = -position
                    pending_reversal = True
                    reversal_owner = 1
                    reversal_target_signal = p1_sig
                    reversal_trigger_time = current_time
                    priority_2_locked = True
                    priority_3_locked = True
                    priority_4_locked = True
                    priority_5_locked = True
                    last_signal_time = current_time
                elif abs(p2_sig) > 0.01:
                    final_sig = -position
                    pending_reversal = True
                    reversal_owner = 2
                    reversal_target_signal = p2_sig
                    reversal_trigger_time = current_time
                    priority_3_locked = True
                    priority_4_locked = True
                    priority_5_locked = True
                    last_signal_time = current_time
                elif abs(p3_sig) > 0.01:
                    final_sig = -position
                    pending_reversal = True
                    reversal_owner = 3
                    reversal_target_signal = p3_sig
                    reversal_trigger_time = current_time
                    priority_4_locked = True
                    priority_5_locked = True
                    last_signal_time = current_time
                elif cooldown_ok and abs(p4_sig) > 0.01:
                    final_sig = p4_sig
                    last_signal_time = current_time
            elif position_owner == 5:
                if abs(p1_sig) > 0.01:
                    final_sig = -position
                    pending_reversal = True
                    reversal_owner = 1
                    reversal_target_signal = p1_sig
                    reversal_trigger_time = current_time
                    priority_2_locked = True
                    priority_3_locked = True
                    priority_4_locked = True
                    priority_5_locked = True
                    last_signal_time = current_time
                elif abs(p2_sig) > 0.01:
                    final_sig = -position
                    pending_reversal = True
                    reversal_owner = 2
                    reversal_target_signal = p2_sig
                    reversal_trigger_time = current_time
                    priority_3_locked = True
                    priority_4_locked = True
                    priority_5_locked = True
                    last_signal_time = current_time
                elif abs(p3_sig) > 0.01:
                    final_sig = -position
                    pending_reversal = True
                    reversal_owner = 3
                    reversal_target_signal = p3_sig
                    reversal_trigger_time = current_time
                    priority_4_locked = True
                    priority_5_locked = True
                    last_signal_time = current_time
                elif abs(p4_sig) > 0.01:
                    final_sig = -position
                    pending_reversal = True
                    reversal_owner = 4
                    reversal_target_signal = p4_sig
                    reversal_trigger_time = current_time
                    priority_5_locked = True
                    last_signal_time = current_time
                elif cooldown_ok and abs(p5_sig) > 0.01:
                    final_sig = p5_sig
                    last_signal_time = current_time

        if final_sig != 0:
            proposed_position = position + final_sig
            if proposed_position > 1.0 + EPS:
                final_sig = 1.0 - position
            elif proposed_position < -1.0 - EPS:
                final_sig = -1.0 - position
            position += final_sig
            if abs(position) < EPS:
                position = 0.0
                position_owner = 0
        final_sig = round(final_sig, 2)
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
        required_cols = [
            config['TIME_COLUMN'], config['PRICE_COLUMN'],
            config['P1_FEATURE_COLUMN'], config['P4_FEATURE_COLUMN'], 
        ]
        required_cols = list(dict.fromkeys(required_cols))
        
        # Load Data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, usecols=required_cols)
        else:
            df = pd.read_parquet(file_path, columns=required_cols)
        
        if df.empty: return None
        df = df.reset_index(drop=True)
        
        if pd.api.types.is_numeric_dtype(df[config['TIME_COLUMN']]):
             df['Time_sec'] = df[config['TIME_COLUMN']].astype(int)
        else:
             df['Time_sec'] = pd.to_timedelta(df[config['TIME_COLUMN']].astype(str)).dt.total_seconds().astype(int)
        
        # Signals
        p1_signals, df_with_features = generate_p1_signals(df, config)
        p2_signals = generate_p2_signals(df, config)
        p3_signals, _ = generate_p3_signals(df, config) 
        p4_signals = generate_p4_signals(df, config) 
        p5_signals = generate_p5_mean_sigma_signals(df, config)
        time_values = df['Time_sec'].values

        # Manager
        final_signals = apply_manager_logic(
            time_values, p1_signals, p2_signals, p3_signals, p4_signals, p5_signals,
            config['UNIVERSAL_COOLDOWN']
        )
        
        # Stats
        p1_entries = np.sum(np.abs(p1_signals) > 0.5)
        p2_entries = np.sum(np.abs(p2_signals) == 1)
        p3_entries = np.sum(np.abs(p3_signals) == 1)
        p4_entries = np.sum(np.abs(p4_signals) == 1)
        p5_entries = np.sum(np.abs(p5_signals) > 0.5)
        final_entries = np.sum(np.abs(final_signals) > 0.5)
        
        # Output
        output_df = pd.DataFrame({
            'Day': day_num,  # <--- ADD THIS LINE TO FIX THE KEYERROR
            config['TIME_COLUMN']: df[config['TIME_COLUMN']],
            config['PRICE_COLUMN']: df[config['PRICE_COLUMN']],
            'P1_Signal': p1_signals, 'P2_Signal': p2_signals,
            'P3_Signal': p3_signals, 'P4_Signal': p4_signals,
            'P5_Signal': p5_signals, 'Signal': final_signals
        })
        
        output_path = temp_dir_path / f"day{day_num}.csv"
        output_df.to_csv(output_path, index=False)
        
        # Cleanup
        del df, df_with_features
        del p1_signals, p2_signals, p3_signals, p4_signals, p5_signals, final_signals
        
        return {
            'path': str(output_path), 'day_num': day_num,
            'p1_entries': p1_entries, 'p2_entries': p2_entries,
            'p3_entries': p3_entries, 'p4_entries': p4_entries,
            'p5_entries': p5_entries, 'final_entries': final_entries
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

    signals_df = df[df["Signal"] != 0].copy()
    if len(signals_df) == 0:
        print("No trades found in final signal file.")
        return

    trades = []
    position = None
    entry = None

    for idx, row in signals_df.iterrows():
        signal = row["Signal"]
        
        if pd.isna(signal):
            continue

        if position is None:
            if signal in [1, -1]:
                position = signal
                entry = row
        else:
            if signal == -position:
                trades.append({
                    "day": entry["Day"],
                    "direction": "LONG" if position == 1 else "SHORT",
                    "entry_time": entry["Time"],
                    "entry_price": float(entry["Price"]),
                    "exit_time": row["Time"],
                    "exit_price": float(row["Price"]),
                    "pnl": (float(row["Price"]) - float(entry["Price"])) * position
                })
                position = None
                entry = None

    if len(trades) == 0:
        print("No completed trades found.")
        return

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(SAVE_PATH, index=False)

    print(f"✓ trade_reports.csv saved at: {SAVE_PATH}")

def main(config):
    parser = argparse.ArgumentParser(description="EBX Manager")
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--max-workers', type=str, default=None)
    args = parser.parse_args()

    config['DATA_DIR'] = args.data_dir
    config['MAX_WORKERS'] = int(args.max_workers) if args.max_workers else multiprocessing.cpu_count()

    start_time = time.time()
    
    print("STRATEGY MANAGER STARTED")
    print(f"Workers: {config['MAX_WORKERS']} | Cooldown: {config['UNIVERSAL_COOLDOWN']}s")
    
    temp_dir_path = pathlib.Path(config['TEMP_DIR'])
    if temp_dir_path.exists(): shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)
    
    data_dir = config['DATA_DIR']
    all_files = glob.glob(os.path.join(data_dir, "day*.parquet")) + glob.glob(os.path.join(data_dir, "day*.csv"))
    sorted_files = sorted(all_files, key=extract_day_num)
    
    if not sorted_files:
        print("Error: No data files found.")
        return
    
    processed_files = []
    stats = {'p1': 0, 'p2': 0, 'p3': 0, 'p4': 0, 'p5': 0, 'final': 0}
    
    tasks = [(f, extract_day_num(f), str(temp_dir_path), config) for f in sorted_files]
    
    try:
        with multiprocessing.Pool(processes=config['MAX_WORKERS']) as pool:
            for result in pool.imap_unordered(process_wrapper, tasks):
                try:
                    if result:
                        processed_files.append(result['path'])
                        stats['p1'] += result['p1_entries']
                        stats['p2'] += result['p2_entries']
                        stats['p3'] += result['p3_entries']
                        stats['p4'] += result['p4_entries']
                        stats['p5'] += result['p5_entries']
                        stats['final'] += result['final_entries']
                        print(f"Day{result['day_num']:3d} | P1:{result['p1_entries']:3d} P2:{result['p2_entries']:3d} P3:{result['p3_entries']:3d} P4:{result['p4_entries']:3d} P5:{result['p5_entries']:3d} Final:{result['final_entries']:3d}")
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
                    if i != 0: infile.readline()
                    shutil.copyfileobj(infile, outfile)
        
        final_df = pd.read_csv(output_path)
        total_bars = len(final_df)
        entries = (final_df['Signal'].abs() > 0).sum()
        
        print("STATS")
        print(f"Total Entries: {entries:,}")
        print(f"Breakdown: P1:{stats['p1']} P2:{stats['p2']} P3:{stats['p3']} P4:{stats['p4']} P5:{stats['p5']}")
        print(f"Saved: {output_path}")

        # GENERATE REPORTS
        print("Generating trade reports...")
        generate_trade_reports_csv(output_path)

        print(f"Time: {time.time() - start_time:.2f}s")
        
    finally:
        if temp_dir_path.exists(): shutil.rmtree(temp_dir_path)

if __name__ == "__main__":
    main(CONFIG)