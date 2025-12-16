import dask_cudf
import cudf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import pandas as pd
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from filter import get_pairs_for_day
import math
import sys

warnings.filterwarnings('ignore')

# ==========================================
# SECTION 1: CORE MATH & HELPERS
# ==========================================

def ensure_pandas(df):
    """Converts dask_cudf or cudf DataFrames to Pandas."""
    if isinstance(df, dask_cudf.DataFrame):
        return df.compute().to_pandas()
    elif isinstance(df, cudf.DataFrame):
        return df.to_pandas()
    return df

def rolling_fft_reconstruction(series, window=120, top_k=3):
    """Rolling Fourier Transform reconstruction."""
    n = len(series)
    out = np.full(n, np.nan)
    series = np.array(series, dtype=float)

    for i in range(window, n):
        segment = series[i-window:i]
        fft_vals = np.fft.fft(segment)
        idx = np.argsort(np.abs(fft_vals))[::-1]
        fft_filtered = np.zeros_like(fft_vals)
        fft_filtered[idx[:top_k]] = fft_vals[idx[:top_k]]
        reconstructed = np.fft.ifft(fft_filtered).real
        out[i] = reconstructed[-1]
    return out

def apply_recursive_filter(series, sc):
    """Generic recursive filter: out[i] = out[i-1] + sc[i] * (series[i] - out[i-1])"""
    n = len(series)
    out = np.full(n, np.nan)
    
    start_idx = 0
    if np.isnan(series[0]):
        valid_indices = np.where(~np.isnan(series))[0]
        if len(valid_indices) > 0:
            start_idx = valid_indices[0]
        else:
            return out 
            
    out[start_idx] = series[start_idx]
    
    for i in range(start_idx + 1, n):
        c = sc[i] if not np.isnan(sc[i]) else 0
        target = series[i]
        if not np.isnan(target):
            out[i] = out[i-1] + c * (target - out[i-1])
        else:
            out[i] = out[i-1]
            
    return out

def get_super_smoother_array(prices, period):
    """Returns just the numpy array for SuperSmoother."""
    n = len(prices)
    pi = math.pi
    a1 = math.exp(-1.414 * pi / period)
    b1 = 2 * a1 * math.cos(1.414 * pi / period)
    c1 = 1 - b1 + a1*a1
    c2 = b1
    c3 = -a1*a1
    
    ss = np.zeros(n)
    if n > 0: ss[0] = prices[0]
    if n > 1: ss[1] = prices[1]
    
    for i in range(2, n):
        ss[i] = c1*prices[i] + c2*ss[i-1] + c3*ss[i-2]
    return ss

# ==========================================
# SECTION 2: INDICATOR GENERATORS
# ==========================================

def add_basic_super_smoothers(df, price_col="Price"):
    """Calculates Fast and Slow Super Smoothers from scratch."""
    prices = df[price_col].values
    # Defined periods
    df["EhlersSuperSmoother"] = get_super_smoother_array(prices, period=30)
    df["EhlersSuperSmoother_Slow"] = get_super_smoother_array(prices, period=1800)
    return df

def add_kama_suite(df, price_col="Price"):
    """Calculates KAMA and Regime detection."""
    prices = df[price_col].values
    
    # Define KAMA Parameters
    period, fast, slow = 10, 20, 60
    
    # Calculate components locally
    direction = df[price_col].diff(period).abs()
    volatility = df[price_col].diff(1).abs().rolling(period).sum()
    er = (direction / volatility).fillna(0)
    
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (((er * (fast_sc - slow_sc)) + slow_sc)**2).values
    
    # Generate KAMA
    df['Kama'] = apply_recursive_filter(prices, sc)
    df['Kama_Slope'] = df['Kama'].diff().fillna(0).abs()
    
    return df

def add_eama_suite(df):
    """
    Calculates EAMA. 
    DEPENDENCY: Requires 'EhlersSuperSmoother' to be calculated first.
    """
    if "EhlersSuperSmoother" not in df.columns:
        raise ValueError("EhlersSuperSmoother must be calculated before EAMA")
        
    ss_series = df["EhlersSuperSmoother"]
    
    # Define EAMA Parameters
    period, fast, slow = 15, 30, 120
    
    # Calculate ER on the SuperSmoother (Fresh calculation)
    direction = ss_series.diff(period).abs()
    volatility = ss_series.diff(1).abs().rolling(period).sum()
    er = (direction / volatility).fillna(0).values
    
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = ((er * (fast_sc - slow_sc)) + slow_sc) ** 2
    
    # Generate EAMA
    df["EAMA"] = apply_recursive_filter(ss_series.values, sc)
    df["EAMA_Slope"] = df["EAMA"].diff().fillna(0)
    df["EAMA_Slope_MA"] = df["EAMA_Slope"].rolling(5).mean()
    
    return df

def add_ftama_suite(df, price_col="Price"):
    """Calculates FFT and FTAMA."""
    # 1. Generate FFT Series
    fft_vals = rolling_fft_reconstruction(df[price_col], window=120, top_k=3)
    df["FFT"] = fft_vals # Store raw FFT
    
    # 2. Define FTAMA Parameters
    period, fast, slow = 15, 30, 120
    
    # 3. Calculate ER on the FFT Series (Fresh calculation)
    fft_s = pd.Series(fft_vals)
    direction = fft_s.diff(period).abs()
    volatility = fft_s.diff(1).abs().rolling(period).sum()
    er = (direction / volatility).fillna(0).values
    
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = ((er * (fast_sc - slow_sc)) + slow_sc) ** 2
    
    # 4. Generate FTAMA
    df["FTAMA"] = apply_recursive_filter(fft_vals, sc)
    return df

def add_hybrid_ham(df, price_col="Price"):
    """Calculates Hybrid Adaptive Moving Average (HAM)."""
    # REQUIREMENTS
    if "EhlersSuperSmoother" not in df.columns:
        raise ValueError("EhlersSuperSmoother must be calculated before HAM")

    prices = df[price_col].values
    n = len(prices)
    fft_slow = rolling_fft_reconstruction(prices, window=180, top_k=3)
    er_period = 30
    fast, slow = 30, 120

    direction = df[price_col].diff(er_period).abs().values
    volatility = df[price_col].diff().abs().rolling(er_period).sum().values

    er = np.zeros(n)
    mask = volatility != 0
    er[mask] = direction[mask] / volatility[mask]

    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    sc = np.nan_to_num(sc, nan=0.0)
    
    fast_ma = df["EhlersSuperSmoother"].values       # super smoother (fast)
    slow_ma = fft_slow                                # FFT-based smoother
    
    ham_vals = sc * fast_ma + (1.0 - sc) * slow_ma

    ham_vals = pd.Series(ham_vals).bfill().ffill().values
    # ----------------------------------------

    df["HAM"] = ham_vals
    threshold = 0.02 # Define a threshold for significant change
    ham_smooth = np.zeros(n)
    
    # Initialize safely
    ham_smooth[0] = ham_vals[0] if not np.isnan(ham_vals[0]) else prices[0]

    # Iterative filter: requires previous step's smooth value
    for i in range(1, n):
        # Only update if the difference between raw HAM and prev Smooth is > threshold
        if abs(ham_vals[i] - ham_smooth[i-1]) > threshold:
            ham_smooth[i] = ham_vals[i]
        else:
            ham_smooth[i] = ham_smooth[i-1]

    df["HAM_Smooth"] = ham_smooth
    df["HAM_Slope"] = np.append([0], np.diff(ham_vals))
    df["HAM_MA"] = pd.Series(ham_vals).rolling(30).mean()
    df["HAM_EMA"] = pd.Series(ham_vals).ewm(span=30).mean()

    return df


def add_jma300(df, price_col="Price"):
    """Calculates Jurik Moving Average approximation."""
    length = 300
    prices = df[price_col].values
    
    vol = np.abs(np.diff(prices, prepend=prices[0]))
    vol_smooth = pd.Series(vol).rolling(window=length, min_periods=1).mean().fillna(0).values
    vol_mean = pd.Series(vol_smooth).rolling(window=length, min_periods=1).mean().replace(0, 1.0).values
    vol_norm = np.clip(vol_smooth / vol_mean, 0.1, 10.0)
    
    alpha = 2.0 / (length + 1.0)
    alpha_adapt = np.clip(alpha * (1 + 0.5 * (vol_norm - 1)), 0.001, 0.999)
    
    out = np.empty(len(prices))
    out[0] = prices[0]
    for i in range(1, len(prices)):
        out[i] = alpha_adapt[i] * prices[i] + (1.0 - alpha_adapt[i]) * out[i-1]
    
    df["JMA300"] = out
    return df

def add_jma1800(df, price_col="Price"):
    """Calculates Jurik Moving Average approximation."""
    length = 1800
    prices = df[price_col].values
    
    vol = np.abs(np.diff(prices, prepend=prices[0]))
    vol_smooth = pd.Series(vol).rolling(window=length, min_periods=1).mean().fillna(0).values
    vol_mean = pd.Series(vol_smooth).rolling(window=length, min_periods=1).mean().replace(0, 1.0).values
    vol_norm = np.clip(vol_smooth / vol_mean, 0.1, 10.0)
    
    alpha = 2.0 / (length + 1.0)
    alpha_adapt = np.clip(alpha * (1 + 0.5 * (vol_norm - 1)), 0.001, 0.999)
    
    out = np.empty(len(prices))
    out[0] = prices[0]
    for i in range(1, len(prices)):
        out[i] = alpha_adapt[i] * prices[i] + (1.0 - alpha_adapt[i]) * out[i-1]
    
    df["JMA1800"] = out
    return df

def resample_to_candles(data_df, period_seconds=90):
    """Resample tick data to OHLC candles - MATCHED TO VISUALIZATION CODE"""
    data_df = data_df.copy()
    
    # Handle Time column - check if already datetime or needs conversion
    if 'Time' in data_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(data_df['Time']):
            data_df['Time'] = pd.to_timedelta(data_df['Time'])
            data_df['DateTime'] = pd.Timestamp('2024-01-01') + data_df['Time']
        else:
            data_df['DateTime'] = data_df['Time']
    
    data_df = data_df.set_index('DateTime')
    
    # Resample to specified period (FIXED: use 's' instead of 'S')
    ohlc = data_df['Price'].resample(f'{period_seconds}s').ohlc()
    ohlc = ohlc.dropna()
    
    return ohlc


def calculate_heikin_ashi(ohlc_df):
    """Calculate Heikin-Ashi candles from OHLC data - FIXED chained assignment"""
    ha_df = pd.DataFrame(index=ohlc_df.index)
    
    # Calculate HA_Close
    ha_df['HA_Close'] = (ohlc_df['open'] + ohlc_df['high'] + 
                          ohlc_df['low'] + ohlc_df['close']) / 4
    
    # Initialize HA_Open array
    ha_open = np.zeros(len(ohlc_df), dtype=np.float64)
    ha_open[0] = (ohlc_df['open'].iloc[0] + ohlc_df['close'].iloc[0]) / 2
    
    # Calculate subsequent HA_Open values using loop (FIXED: proper assignment)
    ha_close_values = ha_df['HA_Close'].values
    for i in range(1, len(ha_open)):
        ha_open[i] = (ha_open[i-1] + ha_close_values[i-1]) / 2
    
    ha_df['HA_Open'] = ha_open
    
    # Calculate HA_High and HA_Low
    ha_df['HA_High'] = pd.concat([ohlc_df['high'], ha_df['HA_Open'], ha_df['HA_Close']], axis=1).max(axis=1)
    ha_df['HA_Low'] = pd.concat([ohlc_df['low'], ha_df['HA_Open'], ha_df['HA_Close']], axis=1).min(axis=1)
    
    return ha_df

def add_hama(df, ohlc_df):
    """Calculates Heikin-Ashi Modified Average (HAMA) and aligns it to original index."""
    # 1. Calculate HA on the resampled data
    ha_df = calculate_heikin_ashi(ohlc_df)
    
    # 2. Calculate the Moving Average on the RESAMPLED data
    # (Doing it here is faster than expanding to ticks first)
    ha_ma = ha_df["HA_Close"].rolling(100).mean()
    
    # 3. Align back to the original DataFrame
    # Ensure original df has DateTime set as index for alignment, or use merge_asof
    # Assuming df has 'DateTime' column from resample_to_candles function:
    
    if 'DateTime' not in df.columns:
         # Fallback if DateTime isn't created yet, though analyze_feature creates it.
         df['DateTime'] = pd.to_timedelta(df['Time']) + pd.Timestamp('2024-01-01')

    # Create a temporary series with the correct datetime index
    hama_series = pd.Series(ha_ma.values, index=ha_df.index)
    
    # Reindex to match the original dataframe's DateTime
    # method='ffill' propagates the 90s candle value across all ticks in that 90s window
    aligned_hama = hama_series.reindex(df['DateTime'], method='ffill')
    
    df["HAMA"] = aligned_hama.values
    return df
  
    return df
def add_PAMA_suite(df, price_col="Price"):
    """
    Calculates PAMA (Proxy/Price Adaptive Moving Average).
    Logic: Calculates an SMA (Period 30) first, then applies 
    KAMA-style adaptive weighting (15/30/120) to that SMA.
    """
    # 1. Generate the Input Source: SMA (Period 30)
    # We calculate this locally to ensure it exists and matches the 
    # timeframe of the other baselines (SuperSmoother/HAMA).
    sma_source = df[price_col].rolling(30).mean()
    
    # 2. Define PAMA Adaptive Parameters
    period = 15   # Efficiency Ratio period
    fast = 30     # Fast SC limit
    slow = 120    # Slow SC limit
    
    # 3. Calculate ER on the SMA Source
    # Direction: Net change of the SMA over 'period'
    direction = sma_source.diff(period).abs()
    
    # Volatility: Sum of absolute changes of the SMA over 'period'
    volatility = sma_source.diff(1).abs().rolling(period).sum()
    
    # Efficiency Ratio
    er = (direction / volatility).fillna(0).values
    
    # 4. Calculate Smoothing Constant (SC)
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = ((er * (fast_sc - slow_sc)) + slow_sc) ** 2
    
    # 5. Generate PAMA using the recursive filter
    # Note: The filter handles the initial NaNs from the SMA calculation
    df["PAMA"] = apply_recursive_filter(sma_source.values, sc)
    
    # 6. Calculate Slopes for analysis
    df["PAMA_Slope"] = df["PAMA"].diff().fillna(0)
    df["PAMA_Slope_MA"] = df["PAMA_Slope"].rolling(5).mean()
    
    return df

def add_zlama_suite(df, price_col="Price"):
    """
    Calculates ZLAMA (Zero Lag Adaptive Moving Average).
    Logic: Calculates a ZLEMA (Period 30) first, then applies 
    KAMA-style adaptive weighting (15/30/120) to that ZLEMA.
    """
    # 1. Generate the Input Source: ZLEMA (Period 30)
    # This matches the baseline period used in EAMA (SuperSmoother) and PAMA (SMA).
    z_per = 30
    lag = int((z_per - 1) / 2)
    
    # Calculate De-lagged data
    z_dat = 2 * df[price_col] - df[price_col].shift(lag)
    
    # Calculate EMA of de-lagged data (The Source Series)
    zlema_source = z_dat.ewm(span=z_per, adjust=False).mean()
    
    # 2. Define ZLAMA Adaptive Parameters
    period = 15   # Efficiency Ratio period
    fast = 30     # Fast SC limit
    slow = 120    # Slow SC limit
    
    # 3. Calculate ER on the ZLEMA Source
    # Direction: Net change of the ZLEMA over 'period'
    direction = zlema_source.diff(period).abs()
    
    # Volatility: Sum of absolute changes of the ZLEMA over 'period'
    volatility = zlema_source.diff(1).abs().rolling(period).sum()
    
    # Efficiency Ratio
    er = (direction / volatility).fillna(0).values
    
    # 4. Calculate Smoothing Constant (SC)
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = ((er * (fast_sc - slow_sc)) + slow_sc) ** 2
    
    # 5. Generate ZLAMA using the recursive filter
    df["ZLAMA"] = apply_recursive_filter(zlema_source.values, sc)
    
    # 6. Calculate Slopes for analysis
    df["ZLAMA_Slope"] = df["ZLAMA"].diff().fillna(0)
    df["ZLAMA_Slope_MA"] = df["ZLAMA_Slope"].rolling(5).mean()
    
    return df

def add_ekf(df):
    """Calculates Extended Kalman Filter Trend."""
    ekf = ExtendedKalmanFilter(dim_x=1, dim_z=1)
    ekf.x = np.array([[df["Price"].iloc[0]]])
    ekf.F = np.array([[1.0]])
    ekf.Q = np.array([[0.07]]) #default 0.05
    ekf.R = np.array([[0.2]]) #default 0.2
    
    def h(x): return np.log(x)
    def H_jac(x): return np.array([[1.0/x[0][0]]])
    
    vals = []
    clean_price = df["Price"].dropna()
    for p in clean_price.values:
        ekf.predict()
        ekf.update(np.array([[np.log(p)]]), HJacobian=H_jac, Hx=h)
        vals.append(ekf.x.item())
        
    df["EKFTrend"] = pd.Series(vals, index=clean_price.index).ffill().shift(1)
    return df

def add_ucm(df):
    """Calculates Unobserved Components Model."""
    n = len(df)
    alpha_ucm, beta_ucm = 0.005, 0.0001
    mu = np.zeros(n)
    beta_slope = np.zeros(n)
    filtered = np.zeros(n)
    
    fvi = df['Price'].first_valid_index()
    if fvi is not None:
        mu[fvi] = df['Price'].loc[fvi]
        filtered[fvi] = mu[fvi]
        for t in range(fvi+1, n):
            price_t = df['Price'].iloc[t]
            if np.isnan(price_t):
                mu[t], beta_slope[t], filtered[t] = mu[t-1], beta_slope[t-1], filtered[t-1]
                continue
            mu_pred = mu[t-1] + beta_slope[t-1]
            mu[t] = mu_pred + alpha_ucm * (price_t - mu_pred)
            beta_slope[t] = beta_slope[t-1] + beta_ucm * (price_t - mu_pred)
            filtered[t] = mu[t] + beta_slope[t]
            
    df["UCM"] = filtered
    return df

# ==========================================
# NEW: EAMA CUSUM CALCULATOR
# ==========================================

def add_eama_cusum(df, drift=0.0005, decay=0.95):

    e = df["EAMA"].values.astype(float)
    n = len(e)

    # EAMA changes
    changes = np.zeros(n)
    for i in range(1, n):
        changes[i] = e[i] - e[i-1]

    cusum_up = 0.0
    cusum_down = 0.0

    up_vals = np.zeros(n)
    down_vals = np.zeros(n)

    for i in range(1, n):
        ch = changes[i]

        if ch > drift:
            cusum_up += ch
            cusum_down = max(0.0, cusum_down * decay)

        elif ch < -drift:
            cusum_down += abs(ch)
            cusum_up = max(0.0, cusum_up * decay)

        else:
            cusum_up = max(0.0, cusum_up * decay)
            cusum_down = max(0.0, cusum_down * decay)

        up_vals[i] = cusum_up
        down_vals[i] = cusum_down

    df["EAMA_Cusum_Up"] = up_vals
    df["EAMA_Cusum_Down"] = down_vals
    df["EAMA_Cusum"] = up_vals - down_vals     # <-- main series you will plot

    return df


def add_standard_indicators(df):
    """Calculates Moving Averages, ZLEMA, BC, TSM from scratch."""
    # Basics
    df['ATR'] = df['Price'].diff().abs().ewm(60).mean()
    df['Slow_MA'] = df['Price'].rolling(1800).mean()
    df['Fast_MA'] = df['Price'].rolling(180).mean()
    df['STD'] = df['Price'].rolling(20).std()
    df['EMA_Fast'] = df['Price'].ewm(span=25, adjust=False).mean()
    df['EMA_Slow'] = df['Price'].ewm(span=1800, adjust=False).mean()
    
    # ZLEMA
    z_per = 1800
    lag = int((z_per - 1) / 2)
    z_dat = 2 * df['Price'] - df['Price'].shift(lag)
    df['ZLEMA'] = z_dat.ewm(span=z_per, adjust=False).mean()

    return df

# ==========================================
# SECTION 3: PLOTTING & MAIN
# ==========================================

def plot_market_analysis(data_df, selected_features=None):
    """Handles the Plotly visualization - single plot only."""
    
    # Exclude technical columns from the feature list used for plotting loop
    exclude_cols = ['Time','Price','Day','PlotIndex','KAMA_Mode','KAMA_Switch', 
                    'HA_Open', 'HA_High', 'HA_Low', 'HA_Close']
    
    feature_columns = [c for c in data_df.columns if c not in exclude_cols]
    if selected_features:
        feature_columns = [c for c in feature_columns if c in selected_features]

    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )

    1# 1. Plot Price
    for day, df_day in data_df.groupby("Day"):
        ts = df_day["Time"].apply(lambda x: str(x).split()[-1]).tolist()
        fig.add_trace(go.Scatter(
            x=df_day["PlotIndex"], y=df_day["Price"], mode='lines',
            name=f"Price Day {day}",
            line=dict(width=2, color='rgba(0,0,0,0.7)'),
            hovertemplate="<b>Day</b>: %{customdata[0]}<br><b>Time</b>: %{customdata[1]}<br><b>Price</b>: %{y:.4f}<extra></extra>",
            customdata=[[day, t] for t in ts],
            showlegend=False
        ), row=1, col=1, secondary_y=False)
    # 1. Plot Price with color-coded segments based on EAMA_Cusum sign (3-state logic)
    # THRESH = 1e-10

    # for day, df_day in data_df.groupby("Day"):
    #     if "EAMA_Cusum" not in df_day.columns:
    #         raise ValueError("EAMA_Cusum must exist before plotting colored price segments")

    #     cusum_vals = df_day["EAMA_Cusum"].values
    #     prices = df_day["Price"].values
    #     xs = df_day["PlotIndex"].values

    #     def get_color(c):
    #         if c > THRESH:
    #             return "green"
    #         elif c < -THRESH:
    #             return "red"
    #         return "black"   # middle region stays black

    #     start = 0
    #     current_color = get_color(cusum_vals[0])

    #     for i in range(1, len(df_day)):
    #         new_color = get_color(cusum_vals[i])

    #         # if color changes → end previous segment and start new one
    #         if new_color != current_color:
    #             fig.add_trace(go.Scatter(
    #                 x=xs[start:i+1],
    #                 y=prices[start:i+1],
    #                 mode="lines",
    #                 name=f"Price Day {day}",
    #                 line=dict(width=2, color=current_color),
    #                 showlegend=False
    #             ))
    #             start = i
    #             current_color = new_color

    #     # final segment
    #     fig.add_trace(go.Scatter(
    #         x=xs[start:],
    #         y=prices[start:],
    #         mode="lines",
    #         name=f"Price Day {day}",
    #         line=dict(width=2, color=current_color),
    #         showlegend=False
    #     ))



    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    
    # 2. Plot Indicators (all on secondary y-axis)
    for i, col in enumerate(feature_columns):
        if col not in data_df.columns: continue
        
        # Clone for normalization
        norm_col = f"{col}_norm"
        data_df[norm_col] = data_df[col]
        
        for day, df_day in data_df.groupby("Day"):
            mask = df_day[norm_col].notna()
            if mask.sum() == 0: continue
            
            fig.add_trace(go.Scatter(
                x=df_day.loc[mask, "PlotIndex"], 
                y=df_day.loc[mask, norm_col],
                mode='lines', 
                name=col, 
                legendgroup=col, 
                visible='legendonly',
                line=dict(color=colors[i % len(colors)], width=1.5),
            ), row=1, col=1, secondary_y=True)

    # 3. Regime Background Shading
    if "KAMA_Mode" in data_df.columns:
        df_plot = data_df[["PlotIndex", "KAMA_Mode"]].copy()
        df_plot['change'] = df_plot['KAMA_Mode'].diff().fillna(0)
        
        starts = df_plot[(df_plot['change'] == 1) | ((df_plot['KAMA_Mode'] == 1) & (df_plot.index == 0))]['PlotIndex'].values
        ends = df_plot[(df_plot['change'] == -1) | ((df_plot['KAMA_Mode'] == 1) & (df_plot.index == len(df_plot)-1))]['PlotIndex'].values

        if len(starts) > 0 and len(ends) > 0:
            if starts[0] > ends[0]: starts = np.insert(starts, 0, df_plot['PlotIndex'].iloc[0])
            if len(starts) > len(ends): ends = np.append(ends, df_plot['PlotIndex'].iloc[-1])

        shapes = [dict(type="rect", xref="x", yref="paper", x0=s, y0=0, x1=e, y1=1, fillcolor="rgba(0,255,0,0.15)", line_width=0, layer="below") for s, e in zip(starts, ends)]
        fig.update_layout(shapes=shapes)

    tick_vals = [data_df[data_df["Day"]==d]["PlotIndex"].iloc[0] for d in range(int(data_df["Day"].max())+1) if not data_df[data_df["Day"]==d].empty]
    tick_txt = [f"Day {d}" for d in range(len(tick_vals))]
    
    fig.update_layout(title='Multi-Day Analysis', hovermode='x unified', height=800, width=1600, plot_bgcolor='white')
    fig.update_xaxes(tickvals=tick_vals, ticktext=tick_txt, row=1, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)')
    
    fig.show()

def analyze_feature(data_df, price_pairs=None, selected_features=None): 
    # 1. Clean Slate: Keep only essential columns to enforce fresh calculation
    data_df = ensure_pandas(data_df)
    
    # We strip down to just the Time/Price to ensure no old columns leak in
    analysis_df = data_df[['Time', 'Price']].copy()
    
    # Re-apply Indexing logic
    analysis_df["Time"] = pd.to_timedelta(analysis_df["Time"])
    analysis_df["Day"] = (analysis_df["Time"].diff() < pd.Timedelta(0)).cumsum()
    analysis_df = analysis_df.dropna(subset=["Price"]).reset_index(drop=True)
    analysis_df["PlotIndex"] = range(len(analysis_df))

    # 2. Sequential Pipeline (Strict Dependency Order)
    
    # Step A: Basic Components
    analysis_df = add_basic_super_smoothers(analysis_df) # Adds SS, SS_Slow
    analysis_df = add_kama_suite(analysis_df)            # Adds Kama
    
    # Step B: Dependent Components
    analysis_df = add_eama_suite(analysis_df)            # Uses SS
    analysis_df = add_eama_cusum(analysis_df)            # Uses EAMA
    analysis_df = add_hybrid_ham(analysis_df)            # Uses SS + Kama + Local ER
    analysis_df = add_zlama_suite(analysis_df)           # Uses ZLEMA
    
    # Step C: Independent/Parallel Components
    analysis_df = add_ftama_suite(analysis_df)
    analysis_df = add_jma300(analysis_df)
    analysis_df = add_jma1800(analysis_df)
    ohlc_df = resample_to_candles(analysis_df, period_seconds=90)
    analysis_df = add_hama(analysis_df, ohlc_df)
    analysis_df = add_PAMA_suite(analysis_df)            # Uses HAMA
    analysis_df = add_ekf(analysis_df)
    analysis_df = add_ucm(analysis_df)
    analysis_df = add_standard_indicators(analysis_df)

    # 3. Visualize
    plot_market_analysis(analysis_df, selected_features)

if __name__=="__main__":
    sys.path.append('.')
    days = [87, 421, 311, 273, 36]

    for d in days:
        print(f"Processing Day {d}...")
        success, pairs, df = get_pairs_for_day(d)
        if not success: continue

        # List of features to visualize
        feats = ['ATR', 'Kama', 'ER', 'EMA_Fast', 'EMA_Slow', 'ZLEMA', 'HAMA',
                  'HAM', 'HAM_Smooth', 'HAM_MA', 'HAM_EMA', 'PAMA', 'Kama_Slope',
                 'Slow_MA', 'Fast_MA', 'UCM', 'EAMA', 'EAMA_Slope', 'EAMA_Slope_MA',
                 'EKFTrend', 'EhlersSuperSmoother', 'JMA300', 'JMA1800', 'EhlersSuperSmoother_Slow']
        
        # Pass the raw dataframe; analyze_feature will strip it and rebuild indicators
        analyze_feature(cudf.from_pandas(df), price_pairs=pairs, selected_features=feats)