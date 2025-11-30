import cudf
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

# ==========================================================
# Configuration
# ==========================================================
DATA_DIR = Path("/data/quant14/EBX")
DAY_TO_PLOT = 87
PRICE_COLUMN = "Price"
TIME_COLUMN = "Time"

# ==========================================================
# Helper Functions
# ==========================================================

def wma(arr, window):
    weights = np.arange(1, window + 1)
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        window_slice = arr[i-window+1 : i+1]
        out[i] = np.dot(window_slice, weights) / weights.sum()
    return out

def hma(series, length):
    half = int(length / 2)
    sqrt_l = int(np.sqrt(length))
    wma_full = wma(series, length)
    wma_half = wma(series, half)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_l)

def kama(series, window=30):
    price = pd.Series(series)
    signal = abs(price.diff(window))
    noise = abs(price.diff()).rolling(window=window).sum()
    er = (signal / noise.replace(0, np.nan)).fillna(0)

    sc_fast = 2 / (2 + 1)
    sc_slow = 2 / (30 + 1)
    sc = ((er * (sc_fast - sc_slow)) + sc_slow) ** 2

    kama_arr = np.full(len(price), np.nan)
    kama_arr[window] = price.iloc[window]

    for i in range(window + 1, len(price)):
        kama_arr[i] = kama_arr[i-1] + sc.iloc[i] * (price.iloc[i] - kama_arr[i-1])

    return kama_arr

# ==========================================================
# Load Data
# ==========================================================
file_path = DATA_DIR / f"day{DAY_TO_PLOT}.parquet"
print(f"Loading {file_path} ...")
df = cudf.read_parquet(file_path)

df_pd = df[[TIME_COLUMN, PRICE_COLUMN]].to_pandas()
df_pd[TIME_COLUMN] = pd.to_timedelta(df_pd[TIME_COLUMN])
df_pd = df_pd.set_index(TIME_COLUMN).sort_index()

# ==========================================================
# Indicators computed on 1-second raw data
# ==========================================================
prices = df_pd[PRICE_COLUMN].values

df_pd["EMA_120"] = df_pd[PRICE_COLUMN].ewm(span=120, adjust=False).mean()
df_pd["SMA_180"] = df_pd[PRICE_COLUMN].rolling(180).mean()
df_pd["HMA_9"]   = hma(prices, 9)
df_pd["KAMA"]    = kama(prices, window=30)
df_pd["KAMA_Slope"] = df_pd["KAMA"].diff().shift(1)

# ==========================================================
# Resample to 5-second OHLC
# ==========================================================
ohlc = df_pd[PRICE_COLUMN].resample("5s").ohlc()
ohlc = ohlc.dropna()

# ==========================================================
# Heiken-Ashi Construction
# ==========================================================
ha = pd.DataFrame(index=ohlc.index)

ha["close"] = (ohlc["open"] + ohlc["high"] + ohlc["low"] + ohlc["close"]) / 4

ha["open"] = np.nan
ha.iloc[0, ha.columns.get_loc("open")] = (ohlc["open"].iloc[0] + ohlc["close"].iloc[0]) / 2
for i in range(1, len(ohlc)):
    ha.iloc[i, ha.columns.get_loc("open")] = (ha["open"].iloc[i-1] + ha["close"].iloc[i-1]) / 2

ha["high"] = ha.loc[:, ["open", "close"]].join(ohlc["high"]).max(axis=1)
ha["low"]  = ha.loc[:, ["open", "close"]].join(ohlc["low"]).min(axis=1)

# ==========================================================
# Resample Indicators to 5s (plot alignment)
# ==========================================================
indicators = df_pd[["EMA_120", "SMA_180", "HMA_9", "KAMA", "KAMA_Slope"]].resample("5s").last()

# ==========================================================
# Plotly Chart
# ==========================================================
fig = go.Figure()

# --- Heiken-Ashi Candles ---
fig.add_trace(go.Candlestick(
    x=ha.index,
    open=ha["open"],
    high=ha["high"],
    low=ha["low"],
    close=ha["close"],
    name="Heiken-Ashi (5s)",
    increasing_line_color="green",
    decreasing_line_color="red",
    opacity=0.7
))

# --- Indicators (from 1s, aligned to 5s) ---
fig.add_trace(go.Scatter(
    x=indicators.index, y=indicators["HMA_9"],
    mode="lines", name="HMA 9", line=dict(color="purple", width=2)
))

fig.add_trace(go.Scatter(
    x=indicators.index, y=indicators["EMA_120"],
    mode="lines", name="EMA 120", line=dict(color="blue", width=1.5)
))

fig.add_trace(go.Scatter(
    x=indicators.index, y=indicators["SMA_180"],
    mode="lines", name="SMA 180", line=dict(color="orange", width=1.5)
))

fig.add_trace(go.Scatter(
    x=indicators.index, y=indicators["KAMA"],
    mode="lines", name="KAMA", line=dict(color="black", width=2)
))

fig.add_trace(go.Scatter(
    x=indicators.index, y=indicators["KAMA_Slope"],
    mode="lines", name="KAMA Slope", line=dict(color="gray", width=1, dash="dot")
))

# ==========================================================
# Layout
# ==========================================================
fig.update_layout(
    title=f"Day {DAY_TO_PLOT} — Heiken-Ashi (5s) with HMA9 + EMA120 + SMA180 + KAMA",
    xaxis_title="Time",
    yaxis_title="Price",
    template="plotly_white",
    height=800
)

fig.show()
