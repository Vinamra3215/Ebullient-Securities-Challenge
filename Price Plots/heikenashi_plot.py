import cudf
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import numpy as np

# ==========================================================
# CONFIGURATION
# ==========================================================
DATA_DIR = Path("/data/quant14/EBX")  # directory containing dayX.parquet files
DAY_TO_PLOT = 87                      # Day number
PRICE_COLUMN = "Price"
TIME_COLUMN = "Time"
RESAMPLE_SECONDS = 5                  # <-- 5-second candles
# ==========================================================

# Construct the parquet file path
file_path = DATA_DIR / f"day{DAY_TO_PLOT}.parquet"

# Load data using GPU (cuDF)
print(f"\nLoading {file_path} ...")
df = cudf.read_parquet(file_path)
print(f"Loaded {len(df):,} rows from Day {DAY_TO_PLOT}")

# ----------------------------------------------------------
# Convert to pandas for processing and plotting
# ----------------------------------------------------------
df_pd = df[[TIME_COLUMN, PRICE_COLUMN]].to_pandas()
df_pd[TIME_COLUMN] = pd.to_timedelta(df_pd[TIME_COLUMN])
df_pd = df_pd.set_index(TIME_COLUMN)

# ----------------------------------------------------------
# Step 1: RESAMPLE to Normal OHLC (5-second)
# ----------------------------------------------------------
ohlc = df_pd[PRICE_COLUMN].resample(f"{RESAMPLE_SECONDS}S").ohlc().dropna()

# ----------------------------------------------------------
# Step 2: Convert OHLC → Heiken-Ashi
# ----------------------------------------------------------

# Extract series
open_ = ohlc['open'].copy()
high_ = ohlc['high'].copy()
low_ = ohlc['low'].copy()
close_ = ohlc['close'].copy()

# Heiken-Ashi arrays
ha_open = np.zeros(len(ohlc))
ha_high = np.zeros(len(ohlc))
ha_low = np.zeros(len(ohlc))
ha_close = np.zeros(len(ohlc))

# HA Close = (O + H + L + C) / 4
ha_close = (open_ + high_ + low_ + close_) / 4

# HA Open first value = (O + C)/2
ha_open[0] = (open_.iloc[0] + close_.iloc[0]) / 2

# Other HA Opens = (prev_HA_Open + prev_HA_Close) / 2
for i in range(1, len(ohlc)):
    ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2

# HA High = max(H, HA_Open, HA_Close)
ha_high = np.maximum.reduce([high_, ha_open, ha_close])

# HA Low = min(L, HA_Open, HA_Close)
ha_low = np.minimum.reduce([low_, ha_open, ha_close])

# Create DataFrame for plotting
ha_df = pd.DataFrame({
    "open": ha_open,
    "high": ha_high,
    "low": ha_low,
    "close": ha_close
}, index=ohlc.index)

print(f"\nGenerated {len(ha_df)} Heiken-Ashi candles")

# ----------------------------------------------------------
# Step 3: Plot Heiken-Ashi Candlesticks
# ----------------------------------------------------------
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=ha_df.index,
    open=ha_df['open'],
    high=ha_df['high'],
    low=ha_df['low'],
    close=ha_df['close'],
    name="Heiken-Ashi (5s)"
))

fig.update_layout(
    title=f"Day {DAY_TO_PLOT} — Heiken-Ashi Candlestick Chart (5s)",
    xaxis_title="Time",
    yaxis_title="Price ($)",
    template="plotly_white",
    hovermode="x unified",
    height=750,
    showlegend=False
)

fig.update_xaxes(rangeslider_visible=False)
fig.show()
