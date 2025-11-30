import cudf
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

# ==========================================================
# CONFIGURATION
# ==========================================================
DATA_DIR = Path("/data/quant14/EBX")  # directory containing dayX.parquet files
DAY_TO_PLOT = 87                      # Day number
PRICE_COLUMN = "Price"
TIME_COLUMN = "Time"
RESAMPLE_SECONDS = 5                  # <-- 5-second candlesticks
# ==========================================================

# Construct the parquet file path
file_path = DATA_DIR / f"day{DAY_TO_PLOT}.parquet"

# Load data using GPU (cuDF)
print(f"\nLoading {file_path} ...")
df = cudf.read_parquet(file_path)
print(f"Loaded {len(df):,} rows from Day {DAY_TO_PLOT}")

# ----------------------------------------------------------
# Convert to pandas for plotting
# ----------------------------------------------------------
df_pd = df[[TIME_COLUMN, PRICE_COLUMN]].to_pandas()

# Convert 'Time' into a proper datetime-like index
df_pd[TIME_COLUMN] = pd.to_timedelta(df_pd[TIME_COLUMN])
df_pd = df_pd.set_index(TIME_COLUMN)

# ----------------------------------------------------------
# RESAMPLE TO 5-SECOND OHLC
# ----------------------------------------------------------
ohlc = df_pd[PRICE_COLUMN].resample(f"{RESAMPLE_SECONDS}S").ohlc()

# Drop empty buckets (if any)
ohlc = ohlc.dropna()

print("\nGenerated 5-second OHLC bars:", len(ohlc))
print(ohlc.head())

# ----------------------------------------------------------
# Plot: Candlestick Chart
# ----------------------------------------------------------
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=ohlc.index,
    open=ohlc['open'],
    high=ohlc['high'],
    low=ohlc['low'],
    close=ohlc['close'],
    name="5s Candles"
))

fig.update_layout(
    title=f"Day {DAY_TO_PLOT} — 5-Second OHLC Candlestick Chart",
    xaxis_title="Time",
    yaxis_title="Price ($)",
    template="plotly_white",
    hovermode="x unified",
    height=750,
    showlegend=False
)

fig.update_xaxes(rangeslider_visible=False)

fig.show()
