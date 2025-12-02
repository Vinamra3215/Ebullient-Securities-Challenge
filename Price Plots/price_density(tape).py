import cudf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================================
# CONFIGURATION
# ==========================================================
DATA_DIR = Path("/data/quant14/EBX")
DAY_TO_PLOT = 87
PRICE_COLUMN = "Price"
TIME_COLUMN = "Time"
NUM_BUCKETS = 80     # number of price buckets for density (increase for higher resolution)
# ==========================================================

# ----------------------------------------------------------
# Load Data
# ----------------------------------------------------------
file_path = DATA_DIR / f"day{DAY_TO_PLOT}.parquet"
print(f"\nLoading {file_path} ...")
df = cudf.read_parquet(file_path)
print(f"Loaded {len(df):,} rows from Day {DAY_TO_PLOT}")

df_pd = df[[TIME_COLUMN, PRICE_COLUMN]].to_pandas()
df_pd[TIME_COLUMN] = pd.to_timedelta(df_pd[TIME_COLUMN])

print(f"Price range: ${df_pd[PRICE_COLUMN].min():.2f} - ${df_pd[PRICE_COLUMN].max():.2f}")


# ==========================================================
# TAPE READING — PRICE DENSITY (TIME AT PRICE)
# ==========================================================
prices = df_pd[PRICE_COLUMN].to_numpy(dtype=float)
times = df_pd[TIME_COLUMN].to_numpy()

# Convert timedelta → seconds
time_sec = (times - times[0]).astype('timedelta64[ms]').astype(float) / 1000.0

# Time difference between ticks
dt = np.diff(time_sec)
dt = np.append(dt, dt[-1])   # last tick uses same dt

# Create equal-width price buckets
p_min, p_max = prices.min(), prices.max()
bins = np.linspace(p_min, p_max, NUM_BUCKETS)

# Which bucket each price belongs to
bucket_index = np.digitize(prices, bins) - 1
bucket_index = np.clip(bucket_index, 0, NUM_BUCKETS - 1)

# Time spent in each bucket
price_density = np.zeros(NUM_BUCKETS)
for i in range(len(prices)):
    price_density[bucket_index[i]] += dt[i]

# Normalize 0–1 for easier plotting
price_density_norm = price_density / price_density.max()

# Midpoints for plotting
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_centers = np.append(bin_centers, bins[-1] - (bins[-1] - bins[-2]) / 2)


# ==========================================================
# PLOT PRICE + PRICE DENSITY
# ==========================================================
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=False,
    vertical_spacing=0.12,
    row_heights=[0.75, 0.25],
    subplot_titles=(
        f"Price — Day {DAY_TO_PLOT}",
        "Tape Reading: Price Density (Time at Price)"
    )
)

# -------------------------------
# PRICE PANEL (Top)
# -------------------------------
fig.add_trace(
    go.Scatter(
        x=df_pd[TIME_COLUMN],
        y=df_pd[PRICE_COLUMN],
        mode="lines",
        line=dict(color="black", width=2),
        name="Price",
    ),
    row=1, col=1
)

# -------------------------------
# PRICE DENSITY PANEL (Bottom)
# -------------------------------
fig.add_trace(
    go.Bar(
        x=bin_centers,
        y=price_density_norm,
        name="Price Density",
        marker=dict(color="blue", opacity=0.7)
    ),
    row=2, col=1
)

fig.update_layout(
    title=f"Day {DAY_TO_PLOT} — Price Chart + Tape Reading Density",
    template="plotly_white",
    hovermode="x unified",
    height=900,
    showlegend=False
)

fig.update_yaxes(title_text="Price ($)", row=1, col=1)
fig.update_yaxes(title_text="Density", row=2, col=1)

fig.show()
