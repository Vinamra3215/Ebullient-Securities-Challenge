import cudf
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

# ==========================================================
# CONFIGURATION
# ==========================================================
DATA_DIR = Path("/data/quant14/EBX")  # directory containing dayX.parquet files
DAY_TO_PLOT = 87            # <-- change this to any day number you want to visualize
PRICE_COLUMN = "Price"
TIME_COLUMN = "Time"                  # make sure your parquet has this column
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
df_pd[TIME_COLUMN] = pd.to_timedelta(df_pd[TIME_COLUMN])

print(f"Data columns: {list(df_pd.columns)}")
print(f"Price range: ${df_pd[PRICE_COLUMN].min():.2f} - ${df_pd[PRICE_COLUMN].max():.2f}")

# ----------------------------------------------------------
# Plot: Price vs Time
# ----------------------------------------------------------
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df_pd[TIME_COLUMN],
        y=df_pd[PRICE_COLUMN],
        mode="lines",
        line=dict(color="black", width=2),
        name="Price",
        hovertemplate="<b>Time:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>"
    )
)

fig.update_layout(
    title=f"Day {DAY_TO_PLOT} — Price vs Time",
    xaxis_title="Time",
    yaxis_title="Price ($)",
    template="plotly_white",
    hovermode="x unified",
    height=700,
    showlegend=True
)

fig.show()
