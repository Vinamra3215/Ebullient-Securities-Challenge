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

VR_LAG = 20         # typical: 5,10,20,40
VR_WINDOW = 100     # window for rolling var (larger window → smoother VR)
# ==========================================================


# ==========================================================
# ROLLING VARIANCE RATIO TEST
# ==========================================================
def rolling_variance_ratio(prices, lag=20, window=300):
    """
    Returns rolling VR(k) computed correctly:
    VR_t(k) = Var(k-step returns) / (k * Var(1-step returns))
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) < window + lag:
        return np.full(len(prices), np.nan)

    logp = np.log(prices)
    r = np.diff(logp)

    vr = np.full(len(prices), np.nan)

    for t in range(window, len(prices) - lag):
        r_win = r[t - window:t]

        var1 = np.var(r_win, ddof=1)
        if var1 == 0:
            continue

        r_k = pd.Series(r_win).rolling(lag).sum().dropna().values
        if len(r_k) < 2:
            continue

        var_k = np.var(r_k, ddof=1)

        vr[t] = var_k / (lag * var1)

    return vr


# ==========================================================
# LOAD PARQUET WITH cuDF
# ==========================================================
file_path = DATA_DIR / f"day{DAY_TO_PLOT}.parquet"
print(f"\nLoading {file_path} ...")

df = cudf.read_parquet(file_path)
df_pd = df[[TIME_COLUMN, PRICE_COLUMN]].to_pandas()
df_pd[TIME_COLUMN] = pd.to_timedelta(df_pd[TIME_COLUMN])

prices = df_pd[PRICE_COLUMN].to_numpy(dtype=float)

print(f"Loaded {len(df_pd):,} rows")
print(f"Price range: {prices.min():.2f} – {prices.max():.2f}")


# ==========================================================
# COMPUTE VARIANCE RATIO TEST
# ==========================================================
df_pd["VR"] = rolling_variance_ratio(prices, lag=VR_LAG, window=VR_WINDOW)
df_pd["VR_smooth"] = df_pd["VR"].rolling(20).mean()


# ==========================================================
# COLOR PRICE BASED ON VR
# (red if VR < 1, black otherwise)
# ==========================================================
times = df_pd[TIME_COLUMN].values
price_vals = df_pd[PRICE_COLUMN].values
vr_vals = df_pd["VR_smooth"].values

segments = []
current_x = []
current_y = []
current_color = None

for i in range(len(df_pd)):
    vr = vr_vals[i]
    color = "red" if (not np.isnan(vr) and vr < 1) else "black"

    if current_color is None:
        current_color = color

    if color != current_color:
        segments.append((current_x, current_y, current_color))
        current_x = []
        current_y = []
        current_color = color

    current_x.append(times[i])
    current_y.append(price_vals[i])

# append last segment
segments.append((current_x, current_y, current_color))


# ==========================================================
# PLOT PRICE + VR TEST
# ==========================================================
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.12,
    row_heights=[0.7, 0.3],
    subplot_titles=(
        f"Price — Day {DAY_TO_PLOT}",
        f"Variance Ratio (Lag={VR_LAG}, Window={VR_WINDOW})"
    )
)


# -------------------------------
# PRICE PANEL WITH COLORING
# -------------------------------
for xs, ys, col in segments:
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color=col, width=2),
            name=f"Price ({col})",
            showlegend=False
        ),
        row=1, col=1
    )


# -------------------------------
# VR PANEL
# -------------------------------
fig.add_trace(
    go.Scatter(
        x=df_pd[TIME_COLUMN],
        y=df_pd["VR_smooth"],
        mode="lines",
        line=dict(color="blue", width=2),
        name="VR (smoothed)"
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=df_pd[TIME_COLUMN],
        y=[1] * len(df_pd),
        mode="lines",
        line=dict(color="gray", width=1, dash="dash"),
        name="VR = 1"
    ),
    row=2, col=1
)

# -------------------------------
# LAYOUT
# -------------------------------
fig.update_layout(
    title=f"Day {DAY_TO_PLOT} — Price Colored by Variance Ratio Regime",
    template="plotly_white",
    hovermode="x unified",
    height=900,
    showlegend=True
)

fig.update_yaxes(title_text="Price ($)", row=1, col=1)
fig.update_yaxes(title_text="Variance Ratio", row=2, col=1)

fig.show()
