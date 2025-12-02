import cudf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import math

# ==========================================================
# CONFIGURATION
# ==========================================================
DATA_DIR = Path("/data/quant14/EBX")
DAY_TO_PLOT = 87
PRICE_COLUMN = "Price"
TIME_COLUMN = "Time"

VR_LAG = 20
VR_WINDOW = 100
SUPERSMOOTH_PERIOD = 30         # You can change this
# ==========================================================


# ==========================================================
# EHLERS BASE_SUPERSMOOTHER (2-pole low-lag filter)
# ==========================================================
def base_supersmoother(series, period=30):

    series = np.asarray(series, dtype=float)
    n = len(series)

    ss = np.zeros(n)

    # Ehlers coefficients
    pi = math.pi
    a1 = math.exp(-1.414 * pi / period)
    b1 = 2 * a1 * math.cos(1.414 * pi / period)

    c1 = 1 - b1 + (a1*a1)      # feed-forward coefficient
    c2 = b1                    # 1st feedback term
    c3 = -(a1*a1)              # 2nd feedback term

    # Initial conditions
    ss[0] = series[0]
    ss[1] = series[1]

    for i in range(2, n):
        ss[i] = c1 * series[i] + c2 * ss[i-1] + c3 * ss[i-2]

    return ss


# ==========================================================
# VARIANCE RATIO ON SMOOTHED SERIES
# ==========================================================
def rolling_variance_ratio(series, lag=20, window=100):

    series = np.asarray(series, dtype=float)
    logp = np.log(series)

    r = np.diff(logp)
    n = len(series)

    vr = np.full(n, np.nan)

    for t in range(window, n - lag):

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
# LOAD PARQUET
# ==========================================================
file_path = DATA_DIR / f"day{DAY_TO_PLOT}.parquet"
print(f"\nLoading {file_path} ...")

df = cudf.read_parquet(file_path)
df_pd = df[[TIME_COLUMN, PRICE_COLUMN]].to_pandas()
df_pd[TIME_COLUMN] = pd.to_timedelta(df_pd[TIME_COLUMN])

prices = df_pd[PRICE_COLUMN].astype(float).to_numpy()

print(f"Loaded {len(prices):,} rows")
print(f"Price range: {prices.min():.4f} – {prices.max():.4f}")


# ==========================================================
# COMPUTE BASE_SUPERSMOOTHER
# ==========================================================
smoothed = base_supersmoother(prices, period=SUPERSMOOTH_PERIOD)
df_pd["Smooth"] = smoothed


# ==========================================================
# COMPUTE VR ON SMOOTHED SERIES
# ==========================================================
df_pd["VR"] = rolling_variance_ratio(smoothed, lag=VR_LAG, window=VR_WINDOW)
df_pd["VR_smooth"] = df_pd["VR"].rolling(20).mean()

# ==========================================================
# 30th Percentile of VR_smooth
# ==========================================================
valid_vr = df_pd["VR_smooth"].dropna().values

vr_30 = 0

if len(valid_vr) > 0:
    vr_30 = np.percentile(valid_vr, 30)
else:
    vr_30 = np.nan

print(f"30th Percentile of Variance Ratio (VR_smooth): {vr_30:.4f}")


# ==========================================================
# COLOR PRICE BY VR REGIME (<1 red, >=1 black)
# ==========================================================
times = df_pd[TIME_COLUMN].values
ys = df_pd["Smooth"].values
vr_vals = df_pd["VR_smooth"].values

segments = []
cur_x = []
cur_y = []
cur_color = None

for i in range(len(df_pd)):
    vr = vr_vals[i]
    color = "red" if (not np.isnan(vr) and vr < vr_30) else "black"

    if cur_color is None:
        cur_color = color

    if color != cur_color:
        segments.append((cur_x, cur_y, cur_color))
        cur_x = []
        cur_y = []
        cur_color = color

    cur_x.append(times[i])
    cur_y.append(ys[i])

segments.append((cur_x, cur_y, cur_color))


# ==========================================================
# PLOT
# ==========================================================
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.12,
    row_heights=[0.7, 0.3],
    subplot_titles=(
        f"Base Supersmoother (p={SUPERSMOOTH_PERIOD}) Colored by VR",
        f"Variance Ratio (Lag={VR_LAG}, Window={VR_WINDOW})"
    )
)

# Smoothed price segments
for xs, ys, col in segments:
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color=col, width=2),
            showlegend=False
        ),
        row=1, col=1
    )

# VR Panel
fig.add_trace(
    go.Scatter(
        x=df_pd[TIME_COLUMN],
        y=df_pd["VR_smooth"],
        mode="lines",
        line=dict(color="blue", width=2),
        name="VR (Smoothed)"
    ),
    row=2, col=1
)

# 30th Percentile Line
fig.add_trace(
    go.Scatter(
        x=df_pd[TIME_COLUMN],
        y=[vr_30] * len(df_pd),
        mode="lines",
        line=dict(color="red", width=1.5, dash="dot"),
        name="30th Percentile"
    ),
    row=2, col=1
)


fig.update_layout(
    title=f"Day {DAY_TO_PLOT} — Variance Ratio Test on Base_Supersmoother",
    template="plotly_white",
    hovermode="x unified",
    height=900
)

fig.update_yaxes(title_text="Smoothed Price", row=1, col=1)
fig.update_yaxes(title_text="Variance Ratio", row=2, col=1)

fig.show()
