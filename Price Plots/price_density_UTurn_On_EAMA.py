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

NUM_BUCKETS = 80
ROLL_SEC = 180          # rolling window for U-turns
U_TURN_THRESHOLD = 0.07   # threshold to color price red
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

prices = df_pd[PRICE_COLUMN].to_numpy(dtype=float)
times = df_pd[TIME_COLUMN].to_numpy()

print(f"Price range: {prices.min():.4f} – {prices.max():.4f}")



# ==========================================================
# TIME → SECONDS
# ==========================================================
time_sec = (times - times[0]).astype("timedelta64[ms]").astype(float) / 1000.0

# Δt between ticks
dt = np.diff(time_sec)
dt = np.append(dt, dt[-1])



# ==========================================================
# ⭐ STEP 1: COMPUTE EAMA (used for U-Turns)
# ==========================================================
def compute_EAMA(series, eama_period=10, eama_fast=15, eama_slow=40):
    series = np.asarray(series, dtype=float)
    n = len(series)

    # Direction & volatility
    direction = pd.Series(series).diff(eama_period).abs()
    volatility = pd.Series(series).diff().abs().rolling(eama_period).sum()
    er = (direction / volatility.replace(0, np.nan)).fillna(0).values

    fast_sc = 2 / (eama_fast + 1)
    slow_sc = 2 / (eama_slow + 1)
    sc = ((er * (fast_sc - slow_sc)) + slow_sc) ** 2

    eama = np.zeros(n)
    eama[0] = series[0]

    for i in range(1, n):
        eama[i] = eama[i - 1] + sc[i] * (series[i] - eama[i - 1])

    return eama


print("\nComputing EAMA for U-turn detection...")
EAMA = compute_EAMA(prices)
df_pd["EAMA"] = EAMA



# ==========================================================
# PRICE DENSITY (Tape Reading)
# ==========================================================
p_min, p_max = prices.min(), prices.max()
bins = np.linspace(p_min, p_max, NUM_BUCKETS)

bucket_idx = np.digitize(prices, bins) - 1
bucket_idx = np.clip(bucket_idx, 0, NUM_BUCKETS - 1)

price_density = np.zeros(NUM_BUCKETS)
for i in range(len(prices)):
    price_density[bucket_idx[i]] += dt[i]

price_density_norm = price_density / np.max(price_density)

bin_centers = (bins[:-1] + bins[1:]) / 2
bin_centers = np.append(bin_centers, bins[-1] - (bins[-1] - bins[-2]) / 2)



# ==========================================================
# ⭐ STEP 2: ROLLING U-TURN COUNT — now uses EAMA
# ==========================================================
print("\nComputing U-turn counts using EAMA...")

diff_p = np.diff(EAMA)
direction = np.sign(diff_p)
direction = np.append(direction, 0)

U_count = np.zeros(len(EAMA))

left = 0
n = len(EAMA)

for right in range(1, n):

    while time_sec[right] - time_sec[left] > ROLL_SEC:
        left += 1

    d_win = direction[left:right+1]
    d_win = d_win[d_win != 0]

    if len(d_win) > 1:
        flips = np.sum(np.diff(np.sign(d_win)) != 0)
    else:
        flips = 0

    U_count[right] = flips

df_pd["U_Turns"] = U_count / 180



# ==========================================================
# PRICE COLORING BASED ON U-TURN THRESHOLD
# ==========================================================
segments = []
cur_x, cur_y = [], []
cur_color = None

for i in range(len(prices)):

    uval = U_count[i]
    color = "red" if uval > U_TURN_THRESHOLD else "black"

    if cur_color is None:
        cur_color = color

    if color != cur_color:
        segments.append((cur_x, cur_y, cur_color))
        cur_x, cur_y = [], []
        cur_color = color

    cur_x.append(times[i])
    cur_y.append(prices[i])

segments.append((cur_x, cur_y, cur_color))



# ==========================================================
# PLOTTING
# ==========================================================
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=False,
    vertical_spacing=0.10,
    row_heights=[0.55, 0.25, 0.20],
    subplot_titles=[
        f"Price — U-turn Coloring (>{U_TURN_THRESHOLD}) on EAMA",
        "Price Density (Tape Reading)",
        "Rolling U-turn Count (Choppiness 180 sec, on EAMA)"
    ]
)


# -------------------------------
# (1) PRICE PANEL WITH COLORING
# -------------------------------
for xs, ys, col in segments:
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=col, width=2),
            showlegend=False
        ),
        row=1, col=1
    )



# -------------------------------
# (2) PRICE DENSITY BARPLOT
# -------------------------------
fig.add_trace(
    go.Bar(
        x=bin_centers,
        y=price_density_norm,
        marker=dict(color="blue", opacity=0.7),
        name="Price Density"
    ),
    row=2, col=1
)



# -------------------------------
# (3) U-TURN PANEL
# -------------------------------
fig.add_trace(
    go.Scatter(
        x=df_pd[TIME_COLUMN],
        y=df_pd["U_Turns"],
        mode="lines",
        line=dict(color="purple", width=2),
        name="U-Turns"
    ),
    row=3, col=1
)



# -------------------------------
# LAYOUT
# -------------------------------
fig.update_layout(
    title=f"Tape Reading + U-Turn Choppiness (EAMA) — Day {DAY_TO_PLOT}",
    template="plotly_white",
    hovermode="x unified",
    height=950,
    showlegend=False
)

fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Density", row=2, col=1)
fig.update_yaxes(title_text="U-Turns", row=3, col=1)

fig.show()
