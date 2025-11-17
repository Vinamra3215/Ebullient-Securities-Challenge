import cudf
import dask_cudf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numba import jit
from pathlib import Path

# =============================================================
# CONFIG
# =============================================================

CONFIG = {
    "DATA_DIR": "/data/quant14/EBY",
    "DAY": 103,                 # Day to visualize
    "PB9_LOOKBACKS": [f"PB9_T{i}" for i in range(1, 13)],  # PB9_T1...T12
    "ADX_WINDOW": 14,
    "ADX_THRESHOLD": 25.0,
    "PRICE_COLUMN": "Price"
}

# =============================================================
# ADX Calculation from PB9 lookbacks
# =============================================================

@jit(nopython=True, fastmath=True)
def calculate_pb9_adx_from_lookbacks(pb9_matrix, window=14):
    """
    Custom ADX-like indicator from PB9 lookbacks (PB9_T1...PB9_T12)
    """
    n, m = pb9_matrix.shape
    adx = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        diffs = pb9_matrix[i, :] - pb9_matrix[i - 15, :]
        up = np.sum(diffs[diffs > 0])
        down = -np.sum(diffs[diffs < 0])
        plus_dm[i] = up
        minus_dm[i] = down
        tr[i] = np.sum(np.abs(diffs)) / m

    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)

    for i in range(window, n):
        avg_tr = np.mean(tr[i - window + 1:i + 1])
        avg_plus_dm = np.mean(plus_dm[i - window + 1:i + 1])
        avg_minus_dm = np.mean(minus_dm[i - window + 1:i + 1])

        if avg_tr == 0:
            continue

        plus_di[i] = 100 * (avg_plus_dm / avg_tr)
        minus_di[i] = 100 * (avg_minus_dm / avg_tr)
        diff = abs(plus_di[i] - minus_di[i])
        summ = plus_di[i] + minus_di[i]
        dx[i] = 100 * (diff / summ) if summ != 0 else 0.0
        adx[i] = np.mean(dx[i - window + 1:i + 1])

    return adx


# =============================================================
# Load Data and Compute ADX
# =============================================================

def load_day_df(day_num):
    file_path = Path(CONFIG["DATA_DIR"]) / f"day{day_num}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Day {day_num} file not found: {file_path}")
    
    df = cudf.read_parquet(file_path).to_pandas()
    pb9_cols = CONFIG["PB9_LOOKBACKS"]
    
    for col in pb9_cols + [CONFIG["PRICE_COLUMN"], "Time"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    
    df = df.dropna(subset=pb9_cols + [CONFIG["PRICE_COLUMN"]]).reset_index(drop=True)
    return df


# =============================================================
# Visualization
# =============================================================

def plot_price_with_adx(df):
    pb9_matrix = df[CONFIG["PB9_LOOKBACKS"]].values.astype(np.float64)
    adx = calculate_pb9_adx_from_lookbacks(pb9_matrix, window=CONFIG["ADX_WINDOW"])
    df["ADX"] = adx
    df["Time"] = pd.to_timedelta(df["Time"])
    
    # Identify zones where ADX >= threshold
    threshold = CONFIG["ADX_THRESHOLD"]
    strong_trend = df["ADX"] >= threshold
    
    # Create figure
    fig = go.Figure()
    
    # Plot price
    fig.add_trace(go.Scatter(
        x=df["Time"],
        y=df["Price"],
        mode="lines",
        name="Price",
        line=dict(color="black", width=2),
        hovertemplate="Time: %{x}<br>Price: %{y:.4f}<extra></extra>"
    ))
    
    # Overlay ADX (secondary axis)
    fig.add_trace(go.Scatter(
        x=df["Time"],
        y=df["ADX"],
        mode="lines",
        name="PB9_ADX",
        line=dict(color="blue", width=1.5, dash="dot"),
        yaxis="y2",
        hovertemplate="Time: %{x}<br>PB9_ADX: %{y:.2f}<extra></extra>"
    ))
    
    # Add shaded regions where ADX >= threshold
    start_idx = None
    for i in range(1, len(df)):
        if strong_trend.iloc[i] and start_idx is None:
            start_idx = i
        elif not strong_trend.iloc[i] and start_idx is not None:
            fig.add_vrect(
                x0=df["Time"].iloc[start_idx],
                x1=df["Time"].iloc[i],
                fillcolor="rgba(255, 165, 0, 0.2)",
                layer="below",
                line_width=0,
                annotation_text="ADX≥25",
                annotation_position="top left"
            )
            start_idx = None

    # Layout
    fig.update_layout(
        title=f"Day {CONFIG['DAY']} — Price with PB9_ADX (Threshold {CONFIG['ADX_THRESHOLD']})",
        xaxis=dict(title="Time", showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title="Price", side="left", showgrid=True, gridcolor="lightgray"),
        yaxis2=dict(title="PB9_ADX", overlaying="y", side="right", showgrid=False),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        height=600,
        width=1500,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    fig.write_html(f"Day{CONFIG['DAY']}_PB9_ADX.html")
    fig.show()


# =============================================================
# MAIN EXECUTION
# =============================================================

if __name__ == "__main__":
    print(f"Loading Day {CONFIG['DAY']} data...")
    df_day = load_day_df(CONFIG["DAY"])
    print("Computing PB9_ADX and plotting...")
    plot_price_with_adx(df_day)
    print(f"✅ Plot saved as Day{CONFIG['DAY']}_PB9_ADX.html")
