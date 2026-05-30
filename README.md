<![CDATA[<div align="center">

# 🏆 Ebullient Securities — Algorithmic Strategy Development

### **Inter IIT Tech Meet 14.0 · High-Prep Problem Statement**

*Quantitative trading strategies on multi-feature anonymized time-series data (EBX & EBY)*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-2.2-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.3-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Numba](https://img.shields.io/badge/Numba-JIT-00A3E0?style=for-the-badge&logo=numba&logoColor=white)](https://numba.pydata.org)

---

**Team 67** · IIT Patna

</div>

---

## 📖 Table of Contents

- [About the Challenge](#-about-the-challenge)
- [Strategy Overview — Ragnarök Trend Protocol](#-strategy-overview--ragnarök-trend-protocol)
- [Market Structure Analysis](#-market-structure-analysis)
- [Custom Indicator — AURA](#-custom-indicator--aura)
- [Sub-Strategies](#-sub-strategies)
- [Dynamic Priority-Based Strategy Manager](#-dynamic-priority-based-strategy-manager)
- [Performance Metrics](#-performance-metrics)
- [Repository Structure](#-repository-structure)
- [Setup & Usage](#-setup--usage)
- [Tech Stack](#-tech-stack)

---

## 🏦 About the Challenge

**Ebullient Securities** — one of India's largest Prop Options Trading firms — posed a high-prep challenge at **Inter IIT Tech Meet 14.0** focused on **systematic algorithmic strategy development** over anonymized financial time-series data.

### Problem Statement

Two anonymized 1-second interval time-series datasets (**EBX** & **EBY**) were provided, each containing:
- A core price series (index-like data reset to 100 each day)
- Hundreds of masked features across categories — *Price-Based, Volatility-Based, Volume-Based, Alternate-Data-Based*

### Objective

Design, test, and evaluate a **systematic intraday trading strategy** that:
- Demonstrates **robustness** across both EBX and EBY
- Maintains a **consistent Calmar ratio** even when days are reordered
- Operates with **no forward bias** (signals at time *t* cannot look beyond *t*)
- Ends each day **flat** (no inter-day carry of positions)

### Competition Phases

| Phase | Weightage | Constraints |
|:------|:---------:|:------------|
| **Mid-Term Eval** | 30% | Only given features; ±1 unit trades; no scaling |
| **End-Term Eval** | 60% | Derived features allowed; variable trade sizes; model blending permitted |
| **Presentation** | 10% | Approach, results, and evolution documentation |

---

## 🧠 Strategy Overview — Ragnarök Trend Protocol

Our final strategy is a **multi-regime, multi-strategy framework** that dynamically selects the most appropriate sub-strategy based on real-time market conditions. The system follows a three-stage pipeline:

```
┌─────────────────────┐     ┌──────────────────────────┐     ┌─────────────────────┐
│  Market Regime      │────▶│  Strategy Selection      │────▶│  Signal Generation  │
│  Classification     │     │  (Priority Locking)      │     │  & Risk Management  │
└─────────────────────┘     └──────────────────────────┘     └─────────────────────┘
     ▲                              ▲                               │
     │   30-min σ & µ analysis      │   Dynamic priority queue      │   Trailing stops,
     │   ATR / STD thresholds       │   Day-level lock              │   Take-profits,
     │   Volatility percentiles     │   Strategy-specific gates     │   Cooldown periods
```

---

## 📊 Market Structure Analysis

We performed an extensive **two-factor statistical analysis** of EBX and EBY:

### Key Findings

| Metric | EBX | EBY |
|:-------|:---:|:---:|
| Overall Mean µ | 100.015 | 100.002 |
| Overall Std. Dev. σ | 0.535 | 0.534 |
| σ 50th percentile | 0.07 | 0.08 |
| σ 75th percentile | 0.11 | 0.12 |
| Avg. High – Price@30min | 0.47 | 0.49 |
| Avg. Price@30min – Low | 0.52 | 0.49 |

### Regime Discovery

The combination of **early-day mean (µ₀₋₃₀)** and **early-day volatility (σ₀₋₃₀)** produced four distinct predictive regimes:

- **High µ + High σ** → Strongest continuation (82% of days closed above 100)
- **Low µ + High σ** → Strongest downside trend (71.5% closed below 100)
- **Low σ regimes** → Weaker, noisier outcomes regardless of direction
- **High σ in first 1–5 min (EBX)** → High unpredictability for the rest of the day

---

## 🔬 Custom Indicator — AURA

> **A**daptive **U**ltra-smooth **R**esponsive **A**verage

After analyzing multiple established moving averages (JMA, Ehler's Super Smoother, KAMA, ZLEMA) and finding none with the ideal balance between smoothness, responsiveness, lag, and seasonality reduction — we designed **AURA**, a proprietary indicator that blends:

- **KAMA's** adaptive smoothing logic
- **Ehler's Super Smoother's** noise reduction and responsive capabilities

AURA serves as the core trend-following filter across multiple sub-strategies.

---

## ⚡ Sub-Strategies

### 1. Denoised Trend Pattern *(Highest Priority)*
Stabilizes directional signals by filtering noise using AURA-based trend detection. Prevents micro-volatility-driven false signals and captures strong trend-following setups on high-volatility days.

### 2. Sigma-Triggered Alpha Reactor (STAR)
Activates during specific volatility windows by comparing early-session σ and µ with overall thresholds. Implements a **sequential, reversal-based structure** with three phases:
- **Gap-Fill Entry** → Counter-trend at 30-min mark
- **Flip** → Close initial position + open main position in favored direction
- **Position Management** → Trailing stops + reversal targets

### 3. Volatility-Filtered KAMA Trend
Performs day segregation using ATR and STD thresholds. On filtered high-quality tradable days, applies KAMA filter slope to detect trend direction and generates long/short signals with take-profit barriers and minimum cooldown durations.

### 4. Hawkes Adaptive Trend Strategy
Captures large, sustained price movements while avoiding choppy conditions:
- **Ehlers Super Smoother → AURA filter** for adaptive trend
- **Hawkes Process** for impulse detection (momentum bursts vs. noise)
- **Dynamic Volatility Walls** for support/resistance
- **ATR Volatility Gate** to prevent low-vol trading
- **Aggressive Trailing Stop** for drawdown minimization

### 5. SuperNova Crossover
Volatility-gated AURA-JMA crossover that only activates when sufficient market movement is detected. Trades crossover signals between a slower AURA and JMA, with independent signal-level exits that trigger delayed counter-directional entries.

---

## 🔄 Dynamic Priority-Based Strategy Manager

The strategy selector uses a **priority-based locking method** where each strategy has a pre-assigned priority and the lock resets daily.

### EBX Strategy Chain
```
Denoised Trend Pattern ──▶ Volatility-Filtered KAMA Trend ──▶ STAR
      (Primary)                   (Secondary)                  (Tertiary)
```

### EBY Strategy Chain
```
Denoised Trend Pattern ──▶ STAR ──▶ Volatility-Filtered KAMA ──▶ Money Heist
      (Primary)           (2nd)          (3rd)                      (4th)
```

---

## 📈 Performance Metrics

| Metric | EBY | EBX |
|:-------|:---:|:---:|
| **Final Returns** | 34.75% | 39.68% |
| **Annualized Returns** | 31.39% | 20.41% |
| **Calmar Ratio** | 15.94 | 10.36 |
| **Annualized Calmar** | 14.40 | 5.33 |
| **Maximum Drawdown** | -2.18% | -3.83% |
| **Sharpe Ratio** | 2.58 | 2.48 |

---

## 📁 Repository Structure

```
Ebullient-Securities-Challenge/
│
├── Final Zip Structure/           # 🏁 Final submission package
│   ├── EBX.py                     # Signal generator for EBX
│   ├── EBY.py                     # Signal generator for EBY
│   ├── includes.py                # Core data classes & backtesting framework
│   ├── mainIIT.py                 # Backtest runner (reads signals → places orders)
│   ├── backtesterIIT.py           # Backtester engine
│   ├── configIIT.json             # Backtester configuration
│   ├── requirements.txt           # Python dependencies
│   └── *.pdf                      # Presentation & performance reports
│
├── Final Eval Strategy/           # 🧪 End-term strategy development
│   ├── EBX.py                     # Volatility-Filtered KAMA Trend (EBX)
│   ├── EBY.py                     # Volatility-Filtered KAMA Trend (EBY)
│   ├── Final_Combined_EBX.py      # Combined multi-strategy for EBX
│   ├── Final_Combined_EBY.py      # Combined multi-strategy for EBY
│   ├── Smart Buy And Hold EBX/    # Smart buy-and-hold variants (EBX)
│   └── Smart Buy And Hold EBY/    # Smart buy-and-hold variants (EBY)
│
├── Mid Eval Strategy/             # 📝 Mid-term strategy (±1 unit trades)
│   ├── EBX.py                     # Mid-term EBX strategy
│   ├── EBY.py                     # Mid-term EBY strategy
│   └── backtester.py              # Mid-term backtester
│
├── Feature Analysis/              # 🔍 Feature engineering & correlation analysis
│   ├── feature_analysis_dask.py   # Distributed feature analysis (Dask)
│   ├── corr_mat.py                # Correlation matrix computation
│   ├── filter.py                  # Feature filtering & selection
│   └── visualize_feature_analysis.py  # Visualization utilities
│
├── Days Segregation/              # 📅 Day classification by volatility regime
│   ├── using_ATR.py               # ATR-based day segregation
│   ├── using_STD.py               # Standard deviation-based segregation
│   ├── using_ADX.py               # ADX-based segregation
│   ├── using_STD_ATR.py           # Combined STD + ATR segregation
│   └── score.py                   # Day quality scoring
│
├── Price Plots/                   # 📉 Price analysis & visualization
│   ├── KAMA+EMA+SMA+HMA+Heiken.py        # Moving average comparison
│   ├── variance_ratio_test_on_price.py     # Variance ratio tests
│   ├── variance_ratio_test_on_supersmoother.py  # VRT on smoothed data
│   ├── price_density(tape).py              # Price density analysis
│   └── resample_candlestick.py             # Candlestick resampling
│
├── Histogram/                     # 📊 Distribution analysis
│   └── histogram.py               # PnL / feature histograms
│
└── backtester/                    # ⚙️ Core backtesting engine
    └── backtester.py              # Full backtester with PnL, drawdown, Sharpe/Calmar
```

---

## 🚀 Setup & Usage

### Prerequisites

- Python 3.10+
- pip / conda

### Installation

```bash
git clone https://github.com/Vinamra3215/Ebullient-Securities-Challenge.git
cd Ebullient-Securities-Challenge
pip install -r "Final Zip Structure/requirements.txt"
```

### Running the Strategy

**1. Configure the backtest** — edit `Final Zip Structure/configIIT.json`:

```json
{
    "data_path": "<path-to-data-directory>",
    "start_date": 0,
    "end_date": 510,
    "timer": 600,
    "tcost": 2,
    "broadcast": ["EBX", "EBY"]
}
```

**2. Generate trading signals:**

```bash
python "Final Zip Structure/EBX.py" --data-dir <path-to-EBX-CSVs>
python "Final Zip Structure/EBY.py" --data-dir <path-to-EBY-CSVs>
```

**3. Run the backtest:**

```bash
python "Final Zip Structure/mainIIT.py" "Final Zip Structure/configIIT.json"
```

---

## 🛠 Tech Stack

| Component | Technology |
|:----------|:-----------|
| Core Language | Python 3.10+ |
| Data Processing | Pandas 2.3, NumPy 2.2 |
| JIT Compilation | Numba 0.61 (machine-code-level speed) |
| State Estimation | pyKalman 0.10 (Kalman filtering) |
| Parallel Processing | Python `multiprocessing` |
| Distributed Computing | Dask + RAPIDS cuDF (for feature analysis) |
| Visualization | Plotly (interactive HTML reports) |

---

## 📄 Key Deliverables

| Document | Description |
|:---------|:------------|
| `Final Zip Structure/Team_67_presentation.pdf` | Final presentation deck |
| `Final Zip Structure/end_eval_ebullient_67_report.pdf` | Detailed strategy report |
| `Final Zip Structure/end_eval_ebullient_67_performance_report.pdf` | Full performance analysis |

---

<div align="center">

### Built with ❤️ for Inter IIT Tech Meet 14.0

*Ebullient Securities — Quantitative Proprietary Trading Challenge*

</div>
]]>


