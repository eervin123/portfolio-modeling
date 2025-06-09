import streamlit as st
import yfinance as yf
import bt
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="Portfolio Backtest with BFC Allocation", layout="wide")


# Cache the data download
@st.cache_data
def download_data():
    tickers = ["SPY", "AGG"]
    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            data = yf.download(tickers, start="2015-01-01", end="2025-01-01")["Close"]
            if not data.empty:
                return data.ffill().dropna()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    raise Exception("Failed to download data after multiple attempts")


# Cache the BFC Net Return series
@st.cache_data
def get_bfc_net_series():
    df = pd.read_csv(
        "Blockforce Capital MultiStrat Fund Portfolio - Daily Return (Gross&Net) (1).csv"
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    # Convert Net Return from string percent to float
    df["Net Return"] = df["Net Return"].str.replace("%", "").astype(float) / 100
    # Build price series (start at 100)
    bfc_net = (1 + df["Net Return"]).cumprod() * 100
    bfc_net.name = "BFC Net"
    return bfc_net


# Download data
st.write("Loading data...")
data = download_data()
bfc_net = get_bfc_net_series()

# Align dates
all_data = data.copy()
all_data["BFC Net"] = bfc_net
all_data = all_data.ffill().dropna()

# Title and description
st.title("Portfolio Backtest with Blockforce Capital Allocation")
st.write(
    "Adjust the Blockforce Capital allocation using the slider below. The allocation will be taken proportionally from SPY (60%) and AGG (40%)."
)

# Slider for BFC allocation
bfc_allocation = st.slider("Blockforce Capital Allocation (%)", 1, 10, 1, step=1)

# Calculate weights
spy_base = 0.60
agg_base = 0.40
bfc_weight = bfc_allocation / 100

# Adjust SPY and AGG weights proportionally
spy_reduction = bfc_weight * (spy_base / (spy_base + agg_base))
agg_reduction = bfc_weight * (agg_base / (spy_base + agg_base))

custom_weights = {
    "SPY": spy_base - spy_reduction,
    "AGG": agg_base - agg_reduction,
    "BFC Net": bfc_weight,
}


# Cache the backtest results
@st.cache_data
def run_backtest(data, weights):
    s = bt.Strategy(
        "Custom Portfolio",
        [
            bt.algos.RunQuarterly(),
            bt.algos.SelectAll(),
            bt.algos.WeighSpecified(**weights),
            bt.algos.Rebalance(),
        ],
    )
    return bt.run(bt.Backtest(s, data))


# Run backtests
benchmark_60_40 = run_backtest(all_data, {"SPY": 0.60, "AGG": 0.40})
benchmark_spy = run_backtest(all_data, {"SPY": 1.0})
benchmark_agg = run_backtest(all_data, {"AGG": 1.0})
custom_portfolio = run_backtest(all_data, custom_weights)

# Create the plot
fig = go.Figure()

# Add benchmark portfolios
fig.add_trace(
    go.Scatter(
        x=benchmark_60_40[0].prices.index,
        y=benchmark_60_40[0].prices,
        mode="lines",
        name="60/40 Portfolio",
        line=dict(width=2, color="blue"),
    )
)

fig.add_trace(
    go.Scatter(
        x=benchmark_spy[0].prices.index,
        y=benchmark_spy[0].prices,
        mode="lines",
        name="100% SPY",
        line=dict(width=2, color="green"),
    )
)

fig.add_trace(
    go.Scatter(
        x=benchmark_agg[0].prices.index,
        y=benchmark_agg[0].prices,
        mode="lines",
        name="100% AGG",
        line=dict(width=2, color="red"),
    )
)

# Add custom portfolio
fig.add_trace(
    go.Scatter(
        x=custom_portfolio[0].prices.index,
        y=custom_portfolio[0].prices,
        mode="lines",
        name=f"Custom Portfolio ({bfc_allocation}% BFC Net)",
        line=dict(width=3, color="purple"),
    )
)

# Update layout
fig.update_layout(
    title="Portfolio Performance Comparison",
    xaxis_title="Date",
    yaxis_title="Value (Base 100)",
    showlegend=True,
    height=600,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)


# Helper function to format stats
def format_stat(val):
    if isinstance(val, (list, np.ndarray)) and len(val) == 1:
        val = val[0]
    if isinstance(val, pd.Timestamp):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, float):
        if -1 < val < 1:
            return f"{val:.2%}"
        else:
            return f"{val:.4f}"
    return str(val)


# --- Top: Performance Plot and Pie Chart ---
col_plot, col_pie = st.columns([2, 1])

with col_plot:
    st.plotly_chart(fig, use_container_width=True)

with col_pie:
    st.subheader("Current Portfolio Weights")
    pie_fig = go.Figure(
        go.Pie(
            labels=["SPY", "AGG", "BFC Net"],
            values=[
                custom_weights["SPY"],
                custom_weights["AGG"],
                custom_weights["BFC Net"],
            ],
            marker=dict(colors=["green", "red", "#f7931a"]),
            textinfo="label+percent",
            hole=0.4,
        )
    )
    pie_fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=350)
    st.plotly_chart(pie_fig, use_container_width=True)

# --- Stats Table and Bar Subplots Side-by-Side ---
col_table, col_bars = st.columns([1, 2])

with col_table:
    st.subheader("Portfolio Statistics")
    custom_stats_clean = [
        format_stat(x) for x in custom_portfolio.stats.iloc[:, 0].values
    ]
    benchmark_stats_clean = [
        format_stat(x) for x in benchmark_60_40.stats.iloc[:, 0].values
    ]
    stats_df = pd.DataFrame(
        {
            "Metric": custom_portfolio.stats.index,
            f"Custom Portfolio ({bfc_allocation}% BFC Net)": custom_stats_clean,
            "60/40 Portfolio": benchmark_stats_clean,
        }
    )
    st.dataframe(stats_df, hide_index=True, use_container_width=False, height=800)

with col_bars:
    bar_metrics = [
        ("Total Return", "total_return"),
        ("CAGR", "cagr"),
        ("Max Drawdown", "max_drawdown"),
        ("Calmar", "calmar"),
        ("Sharpe", "daily_sharpe"),
        ("Sortino", "daily_sortino"),
        ("Best Month", "best_month"),
        ("Worst Month", "worst_month"),
        ("Best Year", "best_year"),
        ("Worst Year", "worst_year"),
    ]
    n_metrics = len(bar_metrics)
    ncols = 2
    nrows = (n_metrics + ncols - 1) // ncols
    sub_fig = make_subplots(
        rows=nrows, cols=ncols, subplot_titles=[m[0] for m in bar_metrics]
    )
    for i, (label, key) in enumerate(bar_metrics):
        row = i // ncols + 1
        col = i % ncols + 1
        try:
            custom_val = custom_portfolio.stats.iloc[:, 0].loc[key]
        except Exception:
            custom_val = None
        try:
            bm_val = benchmark_60_40.stats.iloc[:, 0].loc[key]
        except Exception:
            bm_val = None
        # For drawdown and worsts, use abs value for bar length
        if "drawdown" in key or "worst" in key:
            try:
                custom_val = abs(float(custom_val))
            except Exception:
                pass
            try:
                bm_val = abs(float(bm_val))
            except Exception:
                pass
        sub_fig.add_trace(
            go.Bar(
                x=[custom_val],
                y=["Custom"],
                orientation="h",
                marker_color="purple",
                showlegend=False,
                hovertemplate=(
                    "%{x:.2%}"
                    if isinstance(custom_val, float) and -1 < custom_val < 1
                    else "%{x}"
                ),
            ),
            row=row,
            col=col,
        )
        sub_fig.add_trace(
            go.Bar(
                x=[bm_val],
                y=["60/40"],
                orientation="h",
                marker_color="blue",
                showlegend=False,
                hovertemplate=(
                    "%{x:.2%}"
                    if isinstance(bm_val, float) and -1 < bm_val < 1
                    else "%{x}"
                ),
            ),
            row=row,
            col=col,
        )
        sub_fig.update_xaxes(showticklabels=False, row=row, col=col)
        sub_fig.update_yaxes(showticklabels=True, row=row, col=col)
    sub_fig.update_layout(
        height=60 * n_metrics + 100,
        title_text="Key Metrics Comparison",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    st.plotly_chart(sub_fig, use_container_width=True)
