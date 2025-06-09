import streamlit as st
import yfinance as yf
import bt
import pandas as pd
import plotly.graph_objects as go
import time
import numpy as np

# Set page config
st.set_page_config(page_title="Portfolio Backtest", layout="wide")


# Cache the data download
@st.cache_data
def download_data():
    tickers = ["SPY", "AGG", "BTC-USD"]
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


# Download data
st.write("Loading data...")
data = download_data()

# Title and description
st.title("Portfolio Backtest with Bitcoin Allocation")
st.write(
    "Adjust the Bitcoin allocation using the slider below. The allocation will be taken proportionally from SPY (60%) and AGG (40%)."
)

# Slider for BTC allocation
btc_allocation = st.slider("Bitcoin Allocation (%)", 0, 10, 0, step=1)

# Calculate weights
spy_base = 0.60
agg_base = 0.40
btc_weight = btc_allocation / 100

# Adjust SPY and AGG weights proportionally
spy_reduction = btc_weight * (spy_base / (spy_base + agg_base))
agg_reduction = btc_weight * (agg_base / (spy_base + agg_base))

custom_weights = {
    "SPY": spy_base - spy_reduction,
    "AGG": agg_base - agg_reduction,
    "BTC-USD": btc_weight,
}

# Run backtests
benchmark_60_40 = run_backtest(data, {"SPY": 0.60, "AGG": 0.40})
benchmark_spy = run_backtest(data, {"SPY": 1.0})
benchmark_agg = run_backtest(data, {"AGG": 1.0})
custom_portfolio = run_backtest(data, custom_weights)

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
        name=f"Custom Portfolio ({btc_allocation}% BTC)",
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

# Create two columns for plot and weights
col_plot, col_weights = st.columns([2, 1])

with col_plot:
    st.plotly_chart(fig, use_container_width=True)

with col_weights:
    st.subheader("Current Portfolio Weights")
    pie_fig = go.Figure(
        go.Pie(
            labels=["SPY", "AGG", "BTC-USD"],
            values=[
                custom_weights["SPY"],
                custom_weights["AGG"],
                custom_weights["BTC-USD"],
            ],
            marker=dict(colors=["green", "red", "#f7931a"]),  # BTC-USD is now orange
            textinfo="label+percent",
            hole=0.4,
        )
    )
    pie_fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=350)
    st.plotly_chart(pie_fig, use_container_width=True)


# Helper function to format stats
def format_stat(val):
    if isinstance(val, (list, np.ndarray)) and len(val) == 1:
        val = val[0]
    if isinstance(val, pd.Timestamp):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, float):
        # Show as percent if between -1 and 1, else as float
        if -1 < val < 1:
            return f"{val:.2%}"
        else:
            return f"{val:.4f}"
    return str(val)


# Display portfolio statistics
st.subheader("Portfolio Statistics")

# Format stats for both portfolios
custom_stats_clean = [format_stat(x) for x in custom_portfolio.stats.values]
benchmark_stats_clean = [format_stat(x) for x in benchmark_60_40.stats.values]

stats_df = pd.DataFrame(
    {
        "Metric": custom_portfolio.stats.index,
        f"Custom Portfolio ({btc_allocation}% BTC)": custom_stats_clean,
        "60/40 Portfolio": benchmark_stats_clean,
    }
)

st.dataframe(stats_df)
