import streamlit as st
import yfinance as yf
import bt
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Blockforce Capital Portfolio Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Show Blockforce logo from local assets
st.image("assets/image.png", width=260)

# Custom header and subtitle using markdown
st.markdown(
    "<h1 style='color:#f7931a; font-size:2.5rem; font-weight:bold;'>Portfolio Analysis Dashboard</h1>",
    unsafe_allow_html=True,
)
st.subheader(
    "Analyze the impact of adding Blockforce Capital allocation to your portfolio"
)


# Cache the data download
@st.cache_data
def download_data():
    tickers = ["SPY", "AGG"]
    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            data = yf.download(tickers, start="2015-01-01", end="2026-01-01")["Close"]
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
st.markdown(
    "<h2 style='color:#f7931a;'>Impact of adding BFC Allocation to 60/40</h2>",
    unsafe_allow_html=True,
)
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

# Add Blockforce-only backtest
benchmark_bfc = run_backtest(all_data, {"BFC Net": 1.0})

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
        line=dict(width=2, color="#4a90e2"),
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
    plot_bgcolor="#1a1a1a",
    paper_bgcolor="#1a1a1a",
    font=dict(color="white"),
    xaxis=dict(gridcolor="#333333"),
    yaxis=dict(gridcolor="#333333"),
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
            marker=dict(colors=["green", "#4a90e2", "#f7931a"]),
            textinfo="label+percent",
            hole=0.4,
        )
    )
    pie_fig.update_layout(
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),
        height=350,
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font=dict(color="white"),
    )
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
    spy_stats_clean = [format_stat(x) for x in benchmark_spy.stats.iloc[:, 0].values]
    bfc_stats_clean = [format_stat(x) for x in benchmark_bfc.stats.iloc[:, 0].values]
    stats_df = pd.DataFrame(
        {
            "Metric": custom_portfolio.stats.index,
            f"Custom Portfolio ({bfc_allocation}% BFC Net)": custom_stats_clean,
            "60/40 Portfolio": benchmark_stats_clean,
            "100% SPY": spy_stats_clean,
            "100% BFC Net": bfc_stats_clean,
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
        try:
            spy_val = benchmark_spy.stats.iloc[:, 0].loc[key]
        except Exception:
            spy_val = None
        try:
            agg_val = benchmark_agg.stats.iloc[:, 0].loc[key]
        except Exception:
            agg_val = None
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
            try:
                spy_val = abs(float(spy_val))
            except Exception:
                pass
            try:
                agg_val = abs(float(agg_val))
            except Exception:
                pass
        sub_fig.add_trace(
            go.Bar(
                x=[custom_val],
                y=["Custom"],
                orientation="h",
                marker_color="#f7931a",
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
        sub_fig.add_trace(
            go.Bar(
                x=[spy_val],
                y=["SPY"],
                orientation="h",
                marker_color="green",
                showlegend=False,
                hovertemplate=(
                    "%{x:.2%}"
                    if isinstance(spy_val, float) and -1 < spy_val < 1
                    else "%{x}"
                ),
            ),
            row=row,
            col=col,
        )
        sub_fig.add_trace(
            go.Bar(
                x=[agg_val],
                y=["AGG"],
                orientation="h",
                marker_color="#4a90e2",
                showlegend=False,
                hovertemplate=(
                    "%{x:.2%}"
                    if isinstance(agg_val, float) and -1 < agg_val < 1
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
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font=dict(color="white"),
    )
    st.plotly_chart(sub_fig, use_container_width=True)

# Update bar colors to use Blockforce orange
for trace in sub_fig.data:
    if trace.marker.color == "purple":
        trace.marker.color = "#f7931a"  # Blockforce orange
    elif trace.marker.color == "blue":
        trace.marker.color = "#ffffff"  # White for comparison

# Add a disclosure footnote at the end
st.caption(
    "Note: AGG returns shown are price returns only (do not include reinvested interest or distributions). "
    "This is a limitation of freely available data sources like yfinance."
)


# --- Monthly Returns Table Section ---
def display_monthly_returns():
    # Helper to get monthly returns from price series
    def get_monthly_returns(prices):
        monthly = prices.resample("ME").last().pct_change()
        monthly.index = monthly.index.to_period("M")
        return monthly

    # Helper to get annual returns from price series
    def get_annual_returns(prices):
        # For partial years, calculate based on available months
        annual = prices.resample("YE").last().pct_change()
        annual.index = annual.index.to_period("Y")
        return annual

    # Get price series for each
    series = {
        f"Custom Portfolio ({bfc_allocation}% BFC Net)": custom_portfolio[0].prices,
        "60/40 Portfolio": benchmark_60_40[0].prices,
        "100% SPY": benchmark_spy[0].prices,
        "100% AGG": benchmark_agg[0].prices,
        "100% BFC Net": benchmark_bfc[0].prices,
    }

    # Debug: Print date ranges for each series
    # st.write("Debug - Date ranges:")
    # for name, prices in series.items():
    #     st.write(f"{name}: {prices.index.min()} to {prices.index.max()}")

    # Compute monthly and annual returns
    monthly_returns = {k: get_monthly_returns(v) for k, v in series.items()}
    annual_returns = {k: get_annual_returns(v) for k, v in series.items()}

    # Build table
    rows = []
    # Ensure we include all years from 2019 to 2025
    years = list(range(2019, 2026))


    for year in years:
        rows.append(
            [str(year), "", "", "", "", "", "", "", "", "", "", "", "", ""]
        )  # Blank row for year
        for name in series.keys():
            row = ["", name]  # Year col blank for asset rows

            # Monthly returns for this year
            monthly_vals = []
            for m in range(1, 13):
                period = pd.Period(f"{year}-{m:02d}")
                val = (
                    monthly_returns[name].loc[period]
                    if period in monthly_returns[name].index
                    else float("nan")
                )
                monthly_vals.append(val)
                row.append(f"{val*100:.1f}%" if pd.notnull(val) else "")

            # Calculate annual return for partial years
            if year in [2019, 2025]:
                # Filter out NaN values and calculate geometric return
                valid_returns = [r for r in monthly_vals if pd.notnull(r)]
                if valid_returns:
                    # Calculate geometric return for available months
                    ann_return = (1 + pd.Series(valid_returns)).prod() - 1
                    row.append(f"{ann_return*100:.1f}%*")
                else:
                    row.append("")
            else:
                # For complete years, use the standard annual return
                period = pd.Period(f"{year}")
                ann = (
                    annual_returns[name].loc[period]
                    if period in annual_returns[name].index
                    else float("nan")
                )
                row.append(f"{ann*100:.1f}%" if pd.notnull(ann) else "")

            rows.append(row)

    columns = ["Year", "Asset"] + [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
        "Annual",
    ]
    df = pd.DataFrame(rows, columns=columns)

    st.markdown("## Monthly and Annual Returns by Portfolio")
    st.table(df)

    # Add footnote for partial years
    st.markdown(
        "* Partial year (2019 or 2025) - returns calculated based on available months only"
    )


display_monthly_returns()
