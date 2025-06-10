import streamlit as st
import yfinance as yf
import bt
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time
from plotly.subplots import make_subplots
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

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

# Blockforce Capital Color Palette
blockforce_colors = {
    "primary_darkest": "#13151e",
    "primary_dark": "#39384f",
    "primary_medium": "#615987",
    "primary_light": "#928baf",
    "accent_gold": "#ffc100",
    "accent_turquoise": "#77e8e3",
    "accent_sky": "#769de5",
    "accent_coral": "#e86449",
    "background_dark": "#0A0F1E",
    "background_card": "#141B2E",
    "background_hover": "#1E2A3B",
    "text_primary": "#FFFFFF",
    "text_secondary": "#769de5",
    "text_muted": "#928baf",
}


# Cache the data download
@st.cache_data
def download_data():
    tickers = ["SPY", "AGG"]
    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers, start="2015-01-01", end="2026-01-01", auto_adjust=True
            )["Close"]
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
bfc_allocation = st.slider("Blockforce Capital Allocation (%)", 1, 25, 10, step=1)

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
def format_stat(val, metric=None):
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


# Custom JS valueFormatter for row-based formatting
custom_formatter = JsCode(
    """
function(params) {
    var metric = params.data['Metric'];
    var value = params.value;
    if (value == null || value === '') return '';
    // Format as date string
    if (metric === 'start' || metric === 'end') {
        return value;
    }
    // Format as percent
    if ([
        'total_return', 'cagr', 'max_drawdown', 'calmar', 'mtd', 'three_month', 'six_month', 'ytd', 'one_year', 'three_year', 'five_year', 'ten_year', 'best_month', 'worst_month', 'best_year', 'worst_year'
    ].includes(metric)) {
        return (value * 100).toFixed(2) + '%';
    }
    // Format as decimal (Sharpe, Sortino, Calmar)
    if ([
        'sharpe', 'daily_sharpe', 'sortino', 'daily_sortino', 'calmar'
    ].includes(metric)) {
        return value.toFixed(2);
    }
    // Default: show as is
    return value;
}
"""
)


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
    # Get all available metrics from the stats DataFrame
    stats_source = custom_portfolio.stats
    all_metrics = list(stats_source.index)

    # Define groups and their metrics (order and group similar to bt's display)
    stat_sections = [
        ("Period", ["start", "end", "rf"]),
        (
            "Returns",
            [
                "total_return",
                "cagr",
                "mtd",
                "three_month",
                "six_month",
                "ytd",
                "one_year",
                "three_year",
                "five_year",
                "ten_year",
                "since_incep_ann",
            ],
        ),
        (
            "Risk",
            [
                "max_drawdown",
                "calmar",
                "daily_sharpe",
                "daily_sortino",
                "sharpe",
                "sortino",
                "daily_mean_ann",
                "daily_vol_ann",
                "daily_skew",
                "daily_kurt",
                "monthly_sharpe",
                "monthly_sortino",
                "monthly_mean_ann",
                "monthly_vol_ann",
                "monthly_skew",
                "monthly_kurt",
                "yearly_sharpe",
                "yearly_sortino",
                "yearly_mean",
                "yearly_vol",
                "yearly_skew",
                "yearly_kurt",
            ],
        ),
        (
            "Best/Worst",
            [
                "best_day",
                "worst_day",
                "best_month",
                "worst_month",
                "best_year",
                "worst_year",
            ],
        ),
        (
            "Drawdowns & Win Rates",
            [
                "avg_drawdown",
                "avg_drawdown_days",
                "avg_up_month",
                "avg_down_month",
                "win_year_perc",
                "win_12m_perc",
            ],
        ),
    ]

    # Build a list of (section, metric) tuples for display order
    display_rows = []
    for section, metrics in stat_sections:
        # Only add section if at least one metric is present
        present_metrics = [m for m in metrics if m in all_metrics]
        if present_metrics:
            display_rows.append((section, None))  # Section header
            for m in present_metrics:
                display_rows.append((section, m))

    # Build DataFrame for display
    rows = []
    for section, metric in display_rows:
        if metric is None:
            # Section header row
            row = {"Metric": section}
            for col in [
                f"Custom Portfolio ({bfc_allocation}% BFC Net)",
                "60/40 Portfolio",
                "100% SPY",
                "100% BFC Net",
            ]:
                row[col] = ""
            row["_is_section"] = True
        else:
            row = {"Metric": metric.replace("_", " ").title()}
            for label, stats_obj in zip(
                [
                    f"Custom Portfolio ({bfc_allocation}% BFC Net)",
                    "60/40 Portfolio",
                    "100% SPY",
                    "100% BFC Net",
                ],
                [custom_portfolio, benchmark_60_40, benchmark_spy, benchmark_bfc],
            ):
                val = (
                    stats_obj.stats.loc[metric, stats_obj.stats.columns[0]]
                    if metric in stats_obj.stats.index
                    else None
                )
                row[label] = val
            row["_is_section"] = False
        rows.append(row)
    stats_df = pd.DataFrame(rows)
    print(stats_df)
    # Custom value formatter for stats table (explicit, robust)
    stats_formatter = JsCode(
        """
    function(params) {
        if (params.value == null || params.value === '') return '';
        var metric = params.data.Metric;
        // Date metrics
        if ([
            'Start', 'End'
        ].includes(metric)) {
            return params.value;
        }
        // Percent metrics
        if ([
            'Total Return', 'Cagr', 'Mtd', 'Three Month', 'Six Month', 'Ytd',
            'One Year', 'Three Year', 'Five Year', 'Ten Year',
            'Best Month', 'Worst Month', 'Best Year', 'Worst Year',
            'Avg Up Month', 'Avg Down Month', 'Win Year Perc', 'Win 12m Perc',
            'Max Drawdown', 'Avg Drawdown', 'Drawdown',
            'Yearly Mean', 'Yearly Vol', 'Best Day', 'Worst Day'
        ].includes(metric)) {
            return (params.value * 100).toFixed(2) + '%';
        }
        // Decimal metrics
        if ([
            'Calmar', 'Daily Sharpe', 'Daily Sortino', 'Monthly Sharpe', 'Monthly Sortino',
            'Yearly Sharpe', 'Yearly Sortino', 'Sharpe', 'Sortino'
        ].includes(metric)) {
            return params.value.toFixed(2);
        }
        // Integer/float metrics
        if ([
            'Avg Drawdown Days'
        ].includes(metric)) {
            return params.value.toFixed(2);
        }
        // Skew/Kurtosis (raw float)
        if ([
            'Daily Skew', 'Daily Kurt', 'Monthly Skew', 'Monthly Kurt', 'Yearly Skew', 'Yearly Kurt'
        ].includes(metric)) {
            return params.value.toFixed(2);
        }
        // Default
        return params.value;
    }
    """
    )

    # Build grid options for stats table
    gb_stats = GridOptionsBuilder.from_dataframe(stats_df)
    # Configure columns
    for col in stats_df.columns:
        if col == f"Custom Portfolio ({bfc_allocation}% BFC Net)":
            gb_stats.configure_column(
                col,
                valueFormatter=stats_formatter,
                cellStyle={
                    "textAlign": "right",
                    "color": blockforce_colors["accent_turquoise"],
                    "fontWeight": "bold",
                },
            )
        elif col not in ["Metric", "_is_section"]:
            gb_stats.configure_column(
                col,
                valueFormatter=stats_formatter,
                cellStyle={
                    "textAlign": "right",
                    "color": blockforce_colors["text_primary"],
                },
            )
    # Configure Metric column
    gb_stats.configure_column(
        "Metric",
        cellStyle={
            "textAlign": "left",
            "fontWeight": "bold",
            "color": blockforce_colors["text_primary"],
        },
        minWidth=100,
    )
    # Hide the _is_section column
    gb_stats.configure_column("_is_section", hide=True)
    # Enhanced section header styling
    stats_row_style = JsCode(
        f"""
    function(params) {{
        if (params.data._is_section) {{
            return {{
                'backgroundColor': '{blockforce_colors['primary_dark']}',
                'color': '{blockforce_colors['accent_turquoise']}',
                'fontWeight': 'bold',
                'fontSize': '1.1em',
                'borderBottom': '2px solid {blockforce_colors['accent_turquoise']}',
                'paddingTop': '0px',
                'paddingBottom': '0px'
            }};
        }}
        let dark = '{blockforce_colors['background_card']}';
        let darker = '{blockforce_colors['background_dark']}';
        let bg = (params.node.rowIndex % 2 === 0) ? dark : darker;
        return {{'backgroundColor': bg, 'color': '{blockforce_colors['text_primary']}', 'fontWeight': 'normal', 'fontSize': '1em'}};
    }}
    """
    )
    gb_stats.configure_grid_options(
        getRowStyle=stats_row_style,
        headerHeight=36,
        rowHeight=24,
        suppressRowClickSelection=True,
        suppressCellSelection=True,
        domLayout="normal",
        gridStyle={
            "backgroundColor": blockforce_colors["background_card"],
            "color": blockforce_colors["text_primary"],
        },
    )
    # Pass the full DataFrame (with _is_section) to AgGr
    AgGrid(
        stats_df,
        gridOptions=gb_stats.build(),
        fit_columns_on_grid_load=True,
        theme="alpine-dark",
        height=950,
        enable_enterprise_modules=False,
        allow_unsafe_jscode=True,
    )

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

    monthly_returns = {k: get_monthly_returns(v) for k, v in series.items()}
    annual_returns = {k: get_annual_returns(v) for k, v in series.items()}

    # Build a flat table: one row per asset per year, with manual year separator rows
    rows = []
    years = sorted(list(set([d.year for d in all_data.index])))
    months = [
        pd.Timestamp(year=2000, month=m, day=1).strftime("%b") for m in range(1, 13)
    ]

    for year in reversed(years):
        # Insert a separator row for the year
        sep_row = {
            "Year": str(year),
            "Asset": "",
            **{m: "" for m in months},
            "Annual": "",
        }
        rows.append(sep_row)

        for name in series.keys():
            row = {"Year": "", "Asset": name}
            # Monthly returns for this year
            for m in range(1, 13):
                period = pd.Period(f"{year}-{m:02d}")
                val = (
                    monthly_returns[name].loc[period]
                    if period in monthly_returns[name].index
                    else None
                )
                row[pd.Timestamp(year=year, month=m, day=1).strftime("%b")] = (
                    f"{val*100:.1f}%" if pd.notnull(val) else ""
                )

            # Annual return - special handling for partial years
            if year in [2019, 2025]:
                # Get all available monthly returns for the year
                monthly_vals = [
                    monthly_returns[name].loc[pd.Period(f"{year}-{m:02d}")]
                    for m in range(1, 13)
                    if pd.Period(f"{year}-{m:02d}") in monthly_returns[name].index
                ]
                if monthly_vals:
                    # Calculate annual return from available months
                    ann_return = (1 + pd.Series(monthly_vals)).prod() - 1
                    row["Annual"] = f"{ann_return*100:.1f}%*"
                else:
                    row["Annual"] = ""
            else:
                # Normal year - use the annual return directly
                period = pd.Period(f"{year}")
                ann = (
                    annual_returns[name].loc[period]
                    if period in annual_returns[name].index
                    else None
                )
                row["Annual"] = f"{ann*100:.1f}%" if pd.notnull(ann) else ""
            rows.append(row)

    columns = ["Year", "Asset"] + months + ["Annual"]
    df = pd.DataFrame(rows, columns=columns)

    st.markdown("## Monthly and Annual Returns by Portfolio")

    # Build grid options
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        resizable=True, filterable=False, sortable=False, wrapText=True, autoHeight=True
    )
    gb.configure_grid_options(domLayout="normal", rowHeight=24)
    gb.configure_column("Asset", wrapText=False, minWidth=200, maxWidth=300)
    gb.configure_column("Year", wrapText=False, minWidth=80, maxWidth=100)

    # Restore the original row_style_js for the monthly returns table
    row_style_js = JsCode(
        f"""
    function(params) {{
        // Year separator row: Asset is empty
        if (!params.data.Asset) {{
            return {{
                'backgroundColor': '{blockforce_colors['primary_dark']}',
                'color': '{blockforce_colors['accent_turquoise']}',
                'fontWeight': 'normal',
                'fontSize': '1em',
                'borderBottom': '2px solid {blockforce_colors['accent_turquoise']}'
            }};
        }}
        // Highlight Custom Portfolio row (teal font)
        if (params.data.Asset && params.data.Asset.startsWith('Custom Portfolio')) {{
            return {{
                'color': '{blockforce_colors['accent_turquoise']}',
                'fontWeight': 'bold',
                'backgroundColor': '{blockforce_colors['background_dark']}'
            }};
        }}
        // Alternate year backgrounds (for asset rows)
        let yearSepIdx = params.node.rowIndex;
        while (yearSepIdx > 0 && params.api.getDisplayedRowAtIndex(yearSepIdx).data.Asset) {{
            yearSepIdx--;
        }}
        if (yearSepIdx >= 0) {{
            let yearSep = params.api.getDisplayedRowAtIndex(yearSepIdx);
            if (yearSep && yearSep.data && yearSep.data.Year) {{
                let year = parseInt(yearSep.data.Year);
                if (!isNaN(year)) {{
                    return {{'backgroundColor': (year % 2 === 0) ? '{blockforce_colors['background_card']}' : '{blockforce_colors['background_dark']}', 'color': '{blockforce_colors['text_primary']}'}};
                }}
            }}
        }}
        return {{'color': '{blockforce_colors['text_primary']}'}};
    }}
    """
    )
    gb.configure_grid_options(getRowStyle=row_style_js)

    AgGrid(
        df,
        gridOptions=gb.build(),
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        theme="alpine-dark",
        height=920,
        enable_enterprise_modules=False,
    )

    st.markdown(
        "* Partial year (2019 or 2025) - returns calculated based on available months only"
    )


display_monthly_returns()
