# %%
import yfinance as yf
import bt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import matplotlib.pyplot as plt
import plotly.io as pio

# Set the default renderer to browser
pio.renderers.default = "browser"

# Download historical data
tickers = ["SPY", "AGG", "BTC-USD"]
print("Downloading data...")

# Download data with retries and delay
max_retries = 3
retry_delay = 5  # seconds

for attempt in range(max_retries):
    try:
        data = yf.download(tickers, start="2015-01-01", end="2025-01-01")["Close"]
        if not data.empty:
            break
    except Exception as e:
        print(f"Attempt {attempt + 1} failed: {str(e)}")
        if attempt < max_retries - 1:
            print(f"Waiting {retry_delay} seconds before retrying...")
            time.sleep(retry_delay)
        else:
            raise Exception("Failed to download data after multiple attempts")

if data.empty:
    raise Exception("No data was downloaded")

data = data.ffill().dropna()

# Portfolio weights
weights = {"SPY": 0.55, "AGG": 0.35, "BTC-USD": 0.10}


# Define the strategy
def strategy():
    s = bt.Strategy(
        "55/35/10 Quarterly Rebalance",
        [
            bt.algos.RunQuarterly(),
            bt.algos.SelectAll(),
            bt.algos.WeighSpecified(**weights),
            bt.algos.Rebalance(),
        ],
    )
    return bt.Backtest(s, data)


print("Running backtest...")
# Run the backtest
test = strategy()
try:
    result = bt.run(test)
    print("Backtest completed successfully")
    print("\nPortfolio Statistics:")
    print(result.stats)

    print("\nDisplaying statistics...")
    result.display()

    print("\nGenerating Plotly visualization...")
    # Create the plot
    fig = go.Figure()

    # Add portfolio performance
    fig.add_trace(
        go.Scatter(
            x=result[0].prices.index,
            y=result[0].prices,
            mode="lines",
            name="Portfolio",
            line=dict(width=2, color="black"),
        )
    )

    # Add individual asset performances
    for asset in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data[asset].index,
                y=data[asset] / data[asset].iloc[0] * 100,  # Normalize to 100
                mode="lines",
                name=asset,
                line=dict(width=1),
            )
        )

    # Update layout
    fig.update_layout(
        title="55/35/10 Portfolio Backtest",
        xaxis_title="Date",
        yaxis_title="Value (Base 100)",
        showlegend=True,
        height=800,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    fig.show()
except Exception as e:
    print(f"Error during backtest: {str(e)}")
    print("\nData shape:", data.shape)
    print("\nData head:")
    print(data.head())
    print("\nData tail:")
    print(data.tail())
# %%
