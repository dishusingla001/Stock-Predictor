import yfinance as yf
import pandas as pd
import os

stocks = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS"
]

os.makedirs("data/raw", exist_ok=True)

for ticker in stocks:

    print(f"Downloading {ticker}...")

    data = yf.download(
        ticker,
        start="2015-01-01",
        end="2026-01-01",
        auto_adjust=False,
        progress=False
    )

    # Handle MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Convert Date index into a normal column
    data.reset_index(inplace=True)

    filename = f"data/raw/{ticker.replace('.NS', '')}.csv"

    data.to_csv(filename, index=False)

    print(f"Saved: {filename}")