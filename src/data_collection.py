import yfinance as yf
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
        auto_adjust=False
    )

    filename = f"data/raw/{ticker.replace('.NS', '')}.csv"

    data.to_csv(filename)

    print(f"Saved: {filename}")