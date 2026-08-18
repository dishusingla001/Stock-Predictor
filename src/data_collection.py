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
        auto_adjust=False,
        progress=False
    )

    # Convert MultiIndex columns into normal columns
    if isinstance(data.columns, __import__("pandas").MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Convert index (Date) into a normal column
    data.reset_index(inplace=True)

    filename = f"data/raw/{ticker.replace('.NS', '')}.csv"

    data.to_csv(filename, index=False)

    print(f"Saved: {filename}")