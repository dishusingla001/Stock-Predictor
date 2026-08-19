import pandas as pd
import os

from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange


# ==========================================
# Feature Engineering Function
# ==========================================

def create_features(stock_name):

    print(f"\nProcessing {stock_name}...")

    # --------------------------------------
    # 1. Load Data
    # --------------------------------------

    input_file = f"data/raw/{stock_name}.csv"

    data = pd.read_csv(input_file)

    data["Date"] = pd.to_datetime(data["Date"])

    data = data.sort_values("Date")


    # --------------------------------------
    # 2. Trend Features
    # --------------------------------------

    data["SMA10"] = (
        data["Close"]
        .rolling(window=10)
        .mean()
    )

    data["SMA20"] = (
        data["Close"]
        .rolling(window=20)
        .mean()
    )

    data["SMA50"] = (
        data["Close"]
        .rolling(window=50)
        .mean()
    )

    data["EMA20"] = (
        data["Close"]
        .ewm(span=20, adjust=False)
        .mean()
    )

    data["EMA50"] = (
        data["Close"]
        .ewm(span=50, adjust=False)
        .mean()
    )


    # --------------------------------------
    # 3. Momentum Features
    # --------------------------------------

    rsi = RSIIndicator(
        close=data["Close"],
        window=14
    )

    data["RSI"] = rsi.rsi()


    macd = MACD(
        close=data["Close"]
    )

    data["MACD"] = macd.macd()

    data["MACD_signal"] = macd.macd_signal()

    data["MACD_histogram"] = macd.macd_diff()


    roc = ROCIndicator(
        close=data["Close"],
        window=12
    )

    data["ROC"] = roc.roc()


    # --------------------------------------
    # 4. Volatility Features
    # --------------------------------------

    bb = BollingerBands(
        close=data["Close"],
        window=20,
        window_dev=2
    )

    data["BB_upper"] = bb.bollinger_hband()

    data["BB_middle"] = bb.bollinger_mavg()

    data["BB_lower"] = bb.bollinger_lband()


    atr = AverageTrueRange(
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        window=14
    )

    data["ATR"] = atr.average_true_range()


    # --------------------------------------
    # 5. Volume Features
    # --------------------------------------

    data["Volume_change"] = (
        data["Volume"].pct_change()
    )

    data["Volume_SMA"] = (
        data["Volume"]
        .rolling(window=20)
        .mean()
    )

    data["Volume_ratio"] = (
        data["Volume"] / data["Volume_SMA"]
    )


    # --------------------------------------
    # 6. Create Target Variable
    # --------------------------------------

    data["Future_Close"] = (
        data["Close"].shift(-5)
    )

    data["Target"] = (
        data["Future_Close"] > data["Close"]
    ).astype(int)


    # --------------------------------------
    # 7. Remove Missing Values
    # --------------------------------------

    data.dropna(inplace=True)


    # --------------------------------------
    # 8. Save Processed Data
    # --------------------------------------

    os.makedirs("data/processed", exist_ok=True)

    output_file = (
        f"data/processed/{stock_name}_features.csv"
    )

    data.to_csv(
        output_file,
        index=False
    )

    print(f"Saved: {output_file}")

    print(f"Rows: {len(data)}")

    print(
        f"UP: {sum(data['Target'] == 1)} | "
        f"DOWN: {sum(data['Target'] == 0)}"
    )


# ==========================================
# Process All Stocks
# ==========================================

stocks = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "INFY",
    "ICICIBANK"
]


for stock in stocks:

    create_features(stock)


print("\n================================")
print("Feature engineering completed!")
print("================================")


# ==========================================
# PHASE 3 - ML DATASET PREPARATION
# ==========================================

for stock in stocks:

    data = pd.read_csv(
        f"data/processed/{stock}_features.csv"
    )

    # 1. Create Target
    # 1 = tomorrow's Close is higher
    # 0 = tomorrow's Close is lower or equal
    data["Target"] = (
        data["Close"].shift(-1) > data["Close"]
    ).astype(int)

    # 2. Remove the final row
    # because there is no tomorrow's price
    data = data.iloc[:-1]

    # 3. Remove rows containing NaN
    # caused by technical indicators
    data = data.dropna()

    # 4. Save ML-ready dataset
    data.to_csv(
        f"data/processed/{stock}_ML.csv",
        index=False
    )

    # 5. Check the dataset
    print(f"\n========== {stock} ML DATASET ==========")
    print(data.head())
    print("\nDataset Shape:")
    print(data.shape)
    print("\nTarget Distribution:")
    print(data["Target"].value_counts())