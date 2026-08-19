import pandas as pd


# ==========================================
# 1. Load ML Dataset
# ==========================================

features = [
    "RSI",
    "EMA20",
    "EMA50",
    "MACD",
    "ROC",
    "ATR",
    "Volume"
]

stocks = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "INFY",
    "ICICIBANK"
]


for stock in stocks:

    print(f"\n========== {stock} ==========")

    data = pd.read_csv(
        f"data/processed/{stock}_ML.csv"
    )

    print("Dataset shape:", data.shape)
    print("Columns:", data.columns.tolist())

    # ==========================================
    # Separate Features and Target
    # ==========================================

    X = data[features]
    y = data["Target"]

    # ==========================================
    # Check Features and Target
    # ==========================================

    print("\nFeatures:")
    print(X.head())

    print("\nTarget:")
    print(y.head())

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)
