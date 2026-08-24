import pandas as pd
from sklearn.preprocessing import StandardScaler


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
    # Time-Based Train/Test Split
    # ==========================================

    split_index = int(len(X) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # ==========================================
    # Feature Scaling
    # ==========================================

    scaler = StandardScaler()

    # Fit only on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Use the same scaler on test data
    X_test_scaled = scaler.transform(X_test)

    print("\n========== SCALED DATA ==========")
    print("X_train_scaled shape:", X_train_scaled.shape)
    print("X_test_scaled shape:", X_test_scaled.shape)
    print("\nFirst 5 scaled training rows:")
    print(X_train_scaled[:5])

    # ==========================================
    # Check Split
    # ==========================================

    print("\n========== TRAIN / TEST SPLIT ==========")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # ==========================================
    # Check Dates
    # ==========================================

    print("\n========== DATE RANGE ==========")
    print("Training period:")
    print(data["Date"].iloc[0])
    print("to")
    print(data["Date"].iloc[split_index - 1])
    print("\nTesting period:")
    print(data["Date"].iloc[split_index])
    print("to")
    print(data["Date"].iloc[-1])

    # ==========================================
    # Check Features and Target
    # ==========================================

    print("\nFeatures:")
    print(X.head())

    print("\nTarget:")
    print(y.head())

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)
