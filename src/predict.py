import os

import joblib
import pandas as pd


# =========================
# CONFIGURATION
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

stocks = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "INFY",
    "ICICIBANK"
]

features = [
    "RSI",
    "EMA20",
    "EMA50",
    "MACD",
    "ROC",
    "ATR",
    "Volume"
]


# =========================
# PREDICTION
# =========================

print("\n========================================")
print("      5-DAY STOCK PREDICTION")
print("========================================\n")

metadata_path = os.path.join(MODEL_DIR, "model_metadata.csv")
metadata = pd.read_csv(metadata_path).set_index("Stock")
results = []

for stock in stocks:

    print(f"Processing {stock}...")

    data_path = os.path.join(
        PROCESSED_DIR,
        f"{stock}_ML.csv"
    )

    data = pd.read_csv(data_path)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date")

    latest = data.iloc[-1]
    model_info = metadata.loc[stock]

    model_path = os.path.join(BASE_DIR, model_info["Model_File"])
    scaler_path = os.path.join(BASE_DIR, model_info["Scaler_File"])

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X_latest = pd.DataFrame(
        [[latest[feature] for feature in features]],
        columns=features
    )

    # Logistic Regression was trained on scaled features. Tree models were not.
    if model_info["Model"] == "Logistic Regression":
        X_input = scaler.transform(X_latest)
    else:
        X_input = X_latest

    prediction = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]

    down_probability = probabilities[0]
    up_probability = probabilities[1]

    if prediction == 1:
        direction = "UP"
        probability = up_probability
    else:
        direction = "DOWN"
        probability = down_probability

    print("----------------------------------------")
    print(f"Stock        : {stock}")
    print(f"Model        : {model_info['Model']}")
    print(f"Data Date    : {latest['Date'].date()}")
    print(f"Close Price  : {latest['Close']:.2f}")
    print(f"Prediction   : {direction}")
    print(f"Probability  : {probability * 100:.2f}%")
    print(f"UP Probability   : {up_probability * 100:.2f}%")
    print(f"DOWN Probability : {down_probability * 100:.2f}%")
    print("----------------------------------------\n")

    results.append({
        "Stock": stock,
        "Date": latest["Date"].date(),
        "Close": latest["Close"],
        "Model": model_info["Model"],
        "Prediction": direction,
        "Probability": probability,
        "UP_Probability": up_probability,
        "DOWN_Probability": down_probability
    })


# =========================
# SAVE RESULTS
# =========================

results_df = pd.DataFrame(results)

output_path = os.path.join(
    PROCESSED_DIR,
    "predictions.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\n========================================")
print("        FINAL PREDICTIONS")
print("========================================\n")

print(
    results_df[
        [
            "Stock",
            "Date",
            "Close",
            "Prediction",
            "Probability"
        ]
    ].to_string(index=False)
)

print("\nPrediction file saved at:")
print(output_path)

print("\nNOTE:")
print("These predictions use the latest date")
print("available in your processed dataset.")
