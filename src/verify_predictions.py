import os

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ========================================
# CONFIGURATION
# ========================================

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

TEST_DAYS = 100


# ========================================
# HEADER
# ========================================

print("\n========================================")
print("     5-DAY PREDICTION VERIFICATION")
print("========================================\n")

metadata_path = os.path.join(MODEL_DIR, "model_metadata.csv")
metadata = pd.read_csv(metadata_path).set_index("Stock")
all_results = []


# ========================================
# PROCESS EACH STOCK
# ========================================

for stock in stocks:

    print(f"\nProcessing {stock}...")
    print("----------------------------------------")

    data_path = os.path.join(
        PROCESSED_DIR,
        f"{stock}_ML.csv"
    )

    data = pd.read_csv(data_path)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    model_path = os.path.join(
        BASE_DIR,
        metadata.loc[stock, "Model_File"]
    )
    scaler_path = os.path.join(
        BASE_DIR,
        metadata.loc[stock, "Scaler_File"]
    )

    model_name = metadata.loc[stock, "Model"]
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    verification_data = data.tail(TEST_DAYS).copy()

    predictions = []
    probabilities = []
    actuals = []

    for _, row in verification_data.iterrows():

        X = pd.DataFrame(
            [[row[feature] for feature in features]],
            columns=features
        )

        # Logistic Regression was trained on scaled features; tree models were not.
        if model_name == "Logistic Regression":
            X_input = scaler.transform(X)
        else:
            X_input = X

        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0]

        predictions.append(prediction)
        probabilities.append(
            probability[1] if prediction == 1 else probability[0]
        )
        actuals.append(row["Target"])

    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(
        actuals,
        predictions,
        zero_division=0
    )
    recall = recall_score(
        actuals,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        actuals,
        predictions,
        zero_division=0
    )
    cm = confusion_matrix(actuals, predictions)

    for i, (_, row) in enumerate(verification_data.iterrows()):

        predicted_direction = (
            "UP" if predictions[i] == 1 else "DOWN"
        )
        actual_direction = (
            "UP" if actuals[i] == 1 else "DOWN"
        )
        result = (
            "CORRECT"
            if predictions[i] == actuals[i]
            else "WRONG"
        )

        all_results.append({
            "Stock": stock,
            "Date": row["Date"].date(),
            "Close": row["Close"],
            "Predicted": predicted_direction,
            "Probability": probabilities[i],
            "Future_Close": row["Future_Close"],
            "Actual": actual_direction,
            "Result": result
        })

    correct = sum(
        predictions[i] == actuals[i]
        for i in range(len(predictions))
    )
    wrong = len(predictions) - correct

    print(f"Model              : {model_name}")
    print(f"Verification Days   : {len(verification_data)}")
    print(
        f"Date Range          : "
        f"{verification_data['Date'].iloc[0].date()} "
        f"to "
        f"{verification_data['Date'].iloc[-1].date()}"
    )
    print(f"Accuracy            : {accuracy * 100:.2f}%")
    print(f"Precision           : {precision * 100:.2f}%")
    print(f"Recall              : {recall * 100:.2f}%")
    print(f"F1 Score            : {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nCorrect Predictions : {correct}")
    print(f"Wrong Predictions   : {wrong}")


# ========================================
# SAVE DETAILED RESULTS
# ========================================

results_df = pd.DataFrame(all_results)

output_path = os.path.join(
    PROCESSED_DIR,
    "verification_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


# ========================================
# FINAL SUMMARY
# ========================================

print("\n")
print("========================================")
print("        VERIFICATION COMPLETED")
print("========================================\n")

summary = []

for stock in stocks:

    stock_data = results_df[
        results_df["Stock"] == stock
    ]

    correct = (
        stock_data["Result"] == "CORRECT"
    ).sum()

    total = len(stock_data)

    summary.append({
        "Stock": stock,
        "Test_Days": total,
        "Correct": correct,
        "Wrong": total - correct,
        "Accuracy": correct / total * 100
    })

summary_df = pd.DataFrame(summary)

print(summary_df.to_string(index=False))

print("\nDetailed verification file:")
print(output_path)
