import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


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

results = []


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

    # ==========================================
    # Logistic Regression
    # ==========================================

    logistic_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    logistic_model.fit(X_train_scaled, y_train)

    # ==========================================
    # Predictions and Evaluation
    # ==========================================

    y_pred = logistic_model.predict(X_test_scaled)
    y_probability = logistic_model.predict_proba(X_test_scaled)
    up_probability = y_probability[:, 1]

    results.append({
        "Stock": stock,
        "Model": "Logistic Regression",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, up_probability)
    })

    print("\n========== LOGISTIC REGRESSION ==========")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, up_probability):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ==========================================
    # Random Forest
    # ==========================================

    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )

    random_forest_model.fit(X_train, y_train)

    rf_pred = random_forest_model.predict(X_test)
    rf_probability = random_forest_model.predict_proba(X_test)
    rf_up_probability = rf_probability[:, 1]

    results.append({
        "Stock": stock,
        "Model": "Random Forest",
        "Accuracy": accuracy_score(y_test, rf_pred),
        "Precision": precision_score(y_test, rf_pred, zero_division=0),
        "Recall": recall_score(y_test, rf_pred, zero_division=0),
        "F1": f1_score(y_test, rf_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, rf_up_probability)
    })

    print("\n========== RANDOM FOREST ==========")
    print(f"Accuracy : {accuracy_score(y_test, rf_pred):.4f}")
    print(f"Precision: {precision_score(y_test, rf_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, rf_pred, zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_test, rf_pred, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, rf_up_probability):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, rf_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, rf_pred, zero_division=0))

    # ==========================================
    # XGBoost
    # ==========================================

    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    xgb_model.fit(X_train, y_train)

    xgb_pred = xgb_model.predict(X_test)
    xgb_probability = xgb_model.predict_proba(X_test)
    xgb_up_probability = xgb_probability[:, 1]

    results.append({
        "Stock": stock,
        "Model": "XGBoost",
        "Accuracy": accuracy_score(y_test, xgb_pred),
        "Precision": precision_score(y_test, xgb_pred, zero_division=0),
        "Recall": recall_score(y_test, xgb_pred, zero_division=0),
        "F1": f1_score(y_test, xgb_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, xgb_up_probability)
    })

    print("\n========== XGBOOST ==========")
    print(f"Accuracy : {accuracy_score(y_test, xgb_pred):.4f}")
    print(f"Precision: {precision_score(y_test, xgb_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, xgb_pred, zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_test, xgb_pred, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, xgb_up_probability):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, xgb_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, xgb_pred, zero_division=0))


# ==========================================
# 8. Model Comparison
# ==========================================

results_df = pd.DataFrame(results)

print("\n==========================================")
print("              MODEL COMPARISON")
print("==========================================")
print(results_df.to_string(index=False))

results_df.to_csv(
    "data/processed/model_results.csv",
    index=False
)

best_models = (
    results_df
    .sort_values(
        ["Stock", "ROC-AUC"],
        ascending=[True, False]
    )
    .groupby("Stock")
    .first()
    .reset_index()
)

print("\n==========================================")
print("          BEST MODEL PER STOCK")
print("==========================================")
print(
    best_models[
        [
            "Stock",
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC"
        ]
    ].to_string(index=False)
)

average_results = (
    results_df
    .groupby("Model")[[
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ]]
    .mean()
    .sort_values("ROC-AUC", ascending=False)
)

print("\n==========================================")
print("        AVERAGE MODEL PERFORMANCE")
print("==========================================")
print(average_results.to_string())
