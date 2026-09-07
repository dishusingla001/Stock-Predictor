import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
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
trained_models = {}
scalers = {}

tscv = TimeSeriesSplit(n_splits=5)

rf_param_grid = {
    "n_estimators": [100],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", None]
}

xgb_param_grid = {
    "n_estimators": [100],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 10]
}


for stock in stocks:

    print(f"\n========== {stock} ==========")

    data = pd.read_csv(
        f"data/processed/{stock}_ML.csv"
    )

    print("Dataset shape:", data.shape)
    print("Columns:", data.columns.tolist())

    # ==========================================
    # 5-DAY TARGET VERIFICATION
    # ==========================================

    print("\n========== 5-DAY TARGET VERIFICATION ==========")

    print(data[["Date", "Close", "Future_Close", "Target"]].tail(10))

    expected_target = (
        data["Future_Close"] > data["Close"]
    ).astype(int)

    print(
        "Target matches 5-day calculation:",
        (data["Target"] == expected_target).all()
    )

    print("\nTarget distribution:")
    print(data["Target"].value_counts())

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
    scalers[stock] = scaler

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
    trained_models.setdefault(stock, {})["Logistic Regression"] = logistic_model

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

    random_forest_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(
            random_state=42,
            n_jobs=1
        ),
        param_distributions=rf_param_grid,
        n_iter=1,
        scoring="roc_auc",
        cv=tscv,
        random_state=42,
        n_jobs=1,
        refit=True
    )

    random_forest_search.fit(X_train, y_train)
    random_forest_model = random_forest_search.best_estimator_
    trained_models.setdefault(stock, {})["Random Forest"] = random_forest_model

    print("\n========== RANDOM FOREST TUNING ==========")
    print("Best CV ROC-AUC:", f"{random_forest_search.best_score_:.4f}")
    print("Best parameters:", random_forest_search.best_params_)

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

    xgb_search = RandomizedSearchCV(
        estimator=XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_jobs=1
        ),
        param_distributions=xgb_param_grid,
        n_iter=1,
        scoring="roc_auc",
        cv=tscv,
        random_state=42,
        n_jobs=1,
        refit=True
    )

    xgb_search.fit(X_train, y_train)
    xgb_model = xgb_search.best_estimator_
    trained_models.setdefault(stock, {})["XGBoost"] = xgb_model

    print("\n========== XGBOOST TUNING ==========")
    print("Best CV ROC-AUC:", f"{xgb_search.best_score_:.4f}")
    print("Best parameters:", xgb_search.best_params_)

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

os.makedirs("models", exist_ok=True)
model_metadata = []

for _, best_model_row in best_models.iterrows():
    stock = best_model_row["Stock"]
    model_name = best_model_row["Model"]

    joblib.dump(
        trained_models[stock][model_name],
        f"models/{stock}_model.joblib"
    )
    joblib.dump(
        scalers[stock],
        f"models/{stock}_scaler.joblib"
    )

    model_metadata.append({
        "Stock": stock,
        "Model": model_name,
        "Features": ",".join(features),
        "Horizon_Days": 5,
        "Model_File": f"models/{stock}_model.joblib",
        "Scaler_File": f"models/{stock}_scaler.joblib"
    })

metadata_df = pd.DataFrame(model_metadata)
metadata_df.to_csv("models/model_metadata.csv", index=False)

print("\n==========================================")
print("             SAVED BEST MODELS")
print("==========================================")
print(metadata_df.to_string(index=False))

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
