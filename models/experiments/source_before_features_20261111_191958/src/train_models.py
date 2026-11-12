"""Tune on training folds, select on validation, evaluate the winner on test."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES = ["RSI", "EMA20", "EMA50", "MACD", "ROC", "ATR", "Volume"]
STOCKS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
HORIZON = 5
SEARCH_ITERATIONS = 20
CV_SPLITS = 5


def split_periods(data):
    """60/20/20 boundaries, purging five rows before validation and test."""
    data = data.sort_values("Date").reset_index(drop=True)
    if data["Date"].isna().any() or data["Date"].duplicated().any():
        raise ValueError("Dates must be valid and unique.")
    validation_start = int(len(data) * 0.60)
    test_start = int(len(data) * 0.80)
    train = data.iloc[:validation_start - HORIZON].copy()
    validation = data.iloc[validation_start:test_start - HORIZON].copy()
    test = data.iloc[test_start:].copy()
    if any(len(part) < CV_SPLITS + HORIZON for part in (train, validation, test)):
        raise ValueError("Insufficient history for chronological training/validation/test.")
    for part in (train, validation, test):
        if set(part["Target"].unique()) != {0, 1}:
            raise ValueError("Each period must contain both target classes for ROC-AUC.")
    return data, train, validation, test


def search_spaces():
    # The pipeline fits scaling inside each CV training fold, avoiding leakage.
    return {
        "Logistic Regression": (
            Pipeline([("scaler", StandardScaler()),
                      ("model", LogisticRegression(max_iter=2000, random_state=42))]),
            {"model__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
             "model__class_weight": [None, "balanced"]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42, n_jobs=1),
            {"n_estimators": [100, 200, 300],
             "max_depth": [None, 5, 10, 15],
             "min_samples_split": [2, 5, 10, 20],
             "min_samples_leaf": [1, 2, 4, 8],
             "max_features": ["sqrt", "log2", None],
             "class_weight": [None, "balanced"]},
        ),
        "XGBoost": (
            XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=1),
            {"n_estimators": [100, 200, 300],
             "max_depth": [2, 3, 4, 5, 6],
             "learning_rate": [0.01, 0.03, 0.05, 0.1],
             "subsample": [0.7, 0.8, 0.9, 1.0],
             "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
             "min_child_weight": [1, 3, 5, 10],
             "reg_lambda": [1.0, 5.0, 10.0]},
        ),
    }


def evaluate(model, data):
    predicted = model.predict(data[FEATURES])
    probability = model.predict_proba(data[FEATURES])[:, 1]
    target = data["Target"]
    return {
        "Accuracy": accuracy_score(target, predicted),
        "Precision": precision_score(target, predicted, zero_division=0),
        "Recall": recall_score(target, predicted, zero_division=0),
        "F1": f1_score(target, predicted, zero_division=0),
        "ROC-AUC": roc_auc_score(target, probability),
    }


def select_winner(validation_results):
    # Test metrics never participate in model selection.
    return max(validation_results, key=lambda row: row["ROC-AUC"])["Model"]


def train_stock(stock):
    data = pd.read_csv(BASE_DIR / "data" / "processed" / f"{stock}_ML.csv",
                       parse_dates=["Date"])
    required = FEATURES + ["Close", "Future_Close", "Target"]
    if not np.isfinite(data[required].to_numpy(dtype=float)).all():
        raise ValueError(f"{stock}: missing or non-finite training values.")
    if not (data["Target"] == (data["Future_Close"] > data["Close"]).astype(int)).all():
        raise ValueError(f"{stock}: inconsistent targets.")
    data, train, validation, test = split_periods(data)
    periods = []
    print(f"\n========== {stock} ==========", flush=True)
    for name, part in [("Train", train), ("Validation", validation), ("Test", test)]:
        periods.append({"Stock": stock, "Period": name, "Rows": len(part),
                        "Start": part["Date"].iloc[0].date(),
                        "End": part["Date"].iloc[-1].date()})
        print(periods[-1], flush=True)

    cv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=HORIZON)
    for fit_idx, score_idx in cv.split(train):
        if train.iloc[fit_idx]["Target"].nunique() != 2 or train.iloc[score_idx]["Target"].nunique() != 2:
            raise ValueError(f"{stock}: a CV fold lacks both classes.")
    validation_results, searches, tuning_rows = [], {}, []
    for name, (estimator, parameters) in search_spaces().items():
        iterations = min(SEARCH_ITERATIONS, int(np.prod([len(v) for v in parameters.values()])))
        print(f"Tuning {name}: {iterations} candidates x {CV_SPLITS} folds", flush=True)
        search = RandomizedSearchCV(
            estimator, parameters, n_iter=iterations, scoring="roc_auc",
            cv=cv, random_state=42, n_jobs=2, refit=True, error_score="raise",
        )
        search.fit(train[FEATURES], train["Target"])
        searches[name] = search
        row = {"Stock": stock, "Model": name, "Period": "Validation",
               **evaluate(search.best_estimator_, validation),
               "CV_ROC_AUC": search.best_score_,
               "Best_Parameters": json.dumps(search.best_params_, sort_keys=True)}
        validation_results.append(row)
        for i, params in enumerate(search.cv_results_["params"]):
            tuning_rows.append({"Stock": stock, "Model": name,
                                "Parameters": json.dumps(params, sort_keys=True),
                                "Mean_CV_ROC_AUC": search.cv_results_["mean_test_score"][i],
                                "Std_CV_ROC_AUC": search.cv_results_["std_test_score"][i],
                                "Rank": search.cv_results_["rank_test_score"][i]})
        print(f"CV ROC-AUC {search.best_score_:.4f}; validation ROC-AUC {row['ROC-AUC']:.4f}", flush=True)

    winner = select_winner(validation_results)
    # With the choice frozen, refit on all pre-test history except the label gap.
    refit_data = data.iloc[:test.index[0] - HORIZON]
    final_model = clone(searches[winner].best_estimator_)
    final_model.fit(refit_data[FEATURES], refit_data["Target"])
    test_result = {"Stock": stock, "Model": winner, "Period": "Test",
                   **evaluate(final_model, test)}
    majority_class = int(refit_data["Target"].mode().iloc[0])
    test_result["Training_Majority_Baseline_Accuracy"] = accuracy_score(
        test["Target"], np.full(len(test), majority_class))
    print(f"Selected on validation: {winner}; final test: {test_result}", flush=True)

    # Preserve the separate model/scaler contract used by predict and verify.
    if winner == "Logistic Regression":
        scaler = final_model.named_steps["scaler"]
        saved_model = final_model.named_steps["model"]
    else:
        scaler = StandardScaler().fit(refit_data[FEATURES])
        saved_model = final_model
    model_path = f"models/{stock}_model.joblib"
    scaler_path = f"models/{stock}_scaler.joblib"
    joblib.dump(saved_model, BASE_DIR / model_path)
    joblib.dump(scaler, BASE_DIR / scaler_path)
    metadata = {
        "Stock": stock, "Model": winner, "Features": ",".join(FEATURES),
        "Horizon_Days": HORIZON, "Model_File": model_path, "Scaler_File": scaler_path,
        "Selection_Metric": "Validation ROC-AUC", "Gap_Rows": HORIZON,
        "Refit_End": refit_data["Date"].iloc[-1].date(),
        "Validation_Start": validation["Date"].iloc[0].date(),
        "Validation_End": validation["Date"].iloc[-1].date(),
        "Test_Start": test["Date"].iloc[0].date(), "Test_End": test["Date"].iloc[-1].date(),
        "Best_Parameters": json.dumps(searches[winner].best_params_, sort_keys=True),
    }
    return validation_results, test_result, tuning_rows, periods, metadata


def main():
    (BASE_DIR / "models").mkdir(exist_ok=True)
    processed = BASE_DIR / "data" / "processed"
    results, tests, tuning, periods, metadata = [], [], [], [], []
    for stock in STOCKS:
        stock_results, test, stock_tuning, stock_periods, info = train_stock(stock)
        results.extend(stock_results)
        tests.append(test)
        tuning.extend(stock_tuning)
        periods.extend(stock_periods)
        metadata.append(info)
    pd.DataFrame(results).to_csv(processed / "model_results.csv", index=False)
    pd.DataFrame(tests).to_csv(processed / "test_results.csv", index=False)
    pd.DataFrame(tuning).to_csv(processed / "tuning_results.csv", index=False)
    pd.DataFrame(periods).to_csv(processed / "split_periods.csv", index=False)
    pd.DataFrame(metadata).to_csv(BASE_DIR / "models" / "model_metadata.csv", index=False)
    print("\nFinal test results (selection was based only on validation):", flush=True)
    print(pd.DataFrame(tests).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
