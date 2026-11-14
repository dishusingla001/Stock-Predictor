"""Tune on training folds, select on validation, evaluate the winner on test."""
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
from model_features import FEATURE_SETS, prepare_features, up_probability
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


def classification_metrics(target, probability, threshold=0.5):
    predicted = (np.asarray(probability) >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(target, predicted),
        "Balanced_Accuracy": balanced_accuracy_score(target, predicted),
        "Precision": precision_score(target, predicted, zero_division=0),
        "Recall": recall_score(target, predicted, zero_division=0),
        "F1": f1_score(target, predicted, zero_division=0),
        "ROC-AUC": roc_auc_score(target, probability),
        "Predicted_UP_Rate": float(predicted.mean()),
        "TN": int(confusion_matrix(target, predicted, labels=[0, 1])[0, 0]),
        "FP": int(confusion_matrix(target, predicted, labels=[0, 1])[0, 1]),
        "FN": int(confusion_matrix(target, predicted, labels=[0, 1])[1, 0]),
        "TP": int(confusion_matrix(target, predicted, labels=[0, 1])[1, 1]),
    }


def tune_threshold(target, probability):
    # OOF training predictions only. Tie-breaks favor balanced accuracy and 0.5.
    candidates = []
    for threshold in np.linspace(0.35, 0.65, 31):
        predicted = (np.asarray(probability) >= threshold).astype(int)
        candidates.append((accuracy_score(target, predicted),
                           balanced_accuracy_score(target, predicted),
                           -abs(threshold - 0.5), float(threshold)))
    return max(candidates)[-1]


def select_winner(validation_results):
    """Only validation rows are accepted; no test metric participates."""
    if not validation_results or any(row["Period"] != "Validation" for row in validation_results):
        raise ValueError("Selection requires validation results only.")
    return max(validation_results, key=lambda row: (
        row["Accuracy"], row["Balanced_Accuracy"], row["ROC-AUC"],
        -abs(row["Decision_Threshold"] - 0.5),
    ))


def train_stock(stock):
    data = prepare_features(pd.read_csv(
        BASE_DIR / "data" / "processed" / f"{stock}_ML.csv", parse_dates=["Date"]))
    required = FEATURE_SETS["combined"] + ["Close", "Future_Close", "Target"]
    if not np.isfinite(data[required].to_numpy(dtype=float)).all():
        raise ValueError(f"{stock}: missing or non-finite training values.")
    if not (data["Target"] == (data["Future_Close"] > data["Close"]).astype(int)).all():
        raise ValueError(f"{stock}: inconsistent targets.")
    data, train, validation, test = split_periods(data)
    periods = []
    print(f"\n========== {stock} ==========", flush=True)
    for name, part in [("Train", train), ("Validation", validation), ("Test", test)]:
        periods.append({"Stock": stock, "Period": name, "Rows": len(part),
                        "Start": part["Date"].iloc[0].date(), "End": part["Date"].iloc[-1].date()})
    cv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=HORIZON)
    folds = list(cv.split(train))
    for fit_idx, score_idx in folds:
        if any(train.iloc[idx]["Target"].nunique() != 2 for idx in (fit_idx, score_idx)):
            raise ValueError(f"{stock}: a CV fold lacks both classes.")

    results, searches, tuning_rows = [], {}, []
    for feature_set, features in FEATURE_SETS.items():
        for name, (estimator, parameters) in search_spaces().items():
            iterations = min(SEARCH_ITERATIONS, int(np.prod([len(v) for v in parameters.values()])))
            print(f"{stock}: {feature_set} / {name}: {iterations} candidates x {CV_SPLITS} folds", flush=True)
            search = RandomizedSearchCV(
                estimator, parameters, n_iter=iterations,
                scoring={"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "roc_auc": "roc_auc"},
                refit="accuracy", cv=folds, random_state=42, n_jobs=2, error_score="raise",
            )
            search.fit(train[features], train["Target"])
            searches[(feature_set, name)] = search
            oof_target, oof_probability = [], []
            for fit_idx, score_idx in folds:
                fold_model = clone(search.best_estimator_)
                fold_model.fit(train.iloc[fit_idx][features], train.iloc[fit_idx]["Target"])
                oof_target.extend(train.iloc[score_idx]["Target"].tolist())
                oof_probability.extend(up_probability(fold_model, train.iloc[score_idx][features]).tolist())
            tuned = tune_threshold(oof_target, oof_probability)
            probability = up_probability(search.best_estimator_, validation[features])
            # Compare default and OOF-tuned threshold on the same validation period.
            for threshold in sorted({0.5, tuned}):
                results.append({
                    "Stock": stock, "Model": name, "Feature_Set": feature_set,
                    "Features": ",".join(features), "Period": "Validation",
                    "Decision_Threshold": threshold,
                    "Threshold_Source": "Default" if threshold == 0.5 else "Training OOF",
                    **classification_metrics(validation["Target"], probability, threshold),
                    "CV_Accuracy": search.best_score_,
                    "CV_ROC_AUC": search.cv_results_["mean_test_roc_auc"][search.best_index_],
                    "Best_Parameters": json.dumps(search.best_params_, sort_keys=True),
                })
            for i, params in enumerate(search.cv_results_["params"]):
                tuning_rows.append({
                    "Stock": stock, "Model": name, "Feature_Set": feature_set,
                    "Parameters": json.dumps(params, sort_keys=True),
                    "Mean_CV_Accuracy": search.cv_results_["mean_test_accuracy"][i],
                    "Mean_CV_Balanced_Accuracy": search.cv_results_["mean_test_balanced_accuracy"][i],
                    "Mean_CV_ROC_AUC": search.cv_results_["mean_test_roc_auc"][i],
                    "Std_CV_Accuracy": search.cv_results_["std_test_accuracy"][i],
                    "Rank": search.cv_results_["rank_test_accuracy"][i],
                })
            print(f"  CV accuracy {search.best_score_:.4f}; OOF threshold {tuned:.2f}; "
                  f"validation accuracy {results[-1]['Accuracy']:.4f}", flush=True)

    winner = select_winner(results)
    features = winner["Features"].split(",")
    threshold = winner["Decision_Threshold"]
    chosen = searches[(winner["Feature_Set"], winner["Model"])].best_estimator_
    # Importance uses the train-only model on validation, never the refit model on test.
    def accuracy_at_threshold(estimator, inputs, target):
        return accuracy_score(target, up_probability(estimator, inputs) >= threshold)
    importance = permutation_importance(
        chosen, validation[features], validation["Target"],
        scoring=accuracy_at_threshold, n_repeats=10, random_state=42, n_jobs=2,
    )
    importance_rows = [
        {"Stock": stock, "Model": winner["Model"], "Feature_Set": winner["Feature_Set"],
         "Period": "Validation", "Feature": name,
         "Accuracy_Decrease_Mean": importance.importances_mean[i],
         "Accuracy_Decrease_Std": importance.importances_std[i]}
        for i, name in enumerate(features)
    ]
    refit_data = data.iloc[:test.index[0] - HORIZON]
    final_model = clone(chosen)
    final_model.fit(refit_data[features], refit_data["Target"])
    test_result = {
        "Stock": stock, "Model": winner["Model"], "Feature_Set": winner["Feature_Set"],
        "Period": "Test", "Decision_Threshold": threshold,
        **classification_metrics(test["Target"], up_probability(final_model, test[features]), threshold),
    }
    majority = int(refit_data["Target"].mode().iloc[0])
    test_result["Training_Majority_Baseline_Accuracy"] = accuracy_score(
        test["Target"], np.full(len(test), majority))
    test_result["Training_Majority_Baseline_Class"] = majority
    print(f"Selected using validation: {winner['Feature_Set']} / {winner['Model']} / {threshold:.2f}; "
          f"test accuracy {test_result['Accuracy']:.4f}", flush=True)
    if winner["Model"] == "Logistic Regression":
        scaler = final_model.named_steps["scaler"]
        saved_model = final_model.named_steps["model"]
    else:
        scaler = StandardScaler().fit(refit_data[features])
        saved_model = final_model
    metadata = {
        "Stock": stock, "Model": winner["Model"], "Features": ",".join(features),
        "Feature_Set": winner["Feature_Set"], "Decision_Threshold": threshold,
        "Threshold_Source": winner["Threshold_Source"],
        "Horizon_Days": HORIZON, "Model_File": f"models/{stock}_model.joblib",
        "Scaler_File": f"models/{stock}_scaler.joblib",
        "Selection_Metric": "Validation Accuracy; balanced accuracy and ROC-AUC break ties",
        "Gap_Rows": HORIZON, "Refit_End": refit_data["Date"].iloc[-1].date(),
        "Validation_Start": validation["Date"].iloc[0].date(),
        "Validation_End": validation["Date"].iloc[-1].date(),
        "Test_Start": test["Date"].iloc[0].date(), "Test_End": test["Date"].iloc[-1].date(),
        "Best_Parameters": winner["Best_Parameters"],
    }
    return results, test_result, tuning_rows, periods, metadata, importance_rows, saved_model, scaler


def preserve_candidate_run(backup):
    """Archive the full candidate release; reject a mean-accuracy regression as a whole."""
    backup = Path(backup)
    # Refresh candidate outputs before archiving so reports match candidate models.
    for script in ("predict.py", "verify_predictions.py"):
        subprocess.run([sys.executable, "-B", str(BASE_DIR / "src" / script)], check=True)
    processed = BASE_DIR / "data" / "processed"
    models_dir = BASE_DIR / "models"
    candidate = models_dir / "experiments" / datetime.now().strftime("feature_candidate_%Y%m%d_%H%M%S")
    candidate.mkdir(parents=True, exist_ok=False)
    report_names = {
        "model_results.csv", "test_results.csv", "tuning_results.csv", "split_periods.csv",
        "predictions.csv", "verification_results.csv", "feature_importance.csv", "accuracy_comparison.csv",
    }
    for source in list(models_dir.glob("*.joblib")) + list(models_dir.glob("*.csv")):
        shutil.copy2(source, candidate / source.name)
    for name in report_names:
        source = processed / name
        if source.exists():
            shutil.copy2(source, candidate / name)
    for name in ("train_models.py", "model_features.py", "predict.py", "verify_predictions.py"):
        shutil.copy2(BASE_DIR / "src" / name, candidate / name)
    candidate_metadata = pd.read_csv(candidate / "model_metadata.csv")
    for column in ("Model_File", "Scaler_File"):
        candidate_metadata[column] = candidate_metadata[column].map(
            lambda path: str((candidate / Path(path).name).relative_to(BASE_DIR)).replace("\\", "/"))
    candidate_metadata.to_csv(candidate / "model_metadata.csv", index=False)
    accepted = True
    old_mean = new_mean = None
    if (backup / "test_results.csv").exists():
        old = pd.read_csv(backup / "test_results.csv").set_index("Stock")
        new = pd.read_csv(processed / "test_results.csv").set_index("Stock")
        if set(old.index) != set(new.index):
            raise ValueError("Cannot compare releases with different stocks.")
        old_mean = float(old["Accuracy"].mean())
        new_mean = float(new["Accuracy"].mean())
        accepted = new_mean >= old_mean
    if not accepted:
        for source in list(backup.glob("*.joblib")):
            shutil.copy2(source, models_dir / source.name)
        shutil.copy2(backup / "model_metadata.csv", models_dir / "model_metadata.csv")
        for name in report_names:
            if (backup / name).exists():
                shutil.copy2(backup / name, processed / name)
        # Candidate-only reports remain in the archive, avoiding mixed active-model reports.
        for name in ("feature_importance.csv", "accuracy_comparison.csv"):
            destination = processed / name
            if not (backup / name).exists() and destination.exists():
                destination.unlink()
    release = {"Accepted": bool(accepted), "Previous_Mean_Accuracy": old_mean,
               "Candidate_Mean_Accuracy": new_mean,
               "Rule": "Reject the entire candidate set if mean historical test accuracy declines; no per-stock cherry-picking.",
               "Candidate_Directory": str(candidate), "Previous_Snapshot": str(backup)}
    (candidate / "release_decision.json").write_text(json.dumps(release, indent=2), encoding="utf-8")
    print("Candidate artifacts and reports:", candidate, flush=True)
    print("Release decision:", release, flush=True)
    return release


def main():
    processed = BASE_DIR / "data" / "processed"
    models_dir = BASE_DIR / "models"
    # Retain a dated snapshot before any output is replaced, including on repeat runs.
    backup = models_dir / "experiments" / datetime.now().strftime("before_feature_run_%Y%m%d_%H%M%S")
    backup.mkdir(parents=True, exist_ok=False)
    for source in list(models_dir.glob("*.joblib")) + list(models_dir.glob("*.csv")):
        shutil.copy2(source, backup / source.name)
    for source in processed.glob("*.csv"):
        if source.name in {"model_results.csv", "test_results.csv", "tuning_results.csv",
                           "split_periods.csv", "predictions.csv", "verification_results.csv",
                           "feature_importance.csv", "accuracy_comparison.csv"}:
            shutil.copy2(source, backup / source.name)
    print(f"Previous artifacts preserved at {backup}", flush=True)
    previous = pd.read_csv(processed / "test_results.csv") if (processed / "test_results.csv").exists() else None
    results, tests, tuning, periods, metadata, importance, artifacts = [], [], [], [], [], [], []
    for stock in STOCKS:
        rows, test, candidates, splits, info, ranks, model, scaler = train_stock(stock)
        results.extend(rows)
        tests.append(test)
        tuning.extend(candidates)
        periods.extend(splits)
        metadata.append(info)
        importance.extend(ranks)
        artifacts.append((info, model, scaler))
    # Finish all experiments before replacing any active model.
    for info, model, scaler in artifacts:
        joblib.dump(model, BASE_DIR / info["Model_File"])
        joblib.dump(scaler, BASE_DIR / info["Scaler_File"])
    for filename, rows in [
        ("model_results.csv", results), ("test_results.csv", tests),
        ("tuning_results.csv", tuning), ("split_periods.csv", periods),
        ("feature_importance.csv", importance),
    ]:
        pd.DataFrame(rows).to_csv(processed / filename, index=False)
    pd.DataFrame(metadata).to_csv(models_dir / "model_metadata.csv", index=False)
    if previous is not None:
        comparison = pd.DataFrame(tests).merge(
            previous[["Stock", "Model", "Accuracy", "ROC-AUC"]].rename(columns={
                "Model": "Previous_Model", "Accuracy": "Previous_Accuracy", "ROC-AUC": "Previous_ROC_AUC"}),
            on="Stock", validate="one_to_one")
        comparison["Accuracy_Change_Percentage_Points"] = 100 * (comparison["Accuracy"] - comparison["Previous_Accuracy"])
        comparison.to_csv(processed / "accuracy_comparison.csv", index=False)
        print(comparison[["Stock", "Previous_Accuracy", "Accuracy", "Accuracy_Change_Percentage_Points"]].to_string(index=False), flush=True)
    print("\nFinal test results; test metrics were not used to choose models:", flush=True)
    print(pd.DataFrame(tests).to_string(index=False), flush=True)
    preserve_candidate_run(backup)


if __name__ == "__main__":
    main()
