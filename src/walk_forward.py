"""Compare frozen, expanding, and calendar rolling histories with causal refitting."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from model_features import metadata_features, metadata_threshold, prepare_features, up_probability
from train_models import HORIZON, STOCKS, classification_metrics, search_spaces, split_periods

BASE_DIR = Path(__file__).resolve().parents[1]
BLOCK_ROWS = 126
MIN_TRAIN_ROWS = 126
POLICIES = {"frozen": None, "expanding": None, "rolling_3y": 3, "rolling_5y": 5}


def eligible_history(data, origin, policy):
    """Only labels fully resolved before the forecast origin may be fitted."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    origin = pd.Timestamp(origin)
    history = data.loc[(data["Date"] < origin) & (data["Target_Date"] < origin)].copy()
    years = POLICIES[policy]
    if years is not None:
        history = history.loc[history["Date"] >= origin - pd.DateOffset(years=years)]
    if len(history) < MIN_TRAIN_ROWS or history["Target"].nunique() != 2:
        raise ValueError(f"Insufficient two-class history before {origin.date()}.")
    if not (history["Target_Date"] < origin).all():
        raise AssertionError("A training target crosses the prediction boundary.")
    return history


def estimator_from_metadata(info):
    estimator = clone(search_spaces()[info["Model"]][0])
    estimator.set_params(**json.loads(info["Best_Parameters"]))
    return estimator


def fit_model(estimator, history, features):
    model = clone(estimator)
    model.fit(history[features], history["Target"])
    return model


def run_policy(data, evaluation, policy, estimator, features, stock, period,
               threshold=0.5, block_rows=BLOCK_ROWS):
    if block_rows < 1 or evaluation.empty:
        raise ValueError("Evaluation rows and a positive block size are required.")
    evaluation = evaluation.sort_values("Date")
    if evaluation["Date"].duplicated().any():
        raise ValueError("Evaluation dates must be unique.")
    predictions, blocks = [], []
    model = history = None
    for number, start in enumerate(range(0, len(evaluation), block_rows), 1):
        block = evaluation.iloc[start:start + block_rows]
        origin = block["Date"].iloc[0]
        if model is None or policy != "frozen":
            history = eligible_history(data, origin, policy)
            model = fit_model(estimator, history, features)
        probability = up_probability(model, block[features])
        majority = int(history["Target"].mode().iloc[0])
        row = {
            "Stock": stock, "Period": period, "Policy": policy, "Block": number,
            "Train_Rows": len(history), "Train_Start": history["Date"].iloc[0].date(),
            "Train_End": history["Date"].iloc[-1].date(),
            "Train_Label_End": history["Target_Date"].max().date(),
            "Evaluation_Start": origin.date(), "Evaluation_End": block["Date"].iloc[-1].date(),
            "Evaluation_Rows": len(block),
            **classification_metrics(block["Target"], probability, threshold),
            "Majority_Baseline_Accuracy": float((block["Target"] == majority).mean()),
        }
        blocks.append(row)
        for offset, (_, item) in enumerate(block.iterrows()):
            predictions.append({
                "Stock": stock, "Period": period, "Policy": policy, "Block": number,
                "Date": item["Date"].date(), "Close": item["Close"],
                "Future_Close": item["Future_Close"], "Target": int(item["Target"]),
                "UP_Probability": float(probability[offset]),
                "Decision_Threshold": threshold,
                "Predicted": int(probability[offset] >= threshold),
                "Majority_Predicted": majority,
                "Train_End": row["Train_End"], "Train_Label_End": row["Train_Label_End"],
            })
    scores = pd.DataFrame(predictions)
    summary = {
        "Stock": stock, "Period": period, "Policy": policy,
        "Rows": len(scores), "Blocks": len(blocks),
        **classification_metrics(scores["Target"], scores["UP_Probability"], threshold),
        "Worst_Block_Accuracy": min(row["Accuracy"] for row in blocks),
        "Block_Accuracy_Std": float(np.std([row["Accuracy"] for row in blocks])),
        "Majority_Baseline_Accuracy": float((scores["Target"] == scores["Majority_Predicted"]).mean()),
        "Decision_Threshold": threshold,
    }
    return summary, predictions, blocks


def select_policy(rows):
    if not rows or any(row["Period"] != "Validation" for row in rows):
        raise ValueError("Policies must be selected using validation results only.")
    return max(rows, key=lambda row: (
        row["Accuracy"], row["Balanced_Accuracy"], -row["Block_Accuracy_Std"], row["ROC-AUC"],
    ))


def load_stock(stock):
    data = pd.read_csv(BASE_DIR / "data" / "processed" / f"{stock}_ML.csv", parse_dates=["Date"])
    raw = pd.read_csv(BASE_DIR / "data" / "raw" / f"{stock}.csv", parse_dates=["Date"])
    raw = raw.sort_values("Date").reset_index(drop=True)
    if raw["Date"].isna().any() or raw["Date"].duplicated().any():
        raise ValueError(f"{stock}: raw dates must be valid and unique.")
    outcomes = raw[["Date"]].copy()
    outcomes["Target_Date"] = raw["Date"].shift(-HORIZON)
    outcomes["Expected_Future"] = raw["Close"].shift(-HORIZON)
    data = prepare_features(data).merge(outcomes, on="Date", validate="one_to_one")
    if data["Target_Date"].isna().any() or not np.allclose(data["Future_Close"], data["Expected_Future"]):
        raise ValueError(f"{stock}: five-day outcomes do not match raw history.")
    if not (data["Target"] == (data["Future_Close"] > data["Close"]).astype(int)).all():
        raise ValueError(f"{stock}: inconsistent direction targets.")
    return split_periods(data)


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    metadata = pd.read_csv(BASE_DIR / "models" / "model_metadata.csv").set_index("Stock")
    active_paths = list((BASE_DIR / "models").glob("*.joblib")) + [
        BASE_DIR / "models" / "model_metadata.csv",
        BASE_DIR / "data" / "processed" / "predictions.csv",
        BASE_DIR / "data" / "processed" / "verification_results.csv",
        BASE_DIR / "data" / "processed" / "test_results.csv",
    ]
    active_hashes = {str(path.relative_to(BASE_DIR)): file_hash(path) for path in active_paths}
    output = BASE_DIR / "models" / "experiments" / datetime.now().strftime("walk_forward_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    for source in ["walk_forward.py", "model_features.py", "train_models.py"]:
        shutil.copy2(BASE_DIR / "src" / source, output / source)
    metadata.to_csv(output / "reference_metadata.csv")
    protocol = {
        "Block_Rows": BLOCK_ROWS, "Minimum_Training_Rows": MIN_TRAIN_ROWS,
        "Horizon_Trading_Rows": HORIZON, "Policies": POLICIES,
        "Selection": "Validation accuracy; balanced accuracy, stability, then ROC-AUC break ties.",
        "Features_Parameters_Thresholds": "Fixed to active metadata; no retuning in this experiment.",
        "Outcome_Availability": "Raw target date must be strictly before each block's prediction origin.",
        "Test_Use": "Selected policy plus predeclared frozen and expanding controls; no selection from test.",
        "Deployment": "Experiment only; active artifacts are not replaced.",
        "Caveat": "Historical test was examined previously. Results are retrospective. No new data downloaded.",
        "Active_SHA256": active_hashes,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("Experiment directory:", output, flush=True)
    validation_rows, validation_predictions, validation_blocks = [], [], []
    choices, stocks = {}, {}
    # Freeze every stock's choice before running any outer-test evaluation.
    for stock in STOCKS:
        info = metadata.loc[stock]
        features = metadata_features(info)
        threshold = metadata_threshold(info)
        data, train, validation, test = load_stock(stock)
        if not np.isfinite(data[features].to_numpy(dtype=float)).all():
            raise ValueError(f"{stock}: non-finite selected features.")
        estimator = estimator_from_metadata(info)
        stocks[stock] = (data, validation, test, estimator, features, threshold)
        results = []
        for policy in POLICIES:
            row, predictions, blocks = run_policy(
                data, validation, policy, estimator, features, stock, "Validation", threshold)
            results.append(row)
            validation_rows.append(row)
            validation_predictions.extend(predictions)
            validation_blocks.extend(blocks)
            print(f"{stock} validation {policy}: accuracy {row['Accuracy']:.4f}; "
                  f"balanced {row['Balanced_Accuracy']:.4f}; worst block {row['Worst_Block_Accuracy']:.4f}", flush=True)
        choices[stock] = select_policy(results)["Policy"]
        print(f"{stock}: policy selected on validation = {choices[stock]}", flush=True)
        write_csv(output / "validation_results.csv", validation_rows)
        write_csv(output / "validation_predictions.csv", validation_predictions)
        write_csv(output / "validation_blocks.csv", validation_blocks)
    selection = {"Choices": choices, "Selection_Period": "Validation",
                 "Selection_Complete_Before_Test": True}
    (output / "selection_manifest.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    tests, test_predictions, test_blocks, comparisons = [], [], [], []
    for stock in STOCKS:
        data, validation, test, estimator, features, threshold = stocks[stock]
        policy_scores = {}
        for policy in dict.fromkeys(["frozen", "expanding", choices[stock]]):
            row, predictions, blocks = run_policy(
                data, test, policy, estimator, features, stock, "Test", threshold)
            policy_scores[policy] = row
            tests.append(row)
            test_predictions.extend(predictions)
            test_blocks.extend(blocks)
            print(f"{stock} test {policy}: accuracy {row['Accuracy']:.4f}; "
                  f"balanced {row['Balanced_Accuracy']:.4f}", flush=True)
        info = metadata.loc[stock]
        active = joblib.load(BASE_DIR / info["Model_File"])
        scaler = joblib.load(BASE_DIR / info["Scaler_File"])
        inputs = test[features]
        if info["Model"] == "Logistic Regression":
            inputs = scaler.transform(inputs)
        active_scores = classification_metrics(test["Target"], up_probability(active, inputs), threshold)
        # Frozen refit is a reproducibility control against the existing saved model.
        if not np.isclose(active_scores["Accuracy"], policy_scores["frozen"]["Accuracy"]):
            raise AssertionError(f"{stock}: frozen control failed to reproduce active accuracy.")
        chosen = policy_scores[choices[stock]]
        comparisons.append({
            "Stock": stock, "Model": info["Model"], "Selected_Policy": choices[stock],
            "Static_Active_Accuracy": active_scores["Accuracy"],
            "Expanding_Accuracy": policy_scores["expanding"]["Accuracy"],
            "Selected_Accuracy": chosen["Accuracy"],
            "Accuracy_Change_Percentage_Points": 100 * (chosen["Accuracy"] - active_scores["Accuracy"]),
            "Selected_Balanced_Accuracy": chosen["Balanced_Accuracy"],
            "Selected_ROC_AUC": chosen["ROC-AUC"],
            "Selected_UP_Rate": chosen["Predicted_UP_Rate"],
            "Selected_Majority_Baseline_Accuracy": chosen["Majority_Baseline_Accuracy"],
        })
        write_csv(output / "test_results.csv", tests)
        write_csv(output / "test_predictions.csv", test_predictions)
        write_csv(output / "test_blocks.csv", test_blocks)
        write_csv(output / "accuracy_comparison.csv", comparisons)

    # Produce separate latest-date model snapshots for the frozen validation choices.
    # Backtest scores above belong to the sequence of historical models, not these refits.
    snapshots, forecasts = [], []
    for stock in STOCKS:
        data, validation, test, estimator, features, threshold = stocks[stock]
        latest = prepare_features(pd.read_csv(
            BASE_DIR / "data" / "processed" / f"{stock}_prediction.csv", parse_dates=["Date"]))
        if len(latest) != 1 or not np.isfinite(latest[features].to_numpy(dtype=float)).all():
            raise ValueError(f"{stock}: invalid latest prediction row.")
        origin = latest["Date"].iloc[0]
        history = eligible_history(data, origin, choices[stock])
        model = fit_model(estimator, history, features)
        probability = float(up_probability(model, latest[features])[0])
        model_path = output / f"{stock}_pipeline.joblib"
        # Save the complete pipeline, avoiding a separate-scaler feature-order mismatch.
        joblib.dump(model, model_path)
        snapshots.append({
            "Stock": stock, "Model": metadata.loc[stock, "Model"], "Policy": choices[stock],
            "Features": ",".join(features), "Decision_Threshold": threshold,
            "Model_File": str(model_path.relative_to(BASE_DIR)).replace("\\", "/"),
            "Forecast_Date": origin.date(), "Train_Start": history["Date"].iloc[0].date(),
            "Train_End": history["Date"].iloc[-1].date(), "Train_Label_End": history["Target_Date"].max().date(),
            "Train_Rows": len(history), "Evaluation": "Walk-forward backtest uses earlier refits; this latest snapshot is not independently tested.",
        })
        forecasts.append({
            "Stock": stock, "Date": origin.date(), "Close": latest["Close"].iloc[0],
            "Policy": choices[stock], "Prediction": "UP" if probability >= threshold else "DOWN",
            "UP_Probability": probability, "DOWN_Probability": 1 - probability,
            "Decision_Threshold": threshold,
        })
    write_csv(output / "snapshot_metadata.csv", snapshots)
    write_csv(output / "latest_predictions.csv", forecasts)
    for relative, expected in active_hashes.items():
        if file_hash(BASE_DIR / relative) != expected:
            raise AssertionError(f"An active artifact was unexpectedly modified: {relative}")
    comparison = pd.DataFrame(comparisons)
    summary = {
        "Static_Mean_Accuracy": float(comparison["Static_Active_Accuracy"].mean()),
        "Expanding_Mean_Accuracy": float(comparison["Expanding_Accuracy"].mean()),
        "Selected_Mean_Accuracy": float(comparison["Selected_Accuracy"].mean()),
        "Accuracy_Change_Percentage_Points": float(comparison["Accuracy_Change_Percentage_Points"].mean()),
        "Improved_Stocks": int((comparison["Accuracy_Change_Percentage_Points"] > 0).sum()),
        "Choices": choices, "Active_Artifacts_Unchanged": True,
        "Deployment": "Not promoted; periodic retraining policy requires matching inference/verification support.",
        "Data_End": str(pd.read_csv(BASE_DIR / "data" / "raw" / "RELIANCE.csv")["Date"].max()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nHistorical comparison:", flush=True)
    print(comparison.to_string(index=False), flush=True)
    print("Summary:", summary, flush=True)


if __name__ == "__main__":
    main()
