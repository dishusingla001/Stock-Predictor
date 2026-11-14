"""Select contextual features before testing on historical and new 2026 data."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd

from collect_market_context import TICKERS
from market_features import SECTORS, feature_variants, merge_context
from model_features import metadata_features, metadata_threshold, up_probability
from train_models import STOCKS, classification_metrics
from walk_forward import estimator_from_metadata, fit_model

BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_START = pd.Timestamp("2015-03-16")
VALIDATION_START = pd.Timestamp("2021-09-03")
VALIDATION_END = pd.Timestamp("2023-10-18")
HISTORICAL_START = pd.Timestamp("2023-10-27")
HISTORICAL_END = pd.Timestamp("2025-12-23")
FRESH_START = pd.Timestamp("2026-01-01")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_history(data, origin):
    history = data.loc[(data["Date"] < origin) & (data["Target_Date"] < origin)].copy()
    if len(history) < 126 or history["Target"].nunique() != 2:
        raise ValueError("Insufficient two-class training history.")
    if not (history["Target_Date"] < origin).all():
        raise AssertionError("An outcome crosses the fitting boundary.")
    return history


def choose_features(rows):
    if not rows or any(row["Period"] != "Validation" for row in rows):
        raise ValueError("Feature sets must be selected from validation results only.")
    # Prefer fewer features when all predictive metrics tie.
    return max(rows, key=lambda row: (row["Accuracy"], row["Balanced_Accuracy"],
                                      row["ROC-AUC"], -row["Feature_Count"]))["Feature_Set"]


def period_slice(data, start, end=None):
    mask = data["Date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= data["Date"] <= pd.Timestamp(end)
    return data.loc[mask].copy()


def evaluate_model(model, history, evaluation, features, stock, period, variant, threshold):
    probability = up_probability(model, evaluation[features])
    majority = int(history["Target"].mode().iloc[0])
    metrics = {
        "Stock": stock, "Period": period, "Feature_Set": variant, "Feature_Count": len(features),
        "Rows": len(evaluation), "Start": str(evaluation["Date"].min().date()),
        "End": str(evaluation["Date"].max().date()),
        "Train_Start": str(history["Date"].min().date()), "Train_End": str(history["Date"].max().date()),
        "Train_Label_End": str(history["Target_Date"].max().date()),
        "Decision_Threshold": threshold, **classification_metrics(evaluation["Target"], probability, threshold),
        "Majority_Baseline_Accuracy": float((evaluation["Target"] == majority).mean()),
    }
    rows = evaluation[["Date", "Close", "Future_Close", "Target", "Target_Date"]].copy()
    rows["Stock"], rows["Period"], rows["Feature_Set"] = stock, period, variant
    rows["UP_Probability"], rows["Predicted"] = probability, (probability >= threshold).astype(int)
    rows["Decision_Threshold"], rows["Majority_Predicted"] = threshold, majority
    rows["Train_End"], rows["Train_Label_End"] = metrics["Train_End"], metrics["Train_Label_End"]
    return metrics, rows.to_dict("records")


def write(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def main(as_of):
    dataset = BASE_DIR / "data" / f"market_context_{as_of.replace('-', '')}"
    for name in TICKERS:
        if not (dataset / f"{name}.csv").exists():
            raise RuntimeError(f"Missing {name}; run collect_market_context.py first.")
    manifest = json.loads((dataset / "download_manifest.json").read_text())
    if manifest["As_Of_Exclusive"] != as_of or len(manifest["Records"]) != len(TICKERS):
        raise ValueError("Incomplete or mismatched download manifest.")
    for record in manifest["Records"]:
        if sha(dataset / f"{record['Name']}.csv") != record["SHA256"]:
            raise ValueError("A downloaded input no longer matches its recorded hash.")
    protected = list((BASE_DIR / "models").glob("*.joblib")) + list((BASE_DIR / "models").glob("*.csv"))
    protected += list((BASE_DIR / "data" / "raw").glob("*.csv")) + list((BASE_DIR / "data" / "processed").glob("*.csv"))
    hashes = {str(path.relative_to(BASE_DIR)): sha(path) for path in protected}
    output = BASE_DIR / "models" / "experiments" / datetime.now().strftime("market_context_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    for source in ["market_context_experiment.py", "market_features.py", "collect_market_context.py",
                   "model_features.py", "train_models.py", "walk_forward.py"]:
        shutil.copy2(BASE_DIR / "src" / source, output / source)
    shutil.copy2(dataset / "download_manifest.json", output / "download_manifest.json")
    protocol = {
        "As_Of_Exclusive": as_of, "Data_Directory": str(dataset), "Sector_Map": SECTORS,
        "Variants": ["stock_only", "with_market", "with_market_sector"],
        "Validation": [str(VALIDATION_START.date()), str(VALIDATION_END.date())],
        "Historical_Test": [str(HISTORICAL_START.date()), str(HISTORICAL_END.date())],
        "New_2026_Test_Start": str(FRESH_START.date()),
        "Selection": "Validation accuracy, balanced accuracy, ROC-AUC, then fewer features.",
        "Controls": "Fixed active model family/parameters/threshold; all variants use matched dates and identical fitting history.",
        "Target": "Close five stock trading sessions later, computed before filtering missing index dates.",
        "Information_Timing": "After-close predictions. Index data joined on exact dates; no forward or backward filling.",
        "Fresh_Test": "One frozen model per variant fitted only to targets resolved before 2026-01-01. No 2026 refitting or selection.",
        "Coverage": "Missing context dates are reported and excluded for ALL variants; require at least 95% coverage in each evaluation period.",
        "Test_Ablation": "All three predeclared feature sets are scored for diagnosis. Test scores never switch the validation choice.",
        "Deployment": "Isolated experiment; active datasets, reports, and models are preserved.",
        "Input_SHA256": {record["Name"]: record["SHA256"] for record in manifest["Records"]},
        "Active_SHA256": hashes,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("Experiment directory:", output, flush=True)
    metadata = pd.read_csv(BASE_DIR / "models" / "model_metadata.csv").set_index("Stock")
    raw = {name: pd.read_csv(dataset / f"{name}.csv", parse_dates=["Date"]) for name in TICKERS}
    stocks, choices = {}, {}
    validation_rows, validation_predictions, gaps, coverage, revisions = [], [], [], [], []
    for stock in STOCKS:
        info = metadata.loc[stock]
        variants = feature_variants(metadata_features(info))
        full = merge_context(raw[stock], raw["NIFTY50"], raw[SECTORS[stock]])
        finite = np.isfinite(full[variants["with_market_sector"]].to_numpy(dtype=float)).all(axis=1)
        labelled = full["Target"].notna() & (full["Date"] >= TRAIN_START)
        for _, row in full.loc[labelled & ~finite].iterrows():
            gaps.append({"Stock": stock, "Date": row["Date"], "Missing_Market_Date": pd.isna(row["Market_Source_Date"]),
                         "Missing_Sector_Date": pd.isna(row["Sector_Source_Date"]),
                         "Missing_Features": ",".join(name for name in variants["with_market_sector"] if pd.isna(row[name]))})
        expected = full.loc[labelled].copy()
        data = full.loc[labelled & finite].copy()
        for period, start, end in [("Validation", VALIDATION_START, VALIDATION_END),
                                   ("Historical_Test", HISTORICAL_START, HISTORICAL_END),
                                   ("New_2026_Test", FRESH_START, None)]:
            total = len(period_slice(expected, start, end))
            retained = len(period_slice(data, start, end))
            fraction = retained / total if total else 0
            coverage.append({"Stock": stock, "Period": period, "Expected_Rows": total,
                             "Matched_Rows": retained, "Coverage": fraction})
            if fraction < .95 or retained < 30:
                raise ValueError(f"{stock}: insufficient matched coverage in {period}: {fraction:.2%}")
        old = pd.read_csv(BASE_DIR / "data" / "raw" / f"{stock}.csv", parse_dates=["Date"])
        overlap = old[["Date", "Close"]].merge(raw[stock][["Date", "Close"]], on="Date", suffixes=("_old", "_new"))
        difference = (overlap["Close_old"] - overlap["Close_new"]).abs()
        revisions.append({"Stock": stock, "Overlap_Rows": len(overlap), "Changed_Close_Rows": int((difference > .01).sum()),
                          "Max_Absolute_Close_Difference": float(difference.max())})
        full.to_csv(output / f"{stock}_context_features.csv", index=False)
        estimator = estimator_from_metadata(info)
        threshold = metadata_threshold(info)
        history = training_history(data, VALIDATION_START)
        validation = period_slice(data, VALIDATION_START, VALIDATION_END)
        rows = []
        for variant, features in variants.items():
            model = fit_model(estimator, history, features)
            row, predictions = evaluate_model(model, history, validation, features, stock, "Validation", variant, threshold)
            rows.append(row)
            validation_rows.append(row)
            validation_predictions.extend(predictions)
            print(f"{stock} validation {variant}: accuracy {row['Accuracy']:.4f}; balanced {row['Balanced_Accuracy']:.4f}", flush=True)
        choices[stock] = choose_features(rows)
        stocks[stock] = (data, full, variants, estimator, threshold)
        print(f"{stock}: selected {choices[stock]}", flush=True)
        write(output / "validation_results.csv", validation_rows)
        write(output / "validation_predictions.csv", validation_predictions)
        write(output / "missing_context_dates.csv", gaps)
        write(output / "coverage.csv", coverage)
        write(output / "historical_price_revisions.csv", revisions)
    selection = {"Choices": choices, "Selected_From": "Validation",
                 "Frozen_Before_Any_Test": True, "New_2026_Data_Used_For_Selection": False}
    (output / "selection_manifest.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    results, predictions, comparisons, model_records = [], [], [], []
    for period, origin, end in [("Historical_Test", HISTORICAL_START, HISTORICAL_END),
                                 ("New_2026_Test", FRESH_START, None)]:
        for stock in STOCKS:
            data, full, variants, estimator, threshold = stocks[stock]
            history = training_history(data, origin)
            evaluation = period_slice(data, origin, end)
            scores = {}
            for variant, features in variants.items():
                model = fit_model(estimator, history, features)
                row, forecasts = evaluate_model(model, history, evaluation, features, stock, period, variant, threshold)
                results.append(row)
                predictions.extend(forecasts)
                scores[variant] = row
                path = output / f"{stock}_{period}_{variant}_pipeline.joblib"
                joblib.dump(model, path)
                model_records.append({
                    "Stock": stock, "Period": period, "Feature_Set": variant, "Features": ",".join(features),
                    "Decision_Threshold": threshold, "Model_File": str(path.relative_to(BASE_DIR)).replace("\\", "/"),
                    "Train_End": row["Train_End"], "Train_Label_End": row["Train_Label_End"],
                })
                print(f"{stock} {period} {variant}: accuracy {row['Accuracy']:.4f}; AUC {row['ROC-AUC']:.4f}", flush=True)
            baseline, chosen = scores["stock_only"], scores[choices[stock]]
            comparisons.append({
                "Stock": stock, "Period": period, "Selected_Feature_Set": choices[stock],
                "Rows": len(evaluation), "Start": chosen["Start"], "End": chosen["End"],
                "Stock_Only_Accuracy": baseline["Accuracy"], "Selected_Accuracy": chosen["Accuracy"],
                "Change_Percentage_Points": 100 * (chosen["Accuracy"] - baseline["Accuracy"]),
                "Selected_Balanced_Accuracy": chosen["Balanced_Accuracy"],
                "Selected_ROC_AUC": chosen["ROC-AUC"], "Selected_UP_Rate": chosen["Predicted_UP_Rate"],
                "Majority_Baseline_Accuracy": chosen["Majority_Baseline_Accuracy"],
            })
            write(output / "test_results.csv", results)
            write(output / "test_predictions.csv", predictions)
            write(output / "accuracy_comparison.csv", comparisons)
            write(output / "backtest_model_metadata.csv", model_records)
    # These latest refits happen AFTER holdout scoring; their future outcomes are not known.
    latest_predictions, latest_metadata = [], []
    for stock in STOCKS:
        data, full, variants, estimator, threshold = stocks[stock]
        latest = full.tail(1)
        origin = latest["Date"].iloc[0]
        features = variants[choices[stock]]
        if not np.isfinite(latest[features].to_numpy(dtype=float)).all():
            raise ValueError(f"{stock}: newest stock row lacks context; no older fallback.")
        history = training_history(data, origin)
        model = fit_model(estimator, history, features)
        probability = float(up_probability(model, latest[features])[0])
        path = output / f"{stock}_LATEST_pipeline.joblib"
        joblib.dump(model, path)
        latest_predictions.append({
            "Stock": stock, "Date": origin, "Close": latest["Close"].iloc[0],
            "Feature_Set": choices[stock], "Prediction": "UP" if probability >= threshold else "DOWN",
            "UP_Probability": probability, "DOWN_Probability": 1 - probability, "Decision_Threshold": threshold})
        latest_metadata.append({
            "Stock": stock, "Features": ",".join(features), "Feature_Set": choices[stock],
            "Decision_Threshold": threshold, "Model_File": str(path.relative_to(BASE_DIR)).replace("\\", "/"),
            "Forecast_Date": str(origin.date()), "Train_End": str(history["Date"].max().date()),
            "Train_Label_End": str(history["Target_Date"].max().date()),
            "Evaluation": "Latest refit is not independently evaluated; use the separate backtest models for reported test scores."})
    write(output / "latest_predictions.csv", latest_predictions)
    write(output / "latest_model_metadata.csv", latest_metadata)
    for relative, expected in hashes.items():
        if sha(BASE_DIR / relative) != expected:
            raise AssertionError(f"Active artifact changed: {relative}")
    comparison = pd.DataFrame(comparisons)
    summary = {"Choices": choices, "Active_Artifacts_Unchanged": True,
               "New_Data_End": str(raw["RELIANCE"]["Date"].max().date()), "Results": {}}
    for period, rows in comparison.groupby("Period"):
        summary["Results"][period] = {
            "Stock_Only_Mean_Accuracy": float(rows["Stock_Only_Accuracy"].mean()),
            "Selected_Mean_Accuracy": float(rows["Selected_Accuracy"].mean()),
            "Selected_Mean_Balanced_Accuracy": float(rows["Selected_Balanced_Accuracy"].mean()),
            "Selected_Mean_ROC_AUC": float(rows["Selected_ROC_AUC"].mean()),
            "Change_Percentage_Points": float(rows["Change_Percentage_Points"].mean()),
            "Majority_Baseline_Mean_Accuracy": float(rows["Majority_Baseline_Accuracy"].mean()),
            "Improved_Stocks": int((rows["Change_Percentage_Points"] > 0).sum()),
            "Evaluation_Start": rows["Start"].min(), "Evaluation_End": rows["End"].max(),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nMatched-date comparisons:", flush=True)
    print(comparison.to_string(index=False), flush=True)
    print("Summary:", summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", required=True)
    main(parser.parse_args().as_of)
