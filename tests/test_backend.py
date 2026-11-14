"""HTTP behavior, inference parity, dataset provenance, and read-only service checks."""
from datetime import date
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.service import StockService
from src.market_features import stock_features

CORE = ["RSI", "EMA20", "EMA50", "MACD", "ROC", "ATR", "Volume"]


def hash_files(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for relative in ["models", "data/raw", "data/processed"]:
            (self.root / relative).mkdir(parents=True)
        self.raw = self.make_prices(110)
        self.raw.to_csv(self.root / "data/raw/RELIANCE.csv", index=False)
        frame = stock_features(self.raw)
        training = frame.loc[np.isfinite(frame[CORE]).all(axis=1)].iloc[:35]
        self.scaler = StandardScaler().fit(training[CORE])
        target = (training["Future_Close"] > training["Close"]).astype(int)
        self.model = LogisticRegression(random_state=42).fit(self.scaler.transform(training[CORE]), target)
        joblib.dump(self.model, self.root / "models/RELIANCE_model.joblib")
        joblib.dump(self.scaler, self.root / "models/RELIANCE_scaler.joblib")
        self.training_end = str(training["Date"].iloc[-1].date())
        self.metadata = pd.DataFrame([{
            "Stock": "RELIANCE", "Model": "Logistic Regression", "Features": ",".join(CORE),
            "Decision_Threshold": .6, "Horizon_Days": 5,
            "Model_File": "models/RELIANCE_model.joblib", "Scaler_File": "models/RELIANCE_scaler.joblib",
            "Refit_End": self.training_end, "Test_Start": "2024-04-15", "Test_End": "2024-05-15",
        }])
        self.metadata.to_csv(self.root / "models/model_metadata.csv", index=False)
        pd.DataFrame([{
            "Stock": "RELIANCE", "Model": "Logistic Regression", "Period": "Test",
            "Accuracy": .6, "Precision": .61, "Recall": .62, "F1": .615, "ROC-AUC": .64,
            "Training_Majority_Baseline_Accuracy": .55,
        }]).to_csv(self.root / "data/processed/test_results.csv", index=False)
        pd.DataFrame([{
            "Stock": "RELIANCE", "Period": "Test", "Rows": 23,
            "Start": "2024-04-15", "End": "2024-05-15",
        }]).to_csv(self.root / "data/processed/split_periods.csv", index=False)
        self.settings = Settings(project_root=self.root)

    def make_prices(self, count):
        sequence = np.arange(count)
        close = 100 + sequence * .05 + 4 * np.sin(sequence / 4)
        return pd.DataFrame({"Date": pd.bdate_range("2024-01-01", periods=count),
                             "Open": close, "High": close + 1, "Low": close - 1,
                             "Close": close, "Volume": 1000 + sequence})

    def client(self):
        client = TestClient(create_app(self.settings))
        return self.enterContext(client)

    def test_prediction_matches_saved_model_and_metadata_threshold(self):
        client = self.client()
        response = client.get("/api/v1/stocks/reliance.ns/prediction")
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        features = stock_features(self.raw).tail(1)[CORE]
        probability = self.model.predict_proba(self.scaler.transform(features))[0, 1]
        self.assertAlmostEqual(data["up_probability"], probability)
        self.assertAlmostEqual(data["up_probability"] + data["down_probability"], 1)
        self.assertEqual(data["direction"], "UP" if probability >= .6 else "DOWN")
        self.assertEqual(data["decision_threshold"], .6)
        self.assertEqual(data["horizon_trading_days"], 5)
        self.assertEqual(data["data_date"], str(self.raw["Date"].iloc[-1].date()))
        self.assertEqual(data["model_trained_through"], self.training_end)
        self.assertFalse(data["outcome_verified"])
        self.assertEqual(data["model_scope"], "active")
        self.assertNotIn("accuracy", data)

    def test_health_stock_list_and_staleness(self):
        client = self.client()
        with patch("backend.app.service.today_in_india", return_value=date(2026, 11, 12)):
            health = client.get("/health").json()
            stocks = client.get("/api/v1/stocks").json()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["stocks_loaded"], 1)
        self.assertTrue(health["is_stale"])
        self.assertEqual(stocks["count"], 1)
        self.assertEqual(stocks["stocks"][0]["symbol"], "RELIANCE")
        self.assertEqual(stocks["source"]["kind"], "project_raw")
        self.assertTrue(stocks["stocks"][0]["is_stale"])

    def test_history_filters_are_inclusive_ordered_and_limited(self):
        client = self.client()
        start, end = self.raw["Date"].iloc[10], self.raw["Date"].iloc[20]
        response = client.get("/api/v1/stocks/RELIANCE/history",
                              params={"start": str(start.date()), "end": str(end.date()), "limit": 3})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_matching_rows"], 11)
        self.assertEqual(data["returned_rows"], 3)
        self.assertTrue(data["truncated"])
        self.assertEqual([row["date"] for row in data["prices"]],
                         [str(day.date()) for day in self.raw["Date"].iloc[18:21]])
        empty = client.get("/api/v1/stocks/RELIANCE/history", params={"start": "2025-01-01"}).json()
        self.assertEqual(empty["prices"], [])
        self.assertEqual(empty["total_matching_rows"], 0)

    def test_invalid_inputs_and_unknown_stock(self):
        client = self.client()
        self.assertEqual(client.get("/api/v1/stocks/UNKNOWN/prediction").status_code, 404)
        self.assertEqual(client.get("/api/v1/stocks/RELIANCE/history?limit=0").status_code, 422)
        self.assertEqual(client.get("/api/v1/stocks/RELIANCE/history?limit=3001").status_code, 422)
        self.assertEqual(client.get("/api/v1/stocks/RELIANCE/history?start=bad").status_code, 422)
        self.assertEqual(client.get("/api/v1/stocks/RELIANCE/history?start=2025-01-01&end=2024-01-01").status_code, 422)
        self.assertEqual(client.post("/api/v1/stocks/RELIANCE/prediction").status_code, 405)

    def test_newest_complete_snapshot_and_integrity_checks(self):
        directory = self.root / "data/market_context_20261112"
        directory.mkdir()
        raw = self.make_prices(120)
        path = directory / "RELIANCE.csv"
        raw.to_csv(path, index=False)
        manifest = {"As_Of_Exclusive": "2026-11-12",
                    "Records": [{"Name": "RELIANCE", "SHA256": hashlib.sha256(path.read_bytes()).hexdigest()}]}
        (directory / "download_manifest.json").write_text(json.dumps(manifest))
        incomplete = self.root / "data/market_context_20261113"
        incomplete.mkdir()
        (incomplete / "download_manifest.json").write_text(json.dumps({"As_Of_Exclusive": "2026-11-13", "Records": []}))
        client = self.client()
        response = client.get("/api/v1/stocks/RELIANCE/prediction").json()
        self.assertEqual(response["source"]["kind"], "downloaded_snapshot")
        self.assertEqual(response["data_date"], str(raw["Date"].iloc[-1].date()))
        path.write_text(path.read_text() + "\n")
        with self.assertLogs("backend.app.main", level="ERROR"):
            with TestClient(create_app(self.settings)) as bad:
                self.assertEqual(bad.get("/health").status_code, 503)
                self.assertEqual(bad.get("/api/v1/stocks/RELIANCE/prediction").status_code, 503)

    def test_invalid_latest_row_fails_readiness_without_older_fallback(self):
        raw = self.raw.copy()
        raw.loc[raw.index[-1], "Close"] = np.nan
        raw.to_csv(self.root / "data/raw/RELIANCE.csv", index=False)
        with self.assertLogs("backend.app.main", level="ERROR"):
            client = self.client()
        response = client.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(client.get("/api/v1/stocks").status_code, 503)
        self.assertNotIn(str(self.root), response.text)

    def test_unsafe_model_path_and_mismatched_metrics_are_rejected(self):
        self.metadata.loc[0, "Model_File"] = "../outside.joblib"
        self.metadata.to_csv(self.root / "models/model_metadata.csv", index=False)
        with self.assertLogs("backend.app.main", level="ERROR"):
            client = self.client()
        self.assertEqual(client.get("/health").status_code, 503)
        self.metadata.loc[0, "Model_File"] = "models/RELIANCE_model.joblib"
        self.metadata.to_csv(self.root / "models/model_metadata.csv", index=False)
        path = self.root / "data/processed/test_results.csv"
        scores = pd.read_csv(path)
        scores.loc[0, "Model"] = "XGBoost"
        scores.to_csv(path, index=False)
        with self.assertLogs("backend.app.main", level="ERROR"):
            with TestClient(create_app(self.settings)) as mismatched:
                self.assertEqual(mismatched.get("/health").status_code, 503)

    def test_performance_keeps_experimental_scores_separate(self):
        folder = self.root / "models/experiments/market_context_example"
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text(json.dumps({"Results": {"New_2026_Test": {
            "Evaluation_Start": "2026-01-01", "Evaluation_End": "2026-11-03",
            "Stock_Only_Mean_Accuracy": .453, "Selected_Mean_Accuracy": .465,
            "Selected_Mean_Balanced_Accuracy": .507}}}))
        client = self.client()
        overview = client.get("/api/v1/performance").json()
        self.assertEqual(overview["mean_active_historical_accuracy"], .6)
        self.assertEqual(overview["active_models"][0]["start"], "2024-04-15")
        self.assertEqual(overview["active_models"][0]["model_scope"], "active")
        self.assertEqual(overview["experiments"][0]["model_scope"], "experimental")
        self.assertEqual(overview["experiments"][0]["selected_mean_accuracy"], .465)
        single = client.get("/api/v1/stocks/RELIANCE/performance").json()
        self.assertEqual(single["samples"], 23)
        self.assertFalse(single["independent_new_holdout"])

    def test_requests_do_not_write_or_reload_models_and_openapi_is_valid(self):
        with patch("backend.app.service.joblib.load", wraps=joblib.load) as load:
            client = self.client()
            before = hash_files(self.root)
            for _ in range(3):
                self.assertEqual(client.get("/api/v1/stocks/RELIANCE/prediction").status_code, 200)
            self.assertEqual(load.call_count, 2)
        self.assertEqual(hash_files(self.root), before)
        response = client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        self.assertIn("/api/v1/stocks/{symbol}/prediction", response.json()["paths"])

    def test_built_dashboard_and_assets_share_the_api_origin(self):
        directory = self.root / "frontend" / "dist"
        (directory / "assets").mkdir(parents=True)
        (directory / "index.html").write_text("<!doctype html><title>StockSight dashboard</title>")
        (directory / "assets" / "app.js").write_text("console.log('dashboard');")
        client = self.client()
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("StockSight dashboard", response.text)
        self.assertEqual(response.headers["cache-control"], "no-cache")
        self.assertEqual(client.get("/assets/app.js").status_code, 200)
        self.assertEqual(client.get("/api/v1/stocks").status_code, 200)
        self.assertEqual(client.get("/assets/missing.js").status_code, 404)

    def test_cors_allows_local_dashboard_origin(self):
        client = self.client()
        response = client.options("/api/v1/stocks", headers={
            "Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:5173")
        blocked = client.options("/api/v1/stocks", headers={
            "Origin": "https://unconfigured.example", "Access-Control-Request-Method": "GET"})
        self.assertNotIn("access-control-allow-origin", blocked.headers)


if __name__ == "__main__":
    unittest.main()
