"""Load trusted local artifacts once and serve forecasts without mutating project data."""
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.market_features import stock_features
from src.model_features import metadata_features, metadata_threshold, up_probability

from .config import Settings
from .schemas import ExperimentalEvaluation

LOGGER = logging.getLogger(__name__)
INDIA_TIME = timezone(timedelta(hours=5, minutes=30))
STOCK_NAMES = {
    "RELIANCE": "Reliance Industries", "TCS": "Tata Consultancy Services",
    "HDFCBANK": "HDFC Bank", "INFY": "Infosys", "ICICIBANK": "ICICI Bank",
}


class ServiceUnavailable(Exception):
    pass


class UnknownStock(Exception):
    pass


def today_in_india() -> date:
    return datetime.now(INDIA_TIME).date()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StockService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.project_root.resolve()
        try:
            self._load()
        except ServiceUnavailable:
            raise
        except Exception as exc:
            raise ServiceUnavailable("Local models or market data could not be loaded. Check the server log.") from exc

    def _artifact(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to((self.root / "models").resolve()) or path.suffix != ".joblib":
            raise ServiceUnavailable("Active model metadata contains an invalid artifact location.")
        return path

    def _select_dataset(self):
        if self.settings.raw_data_dir is not None:
            return self.settings.raw_data_dir.resolve(), {"kind": "configured_directory", "as_of_exclusive": None}
        for folder in sorted((self.root / "data").glob("market_context_*"), reverse=True):
            manifest_path = folder / "download_manifest.json"
            if not folder.is_dir() or not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = {row["Name"]: row for row in manifest["Records"]}
            # A collector may be in progress. Only complete snapshots can be selected.
            if not set(self.metadata.index).issubset(records):
                continue
            as_of = date.fromisoformat(manifest["As_Of_Exclusive"])
            for stock in self.metadata.index:
                path = folder / f"{stock}.csv"
                if not path.is_file() or sha256(path) != records[stock]["SHA256"]:
                    raise ServiceUnavailable("The newest complete data snapshot failed its integrity check.")
            return folder, {"kind": "downloaded_snapshot", "as_of_exclusive": as_of}
        return self.root / "data" / "raw", {"kind": "project_raw", "as_of_exclusive": None}

    def _prices(self, path):
        frame = pd.read_csv(path, parse_dates=["Date"])
        columns = ["Open", "High", "Low", "Close", "Volume"]
        if frame.empty or frame["Date"].isna().any() or frame["Date"].duplicated().any():
            raise ServiceUnavailable("Stock history has missing, invalid, or duplicate dates.")
        frame = frame.sort_values("Date").reset_index(drop=True)
        values = frame[columns].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (frame[["Open", "High", "Low", "Close"]] <= 0).any().any() or (frame["Volume"] < 0).any():
            raise ServiceUnavailable("Stock history contains invalid price or volume values.")
        if (frame["High"] < frame[["Open", "Close", "Low"]].max(axis=1) - 1e-6).any() or (frame["Low"] > frame[["Open", "Close", "High"]].min(axis=1) + 1e-6).any():
            raise ServiceUnavailable("Stock history contains inconsistent daily prices.")
        if (frame["Volume"] % 1 != 0).any():
            raise ServiceUnavailable("Stock history contains invalid volume values.")
        if frame["Date"].max().date() > today_in_india():
            raise ServiceUnavailable("Stock history contains future-dated prices.")
        if self.source["as_of_exclusive"] is not None and frame["Date"].max().date() >= self.source["as_of_exclusive"]:
            raise ServiceUnavailable("Snapshot contains a bar on or after its exclusive as-of date.")
        return frame

    def _load(self):
        metadata_path = self.root / "models" / "model_metadata.csv"
        self.metadata = pd.read_csv(metadata_path).set_index("Stock")
        if self.metadata.empty or self.metadata.index.duplicated().any():
            raise ServiceUnavailable("Active stock metadata is empty or contains duplicate symbols.")
        if not set(self.metadata.index).issubset(STOCK_NAMES):
            raise ServiceUnavailable("Active metadata contains unsupported stock symbols.")
        self.raw_dir, self.source = self._select_dataset()
        self.models, self.scalers, self.prices, self.features, self.model_ids = {}, {}, {}, {}, {}
        for stock, info in self.metadata.iterrows():
            if int(info["Horizon_Days"]) != 5:
                raise ServiceUnavailable("This API requires a five-trading-day active model.")
            fields = metadata_features(info)
            metadata_threshold(info)
            prices = self._prices(self.raw_dir / f"{stock}.csv")
            frame = stock_features(prices)
            if not set(fields).issubset(frame.columns):
                raise ServiceUnavailable("The active model needs features unsupported by this API.")
            if not np.isfinite(frame.tail(1)[fields].to_numpy(dtype=float)).all():
                raise ServiceUnavailable("The newest stock row has invalid indicators; no older row was substituted.")
            model_path = self._artifact(str(info["Model_File"]))
            scaler_path = self._artifact(str(info["Scaler_File"]))
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path) if info["Model"] == "Logistic Regression" else None
            # Verify the inference contract at startup rather than on the first user request.
            inputs = frame.tail(1)[fields]
            if scaler is not None:
                inputs = scaler.transform(inputs)
            probability = up_probability(model, inputs)
            if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
                raise ServiceUnavailable("An active model returned invalid probabilities.")
            self.models[stock], self.scalers[stock] = model, scaler
            self.prices[stock], self.features[stock] = prices, frame
            identity = metadata_path.read_bytes() + model_path.read_bytes()
            if scaler is not None:
                identity += scaler_path.read_bytes()
            self.model_ids[stock] = hashlib.sha256(identity).hexdigest()[:16]
        self._load_performance()
        self.experiments = self._load_experiments()

    def _load_performance(self):
        results = pd.read_csv(self.root / "data" / "processed" / "test_results.csv").set_index("Stock")
        splits = pd.read_csv(self.root / "data" / "processed" / "split_periods.csv")
        if results.index.duplicated().any() or set(results.index) != set(self.metadata.index):
            raise ServiceUnavailable("Active performance results do not match the active stock set.")
        self.performance_rows = {}
        metric_columns = {
            "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
            "f1": "F1", "roc_auc": "ROC-AUC",
            "training_majority_baseline_accuracy": "Training_Majority_Baseline_Accuracy",
        }
        for stock, info in self.metadata.iterrows():
            row = results.loc[stock]
            periods = splits.loc[(splits["Stock"] == stock) & (splits["Period"] == "Test")]
            if row["Model"] != info["Model"] or row["Period"] != "Test" or len(periods) != 1:
                raise ServiceUnavailable("Active metrics refer to a different model or evaluation period.")
            split = periods.iloc[0]
            if str(split["Start"]) != str(info["Test_Start"]) or str(split["End"]) != str(info["Test_End"]):
                raise ServiceUnavailable("Active metric dates do not match model metadata.")
            metrics = {name: float(row[column]) for name, column in metric_columns.items()}
            if any(not np.isfinite(value) or not 0 <= value <= 1 for value in metrics.values()):
                raise ServiceUnavailable("Active performance metrics are invalid.")
            self.performance_rows[stock] = {
                "symbol": stock, "model": info["Model"], "model_id": self.model_ids[stock],
                "start": info["Test_Start"], "end": info["Test_End"],
                "samples": int(split["Rows"]), "model_trained_through": info["Refit_End"], **metrics,
            }

    def _load_experiments(self):
        for path in sorted((self.root / "models" / "experiments").glob("market_context_*/summary.json"), reverse=True):
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
                records = []
                for period, row in summary["Results"].items():
                    records.append({
                        "experiment_id": path.parent.name, "evaluation_kind": period,
                        "start": row["Evaluation_Start"], "end": row["Evaluation_End"],
                        "stock_only_mean_accuracy": float(row["Stock_Only_Mean_Accuracy"]),
                        "selected_mean_accuracy": float(row["Selected_Mean_Accuracy"]),
                        "selected_mean_balanced_accuracy": row.get("Selected_Mean_Balanced_Accuracy"),
                    })
                # Validate the entire optional summary before using any of its rows.
                return [ExperimentalEvaluation.model_validate(row).model_dump() for row in records]
            except (OSError, ValueError, KeyError, TypeError):
                LOGGER.warning("Skipping an incomplete experiment summary: %s", path.name)
        return []

    def symbol(self, value):
        stock = value.strip().upper()
        if stock.endswith(".NS"):
            stock = stock[:-3]
        if stock not in self.metadata.index:
            raise UnknownStock("Unknown stock symbol. Use /api/v1/stocks to see supported stocks.")
        return stock

    def freshness(self, data_date):
        age = max(0, (today_in_india() - data_date).days)
        return {"data_age_calendar_days": age, "is_stale": age > self.settings.stale_after_days}

    def health(self):
        latest = min(frame["Date"].iloc[-1].date() for frame in self.prices.values())
        return {"status": "ok", "stocks_loaded": len(self.models), "latest_data_date": latest,
                **self.freshness(latest), "source": self.source}

    def stock_list(self):
        stocks = []
        for stock, info in self.metadata.iterrows():
            row = self.prices[stock].iloc[-1]
            data_date = row["Date"].date()
            stocks.append({
                "symbol": stock, "name": STOCK_NAMES[stock], "model": info["Model"],
                "data_date": data_date, "close": float(row["Close"]), **self.freshness(data_date),
            })
        return {"stocks": stocks, "count": len(stocks), "source": self.source}

    def prediction(self, symbol):
        stock = self.symbol(symbol)
        info = self.metadata.loc[stock]
        fields = metadata_features(info)
        latest = self.features[stock].tail(1)
        inputs = latest[fields]
        scaler = self.scalers[stock]
        if scaler is not None:
            inputs = scaler.transform(inputs)
        probability = float(up_probability(self.models[stock], inputs)[0])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ServiceUnavailable("The active model returned an invalid probability.")
        threshold = metadata_threshold(info)
        row = latest.iloc[0]
        data_date = row["Date"].date()
        return {
            "symbol": stock, "model": info["Model"], "model_id": self.model_ids[stock],
            "model_trained_through": info["Refit_End"], "data_date": data_date,
            "close": float(row["Close"]), "direction": "UP" if probability >= threshold else "DOWN",
            "up_probability": probability, "down_probability": 1 - probability,
            "decision_threshold": threshold, **self.freshness(data_date),
            "source": self.source, "generated_at": datetime.now(timezone.utc),
            "indicators": {name: float(row[name]) for name in fields},
        }

    def history(self, symbol, start=None, end=None, limit=180):
        stock = self.symbol(symbol)
        frame = self.prices[stock]
        subset = frame
        if start is not None:
            subset = subset.loc[subset["Date"] >= pd.Timestamp(start)]
        if end is not None:
            subset = subset.loc[subset["Date"] <= pd.Timestamp(end)]
        total = len(subset)
        subset = subset.tail(limit)
        prices = [
            {"date": row["Date"].date(), "open": float(row["Open"]), "high": float(row["High"]),
             "low": float(row["Low"]), "close": float(row["Close"]), "volume": int(row["Volume"])}
            for _, row in subset.iterrows()
        ]
        return {
            "symbol": stock, "latest_available_date": frame["Date"].iloc[-1].date(),
            "source": self.source, "total_matching_rows": total, "returned_rows": len(prices),
            "truncated": total > len(prices), "prices": prices,
        }

    def performance(self, symbol):
        return self.performance_rows[self.symbol(symbol)].copy()

    def performance_overview(self):
        rows = [row.copy() for row in self.performance_rows.values()]
        return {"active_models": rows, "mean_active_historical_accuracy": float(np.mean([row["accuracy"] for row in rows])),
                "experiments": self.experiments}
