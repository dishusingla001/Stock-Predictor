"""Causality, target-calendar, date-alignment, and selection checks."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from market_features import feature_variants, index_features, merge_context, stock_features
from market_context_experiment import choose_features, training_history, evaluate_model


def prices(count=300):
    days = pd.bdate_range("2019-01-01", periods=count)
    close = 100 + np.arange(count) * .05 + 2 * np.sin(np.arange(count) / 5)
    return pd.DataFrame({"Date": days, "Open": close, "High": close + 1,
                         "Low": close - 1, "Close": close, "Volume": 1000 + np.arange(count)})


class MarketContextTests(unittest.TestCase):
    def test_future_prices_do_not_change_past_features(self):
        stock = prices()
        market = prices()
        sector = prices()
        before = merge_context(stock, market, sector)
        changed = sector.copy()
        changed.loc[changed.index >= 220, "Close"] *= 2
        changed_stock = stock.copy()
        changed_stock.loc[changed_stock.index >= 220, ["Close", "High", "Low"]] *= 2
        after = merge_context(changed_stock, market, changed)
        features = feature_variants(["RSI", "EMA20", "EMA50", "MACD", "ROC", "ATR", "Volume"])["with_market_sector"]
        pd.testing.assert_frame_equal(before.loc[:219, features], after.loc[:219, features])

    def test_missing_index_date_is_not_filled_or_used_to_shift_returns(self):
        stock = prices()
        missing_date = stock["Date"].iloc[100]
        index = prices().drop(index=100)
        merged = merge_context(stock, index, prices())
        self.assertTrue(pd.isna(merged.loc[100, "Market_Return_1D"]))
        self.assertTrue(pd.isna(merged.loc[100, "Market_Source_Date"]))
        self.assertTrue(pd.isna(merged.loc[101, "Market_Return_1D"]))
        self.assertTrue(pd.isna(merged.loc[105, "Market_Return_5D"]))
        self.assertTrue(pd.isna(merged.loc[120, "Market_Return_20D"]))
        self.assertEqual(merged.loc[100, "Date"], missing_date)
        self.assertTrue(np.isfinite(merged.loc[110, "Market_Volatility_20D"]))

    def test_outcome_horizon_survives_missing_context_filter(self):
        stock = prices()
        market = prices().drop(index=[102, 103])
        merged = merge_context(stock, market, prices())
        self.assertEqual(merged.loc[100, "Target_Date"], stock["Date"].iloc[105])
        self.assertEqual(merged.loc[100, "Future_Close"], stock["Close"].iloc[105])
        self.assertTrue(merged.tail(5)["Target"].isna().all())
        self.assertTrue(merged.tail(5)["Target_Date"].isna().all())

    def test_model_fitting_uses_only_mature_labels(self):
        frame = stock_features(prices()).iloc[50:].copy()
        origin = frame["Date"].iloc[170]
        history = training_history(frame, origin)
        self.assertTrue((history["Target_Date"] < origin).all())
        self.assertTrue((history["Date"] < origin).all())
        changed = frame.copy()
        unknown = changed["Target_Date"] >= origin
        changed.loc[unknown, "Target"] = 1 - changed.loc[unknown, "Target"]
        pd.testing.assert_frame_equal(history, training_history(changed, origin))

    def test_feature_selection_rejects_test_results(self):
        rows = [
            {"Period": "Validation", "Feature_Set": "stock_only", "Accuracy": .6,
             "Balanced_Accuracy": .58, "ROC-AUC": .6, "Feature_Count": 7, "Fresh_Accuracy": .1},
            {"Period": "Validation", "Feature_Set": "with_market", "Accuracy": .55,
             "Balanced_Accuracy": .7, "ROC-AUC": .8, "Feature_Count": 15, "Fresh_Accuracy": .99},
        ]
        self.assertEqual(choose_features(rows), "stock_only")
        with self.assertRaises(ValueError):
            choose_features([{**rows[0], "Period": "New_2026_Test"}])
        self.assertEqual({k: len(v) for k, v in feature_variants(["RSI", "EMA20", "EMA50", "MACD", "ROC", "ATR", "Volume"]).items()},
                         {"stock_only": 7, "with_market": 15, "with_market_sector": 24})

    def test_prediction_rows_match_summary_and_training_cutoff(self):
        frame = stock_features(prices()).dropna(subset=["RSI", "Future_Close", "Target_Date"])
        history = training_history(frame, frame["Date"].iloc[180])
        evaluation = frame.iloc[180:]
        estimator = LogisticRegression().fit(history[["RSI"]], history["Target"])
        summary, rows = evaluate_model(estimator, history, evaluation, ["RSI"], "EXAMPLE", "Validation", "stock_only", .5)
        output = pd.DataFrame(rows)
        self.assertAlmostEqual(summary["Accuracy"], float((output["Predicted"] == output["Target"]).mean()))
        self.assertTrue((pd.to_datetime(output["Train_Label_End"]) < pd.to_datetime(output["Date"])).all())


if __name__ == "__main__":
    unittest.main()
