"""Behavioral checks for causal walk-forward training and calendar windows."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import walk_forward as wf


def fixture():
    dates = pd.bdate_range("2015-01-01", periods=1650)
    signal = np.sin(np.arange(1645) / 8) + np.cos(np.arange(1645) / 17)
    return pd.DataFrame({
        "Date": dates[:1645], "Target_Date": dates[5:],
        "signal": signal, "Target": (signal > .1).astype(int),
        "Close": 100 + np.arange(1645) / 100,
        "Future_Close": 100 + np.arange(1645) / 100 + np.where(signal > .1, 1, -1),
    })


class WalkForwardTests(unittest.TestCase):
    def test_calendar_windows_and_target_maturity(self):
        data = fixture()
        origin = data["Date"].iloc[1500]
        for policy, years in wf.POLICIES.items():
            history = wf.eligible_history(data, origin, policy)
            self.assertTrue((history["Target_Date"] < origin).all())
            self.assertLess(history.index[-1] + 5, 1500)
            if years:
                lower = origin - pd.DateOffset(years=years)
                self.assertTrue((history["Date"] >= lower).all())
                excluded = data.loc[(data["Date"] < lower) & (data["Target_Date"] < origin)]
                self.assertTrue(set(excluded.index).isdisjoint(history.index))

    def test_future_changes_cannot_change_earlier_predictions(self):
        data = fixture()
        evaluation = data.iloc[1300:1340]
        estimator = LogisticRegression(random_state=42)
        first = wf.run_policy(data, evaluation, "rolling_3y", estimator, ["signal"], "EXAMPLE", "Validation", block_rows=20)
        changed = data.copy()
        changed.loc[changed.index >= 1320, "signal"] = 999
        changed.loc[changed.index >= 1320, "Target"] = 1 - changed.loc[changed.index >= 1320, "Target"]
        second = wf.run_policy(changed, changed.iloc[1300:1340], "rolling_3y", estimator, ["signal"], "EXAMPLE", "Validation", block_rows=20)
        np.testing.assert_array_equal(
            pd.DataFrame(first[1]).iloc[:20]["UP_Probability"].to_numpy(),
            pd.DataFrame(second[1]).iloc[:20]["UP_Probability"].to_numpy())

    def test_each_row_scored_once_and_histories_advance_causally(self):
        data = fixture()
        evaluation = data.iloc[1300:1361]
        estimator = LogisticRegression(random_state=42)
        for policy in ("frozen", "expanding", "rolling_3y"):
            with patch.object(wf, "fit_model", wraps=wf.fit_model) as fit:
                summary, predictions, blocks = wf.run_policy(
                    data, evaluation, policy, estimator, ["signal"], "EXAMPLE", "Validation", block_rows=20)
            rows = pd.DataFrame(predictions)
            self.assertEqual(len(rows), len(evaluation))
            self.assertFalse(rows["Date"].duplicated().any())
            self.assertEqual(summary["Rows"], 61)
            self.assertEqual(fit.call_count, 1 if policy == "frozen" else 4)
            for block in blocks:
                self.assertLess(block["Train_Label_End"], block["Evaluation_Start"])
            ends = [block["Train_End"] for block in blocks]
            if policy == "frozen":
                self.assertEqual(len(set(ends)), 1)
            else:
                self.assertEqual(ends, sorted(set(ends)))

    def test_validation_selection_ignores_test_scores(self):
        rows = [
            {"Period": "Validation", "Policy": "rolling_3y", "Accuracy": .6,
             "Balanced_Accuracy": .58, "Block_Accuracy_Std": .05, "ROC-AUC": .6, "Test_Accuracy": .1},
            {"Period": "Validation", "Policy": "expanding", "Accuracy": .55,
             "Balanced_Accuracy": .7, "Block_Accuracy_Std": .01, "ROC-AUC": .8, "Test_Accuracy": .99},
        ]
        self.assertEqual(wf.select_policy(rows)["Policy"], "rolling_3y")
        with self.assertRaises(ValueError):
            wf.select_policy([{**rows[0], "Period": "Test"}])

    def test_invalid_policy_and_insufficient_history_fail(self):
        data = fixture()
        with self.assertRaises(ValueError):
            wf.eligible_history(data, data["Date"].iloc[1000], "unknown")
        with self.assertRaises(ValueError):
            wf.eligible_history(data, data["Date"].iloc[50], "rolling_3y")


if __name__ == "__main__":
    unittest.main()
