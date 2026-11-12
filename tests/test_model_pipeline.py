"""Regression checks for leakage boundaries, thresholds, and saved-model inference."""
import io
import sys
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_features import FEATURE_SETS, metadata_features, metadata_threshold, prepare_features, up_probability
import train_models
from train_models import HORIZON, classification_metrics, select_winner, split_periods, tune_threshold


class ModelPipelineTests(unittest.TestCase):
    def test_chronological_gaps_keep_labels_inside_periods(self):
        frame = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=200),
                              "Target": np.arange(200) % 2})
        data, train, validation, test = split_periods(frame)
        self.assertLess(train.index[-1] + HORIZON, validation.index[0])
        self.assertLess(validation.index[-1] + HORIZON, test.index[0])
        refit = data.iloc[:test.index[0] - HORIZON]
        self.assertLess(refit.index[-1] + HORIZON, test.index[0])

    def test_threshold_can_recover_both_classes(self):
        target = [0, 0, 0, 1, 1, 1]
        probability = np.array([0.36, 0.40, 0.42, 0.44, 0.46, 0.48])
        threshold = tune_threshold(target, probability)
        np.testing.assert_array_equal(probability >= threshold, target)
        self.assertEqual(tune_threshold([0, 1], [0.1, 0.9]), 0.5)

    def test_selection_uses_validation_accuracy_and_rejects_test_rows(self):
        rows = [
            {"Period": "Validation", "Accuracy": 0.6, "Balanced_Accuracy": 0.59,
             "ROC-AUC": 0.61, "Decision_Threshold": 0.5, "Test_Accuracy": 0.2},
            {"Period": "Validation", "Accuracy": 0.55, "Balanced_Accuracy": 0.7,
             "ROC-AUC": 0.9, "Decision_Threshold": 0.4, "Test_Accuracy": 0.99},
        ]
        self.assertIs(select_winner(rows), rows[0])
        with self.assertRaises(ValueError):
            select_winner([{**rows[0], "Period": "Test"}])

    def test_features_do_not_use_future_labels(self):
        frame = pd.DataFrame({"Close": [100., 200.], "MACD": [2., 4.],
                              "MACD_histogram": [1., 2.], "ATR": [3., 6.],
                              "Future_Close": [90., 300.]})
        first = prepare_features(frame)
        frame["Future_Close"] = [999., 1.]
        second = prepare_features(frame)
        derived = ["MACD_Relative", "MACD_Histogram_Relative", "ATR_Relative"]
        pd.testing.assert_frame_equal(first[derived], second[derived])
        np.testing.assert_allclose(first[derived], [[.02, .01, .03], [.02, .01, .03]])

    def test_saved_split_model_and_metadata_reproduce_pipeline(self):
        rng = np.random.default_rng(42)
        for feature_set, features in FEATURE_SETS.items():
            inputs = pd.DataFrame(rng.normal(size=(60, len(features))), columns=features)
            target = (inputs.iloc[:, 0] > 0).astype(int)
            pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
            pipeline.fit(inputs, target)
            buffer = io.BytesIO()
            joblib.dump((pipeline.named_steps["model"], pipeline.named_steps["scaler"]), buffer)
            buffer.seek(0)
            model, scaler = joblib.load(buffer)
            info = {"Features": ",".join(features), "Decision_Threshold": 0.43}
            probability = up_probability(model, scaler.transform(inputs[metadata_features(info)]))
            np.testing.assert_allclose(probability, up_probability(pipeline, inputs))
            predicted = probability >= metadata_threshold(info)
            np.testing.assert_array_equal(predicted, up_probability(pipeline, inputs) >= .43)
        self.assertEqual(metadata_threshold({}), .5)
        with self.assertRaises(ValueError):
            metadata_threshold({"Decision_Threshold": float("nan")})

    def test_declining_release_is_archived_and_previous_set_restored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = root / "models"
            processed = root / "data" / "processed"
            backup = models / "experiments" / "previous"
            source = root / "src"
            for folder in (models, processed, backup, source):
                folder.mkdir(parents=True, exist_ok=True)
            for name in ("train_models.py", "model_features.py", "predict.py", "verify_predictions.py"):
                (source / name).write_text("# example source")
            for folder, value, accuracy in ((models, b"candidate", .45), (backup, b"previous", .6)):
                (folder / "EXAMPLE_model.joblib").write_bytes(value)
                (folder / "EXAMPLE_scaler.joblib").write_bytes(value)
                pd.DataFrame([{"Stock": "EXAMPLE", "Model_File": "models/EXAMPLE_model.joblib",
                               "Scaler_File": "models/EXAMPLE_scaler.joblib"}]).to_csv(
                                   folder / "model_metadata.csv", index=False)
                pd.DataFrame([{"Stock": "EXAMPLE", "Accuracy": accuracy}]).to_csv(
                    (processed if folder == models else backup) / "test_results.csv", index=False)
            (processed / "feature_importance.csv").write_text("Feature,Importance\nRSI,0.1\n")
            with patch.object(train_models, "BASE_DIR", root), patch.object(train_models.subprocess, "run") as run:
                decision = train_models.preserve_candidate_run(backup)
            self.assertFalse(decision["Accepted"])
            self.assertEqual(run.call_count, 2)
            self.assertEqual((models / "EXAMPLE_model.joblib").read_bytes(), b"previous")
            self.assertAlmostEqual(pd.read_csv(processed / "test_results.csv")["Accuracy"].iloc[0], .6)
            candidate = Path(decision["Candidate_Directory"])
            self.assertEqual((candidate / "EXAMPLE_model.joblib").read_bytes(), b"candidate")
            archived_meta = pd.read_csv(candidate / "model_metadata.csv")
            self.assertEqual((root / archived_meta["Model_File"].iloc[0]).read_bytes(), b"candidate")
            self.assertFalse((processed / "feature_importance.csv").exists())
            self.assertTrue((candidate / "feature_importance.csv").exists())
            self.assertTrue((candidate / "release_decision.json").exists())

    def test_confusion_matrix_matches_accuracy(self):
        metrics = classification_metrics([0, 0, 1, 1], [.1, .6, .4, .9])
        self.assertEqual([metrics[k] for k in ["TN", "FP", "FN", "TP"]], [1, 1, 1, 1])
        self.assertEqual(metrics["Accuracy"], .5)


if __name__ == "__main__":
    unittest.main()
