"""Feature definitions and inference rules shared by training and consumers."""
import numpy as np

CURRENT_FEATURES = ["RSI", "EMA20", "EMA50", "MACD", "ROC", "ATR", "Volume"]
RELATIVE_FEATURES = [
    "RSI", "ROC", "Return_1D", "Return_5D",
    "EMA20_EMA50_Ratio", "Close_EMA20_Ratio", "Close_EMA50_Ratio",
    "BB_Position", "Volume_ratio", "MACD_Relative",
    "MACD_Histogram_Relative", "ATR_Relative",
]
FEATURE_SETS = {
    "current": CURRENT_FEATURES,
    "relative": RELATIVE_FEATURES,
    "combined": list(dict.fromkeys(CURRENT_FEATURES + RELATIVE_FEATURES
                                  + ["MACD_histogram"])),
}


def prepare_features(data):
    """Derive contemporaneous ratios; no future prices or fitted statistics."""
    data = data.copy()
    close = data["Close"].replace(0, np.nan)
    for source, destination in [
        ("MACD", "MACD_Relative"),
        ("MACD_histogram", "MACD_Histogram_Relative"),
        ("ATR", "ATR_Relative"),
    ]:
        data[destination] = data[source] / close
    return data.replace([np.inf, -np.inf], np.nan)


def metadata_features(info):
    features = str(info["Features"]).split(",")
    if not features or any(not name.strip() for name in features):
        raise ValueError("Model metadata contains an empty feature name.")
    return [name.strip() for name in features]


def metadata_threshold(info):
    threshold = float(info.get("Decision_Threshold", 0.5))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Decision threshold must be finite and between zero and one.")
    return threshold


def up_probability(model, inputs):
    classes = list(model.classes_)
    if classes != [0, 1]:
        raise ValueError("Expected binary model classes [0, 1].")
    return model.predict_proba(inputs)[:, classes.index(1)]
