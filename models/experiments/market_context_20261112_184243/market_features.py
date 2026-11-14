"""Causal stock, broad-market, and sector features for matched-date experiments."""
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange

SECTORS = {"RELIANCE": "NIFTYENERGY", "TCS": "NIFTYIT", "INFY": "NIFTYIT",
           "HDFCBANK": "NIFTYBANK", "ICICIBANK": "NIFTYBANK"}
INDEX_FIELDS = ["Return_1D", "Return_5D", "Return_20D", "Volatility_20D", "Close_EMA20_Ratio"]


def sorted_prices(raw):
    data = raw.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)
    if data["Date"].isna().any() or data["Date"].duplicated().any():
        raise ValueError("Price dates must be valid and unique.")
    if not np.isfinite(data["Close"]).all() or (data["Close"] <= 0).any():
        raise ValueError("Closing prices must be positive and finite.")
    return data


def stock_features(raw):
    data = sorted_prices(raw)
    close = data["Close"]
    data["RSI"] = RSIIndicator(close=close, window=14).rsi()
    data["EMA20"] = close.ewm(span=20, adjust=False).mean()
    data["EMA50"] = close.ewm(span=50, adjust=False).mean()
    data["MACD"] = MACD(close=close).macd()
    data["ROC"] = ROCIndicator(close=close, window=12).roc()
    data["ATR"] = AverageTrueRange(high=data["High"], low=data["Low"], close=close, window=14).average_true_range()
    for days in (1, 5, 20):
        data[f"Stock_Return_{days}D"] = close.pct_change(days, fill_method=None)
    # Compute outcomes before any context-date filtering, preserving five STOCK sessions.
    data["Future_Close"] = close.shift(-5)
    data["Target_Date"] = data["Date"].shift(-5)
    data["Target"] = np.where(data["Future_Close"].notna(),
                              (data["Future_Close"] > close).astype(int), np.nan)
    return data


def index_features(raw, prefix, calendar=None):
    raw = sorted_prices(raw)
    if calendar is not None:
        # Reindex without filling so return horizons follow the STOCK calendar.
        dates = pd.DataFrame({"Date": pd.to_datetime(calendar)})
        raw = dates.merge(raw[["Date", "Close"]], on="Date", how="left", validate="one_to_one")
    close = raw["Close"]
    frame = raw[["Date"]].copy()
    for days in (1, 5, 20):
        frame[f"{prefix}_Return_{days}D"] = close.pct_change(days, fill_method=None)
    frame[f"{prefix}_Volatility_20D"] = close.pct_change(fill_method=None).rolling(20, min_periods=15).std()
    frame[f"{prefix}_Close_EMA20_Ratio"] = close / close.ewm(span=20, adjust=False).mean()
    frame[f"{prefix}_Source_Date"] = frame["Date"].where(close.notna())
    return frame


def merge_context(stock, market, sector):
    data = stock_features(stock)
    data = data.merge(index_features(market, "Market", data["Date"]), on="Date", how="left", validate="one_to_one")
    data = data.merge(index_features(sector, "Sector", data["Date"]), on="Date", how="left", validate="one_to_one")
    for days in (1, 5, 20):
        data[f"Stock_vs_Market_Return_{days}D"] = data[f"Stock_Return_{days}D"] - data[f"Market_Return_{days}D"]
        data[f"Stock_vs_Sector_Return_{days}D"] = data[f"Stock_Return_{days}D"] - data[f"Sector_Return_{days}D"]
    data["Sector_vs_Market_Return_5D"] = data["Sector_Return_5D"] - data["Market_Return_5D"]
    return data.replace([np.inf, -np.inf], np.nan)


def feature_variants(core):
    market = list(core) + [f"Market_{name}" for name in INDEX_FIELDS] + [
        f"Stock_vs_Market_Return_{days}D" for days in (1, 5, 20)]
    full = market + [f"Sector_{name}" for name in INDEX_FIELDS] + [
        f"Stock_vs_Sector_Return_{days}D" for days in (1, 5, 20)] + ["Sector_vs_Market_Return_5D"]
    return {"stock_only": list(core), "with_market": market, "with_market_sector": full}
