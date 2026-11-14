"""Download isolated Yahoo daily history for market-context experiments."""
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
TICKERS = {
    "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "HDFCBANK": "HDFCBANK.NS",
    "INFY": "INFY.NS", "ICICIBANK": "ICICIBANK.NS",
    "NIFTY50": "^NSEI", "NIFTYBANK": "^NSEBANK",
    "NIFTYIT": "^CNXIT", "NIFTYENERGY": "^CNXENERGY",
}


def collect(as_of):
    import yfinance as yf
    date.fromisoformat(as_of)
    output = BASE_DIR / "data" / f"market_context_{as_of.replace('-', '')}"
    output.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(output / "yahoo_cache"))
    records = []
    for name, ticker in TICKERS.items():
        path = output / f"{name}.csv"
        if path.exists():
            frame = pd.read_csv(path, parse_dates=["Date"])
            print(f"Using cached {name}", flush=True)
        else:
            print(f"Downloading {name}: {ticker}", flush=True)
            frame = None
            for attempt in range(2):
                frame = yf.download(
                    ticker, start="2015-01-01", end=as_of, auto_adjust=False,
                    progress=False, threads=False, timeout=30,
                )
                if frame is not None and not frame.empty:
                    break
                print(f"{name}: download attempt {attempt + 1} returned no rows", flush=True)
            if frame is None or frame.empty:
                raise RuntimeError(f"No Yahoo history returned for {ticker}; no substituted data used.")
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)
            frame = frame.reset_index()
            frame["Date"] = pd.to_datetime(frame["Date"]).dt.tz_localize(None).dt.normalize()
        frame = frame.sort_values("Date").reset_index(drop=True)
        required = ["Open", "High", "Low", "Close", "Volume"] if name in {"RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"} else ["Close"]
        if frame["Date"].duplicated().any() or frame["Date"].isna().any():
            raise ValueError(f"{name}: invalid or duplicate dates.")
        if not np.isfinite(frame[required].to_numpy(dtype=float)).all() or (frame["Close"] <= 0).any():
            raise ValueError(f"{name}: invalid price/volume data.")
        if frame["Date"].max() >= pd.Timestamp(as_of):
            raise ValueError(f"{name}: excluded as-of date was returned.")
        frame.to_csv(path, index=False)
        records.append({
            "Name": name, "Yahoo_Ticker": ticker, "Rows": len(frame),
            "Start": str(frame["Date"].min().date()), "End": str(frame["Date"].max().date()),
            "SHA256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "Source": f"https://finance.yahoo.com/quote/{ticker}/history/",
        })
        (output / "download_manifest.json").write_text(
            json.dumps({"As_Of_Exclusive": as_of, "Records": records}, indent=2), encoding="utf-8")
        print(f"{name}: {len(frame)} rows, {records[-1]['Start']} to {records[-1]['End']}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", required=True, help="Exclusive end date; excludes potentially partial current-day bars.")
    print("Downloaded data directory:", collect(parser.parse_args().as_of), flush=True)
