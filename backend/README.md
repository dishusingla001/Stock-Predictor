# StockSight backend

From the repository root, use the existing project virtual environment:

```powershell
.\venv\Scripts\python.exe -m pip install -r backend/requirements.txt
.\venv\Scripts\python.exe -B -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Interactive API documentation: http://127.0.0.1:8000/docs

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | /health | Readiness, available data date, and staleness |
| GET | /api/v1/stocks | Supported symbols, prices, and active model names |
| GET | /api/v1/stocks/RELIANCE/prediction | Active-model five-trading-session forecast |
| GET | /api/v1/stocks/RELIANCE/history?limit=180 | OHLCV history for a chart |
| GET | /api/v1/stocks/RELIANCE/performance | Dated historical test metrics for that active model |
| GET | /api/v1/performance | Active historical metrics and separately labeled experiments |

Symbols are case-insensitive; both `RELIANCE` and `RELIANCE.NS` are accepted.
History accepts inclusive ISO `start` and `end` dates. `limit` is 1?3000
(default 180); the newest matching rows are returned in ascending date order.
Empty date ranges return an empty list. Unknown symbols return 404; invalid
dates/limits return 422. Missing, corrupted, or incompatible assets return 503.

Models come only from the active `models/model_metadata.csv`. Experiments are
never promoted automatically. The API computes fresh indicators from the newest
complete local `data/market_context_<date>` snapshot and verifies its manifest
hashes. If none exists, it uses `data/raw`. An invalid complete snapshot
fails readiness rather than falling back silently. An unfinished download
without all supported stocks is ignored until complete.

All models and price data load once at startup. To refresh prices, run the existing
collector with an exclusive end date, then restart the API. Price/prediction requests make no network downloads, retraining, or CSV writes.
The independent news endpoint fetches external headline feeds on cache expiry. The API does not run the
top-level scripts that overwrite prediction reports.

For an explicit dataset directory:

```powershell
$env:STOCKSIGHT_DATA_DIR = 'D:\path\to\daily-stock-csvs'
```

The directory must contain one OHLCV CSV per supported symbol. Explicit data
directories must contain completed daily bars. Clear the variable to restore
automatic snapshot selection.

Predictions report the data date, model training date, probabilities, threshold,
and a **five trading-session horizon**. They do not estimate an exact future
calendar date without an exchange calendar. Probabilities are model outputs,
not accuracy measurements. The newest forecast outcome is unverified.
`data_age_calendar_days` and `is_stale` use the date in India; more than seven
calendar days is flagged as stale. This is a freshness indicator, not a trading
calendar calculation.

Active-model performance refers to its saved historical test dates.
Experimental 2026 results are nested separately and evaluate different models.
Latest experimental refits have no independent future-outcome score. The API
does not turn any of these outputs into BUY/SELL recommendations.

CORS permits `http://localhost:5173` and `http://127.0.0.1:5173` for the future
dashboard. Override with comma-separated `STOCKSIGHT_CORS_ORIGINS` if needed.
The server is intended for local development and binds to loopback in the
documented command.

Install test dependencies and run all checks:

```powershell
.\venv\Scripts\python.exe -m pip install -r backend/requirements-dev.txt
.\venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

The model-related dependency versions match the environment used to create the
saved artifacts. API implementation follows the [FastAPI documentation](https://fastapi.tiangolo.com/).

After `npm --prefix frontend run build`, restart the backend to serve the
dashboard at http://127.0.0.1:8000 alongside the API. The `/docs` endpoint remains
available. If no dashboard build exists, the root endpoint shows the API index.

## Independent news endpoint

`GET /api/v1/stocks/{symbol}/news` returns up to 12 company-matched English
headlines from the last seven days: title, source, Google News article link,
UTC publication time, VADER tone label, and compound sentiment score (-1 to +1).
The response includes fetch time, positive/neutral/negative counts, cache/stale
flags, methodology, and a stale warning when needed. The compound score is
not a probability. See the root README's News sentiment section for limitations
and scoring thresholds. Install the updated requirements and restart the API.

No API key is needed. The endpoint uses outbound HTTPS to Google News RSS with
an eight-second timeout, bounded one-megabyte responses, no article-body fetches,
a 15-minute in-memory cache, and a one-minute failure cooldown per stock. It
serves stale cache for at most one hour after a successful fetch, explicitly
flagged; with no usable cache it returns 503. Valid empty feeds return 200 with
an empty article list. Unknown symbols return 404, malformed symbols 422.
The route works independently of model readiness and never changes predictions.
Set `STOCKSIGHT_NEWS_ENABLED=false` for offline operation; other endpoints remain
available. Refreshing the panel respects the provider cache.
