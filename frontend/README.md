# StockSight dashboard

The React/Vite frontend uses the existing FastAPI backend and real local data.
It includes a five-stock watchlist, selectable chart ranges, pointer/keyboard
price exploration, five-trading-session forecasts, technical indicators,
and separate historical/experimental evaluation views.

## Run the completed application

From the repository root:

```powershell
npm --prefix frontend ci
npm --prefix frontend run build
.\venv\Scripts\python.exe -B -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Restart an already-running backend after the first
frontend build so it registers the static assets. Subsequent builds are read
from disk; a fresh browser load picks up the new index.

## Develop the frontend

Keep the backend running, then use:

```powershell
npm --prefix frontend run dev
```

Open http://127.0.0.1:5173. Vite proxies `/api` and `/health` to port 8000.
Set `STOCKSIGHT_API_URL` before launching Vite if the backend uses another
address. `VITE_API_BASE_URL` is an optional build-time browser API address;
the default is same-origin and also works in the backend-served production build.

Data dates and model training dates are visible. Chart-month shortcuts represent
approximately 22/66/126/252 trading sessions, with the exact session count shown.
Refresh view reloads the available local snapshot; it does not fetch live prices.
Forecast probabilities are not labeled as accuracy or trading recommendations.
Experimental scores remain separate from active-model historical scores.

## Verify

```powershell
npm --prefix frontend test
npm --prefix frontend run build
npm --prefix frontend run test:e2e
```

Browser tests require the backend. They start and stop their own Vite server
unless `E2E_BASE_URL` points to the completed application. On Windows, tests
use installed Microsoft Edge if available; otherwise install Playwright Chromium
with `npx playwright install chromium` from `frontend`, or set
`PLAYWRIGHT_CHROMIUM_EXECUTABLE` to an installed Chromium browser.

The dashboard uses React/Vite with native SVG charts. No model training,
deployment, or external account integration is performed by the frontend.

## News panel

The selected stock has a separate News & sentiment panel with publisher links,
publication/fetch timestamps in IST, VADER positive/neutral/negative headline
labels, and counts. News requests run independently of price/prediction requests;
loading, empty, unavailable/retry, and stale-cache states are explicit. Switching
stocks clears previous headlines and cancels obsolete requests. Refresh news
reloads the endpoint (respecting its 15-minute cache); Refresh view also reloads
news. No news score is shown as prediction confidence, and forecasts are unchanged.
Browser tests use provider fixtures for deterministic outage, cache, empty,
switching, mobile, and refresh checks. The backend tests exercise actual VADER
classification and feed parsing offline.
