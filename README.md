# StockSight

## Project Overview

StockSight is a machine-learning stock prediction project that analyzes historical market data for **Reliance, TCS, HDFC Bank, Infosys, and ICICI Bank**. It transforms price and volume history into technical indicators such as RSI, EMA, MACD, ROC, ATR, and trading volume, then evaluates classification models to predict whether each stock is likely to move **up or down over the next five trading days**. The project also stores trained models, compares their performance, and generates probability-based predictions from the latest available market data.

## Training and prediction data

Run these commands from the project root after collecting raw data:

```powershell
python src/feature_engineering.py
python src/predict.py
```

Feature engineering produces separate datasets for each stock:

- `{stock}_features.csv` and `{stock}_ML.csv`: indicator-complete historical rows with a known five-trading-day `Future_Close` and `Target`, used for training and verification.
- `{stock}_prediction.csv`: exactly the latest raw row with technical indicators, without `Future_Close` or `Target`, used by `predict.py`.

The latest prediction row is validated before export; missing or non-finite values raise an error instead of silently selecting an older date. Prediction dates reflect the available raw data, so refresh raw data and regenerate features to obtain newer predictions. Existing saved models can be used without retraining for this data separation.

## Feature and accuracy experiments

Run from the project root:

```powershell
python -B -m unittest discover -s tests -v
python -u src/train_models.py
python src/predict.py
python src/verify_predictions.py
```

Training compares three feature sets for each stock: the original seven inputs,
relative indicators/returns/volume, and their combined set. Shared definitions in
`src/model_features.py` derive MACD/Close, MACD histogram/Close, and ATR/Close
without using future labels. Raw volume percentage change is excluded because
zero-volume days produce non-finite values; volume ratio remains available. Prediction and verification read the selected
feature order and decision threshold from model metadata.

Chronological 60% / 20% / 20% boundaries and five-row label gaps are preserved.
Within training, five-fold `TimeSeriesSplit(gap=5)` evaluates all 12 Logistic
Regression configurations and 20 random configurations each for Random Forest
and XGBoost, for every feature set. Scaling is fitted inside each training fold.
Hyperparameters are selected by mean CV accuracy; balanced accuracy and ROC-AUC
are also recorded.

For each tuned configuration, out-of-fold training probabilities select one
threshold from 0.35 through 0.65 in increments of 0.01. Accuracy ties favor
balanced accuracy and proximity to 0.5. The default 0.5 threshold is also retained
as a candidate. Validation accuracy selects the feature set, model, and threshold;
balanced accuracy and ROC-AUC break ties. No test metric participates in selection.
The selected configuration is refitted on pre-test history with the five-row
gap and evaluated once on the historical test period. Thresholds fitted on
earlier models may not transfer perfectly after refitting.

Outputs in `data/processed`:

- `model_results.csv`: validation results for every feature/model/threshold candidate.
- `test_results.csv`: selected models, confusion counts, UP prediction rates,
  accuracy, balanced accuracy, ROC-AUC, and the training-majority baseline.
- `tuning_results.csv`: parameters and CV scores for every search candidate.
- `split_periods.csv`: actual dates and row counts.
- `feature_importance.csv`: validation permutation importance, measured as the
  decrease in accuracy after shuffling a feature (ten repeats). Correlated
  features can share importance; negative values do not prove a useful signal.
  These rankings are diagnostic and do not select features automatically.
- `accuracy_comparison.csv`: before/after historical test accuracy for each stock.
- `models/model_metadata.csv`: feature order, threshold, selected parameters,
  dates, and artifact paths.

Every training run snapshots existing model artifacts and result reports under
`models/experiments/before_feature_run_<timestamp>` before replacing outputs.
All five stock experiments finish before active models are replaced. A complete
candidate snapshot is retained in `models/experiments/feature_candidate_<timestamp>`.
If mean historical test accuracy across all stocks declines, the whole previous
model set and its reports are restored. This release check does not select
individual stock models or trigger additional tuning. See `release_decision.json`
in the candidate archive. Candidate metadata points to its archived model files.

The final historical period has already been inspected by previous project
versions. These scores are retrospective comparisons, not proof of future
performance. The last-100-row verification overlaps the test period. A genuinely
new evaluation requires newer data and frozen model choices. Current raw data
ends on December 31, 2025; this experiment does not refresh market data.

Rolling training windows are evaluated separately below. Market-index features
remain a future experiment so their contribution can be measured independently.

## Walk-forward training-window experiment

Run `python -B -u src/walk_forward.py`. This is a separate experiment: it does
not replace active models, predictions, or verification reports.

The active model family, its saved hyperparameters, its feature list, and its
decision threshold are held fixed for each stock. Four training-history
policies are compared on the existing validation period:

- `frozen`: fit once before evaluation and keep the same model throughout.
- `expanding`: refit on all eligible history every 126 evaluation rows.
- `rolling_3y`: refit on the previous three calendar years every 126 rows.
- `rolling_5y`: refit on the previous five calendar years every 126 rows.

Eligibility uses the target's actual outcome date, derived from the raw
five-trading-row shift. Only targets resolved strictly before a block's first
prediction date can train its model. Earlier evaluation rows may enter later
refits once their outcomes are known, as they would in chronological operation.

Every stock's policy is selected on validation accuracy before any outer-test
run. Balanced accuracy, lower block-score variation, and ROC-AUC break ties.
The selected policy and predeclared frozen/expanding controls are then evaluated
on the existing test dates. Frozen controls must reproduce active-model test
accuracy. Walk-forward test scores describe a sequence of refitted models;
they should not be confused with one model fitted before the entire test set.

Results are written to `models/experiments/walk_forward_<timestamp>`:

- `protocol.json` and `selection_manifest.json`: rules, active-file hashes,
  and choices fixed before test evaluation.
- `validation_results.csv`, `validation_blocks.csv`, and
  `validation_predictions.csv`: policy scores and their underlying forecasts.
- `test_results.csv`, `test_blocks.csv`, `test_predictions.csv`, and
  `accuracy_comparison.csv`: selected policies and matched controls.
- `snapshot_metadata.csv`, `*_pipeline.joblib`, and `latest_predictions.csv`:
  separate latest-date model refits. These use all eligible history under the
  selected window, even for the frozen policy (one fit for the new forecast).
  Their own future outcomes are not verified; historical backtest scores belong
  to earlier model fits, not these latest snapshots.
- `summary.json`: aggregate comparison and confirmation that active files
  remain unchanged.

The fixed parameters and validation period have been used in earlier
experiments. All scores are retrospective. This run does not download fresh
market data, and it does not use test scores to switch individual stock policies.
Promoting a periodic-refitting policy requires inference and verification that
follow that same schedule; these experimental snapshots are not installed as
static active models.

The November 12, 2026 walk-forward run is saved under
`models/experiments/walk_forward_20261112_182739`. Validation selected three-year
windows for RELIANCE and HDFCBANK, expanding history for TCS and INFY, and frozen
history for ICICIBANK. Mean historical test accuracy was 52.23%, versus 53.25%
for the active static models. The predeclared expanding-only control scored
53.36%; its 0.11 percentage-point difference is small and was not used to switch
policies after examining test scores. Active models and reports were preserved.

## Market and sector context experiment

First collect an isolated daily-data snapshot, then run the experiment:

```powershell
python -B -u src/collect_market_context.py --as-of 2026-11-12
python -B -u src/market_context_experiment.py --as-of 2026-11-12
```

The exclusive as-of date excludes potentially incomplete current-day bars.
Downloads are stored in `data/market_context_20261112`, with source tickers,
date ranges, and SHA256 hashes. Existing raw data and active artifacts are preserved.

Three feature sets are compared: the seven active stock inputs, those inputs
plus NIFTY 50 context (15 features), and stock/market/sector context (24 features).
The sector proxies are NIFTY Bank for HDFCBANK/ICICIBANK, NIFTY IT for TCS/INFY,
and the broader NIFTY Energy index for RELIANCE. These are sector proxies,
not a point-in-time reconstruction of index constituents.

Index features include 1/5/20-session returns, trailing return volatility,
and price relative to EMA20. Relative returns compare the stock with its
market and sector. Index prices are reindexed to the stock calendar without
filling. Returns require both endpoints; volatility uses a 20-session window
with at least 15 valid returns. No future data, backfilling, or shifted
future outcomes are used as inputs. Missing context dates and affected
return endpoints are excluded for every candidate, with dates and coverage
reported. At least 95% matched coverage is required for each evaluation period.
Targets are computed on the full stock calendar before any exclusions.

Model families, parameters, and decision thresholds stay fixed to active
metadata. Validation from September 3, 2021 through October 18, 2023 chooses
the feature set. All stock choices are saved before either test is evaluated.
The previous historical period is a retrospective comparison. The newer test
uses only 2026 forecast dates with known five-session outcomes; its models fit
only labels resolved before January 1, 2026 and stay frozen throughout 2026.

Outputs are written to `models/experiments/market_context_<timestamp>`:
protocol/selection manifests, validation and test forecasts, matched-date
accuracy comparisons, coverage and price-revision reports, complete feature
datasets, and the exact historical/2026 model artifacts used for scoring.
Separate latest-date models and forecasts are fitted only after all test
scoring; their future outcomes are not verified.

Historical prices downloaded now can differ from previously saved prices
because of provider revisions or corporate-action adjustments. Matched
stock-only controls use the same refreshed snapshot as context candidates,
so their scores should be compared with each other rather than directly
with the old saved test scores. Active models and reports are not replaced.

The November 12, 2026 context experiment is saved under
`models/experiments/market_context_20261112_184243`. Fresh data ends on November 11;
known five-session outcomes allow scoring forecast dates through November 3.
On matched 2026 dates, selected feature sets averaged 46.48% accuracy versus
45.30% for stock-only models refitted on the same pre-2026 history. Mean balanced
accuracy was 50.69%, so the modest raw-accuracy gain does not establish a strong
predictive model. HDFCBANK improved, TCS declined, and the remaining selected
configurations were unchanged in accuracy. Active artifacts were preserved.
Separate latest-date forecasts are in the experiment's `latest_predictions.csv`.


## Local prediction API

The FastAPI backend serves active-model predictions, price history, and
separately scoped historical/experimental performance. It uses the newest
complete local stock snapshot and validates its manifest before loading.

Run from the project root:

```powershell
.\venv\Scripts\python.exe -m pip install -r backend/requirements.txt
.\venv\Scripts\python.exe -B -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs to try the endpoints.
See [backend/README.md](backend/README.md) for the API contract, data selection,
configuration, dependency installation, and tests. Forecast probabilities,
five-trading-day horizons, data dates, and evaluation periods are explicit;
the API does not overwrite models or prediction reports.

## StockSight dashboard

The React/Vite dashboard connects to the local API. It includes stock selection,
interactive closing-price/volume charts, five-trading-session forecasts,
technical indicators, and separately labeled historical and experimental scores.

Build it from the repository root:

```powershell
npm --prefix frontend ci
npm --prefix frontend run build
```

Start or restart the backend, then open http://127.0.0.1:8000.
The backend serves the built dashboard and API from the same address.
For frontend development, run `npm --prefix frontend run dev` and open
http://127.0.0.1:5173 while the backend is running.
See [frontend/README.md](frontend/README.md) for configuration and browser tests.

## News sentiment

The dashboard's **News & sentiment** panel shows company-specific English
headlines from Google News RSS for the previous seven days, with publisher,
publication time (IST), article link, and positive/neutral/negative headline tone.
It loads independently of forecasts and charts.

`GET /api/v1/stocks/RELIANCE/news` exposes the same view. Install the updated
`backend/requirements.txt`, rebuild the dashboard, and restart the backend.
No API key is required; the server needs outbound HTTPS access to Google News.

VADER 3.3.2 scores only the headline after removing its publisher suffix.
Compound scores >= +0.05 are positive, <= -0.05 negative, and otherwise neutral.
These scores are not probabilities or measured accuracy. VADER is a general
English lexicon/rule-based analyzer, not a validated financial forecasting model;
it may misread financial terms, attribution, mixed news, and context. Only
headlines are analyzed, never full article bodies. News does not enter model
features, change forecasts, or generate trading recommendations.

Feeds are filtered for supported company aliases, valid dates within seven days,
and safe HTTP(S) links, then deduplicated and sorted newest first (up to 12).
Cache entries last 15 minutes. On a provider failure, the last successful fetch
may be served for up to one hour with an explicit stale warning; failures are
retried at most once per minute per stock. An empty valid feed is distinct from
an unavailable provider. Cache state is in memory and resets at restart.
Google News RSS is an external feed without an availability guarantee; article
links go through Google News and coverage may be incomplete.

Set `STOCKSIGHT_NEWS_ENABLED=false` to disable outbound news requests. All
non-news endpoints continue to use local models and data. News requests never
write market data, reports, or model artifacts.

Offline news coverage: `.\venv\Scripts\python.exe -B -m unittest discover -s tests -p test_news.py -v`.
Browser coverage is included in `npm --prefix frontend run test:e2e`.
