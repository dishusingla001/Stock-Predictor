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

Rolling training windows and market-index features remain separate future
experiments so their effects can be measured independently.
