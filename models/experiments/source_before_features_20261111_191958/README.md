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

## Model tuning and validation

Run `python src/train_models.py`, then rerun `python src/predict.py` and `python src/verify_predictions.py` to refresh outputs for the new models.

Training uses chronological 60% / 20% / 20% boundaries. Five rows immediately before validation and test are excluded to prevent five-day targets from crossing those boundaries. Hyperparameters are tuned only within the training period using five-fold `TimeSeriesSplit(gap=5)`: 20 random candidates each for Random Forest and XGBoost, and all 12 Logistic Regression configurations. Logistic Regression scaling is fitted separately inside each training fold.

Each stock's model family is selected using validation ROC-AUC. The selected configuration is then refitted on pre-test history, excluding the five-row test gap, and evaluated on the final test period. Test results do not determine model selection. The saved model and scaler use this refit and remain compatible with prediction and verification scripts.

Outputs:
- `model_results.csv`: validation metrics for all three model families.
- `test_results.csv`: final test metrics for each selected model, plus a constant-class baseline chosen from refit data.
- `tuning_results.csv`: parameters and cross-validation scores for every candidate.
- `split_periods.csv`: actual dates and row counts of each period.
- `models/model_metadata.csv`: selected parameters, dates, and artifact paths.

The final historical period was evaluated by earlier versions of this project; this separation prevents its use in the new selection procedure, but a truly unseen performance check requires newer data. The last-100-row verification is a subset of the test period, not an additional independent holdout.
