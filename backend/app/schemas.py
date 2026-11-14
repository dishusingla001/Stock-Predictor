"""Explicit response contracts keep forecast probabilities separate from evaluations."""
from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Probability = Annotated[float, Field(ge=0, le=1)]


class Response(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DataSource(Response):
    kind: Literal["downloaded_snapshot", "project_raw", "configured_directory"]
    as_of_exclusive: date | None = None


class Health(Response):
    status: Literal["ok", "unavailable"]
    stocks_loaded: int
    latest_data_date: date | None = None
    data_age_calendar_days: int | None = None
    is_stale: bool | None = None
    source: DataSource | None = None
    detail: str | None = None


class Stock(Response):
    symbol: str
    name: str
    exchange: Literal["NSE"] = "NSE"
    currency: Literal["INR"] = "INR"
    model: str
    data_date: date
    close: float
    data_age_calendar_days: int
    is_stale: bool


class StockList(Response):
    stocks: list[Stock]
    count: int
    source: DataSource


class Prediction(Response):
    symbol: str
    model: str
    model_id: str
    model_scope: Literal["active"] = "active"
    model_trained_through: date
    data_date: date
    close: float
    currency: Literal["INR"] = "INR"
    direction: Literal["UP", "DOWN"]
    up_probability: Probability
    down_probability: Probability
    decision_threshold: Probability
    horizon_trading_days: Literal[5] = 5
    horizon_description: str = "Closing-price direction five trading sessions after the data date."
    outcome_verified: Literal[False] = False
    data_age_calendar_days: int
    is_stale: bool
    source: DataSource
    generated_at: datetime
    indicators: dict[str, float]
    probability_description: str = "Model class probabilities; these are not measured prediction accuracy."


class PricePoint(Response):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


class PriceHistory(Response):
    symbol: str
    currency: Literal["INR"] = "INR"
    latest_available_date: date
    source: DataSource
    total_matching_rows: int
    returned_rows: int
    truncated: bool
    prices: list[PricePoint]


class ModelPerformance(Response):
    symbol: str
    model: str
    model_id: str
    model_scope: Literal["active"] = "active"
    evaluation_kind: Literal["historical_test"] = "historical_test"
    start: date
    end: date
    samples: int
    horizon_trading_days: Literal[5] = 5
    model_trained_through: date
    accuracy: Probability
    precision: Probability
    recall: Probability
    f1: Probability
    roc_auc: Probability
    training_majority_baseline_accuracy: Probability
    independent_new_holdout: Literal[False] = False
    evaluation_note: str = "This historical period has already been inspected during development."


class ExperimentalEvaluation(Response):
    experiment_id: str
    model_scope: Literal["experimental"] = "experimental"
    evaluation_kind: str
    start: date
    end: date
    stock_only_mean_accuracy: Probability
    selected_mean_accuracy: Probability
    selected_mean_balanced_accuracy: Probability | None = None
    note: str = "Separate experimental models; these scores do not evaluate the active models or latest refitted snapshots."


class PerformanceOverview(Response):
    active_models: list[ModelPerformance]
    mean_active_historical_accuracy: Probability
    experiments: list[ExperimentalEvaluation]
    interpretation: str = "Compare measured scores only within their stated dates and model scope. Forecast probabilities are separate."


class Error(Response):
    detail: str

# News is an independent view of headline tone, never a forecast probability.
class NewsArticle(Response):
    id: str
    title: str
    source: str
    url: str
    published_at: datetime
    sentiment: Literal["positive", "neutral", "negative"]
    sentiment_score: Annotated[float, Field(ge=-1, le=1)]


class NewsCounts(Response):
    positive: int = Field(ge=0)
    neutral: int = Field(ge=0)
    negative: int = Field(ge=0)


class NewsFeed(Response):
    symbol: str
    provider: Literal["Google News RSS"] = "Google News RSS"
    window_days: Literal[7] = 7
    fetched_at: datetime
    is_cached: bool
    is_stale: bool
    articles: list[NewsArticle]
    sentiment_counts: NewsCounts
    sentiment_method: str = "VADER 3.3.2; English headline only; compound thresholds +0.05/-0.05."
    sentiment_note: str = "Headline tone only, not a probability, measured accuracy, or trading recommendation. Financial context may be misread. Does not affect the stock forecast."
    detail: str | None = None
