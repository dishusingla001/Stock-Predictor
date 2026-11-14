"""Filesystem and local-development configuration for the read-only API."""
from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    raw_data_dir: Path | None = None
    stale_after_days: int = 7
    news_enabled: bool = True
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")

    @classmethod
    def from_environment(cls):
        data = os.environ.get("STOCKSIGHT_DATA_DIR")
        origins = os.environ.get("STOCKSIGHT_CORS_ORIGINS")
        return cls(
            raw_data_dir=Path(data).resolve() if data else None,
            news_enabled=os.environ.get("STOCKSIGHT_NEWS_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"},
            cors_origins=tuple(item.strip() for item in origins.split(",") if item.strip())
            if origins is not None else cls.__dataclass_fields__["cors_origins"].default,
        )
