"""StockSight's local FastAPI application; run from the project root."""
from contextlib import asynccontextmanager
from datetime import date
import logging
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .news import NewsService, NewsUnavailable
from .schemas import Error, Health, ModelPerformance, NewsFeed, PerformanceOverview, Prediction, PriceHistory, StockList
from .service import ServiceUnavailable, StockService, UnknownStock

LOGGER = logging.getLogger(__name__)
Symbol = Annotated[str, Path(min_length=1, max_length=24, pattern=r"^[A-Za-z][A-Za-z0-9]*(?:\.[Nn][Ss])?$")]
ERROR_RESPONSES = {404: {"model": Error}, 503: {"model": Error}}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_environment()

    @asynccontextmanager
    async def lifespan(application):
        application.state.service = None
        application.state.news = NewsService(enabled=settings.news_enabled)
        application.state.startup_error = None
        try:
            application.state.service = StockService(settings)
        except ServiceUnavailable as exc:
            application.state.startup_error = str(exc)
            LOGGER.exception("StockSight readiness check failed")
        yield
        application.state.service = None

    application = FastAPI(
        title="StockSight API", version="0.2.0", lifespan=lifespan,
        description="Local five-trading-day stock forecasts, OHLCV history, and clearly scoped evaluation results. "
                    "Predictions use active saved models; experimental scores are reported separately.",
    )
    application.add_middleware(
        CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=False,
        allow_methods=["GET"], allow_headers=["Accept", "Content-Type"],
    )

    @application.exception_handler(UnknownStock)
    async def unknown_stock_handler(request: Request, exc: UnknownStock):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @application.exception_handler(ServiceUnavailable)
    async def unavailable_handler(request: Request, exc: ServiceUnavailable):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @application.exception_handler(NewsUnavailable)
    async def news_unavailable_handler(request: Request, exc: NewsUnavailable):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    def get_service(request: Request) -> StockService:
        service = getattr(request.app.state, "service", None)
        if service is None:
            raise HTTPException(status_code=503, detail=getattr(request.app.state, "startup_error", None)
                                or "The local prediction service is not ready.")
        return service

    Service = Annotated[StockService, Depends(get_service)]

    dashboard = settings.project_root / "frontend" / "dist"
    if (dashboard / "assets").is_dir():
        application.mount("/assets", StaticFiles(directory=dashboard / "assets"), name="dashboard-assets")

    @application.get("/", include_in_schema=False)
    def root():
        if (dashboard / "index.html").is_file():
            return FileResponse(dashboard / "index.html", headers={"Cache-Control": "no-cache"})
        return {"name": "StockSight API", "docs": "/docs", "api": "/api/v1"}

    @application.get("/health", response_model=Health, responses={503: {"model": Health}}, tags=["Health"])
    def health(request: Request):
        service = getattr(request.app.state, "service", None)
        if service is None:
            response = Health(status="unavailable", stocks_loaded=0,
                              detail=getattr(request.app.state, "startup_error", None) or "Service is starting.")
            return JSONResponse(status_code=503, content=response.model_dump(mode="json"))
        return service.health()

    @application.get("/api/v1/stocks", response_model=StockList, responses=ERROR_RESPONSES, tags=["Stocks"])
    def stocks(service: Service):
        return service.stock_list()

    @application.get("/api/v1/stocks/{symbol}/prediction", response_model=Prediction,
                     responses=ERROR_RESPONSES, tags=["Predictions"])
    def prediction(symbol: Symbol, service: Service):
        return service.prediction(symbol)

    @application.get("/api/v1/stocks/{symbol}/news", response_model=NewsFeed,
                     responses=ERROR_RESPONSES, tags=["News"])
    def stock_news(symbol: Symbol, request: Request):
        """Company-specific headline tone, independent of prediction readiness."""
        return request.app.state.news.get(symbol)

    @application.get("/api/v1/stocks/{symbol}/history", response_model=PriceHistory,
                     responses=ERROR_RESPONSES, tags=["History"])
    def history(symbol: Symbol, service: Service, start: date | None = None, end: date | None = None,
                limit: Annotated[int, Query(ge=1, le=3000)] = 180):
        """Inclusive date filters; return the most recent matching rows in ascending order."""
        if start is not None and end is not None and start > end:
            raise HTTPException(status_code=422, detail="start must be on or before end.")
        return service.history(symbol, start, end, limit)

    @application.get("/api/v1/stocks/{symbol}/performance", response_model=ModelPerformance,
                     responses=ERROR_RESPONSES, tags=["Performance"])
    def stock_performance(symbol: Symbol, service: Service):
        return service.performance(symbol)

    @application.get("/api/v1/performance", response_model=PerformanceOverview,
                     responses=ERROR_RESPONSES, tags=["Performance"])
    def performance(service: Service):
        return service.performance_overview()

    return application


app = create_app()
