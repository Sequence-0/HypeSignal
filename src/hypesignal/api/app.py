"""FastAPI application factory and lifecycle coordination for HypeSignal."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from hypesignal.api.routes.connectors import router as connectors_router
from hypesignal.api.routes.demographics import router as demographics_router
from hypesignal.api.routes.network import router as network_router
from hypesignal.api.routes.sentiment import router as sentiment_router
from hypesignal.api.routes.timeline import router as timeline_router
from hypesignal.api.routes.trends import router as trends_router
from hypesignal.api.schemas import HealthResponse
from hypesignal.connectors.base import PlatformConnector
from hypesignal.connectors.reddit import RedditConnector
from hypesignal.connectors.telegram import TelegramConnector
from hypesignal.connectors.twitter import TwitterConnector
from hypesignal.connectors.youtube import YouTubeConnector
from hypesignal.demographics.demographics_engine import DemographicsEngine
from hypesignal.network.network_engine import NetworkEngine
from hypesignal.nlp.engine import MultiDimensionalSentimentEngine
from hypesignal.nlp.temporal_sentiment import TemporalSentimentTracker
from hypesignal.storage.duckdb_manager import DuckDBManager
from hypesignal.storage.vector_store import VectorStoreManager
from hypesignal.timeline.timeline_manager import TimelineManager
from hypesignal.trends.trends_engine import TrendsEngine

logger = logging.getLogger(__name__)


def create_app(
    db: Optional[DuckDBManager] = None,
    vector_store: Optional[VectorStoreManager] = None,
    timeline: Optional[TimelineManager] = None,
    nlp: Optional[MultiDimensionalSentimentEngine] = None,
    temporal_sentiment: Optional[TemporalSentimentTracker] = None,
    demographics: Optional[DemographicsEngine] = None,
    trends: Optional[TrendsEngine] = None,
    network: Optional[NetworkEngine] = None,
    connectors: Optional[Dict[str, PlatformConnector]] = None,
    device: Optional[str] = None,
    auto_connect: bool = True,
    cors_origins: Optional[list[str]] = None,
    allow_credentials: bool = False,
    title: str = "HypeSignal Analytics API",
    version: str = "0.1.0",
) -> FastAPI:
    """Create and configure the unified HypeSignal FastAPI application."""
    app_db = db
    app_vector = vector_store
    app_timeline = timeline
    app_nlp = nlp
    app_tracker = temporal_sentiment
    app_demographics = demographics
    app_trends = trends
    app_network = network
    app_connectors = connectors

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal app_db, app_vector, app_timeline, app_nlp, app_tracker, app_demographics, app_trends, app_network, app_connectors

        # 1. Initialize DuckDB Storage
        if app_db is None:
            db_path = os.getenv("HYPESIGNAL_DB_PATH", ":memory:")
            logger.info("Initializing default DuckDBManager at '%s'...", db_path)
            app_db = DuckDBManager(db_path=db_path)
        app.state.db = app_db

        # 2. Vector Store (optional / embedded)
        if app_vector is None:
            try:
                app_vector = VectorStoreManager()
            except Exception as e:
                logger.warning("VectorStoreManager initialization skipped: %s", e)
        app.state.vector_store = app_vector

        # 3. Timeline Manager
        if app_timeline is None:
            app_timeline = TimelineManager(db=app_db)
        app.state.timeline = app_timeline

        # 4. Multi-dimensional Sentiment & Emotion Engine
        if app_nlp is None:
            logger.info("Initializing MultiDimensionalSentimentEngine...")
            app_nlp = MultiDimensionalSentimentEngine(device=device)
        app.state.nlp = app_nlp

        # 5. Temporal Sentiment Tracker
        if app_tracker is None:
            app_tracker = TemporalSentimentTracker(engine=app_nlp, db=app_db)
        app.state.temporal_sentiment = app_tracker

        # 6. Demographics & Persona Engine
        if app_demographics is None:
            logger.info("Initializing DemographicsEngine...")
            app_demographics = DemographicsEngine(vector_store=app_vector, device=device)
        app.state.demographics = app_demographics

        # 7. Real-Time Trends & Topic Detection Engine
        if app_trends is None:
            logger.info("Initializing TrendsEngine...")
            app_trends = TrendsEngine(device=device)
        app.state.trends = app_trends

        # 8. Link Analysis & Network Topology Engine
        if app_network is None:
            logger.info("Initializing NetworkEngine...")
            app_network = NetworkEngine()
        app.state.network = app_network

        # 9. Platform Connectors
        if app_connectors is None:
            app_connectors = {
                "twitter": TwitterConnector(),
                "reddit": RedditConnector(),
                "youtube": YouTubeConnector(),
                "telegram": TelegramConnector(),
            }
        app.state.connectors = app_connectors

        # Auto-connect enabled connectors
        if auto_connect:
            for name, conn in app_connectors.items():
                if conn.config.enabled:
                    conn.connect()

        logger.info("HypeSignal FastAPI application initialized successfully.")
        yield

        # Teardown
        logger.info("Shutting down HypeSignal platform connectors...")
        for name, conn in app_connectors.items():
            if conn.is_connected():
                conn.disconnect()

    app = FastAPI(
        title=title,
        version=version,
        description="Offline-first AI-driven Social Media Analytics Framework API",
        lifespan=lifespan,
    )

    # CORS Middleware: Explicit origin allowlist (disallow wildcard credentials)
    safe_origins = cors_origins or [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]
    has_wildcard = "*" in safe_origins
    cred_allowed = False if has_wildcard else allow_credentials

    app.add_middleware(
        CORSMiddleware,
        allow_origins=safe_origins,
        allow_credentials=cred_allowed,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server exception: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred.", "error": str(exc)},
        )

    # Healthcheck endpoints
    @app.get("/health", response_model=HealthResponse, tags=["System Health"])
    @app.get("/api/v1/health", response_model=HealthResponse, tags=["System Health"])
    def healthcheck(request: Request) -> HealthResponse:
        """Healthcheck returning operational state of all underlying analytical engines."""
        return HealthResponse(
            status="ok",
            version=version,
            engines={
                "duckdb": "online" if getattr(request.app.state, "db", None) else "offline",
                "timeline": "online" if getattr(request.app.state, "timeline", None) else "offline",
                "nlp": "online" if getattr(request.app.state, "nlp", None) else "offline",
                "demographics": "online" if getattr(request.app.state, "demographics", None) else "offline",
                "trends": "online" if getattr(request.app.state, "trends", None) else "offline",
                "network": "online" if getattr(request.app.state, "network", None) else "offline",
                "connectors": f"{len(getattr(request.app.state, 'connectors', {}))} registered",
            },
        )

    # Register API v1 Routers
    api_v1_prefix = "/api/v1"
    app.include_router(timeline_router, prefix=api_v1_prefix)
    app.include_router(sentiment_router, prefix=api_v1_prefix)
    app.include_router(demographics_router, prefix=api_v1_prefix)
    app.include_router(trends_router, prefix=api_v1_prefix)
    app.include_router(network_router, prefix=api_v1_prefix)
    app.include_router(connectors_router, prefix=api_v1_prefix)

    return app


# Default application instance for Uvicorn / ASGI servers
app = create_app()
