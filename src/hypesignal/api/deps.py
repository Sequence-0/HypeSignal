"""FastAPI dependency providers for accessing application singletons."""

from __future__ import annotations

from typing import Dict, Optional

from fastapi import HTTPException, Request, status

from hypesignal.connectors.base import PlatformConnector
from hypesignal.demographics.demographics_engine import DemographicsEngine
from hypesignal.network.network_engine import NetworkEngine
from hypesignal.nlp.engine import MultiDimensionalSentimentEngine
from hypesignal.nlp.temporal_sentiment import TemporalSentimentTracker
from hypesignal.storage.duckdb_manager import DuckDBManager
from hypesignal.storage.vector_store import VectorStoreManager
from hypesignal.timeline.timeline_manager import TimelineManager
from hypesignal.trends.trends_engine import TrendsEngine


def get_db(request: Request) -> DuckDBManager:
    """Provide shared DuckDBManager instance."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DuckDB storage engine is not initialized.",
        )
    return db


def get_vector_store(request: Request) -> Optional[VectorStoreManager]:
    """Provide shared VectorStoreManager instance if available."""
    return getattr(request.app.state, "vector_store", None)


def get_timeline(request: Request) -> TimelineManager:
    """Provide shared TimelineManager instance."""
    timeline = getattr(request.app.state, "timeline", None)
    if timeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Timeline manager is not initialized.",
        )
    return timeline


def get_nlp(request: Request) -> MultiDimensionalSentimentEngine:
    """Provide shared MultiDimensionalSentimentEngine instance."""
    nlp = getattr(request.app.state, "nlp", None)
    if nlp is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Multi-dimensional NLP engine is not initialized.",
        )
    return nlp


def get_temporal_sentiment(request: Request) -> TemporalSentimentTracker:
    """Provide shared TemporalSentimentTracker instance."""
    tracker = getattr(request.app.state, "temporal_sentiment", None)
    if tracker is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Temporal sentiment tracker is not initialized.",
        )
    return tracker


def get_demographics(request: Request) -> DemographicsEngine:
    """Provide shared DemographicsEngine instance."""
    demographics = getattr(request.app.state, "demographics", None)
    if demographics is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Demographics engine is not initialized.",
        )
    return demographics


def get_trends(request: Request) -> TrendsEngine:
    """Provide shared TrendsEngine instance."""
    trends = getattr(request.app.state, "trends", None)
    if trends is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trends & Topic engine is not initialized.",
        )
    return trends


def get_network(request: Request) -> NetworkEngine:
    """Provide shared NetworkEngine instance."""
    network = getattr(request.app.state, "network", None)
    if network is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Network & Link Analysis engine is not initialized.",
        )
    return network


def get_connectors(request: Request) -> Dict[str, PlatformConnector]:
    """Provide dictionary of registered platform connectors."""
    connectors = getattr(request.app.state, "connectors", None)
    if connectors is None:
        return {}
    return connectors
