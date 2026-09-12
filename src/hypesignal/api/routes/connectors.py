"""Platform connector status and ingestion management routes."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query, status

from hypesignal.api.deps import get_connectors, get_db
from hypesignal.api.schemas import (
    ConnectorPollRequest,
    ConnectorPollResponse,
    ConnectorsStatusResponse,
)
from hypesignal.connectors.base import PlatformConnector
from hypesignal.models.enums import PlatformType
from hypesignal.storage.duckdb_manager import DuckDBManager

router = APIRouter(prefix="/connectors", tags=["Live Platform Ingestion Connectors"])


@router.get("/status", response_model=ConnectorsStatusResponse)
def get_connectors_status(
    connectors: Dict[str, PlatformConnector] = Depends(get_connectors),
) -> ConnectorsStatusResponse:
    """Retrieve operational status, rate limit metrics, and counters for all connectors."""
    info_map = {
        name: conn.get_info() for name, conn in connectors.items()
    }
    return ConnectorsStatusResponse(connectors=info_map)


@router.post("/{platform}/poll", response_model=ConnectorPollResponse)
def poll_connector(
    platform: str,
    req: ConnectorPollRequest,
    ingest: bool = Query(default=False, description="Whether to immediately ingest polled posts into DuckDB"),
    connectors: Dict[str, PlatformConnector] = Depends(get_connectors),
    db: DuckDBManager = Depends(get_db),
) -> ConnectorPollResponse:
    """Poll a specific platform connector (e.g. twitter, reddit, youtube, telegram)."""
    plat_key = platform.lower().strip()
    connector = connectors.get(plat_key)

    if connector is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Connector '{platform}' not found. Available connectors: {list(connectors.keys())}",
        )

    posts = connector.poll(query=req.query, limit=req.limit)

    if ingest and posts:
        db.insert_posts(posts)

    return ConnectorPollResponse(
        platform=connector.platform,
        count=len(posts),
        posts=posts,
    )
