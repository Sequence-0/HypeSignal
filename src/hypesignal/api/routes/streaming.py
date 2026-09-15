"""Server-Sent Events (SSE) Streaming API route (Component F)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse

from hypesignal.api.streaming import EventBroadcaster

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/streaming", tags=["Real-Time Streaming"])


@router.get(
    "/events",
    summary="Real-time Server-Sent Events (SSE) stream for ingestion, enrichment, and trend alerts",
)
async def stream_events(request: Request) -> EventSourceResponse:
    """Subscribe to the real-time analytics and alerts event stream with bounded backpressure."""
    broadcaster: EventBroadcaster = getattr(request.app.state, "broadcaster", None)
    if broadcaster is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Event broadcaster not initialized",
        )

    return EventSourceResponse(
        broadcaster.sse_generator(request),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
