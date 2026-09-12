"""Timeline and chronological event query routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from hypesignal.api.deps import get_db, get_timeline
from hypesignal.api.schemas import (
    ActivityTimeseriesPoint,
    ActivityTimeseriesResponse,
    CascadeChronologyPoint,
    CascadeChronologyResponse,
    TimelineBoundsResponse,
)
from hypesignal.models.canonical import CanonicalPost
from hypesignal.storage.duckdb_manager import DuckDBManager
from hypesignal.timeline.timeline_manager import TimelineManager

router = APIRouter(prefix="/timeline", tags=["Timeline & Historical Ingestion"])


@router.get("/bounds", response_model=TimelineBoundsResponse)
def get_timeline_bounds(
    timeline: TimelineManager = Depends(get_timeline),
) -> TimelineBoundsResponse:
    """Retrieve the earliest and latest post timestamps present in the historical store."""
    bounds = timeline.get_time_bounds()
    return TimelineBoundsResponse(
        earliest=bounds["earliest"],
        latest=bounds["latest"],
        total_posts=bounds["total_posts"],
    )


@router.get("/slice", response_model=List[CanonicalPost])
def get_timeline_slice(
    start_time: datetime = Query(..., description="Start of observation interval (UTC)"),
    end_time: datetime = Query(..., description="End of observation interval (UTC)"),
    platform: Optional[str] = Query(default=None, description="Optional platform filter (e.g. twitter)"),
    limit: int = Query(default=100, ge=1, le=1000, description="Max posts to return"),
    timeline: TimelineManager = Depends(get_timeline),
) -> List[CanonicalPost]:
    """Retrieve chronological slice of posts within [start_time, end_time]."""
    df = timeline.get_timeline_slice(
        start_time=start_time,
        end_time=end_time,
        platform=platform,
        limit=limit,
    )
    if df.is_empty():
        return []

    # Parse dataframe records into CanonicalPost
    records = df.to_dicts()
    posts: List[CanonicalPost] = []
    for r in records:
        try:
            posts.append(CanonicalPost.model_validate(r))
        except Exception:
            # Fallback for DuckDB struct columns or types
            posts.append(
                CanonicalPost(
                    id=str(r["id"]),
                    platform=r["platform"],
                    author_id=str(r["author_id"]),
                    author_screen_name=r.get("author_screen_name"),
                    text=r["text"],
                    timestamp=r["timestamp"],
                )
            )
    return posts


@router.get("/timeseries", response_model=ActivityTimeseriesResponse)
def get_activity_timeseries(
    interval: str = Query(default="1 hour", description="Time bucket interval (e.g. '1 hour', '1 day')"),
    start_time: Optional[datetime] = Query(default=None, description="Optional start datetime"),
    end_time: Optional[datetime] = Query(default=None, description="Optional end datetime"),
    timeline: TimelineManager = Depends(get_timeline),
) -> ActivityTimeseriesResponse:
    """Aggregate historical post volume into regular chronological time buckets."""
    try:
        df = timeline.get_activity_timeseries(
            interval=interval,
            start_time=start_time,
            end_time=end_time,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    points: List[ActivityTimeseriesPoint] = []
    if not df.is_empty():
        for row in df.to_dicts():
            points.append(
                ActivityTimeseriesPoint(
                    bucket=row["bucket"],
                    post_count=int(row["post_count"]),
                )
            )

    return ActivityTimeseriesResponse(
        interval=interval,
        total_buckets=len(points),
        points=points,
    )


@router.get("/cascade/{cascade_id:path}", response_model=CascadeChronologyResponse)
def get_cascade_chronology(
    cascade_id: str,
    timeline: TimelineManager = Depends(get_timeline),
) -> CascadeChronologyResponse:
    """Retrieve chronological adoption trace for a specific diffusion cascade."""
    df = timeline.get_cascade_chronology(cascade_id=cascade_id)
    if df.is_empty():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cascade '{cascade_id}' not found.",
        )

    events: List[CascadeChronologyPoint] = []
    for r in df.to_dicts():
        events.append(
            CascadeChronologyPoint(
                post_id=str(r["post_id"]),
                user_id=str(r["user_id"]),
                user_screen_name=r.get("user_screen_name"),
                timestamp=r["timestamp"],
                adoption_order=int(r["adoption_order"]),
                seconds_since_origin=float(r["seconds_since_origin"]) if "seconds_since_origin" in r else None,
            )
        )

    return CascadeChronologyResponse(
        cascade_id=cascade_id,
        total_events=len(events),
        events=events,
    )


@router.get("/user/{user_id}", response_model=List[CanonicalPost])
def get_user_timeline(
    user_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    timeline: TimelineManager = Depends(get_timeline),
) -> List[CanonicalPost]:
    """Retrieve chronological post history for an individual user."""
    df = timeline.get_user_timeline(user_id=user_id, limit=limit)
    if df.is_empty():
        return []

    posts: List[CanonicalPost] = []
    for r in df.to_dicts():
        posts.append(
            CanonicalPost(
                id=str(r["id"]),
                platform=r["platform"],
                author_id=str(r["author_id"]),
                author_screen_name=r.get("author_screen_name"),
                text=r["text"],
                timestamp=r["timestamp"],
            )
        )
    return posts
