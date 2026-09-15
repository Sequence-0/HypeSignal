"""Timeline and chronological event query routes."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from hypesignal.api.deps import get_timeline
from hypesignal.api.schemas import (
    ActivityTimeseriesPoint,
    ActivityTimeseriesResponse,
    CascadeChronologyPoint,
    CascadeChronologyResponse,
    TimelineBoundsResponse,
)
from hypesignal.models.canonical import CanonicalPost, PostMetrics
from hypesignal.timeline.timeline_manager import TimelineManager


def _record_to_canonical_post(r: Dict[str, Any]) -> CanonicalPost:
    """Safely deserialize JSON fields from DuckDB and reconstruct a complete CanonicalPost."""
    r_dict = dict(r)
    for json_field in ("metrics", "extra_metadata", "urls", "hashtags", "mentions", "media_urls"):
        val = r_dict.get(json_field)
        if isinstance(val, str):
            try:
                r_dict[json_field] = json.loads(val)
            except Exception:
                pass

    try:
        return CanonicalPost.model_validate(r_dict)
    except Exception:
        # Fallback if types or struct shape still differ
        metrics_val = r_dict.get("metrics")
        if isinstance(metrics_val, dict):
            try:
                metrics_obj = PostMetrics.model_validate(metrics_val)
            except Exception:
                metrics_obj = PostMetrics()
        elif isinstance(metrics_val, PostMetrics):
            metrics_obj = metrics_val
        else:
            metrics_obj = PostMetrics()

        return CanonicalPost(
            id=str(r_dict["id"]),
            platform=r_dict["platform"],
            author_id=str(r_dict["author_id"]),
            author_screen_name=r_dict.get("author_screen_name"),
            text=r_dict.get("text") or "",
            timestamp=r_dict["timestamp"],
            parent_id=r_dict.get("parent_id"),
            reply_to_user_id=r_dict.get("reply_to_user_id"),
            source_client=r_dict.get("source_client"),
            urls=r_dict.get("urls") if isinstance(r_dict.get("urls"), list) else [],
            hashtags=r_dict.get("hashtags") if isinstance(r_dict.get("hashtags"), list) else [],
            mentions=r_dict.get("mentions") if isinstance(r_dict.get("mentions"), list) else [],
            media_urls=r_dict.get("media_urls") if isinstance(r_dict.get("media_urls"), list) else [],
            metrics=metrics_obj,
            extra_metadata=r_dict.get("extra_metadata") if isinstance(r_dict.get("extra_metadata"), dict) else {},
        )

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
    start_time: Optional[datetime] = Query(default=None, description="Start of observation interval (UTC)"),
    end_time: Optional[datetime] = Query(default=None, description="End of observation interval (UTC)"),
    platform: Optional[str] = Query(default=None, description="Optional platform filter (e.g. twitter, youtube, reddit)"),
    keyword: Optional[str] = Query(default=None, description="Optional keyword or phrase search across post text and hashtags"),
    parent_id: Optional[str] = Query(default=None, description="Optional filter by root post, video ID, or thread ID"),
    author_id: Optional[str] = Query(default=None, description="Optional filter by author user ID"),
    limit: int = Query(default=100, ge=1, le=1000, description="Max posts to return"),
    timeline: TimelineManager = Depends(get_timeline),
) -> List[CanonicalPost]:
    """Retrieve chronological slice of posts with keyword, video/thread, platform, and time window filters."""
    df = timeline.get_timeline_slice(
        start_time=start_time,
        end_time=end_time,
        platform=platform,
        keyword=keyword,
        parent_id=parent_id,
        author_id=author_id,
        limit=limit,
    )
    if df.is_empty():
        return []

    return [_record_to_canonical_post(r) for r in df.to_dicts()]


@router.get("/timeseries", response_model=ActivityTimeseriesResponse)
def get_activity_timeseries(
    interval: str = Query(default="1 hour", description="Time bucket interval (e.g. '1 hour', '1 day')"),
    start_time: Optional[datetime] = Query(default=None, description="Optional start datetime"),
    end_time: Optional[datetime] = Query(default=None, description="Optional end datetime"),
    platform: Optional[str] = Query(default=None, description="Optional platform filter (e.g. youtube, twitter)"),
    keyword: Optional[str] = Query(default=None, description="Optional keyword or hashtag filter"),
    timeline: TimelineManager = Depends(get_timeline),
) -> ActivityTimeseriesResponse:
    """Aggregate historical post volume into regular chronological time buckets with optional keyword and platform filtering."""
    try:
        df = timeline.get_activity_timeseries(
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            platform=platform,
            keyword=keyword,
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

    return [_record_to_canonical_post(r) for r in df.to_dicts()]
