"""Real-time trend detection and dynamic topic modeling routes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query

from hypesignal.api.deps import get_db, get_trends
from hypesignal.storage.duckdb_manager import DuckDBManager
from hypesignal.trends.schemas import BurstAlert, DynamicTopicTimeline, TopicRepresentation, TrendOverview
from hypesignal.trends.trends_engine import TrendsEngine

router = APIRouter(prefix="/trends", tags=["Real-Time Trends & Dynamic Topics"])


@router.get("/overview", response_model=TrendOverview)
def get_trend_overview(
    window_duration_minutes: int = Query(default=60, ge=5, le=1440, description="Duration of observation window in minutes"),
    num_baseline_windows: int = Query(default=5, ge=1, le=24, description="Number of baseline historical windows"),
    z_threshold: float = Query(default=2.5, ge=0.5, le=10.0, description="Z-score threshold for burst alerts"),
    min_count: int = Query(default=3, ge=1, le=100, description="Minimum frequency count in current window"),
    topic_doc_limit: int = Query(default=500, ge=10, le=5000, description="Max recent posts to fit into dynamic topics"),
    trends: TrendsEngine = Depends(get_trends),
    db: DuckDBManager = Depends(get_db),
) -> TrendOverview:
    """Run end-to-end dual-tier trend detection: Tier-1 burst alerts + Tier-2 dynamic topic modeling."""
    return trends.analyze_trends(
        db=db,
        window_duration_minutes=window_duration_minutes,
        num_baseline_windows=num_baseline_windows,
        z_threshold=z_threshold,
        min_count=min_count,
        topic_doc_limit=topic_doc_limit,
    )


@router.get("/bursts", response_model=List[BurstAlert])
def get_active_bursts(
    window_duration_minutes: int = Query(default=60, ge=5, le=1440),
    num_baseline_windows: int = Query(default=5, ge=1, le=24),
    z_threshold: float = Query(default=2.5, ge=0.5, le=10.0),
    min_count: int = Query(default=3, ge=1, le=100),
    only_active: bool = Query(default=True, description="Filter only alerts exceeding Z-score threshold"),
    trends: TrendsEngine = Depends(get_trends),
    db: DuckDBManager = Depends(get_db),
) -> List[BurstAlert]:
    """Retrieve Tier-1 statistical burstiness alerts for emerging terms and hashtags."""
    max_ts_row = db.con.execute("SELECT MAX(timestamp) FROM posts;").fetchone()
    if max_ts_row and max_ts_row[0]:
        current_end = max_ts_row[0]
        if current_end.tzinfo is None:
            current_end = current_end.replace(tzinfo=timezone.utc)
    else:
        current_end = datetime.now(timezone.utc)

    alerts = trends.burst_detector.detect_bursts_from_duckdb(
        db=db,
        current_window_end=current_end,
        window_duration_minutes=window_duration_minutes,
        num_baseline_windows=num_baseline_windows,
        z_threshold=z_threshold,
        min_count=min_count,
        target="all",
    )

    if only_active:
        return [b for b in alerts if b.is_burst]
    return alerts


@router.get("/topics")
def get_dynamic_topics(
    hours_back: int = Query(default=24, ge=1, le=720, description="Hours of historical posts to fit"),
    limit: int = Query(default=500, ge=10, le=5000),
    nr_bins: int = Query(default=5, ge=2, le=20),
    trends: TrendsEngine = Depends(get_trends),
    db: DuckDBManager = Depends(get_db),
) -> Dict[str, Any]:
    """Fit Tier-2 BERTopic temporal dynamic topic models and retrieve topic trajectories."""
    max_ts_row = db.con.execute("SELECT MAX(timestamp) FROM posts;").fetchone()
    if max_ts_row and max_ts_row[0]:
        end_time = max_ts_row[0]
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
    else:
        end_time = datetime.now(timezone.utc)

    start_time = end_time - timedelta(hours=hours_back)

    topic_reps, timelines = trends.topic_modeler.fit_from_duckdb(
        db=db,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        nr_bins=nr_bins,
    )

    return {
        "topics": [t.model_dump() for t in (topic_reps or [])],
        "timelines": [tl.model_dump() for tl in (timelines or [])],
        "total_topics": len(topic_reps or []),
    }
