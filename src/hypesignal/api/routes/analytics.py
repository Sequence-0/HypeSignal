"""Unified REST Analytics API endpoints (Component F).

Exposes discussion thread reconstruction, follower demographic profiling with
min_group_size privacy suppression, trend kinematics forecasting, narrative drift,
cross-segment diffusion trajectories, and bridge KOL leaderboard.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request, status

from pydantic import BaseModel, Field

from hypesignal.demographics.audience_profiler import InfluencerAudienceProfile, InfluencerAudienceProfiler
from hypesignal.network.bridge_kols import BridgeKOLLeaderboard
from hypesignal.network.cross_segment_diffusion import CrossSegmentDiffusionReport, CrossSegmentDiffusionTracker
from hypesignal.nlp.thread_sentiment import ThreadSentimentAnalyzer
from hypesignal.timeline.thread_manager import ConversationThreadManager
from hypesignal.trends.narrative_drift import NarrativeDriftAlert, NarrativeDriftTracker
from hypesignal.trends.trend_forecaster import TrendForecast, TrendForecaster

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Unified Analytics"])


class ConversationAnalyticsResponse(BaseModel):
    """Reconstructed conversation metrics, sentiment dynamics, and tree representation."""
    conversation_id: str
    metrics: Dict[str, Any]
    sentiment_dynamics: Dict[str, Any] = Field(default_factory=dict)
    thread_tree: Dict[str, Any]


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationAnalyticsResponse,
    summary="Reconstructed conversation tree, structural metrics, and emotional trajectory",
)
def get_conversation_analytics(conversation_id: str, request: Request) -> ConversationAnalyticsResponse:
    """Reconstruct a discussion thread tree and compute structural and sentiment dynamics."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database offline")

    thread_mgr = ConversationThreadManager(db=db)
    thread = thread_mgr.reconstruct_thread(conversation_id)
    if thread is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation thread '{conversation_id}' not found",
        )

    metrics = {
        "total_posts": thread.total_posts,
        "total_replies": thread.total_replies,
        "max_depth": thread.max_depth,
        "avg_branching_factor": thread.avg_branching_factor,
        "participant_count": len(thread.participant_ids),
        "participant_ids": thread.participant_ids,
        "duration_seconds": thread.duration_seconds,
        "mean_reply_latency_seconds": thread.mean_reply_latency_seconds,
        "depth_distribution": thread.depth_distribution,
    }
    root_node = thread.root_node
    nlp_engine = getattr(request.app.state, "nlp", None)
    sentiment_dynamics = None

    if nlp_engine is not None:
        try:
            sentiment_analyzer = ThreadSentimentAnalyzer(engine=nlp_engine)
            analysis = sentiment_analyzer.analyze_thread_from_db(
                root_post_id=conversation_id,
                thread_manager=thread_mgr,
            )
            if analysis is not None:
                sentiment_dynamics = analysis.model_dump()
        except Exception as e:
            logger.warning("Thread sentiment analysis failed for '%s': %s", conversation_id, e)

    # Fallback to querying stored post_analytics if NLP engine was offline or analysis not run
    if sentiment_dynamics is None:
        try:
            row = db.con.execute(
                "SELECT effective_polarity, sentiment_score, primary_emotion FROM post_analytics WHERE post_id = ?;",
                [conversation_id],
            ).fetchone()
            if row:
                sentiment_dynamics = {
                    "root_post_id": conversation_id,
                    "root_polarity": row[0],
                    "root_sentiment_score": float(row[1]),
                    "root_primary_emotion": row[2],
                    "comment_mean_sentiment": 0.0,
                    "polarity_drift": 0.0,
                    "controversy_index": 0.0,
                    "hostility_velocity": 0.0,
                    "supportive_ratio": 1.0 if row[0] == "positive" else 0.0,
                    "against_ratio": 1.0 if row[0] == "negative" else 0.0,
                    "neutral_ratio": 1.0 if row[0] == "neutral" else 0.0,
                    "total_comments_analyzed": metrics["total_replies"],
                    "dominant_thread_emotion": row[2],
                    "emotional_trajectory": [],
                }
        except Exception as e:
            logger.debug("Fallback post_analytics lookup skipped: %s", e)

    return ConversationAnalyticsResponse(
        conversation_id=conversation_id,
        metrics=metrics,
        sentiment_dynamics=sentiment_dynamics or {},
        thread_tree=root_node.model_dump(),
    )


@router.get(
    "/audience/{user_id}",
    response_model=InfluencerAudienceProfile,
    summary="Follower audience demographics with privacy threshold filtering",
)
def get_audience_analytics(
    user_id: str,
    request: Request,
    min_group_size: int = Query(default=5, ge=1, description="Minimum cohort size for privacy suppression"),
) -> InfluencerAudienceProfile:
    """Aggregate follower demographic distributions with privacy threshold suppression."""
    db = getattr(request.app.state, "db", None)
    demographics_engine = getattr(request.app.state, "demographics", None)

    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database offline")

    profiler = InfluencerAudienceProfiler(
        db=db,
        demographics_engine=demographics_engine,
        min_group_size=min_group_size,
    )

    return profiler.profile_influencer_audience(influencer_id=user_id)


@router.get(
    "/trends/forecast",
    response_model=List[TrendForecast],
    summary="Predictive trend velocity, acceleration, and lifecycle states",
)
def get_trend_forecast(
    request: Request,
    term: Optional[str] = Query(default=None, description="Specific term/hashtag to forecast"),
    window_minutes: float = Query(default=60.0, gt=0, description="Window duration in minutes"),
    limit: int = Query(default=20, ge=1, le=100),
) -> List[TrendForecast]:
    """Compute 1st derivative velocity, 2nd derivative acceleration, and lifecycle state."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database offline")

    forecaster = TrendForecaster()

    if term:
        kinematics = forecaster.forecast_trend_kinematics(
            db=db,
            term=term,
            window_duration_minutes=window_minutes,
        )
        return [kinematics]

    # If no specific term requested, discover active terms in DuckDB using subquery
    query = """
        SELECT tag, count(*) as cnt 
        FROM (
            SELECT unnest(hashtags) as tag 
            FROM posts 
            WHERE timestamp >= (SELECT COALESCE(MAX(timestamp), CURRENT_TIMESTAMP) FROM posts) - INTERVAL (? || ' minutes')
              AND hashtags IS NOT NULL
        )
        GROUP BY tag 
        ORDER BY cnt DESC 
        LIMIT ?;
    """
    try:
        rows = db.con.execute(query, [int(window_minutes * 2), limit]).fetchall()
        terms = [r[0] for r in rows if r[0]]
    except Exception as e:
        logger.debug("Active terms extraction skipped: %s", e)
        terms = []

    results: List[TrendForecast] = []
    for t in terms:
        kin = forecaster.forecast_trend_kinematics(
            db=db,
            term=t,
            window_duration_minutes=window_minutes,
        )
        results.append(kin)

    results.sort(key=lambda x: x.virality_potential_score, reverse=True)
    return results


@router.get(
    "/trends/narrative-drift",
    response_model=NarrativeDriftAlert,
    summary="Semantic centroid cosine drift and sentiment inversion detection",
)
def get_narrative_drift(
    request: Request,
    term: str = Query(..., description="Topic or term to analyze for narrative drift"),
    window_minutes: float = Query(default=60.0, gt=0, description="Window duration in minutes"),
) -> NarrativeDriftAlert:
    """Detect semantic concept drift and sentiment inversions between sliding time windows."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database offline")

    # Reuse cached tracker from app.state or topic modeler embedding model (M1)
    tracker = getattr(request.app.state, "narrative_drift", None)
    if tracker is None:
        trends_engine = getattr(request.app.state, "trends", None)
        st_model = getattr(getattr(trends_engine, "topic_modeler", None), "embedding_model", None)
        tracker = NarrativeDriftTracker(sentence_transformer=st_model)
        request.app.state.narrative_drift = tracker

    return tracker.track_drift_from_duckdb(
        db=db,
        topic=term,
        window_minutes=window_minutes,
    )


@router.get(
    "/diffusion/cross-segment/{cascade_id}",
    response_model=CrossSegmentDiffusionReport,
    summary="Chronological cross-segment information diffusion sequence and sentiment drift",
)
def get_cross_segment_diffusion(
    cascade_id: str,
    request: Request,
    segment_by: str = Query(default="community", description="Dimension to segment by ('community', 'age_bracket', 'persona')"),
) -> CrossSegmentDiffusionReport:
    """Reconstruct chronological cross-segment transitions, transmission latencies, and sentiment drift."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database offline")

    tracker = CrossSegmentDiffusionTracker(db=db)
    try:
        report = tracker.track_cascade_from_db(
            db=db,
            cascade_id=cascade_id,
            segment_by=segment_by,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return report


@router.get(
    "/influencers/bridge-kols",
    response_model=BridgeKOLLeaderboard,
    summary="Ranked inter-community boundary spanners (Bridge KOLs)",
)
def get_bridge_kols(
    request: Request,
    top_k: int = Query(default=20, ge=1, le=100),
    min_neighbors: int = Query(default=2, ge=0),
    min_cross_ratio: float = Query(default=0.0, ge=0.0, le=1.0),
) -> BridgeKOLLeaderboard:
    """Identify boundary spanners bridging distinct Louvain communities."""
    network_engine = getattr(request.app.state, "network", None)
    db = getattr(request.app.state, "db", None)

    if network_engine is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Network engine offline")

    return network_engine.get_bridge_kols(
        top_k=top_k,
        min_neighbors=min_neighbors,
        min_cross_ratio=min_cross_ratio,
        db=db,
    )
