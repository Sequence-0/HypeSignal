"""Multi-dimensional sentiment, emotion, and temporal fluctuation routes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from hypesignal.api.deps import get_db, get_nlp, get_temporal_sentiment
from hypesignal.api.schemas import (
    SentimentAnalyzeRequest,
    SentimentAnalyzeResponse,
    TemporalSentimentPoint,
    TemporalSentimentResponse,
)
from hypesignal.nlp.engine import MultiDimensionalSentimentEngine
from hypesignal.nlp.temporal_sentiment import TemporalSentimentTracker
from hypesignal.storage.duckdb_manager import DuckDBManager

router = APIRouter(prefix="/sentiment", tags=["Multi-Dimensional Sentiment & Emotion"])


@router.post("/analyze", response_model=SentimentAnalyzeResponse)
def analyze_sentiment(
    req: SentimentAnalyzeRequest,
    nlp: MultiDimensionalSentimentEngine = Depends(get_nlp),
) -> SentimentAnalyzeResponse:
    """Analyze one or more social media texts across sentiment, emotion, irony, and stance."""
    texts: List[str] = []
    if req.text:
        texts.append(req.text)
    if req.texts:
        texts.extend(req.texts)

    if not texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either 'text' or 'texts' to analyze.",
        )

    results = nlp.analyze_multidimensional(
        texts=texts,
        target=req.stance_target,
    )

    return SentimentAnalyzeResponse(
        count=len(results),
        results=results,
    )


@router.get("/temporal", response_model=TemporalSentimentResponse)
def get_temporal_sentiment_trajectory(
    interval: str = Query(default="1 hour", description="Time interval (e.g. '1 hour', '1 day')"),
    start_time: Optional[datetime] = Query(default=None, description="Start timestamp (UTC)"),
    end_time: Optional[datetime] = Query(default=None, description="End timestamp (UTC)"),
    limit: int = Query(default=500, ge=1, le=2000, description="Max posts to analyze"),
    tracker: TemporalSentimentTracker = Depends(get_temporal_sentiment),
    db: DuckDBManager = Depends(get_db),
) -> TemporalSentimentResponse:
    """Calculate rolling chronological sentiment, sarcasm, and emotion trajectories from DuckDB."""
    if end_time is None:
        max_ts_row = db.con.execute("SELECT MAX(timestamp) FROM posts;").fetchone()
        if max_ts_row and max_ts_row[0]:
            end_time = max_ts_row[0]
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)
        else:
            end_time = datetime.now(timezone.utc)

    if start_time is None:
        start_time = end_time - timedelta(days=7)

    try:
        df = tracker.track_historical_window(
            start_time=start_time,
            end_time=end_time,
            interval=interval,
            limit=limit,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    points: List[TemporalSentimentPoint] = []
    if not df.is_empty():
        for r in df.to_dicts():
            points.append(
                TemporalSentimentPoint(
                    bucket=r["bucket"],
                    post_count=int(r["post_count"]),
                    mean_sentiment_score=float(r["mean_sentiment_score"]) if r.get("mean_sentiment_score") is not None else None,
                    positive_ratio=float(r["positive_ratio"]) if r.get("positive_ratio") is not None else None,
                    negative_ratio=float(r["negative_ratio"]) if r.get("negative_ratio") is not None else None,
                    neutral_ratio=float(r["neutral_ratio"]) if r.get("neutral_ratio") is not None else None,
                    sarcasm_rate=float(r["sarcasm_rate"]) if r.get("sarcasm_rate") is not None else None,
                    mean_irony_score=float(r["mean_irony_score"]) if r.get("mean_irony_score") is not None else None,
                    dominant_emotion=str(r["dominant_emotion"]) if r.get("dominant_emotion") is not None else None,
                )
            )

    return TemporalSentimentResponse(
        interval=interval,
        total_buckets=len(points),
        points=points,
    )
