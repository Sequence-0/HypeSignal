"""Schemas for statistical burst alerts, dynamic topic modeling, and trend ranking."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class BurstAlert(BaseModel):
    """Statistical burst detection alert for emerging terms or hashtags."""
    term: str
    term_type: str = Field(default="keyword")  # 'hashtag', 'keyword', 'mention'
    current_count: int = Field(..., ge=0)
    baseline_mean: float = Field(..., ge=0.0)
    baseline_std: float = Field(..., ge=0.0)
    z_score: float
    velocity: float
    window_start: datetime
    window_end: datetime
    is_burst: bool = Field(default=False)


class TopicRepresentation(BaseModel):
    """Semantic topic representation extracted by BERTopic."""
    topic_id: int
    name: str
    top_words: List[Tuple[str, float]] = Field(default_factory=list)
    doc_count: int = Field(default=0, ge=0)
    representative_docs: List[str] = Field(default_factory=list)


class DynamicTopicTimeline(BaseModel):
    """Temporal evolution and trajectory of a dynamic topic over chronological bins."""
    topic_id: int
    topic_name: str
    timestamps: List[datetime] = Field(default_factory=list)
    frequencies: List[int] = Field(default_factory=list)
    evolving_words: List[List[str]] = Field(default_factory=list)


class TrendRankingResult(BaseModel):
    """Composite ranking score for a trending narrative or burst event."""
    trend_id: str
    name: str
    trend_type: str = Field(default="burst_term")  # 'burst_term', 'dynamic_topic'
    composite_score: float = Field(..., ge=0.0)
    burst_acceleration: float = Field(default=0.0)
    sentiment_momentum: float = Field(default=0.0)
    participant_diversity: float = Field(default=0.0, ge=0.0, le=1.0)
    total_volume: int = Field(default=0, ge=0)
    unique_authors: int = Field(default=0, ge=0)
    top_keywords: List[str] = Field(default_factory=list)
    sample_texts: List[str] = Field(default_factory=list)


class TrendOverview(BaseModel):
    """High-level snapshot of active bursts, evolving topics, and ranked trends."""
    window_start: datetime
    window_end: datetime
    active_bursts: List[BurstAlert] = Field(default_factory=list)
    dynamic_topics: List[TopicRepresentation] = Field(default_factory=list)
    ranked_trends: List[TrendRankingResult] = Field(default_factory=list)
