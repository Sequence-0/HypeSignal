"""Pydantic request and response schemas for HypeSignal REST API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from hypesignal.connectors.schemas import ConnectorInfo
from hypesignal.demographics.schemas import AggregateDemographics, UserProfile
from hypesignal.models.canonical import CanonicalPost
from hypesignal.models.enums import PlatformType
from hypesignal.network.schemas import CascadeTree, KOLProfile, NetworkOverview
from hypesignal.nlp.schemas import MultiDimensionalResult
from hypesignal.trends.schemas import BurstAlert, DynamicTopicTimeline, TopicRepresentation, TrendOverview


class HealthResponse(BaseModel):
    """System health and operational status response."""
    status: str = Field(default="ok")
    version: str = Field(default="0.1.0")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    engines: Dict[str, str] = Field(default_factory=dict)


class TimelineBoundsResponse(BaseModel):
    """Earliest and latest timestamps in historical timeline."""
    earliest: Optional[datetime] = None
    latest: Optional[datetime] = None
    total_posts: int = 0


class ActivityTimeseriesPoint(BaseModel):
    """Single aggregated bucket in an activity timeseries."""
    bucket: datetime
    post_count: int


class ActivityTimeseriesResponse(BaseModel):
    """Aggregated chronological post activity response."""
    interval: str
    total_buckets: int
    points: List[ActivityTimeseriesPoint]


class CascadeChronologyPoint(BaseModel):
    """Chronological event in a diffusion cascade."""
    model_config = ConfigDict(extra="allow")

    post_id: str
    user_id: str
    user_screen_name: Optional[str] = None
    timestamp: datetime
    adoption_order: int
    seconds_since_origin: Optional[float] = None


class CascadeChronologyResponse(BaseModel):
    """Chronological trace of a cascade's propagation."""
    cascade_id: str
    total_events: int
    events: List[CascadeChronologyPoint]


class SentimentAnalyzeRequest(BaseModel):
    """Text sentiment and emotion analysis request."""
    text: Optional[str] = Field(default=None, max_length=5000, description="Single text snippet to analyze")
    texts: Optional[List[str]] = Field(default=None, max_length=100, description="Batch of text snippets to analyze (max 100)")
    stance_target: Optional[str] = Field(default=None, max_length=100, description="Target entity for stance detection")
    invert_on_irony: bool = Field(default=True, description="Whether to invert positive sentiment if ironic")


class SentimentAnalyzeResponse(BaseModel):
    """Multi-dimensional NLP inference response."""
    count: int
    results: List[MultiDimensionalResult]


class TemporalSentimentPoint(BaseModel):
    """Aggregated sentiment, irony, and emotion metrics for a time bucket."""
    bucket: datetime
    post_count: int
    mean_sentiment_score: Optional[float] = None
    positive_ratio: Optional[float] = None
    negative_ratio: Optional[float] = None
    neutral_ratio: Optional[float] = None
    sarcasm_rate: Optional[float] = None
    mean_irony_score: Optional[float] = None
    dominant_emotion: Optional[str] = None


class TemporalSentimentResponse(BaseModel):
    """Temporal trajectory of sentiment and emotions over time."""
    interval: str
    total_buckets: int
    points: List[TemporalSentimentPoint]


class UserDemographicProfileRequest(BaseModel):
    """Ad-hoc user demographic profiling request."""
    user_id: Optional[str] = Field(default=None, description="Optional user ID if fetching from DB")
    screen_name: Optional[str] = Field(default=None)
    bio: Optional[str] = Field(default=None)
    location_raw: Optional[str] = Field(default=None)
    sample_posts: Optional[List[str]] = Field(default=None, description="Sample post texts for behavioral profiling")


class ConnectorsStatusResponse(BaseModel):
    """Status summary of all platform ingestion connectors."""
    connectors: Dict[str, ConnectorInfo]


class ConnectorPollRequest(BaseModel):
    """Trigger polling on a platform connector."""
    query: str = Field(default="", description="Search query or keyword filter")
    limit: int = Field(default=20, ge=1, le=100, description="Max posts to retrieve")


class ConnectorPollResponse(BaseModel):
    """Result of connector polling."""
    platform: PlatformType
    count: int
    posts: List[CanonicalPost]
