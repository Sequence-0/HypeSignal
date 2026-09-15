"""Real-Time Trend & Dynamic Topic Detection package (Component D).

Combines Tier-1 statistical burst detection (Z-scores and velocity) with
Tier-2 dynamic temporal topic modeling (BERTopic) and multi-factor trend ranking.
"""

from hypesignal.trends.burst_detector import BurstDetector, DEFAULT_STOPWORDS
from hypesignal.trends.narrative_drift import NarrativeDriftAlert, NarrativeDriftTracker
from hypesignal.trends.schemas import (
    BurstAlert,
    DynamicTopicTimeline,
    TopicRepresentation,
    TrendOverview,
    TrendRankingResult,
)
from hypesignal.trends.topic_modeler import DynamicTopicModeler
from hypesignal.trends.trend_forecaster import (
    TrendForecast,
    TrendForecaster,
    TrendLifecycleState,
)
from hypesignal.trends.trend_ranker import TrendRanker
from hypesignal.trends.trends_engine import TrendsEngine

__all__ = [
    "BurstAlert",
    "BurstDetector",
    "DEFAULT_STOPWORDS",
    "DynamicTopicModeler",
    "DynamicTopicTimeline",
    "NarrativeDriftAlert",
    "NarrativeDriftTracker",
    "TopicRepresentation",
    "TrendForecast",
    "TrendForecaster",
    "TrendLifecycleState",
    "TrendOverview",
    "TrendRanker",
    "TrendRankingResult",
    "TrendsEngine",
]
