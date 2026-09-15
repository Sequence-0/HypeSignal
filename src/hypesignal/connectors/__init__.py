"""Connectors and live ingestion stubs for multi-platform social feeds."""

from hypesignal.connectors.base import PlatformConnector, TokenBucketRateLimiter
from hypesignal.connectors.bluesky import BlueskyConnector
from hypesignal.connectors.reddit import RedditConnector
from hypesignal.connectors.schemas import (
    ConnectorConfig,
    ConnectorInfo,
    ConnectorStats,
    ConnectorStatus,
)
from hypesignal.connectors.telegram import TelegramConnector
from hypesignal.connectors.twitter import TwitterConnector
from hypesignal.connectors.youtube import YouTubeConnector

__all__ = [
    "BlueskyConnector",
    "ConnectorConfig",
    "ConnectorInfo",
    "ConnectorStats",
    "ConnectorStatus",
    "PlatformConnector",
    "RedditConnector",
    "TelegramConnector",
    "TokenBucketRateLimiter",
    "TwitterConnector",
    "YouTubeConnector",
]
