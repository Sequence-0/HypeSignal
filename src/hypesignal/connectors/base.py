"""Base abstractions and rate limiting for platform ingestion connectors."""

from __future__ import annotations

import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from hypesignal.connectors.schemas import (
    ConnectorConfig,
    ConnectorInfo,
    ConnectorStats,
    ConnectorStatus,
)
from hypesignal.models.canonical import CanonicalPost
from hypesignal.models.enums import PlatformType

logger = logging.getLogger(__name__)

# Standard regexes for fallback entity extraction
RE_URL = re.compile(r"https?://\S+")
RE_HASHTAG = re.compile(r"#\w+")
RE_MENTION = re.compile(r"@\w+")


class TokenBucketRateLimiter:
    """Thread-safe token bucket rate limiter for API requests."""

    def __init__(self, rate: float, capacity: float) -> None:
        """Initialize rate limiter.
        
        Args:
            rate: Token refill rate in tokens per second.
            capacity: Maximum burst capacity of tokens.
        """
        self.rate = float(rate)
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self, now: float) -> None:
        elapsed = now - self.last_refill
        self.last_refill = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

    def acquire(self, tokens: float = 1.0) -> bool:
        """Attempt to consume the specified number of tokens immediately.
        
        Returns True if tokens were consumed, False if rate limited.
        """
        with self._lock:
            now = time.monotonic()
            self._refill(now)
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def wait_time(self, tokens: float = 1.0) -> float:
        """Calculate required wait time in seconds before `tokens` can be acquired."""
        with self._lock:
            now = time.monotonic()
            self._refill(now)
            if self.tokens >= tokens:
                return 0.0
            deficit = tokens - self.tokens
            return deficit / self.rate if self.rate > 0 else float("inf")

    def reset(self) -> None:
        """Reset tokens back to full capacity."""
        with self._lock:
            self.tokens = self.capacity
            self.last_refill = time.monotonic()


class PlatformConnector(ABC):
    """Abstract base class for live social platform connectors."""

    def __init__(self, config: Optional[ConnectorConfig] = None) -> None:
        """Initialize platform connector with configuration and rate limiter."""
        self.config = config or ConnectorConfig(platform=self.platform)
        self.status = ConnectorStatus.DISCONNECTED
        self.stats = ConnectorStats()
        self._lock = threading.Lock()

        # Refill rate per second: max_requests_per_minute / 60.0
        rate = max(0.01, self.config.max_requests_per_minute / 60.0)
        self.rate_limiter = TokenBucketRateLimiter(
            rate=rate,
            capacity=float(self.config.max_requests_per_minute),
        )

    @property
    @abstractmethod
    def platform(self) -> PlatformType:
        """The social platform associated with this connector."""
        ...

    def record_request(self) -> None:
        """Thread-safely record an outgoing API request."""
        with self._lock:
            self.stats.requests_made += 1

    def record_success(self, count: int) -> None:
        """Thread-safely record successfully ingested messages."""
        with self._lock:
            self.stats.messages_ingested += count
            self.stats.last_active = datetime.now(timezone.utc)
            self.status = ConnectorStatus.CONNECTED

    def record_rate_limited(self) -> None:
        """Thread-safely record a rate limit event."""
        with self._lock:
            self.status = ConnectorStatus.RATE_LIMITED
            self.stats.error_count += 1

    def record_error(self, message: str) -> None:
        """Thread-safely record an ingestion or network error."""
        with self._lock:
            self.status = ConnectorStatus.ERROR
            self.stats.error_count += 1
            self.stats.last_error_message = message

    def connect(self) -> bool:
        """Initialize connection to the platform service."""
        if not self.config.enabled:
            logger.info("Connector for %s is disabled in config.", self.platform.value)
            with self._lock:
                self.status = ConnectorStatus.DISCONNECTED
            return False

        try:
            with self._lock:
                self.status = ConnectorStatus.CONNECTING
            self._do_connect()
            with self._lock:
                self.status = ConnectorStatus.CONNECTED
                self.stats.last_active = datetime.now(timezone.utc)
            logger.info("Connected to %s successfully.", self.platform.value)
            return True
        except Exception as e:
            with self._lock:
                self.status = ConnectorStatus.ERROR
                self.stats.error_count += 1
                self.stats.last_error_message = str(e)
            logger.error("Failed to connect to %s: %s", self.platform.value, e)
            return False

    def disconnect(self) -> bool:
        """Tear down connection to the platform service."""
        try:
            self._do_disconnect()
            with self._lock:
                self.status = ConnectorStatus.DISCONNECTED
            logger.info("Disconnected from %s.", self.platform.value)
            return True
        except Exception as e:
            with self._lock:
                self.status = ConnectorStatus.ERROR
                self.stats.error_count += 1
                self.stats.last_error_message = str(e)
            logger.error("Error disconnecting from %s: %s", self.platform.value, e)
            return False

    def is_connected(self) -> bool:
        """Return True if connector is currently connected."""
        with self._lock:
            return self.status == ConnectorStatus.CONNECTED

    def _do_connect(self) -> None:
        """Subclass hook for specific authentication or socket handshake."""
        pass

    def _do_disconnect(self) -> None:
        """Subclass hook for specific resource cleanup."""
        pass

    @abstractmethod
    def normalize_post(self, raw_payload: Dict[str, Any]) -> CanonicalPost:
        """Normalize a raw platform-specific post/message dictionary into CanonicalPost."""
        ...

    @abstractmethod
    def poll(self, query: str = "", limit: int = 50) -> List[CanonicalPost]:
        """Poll the platform for recent posts matching the query."""
        ...

    def get_info(self) -> ConnectorInfo:
        """Return current status, configuration, and runtime metrics."""
        with self._lock:
            return ConnectorInfo(
                platform=self.platform,
                status=self.status,
                config=self.config.model_copy(),
                stats=self.stats.model_copy(),
            )

    @staticmethod
    def extract_entities(text: str) -> Dict[str, List[str]]:
        """Utility method to extract URLs, hashtags, and mentions from plain text."""
        return {
            "urls": RE_URL.findall(text),
            "hashtags": [h[1:] for h in RE_HASHTAG.findall(text)],
            "mentions": [m[1:] for m in RE_MENTION.findall(text)],
        }
