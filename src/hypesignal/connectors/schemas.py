"""Schemas and data models for live platform ingestion connectors."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from hypesignal.models.enums import PlatformType


class ConnectorStatus(str, Enum):
    """Lifecycle connection status for platform ingestion connectors."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"


class ConnectorConfig(BaseModel):
    """Configuration options for a platform ingestion connector."""
    model_config = ConfigDict(extra="allow")

    platform: PlatformType
    enabled: bool = Field(default=True, description="Whether connector is active")
    poll_interval_seconds: float = Field(default=60.0, ge=1.0, description="Interval between polling cycles")
    max_requests_per_minute: int = Field(default=60, ge=1, description="Rate limit ceiling per minute")
    retry_attempts: int = Field(default=3, ge=0, description="Number of retry attempts on transient network errors")
    credentials: Dict[str, Any] = Field(default_factory=dict, description="Platform API keys, tokens, secrets")


class ConnectorStats(BaseModel):
    """Runtime operational metrics for an ingestion connector."""
    messages_ingested: int = Field(default=0, ge=0)
    requests_made: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)
    last_active: Optional[datetime] = Field(default=None)
    rate_limited_until: Optional[datetime] = Field(default=None)
    last_error_message: Optional[str] = Field(default=None)


class ConnectorInfo(BaseModel):
    """Public summary metadata for an ingestion connector."""
    platform: PlatformType
    status: ConnectorStatus
    config: ConnectorConfig
    stats: ConnectorStats
