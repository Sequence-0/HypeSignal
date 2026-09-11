"""Canonical data models for the HypeSignal framework.

These schemas normalize raw multi-platform feeds (Twitter/X, Telegram,
Reddit, YouTube, etc.) and historical research datasets into a unified,
strongly-typed representation.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hypesignal.models.enums import PlatformType, RelationType


class GeoCoordinates(BaseModel):
    """Geographical coordinate representation (WGS 84)."""
    model_config = ConfigDict(frozen=True)

    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in degrees [-90, 90]")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in degrees [-180, 180]")

    @classmethod
    def from_raw_string(cls, raw: str) -> Optional[GeoCoordinates]:
        """Parse coordinates from raw strings like 'UT: 43.009815,-83.710408' or '43.009815, -83.710408'."""
        if not raw:
            return None
        match = re.search(r"(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)", raw)
        if match:
            try:
                lat = float(match.group(1))
                lon = float(match.group(2))
                if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                    return cls(latitude=lat, longitude=lon)
            except (ValueError, TypeError):
                return None
        return None


class PostMetrics(BaseModel):
    """Engagement metrics associated with a social post or comment."""
    likes: int = Field(default=0, ge=0)
    reposts: int = Field(default=0, ge=0)
    replies: int = Field(default=0, ge=0)
    views: Optional[int] = Field(default=None, ge=0)
    shares: int = Field(default=0, ge=0)


class UserMetrics(BaseModel):
    """Aggregated metrics for a user profile."""
    followers_count: int = Field(default=0, ge=0)
    following_count: int = Field(default=0, ge=0)
    posts_count: int = Field(default=0, ge=0)
    listed_count: int = Field(default=0, ge=0)


class CanonicalPost(BaseModel):
    """Canonical representation of any post, tweet, message, or video comment."""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., description="Unique platform-scoped or global post identifier")
    platform: PlatformType = Field(default=PlatformType.TWITTER)
    author_id: str = Field(..., description="Unique author user ID")
    author_screen_name: Optional[str] = Field(default=None, description="Username/handle of author")
    text: str = Field(..., description="Raw text content")
    timestamp: datetime = Field(..., description="Post publication datetime (UTC normalized)")
    timestamp_ms: Optional[int] = Field(default=None, description="Epoch timestamp in milliseconds")
    parent_id: Optional[str] = Field(default=None, description="Parent post ID if reply or thread")
    reply_to_user_id: Optional[str] = Field(default=None, description="Replied-to user ID")
    source_client: Optional[str] = Field(default=None, description="Client/source application")
    urls: List[str] = Field(default_factory=list, description="Extracted URLs")
    hashtags: List[str] = Field(default_factory=list, description="Extracted hashtags")
    mentions: List[str] = Field(default_factory=list, description="Mentioned user handles")
    metrics: PostMetrics = Field(default_factory=PostMetrics)
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def synchronize_timestamps(self) -> CanonicalPost:
        """Ensure UTC timezone and timestamp_ms consistency.
        
        Assumption:
            For historical social datasets (such as Cheng-Caverlee-Lee) where
            timestamps are provided as naive datetimes without explicit offsets,
            they are treated as UTC (matching Twitter's original REST API UTC standard).
            If an explicit timezone offset is attached, it is converted to UTC.
        """
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
        else:
            self.timestamp = self.timestamp.astimezone(timezone.utc)

        if self.timestamp_ms is None:
            self.timestamp_ms = int(self.timestamp.timestamp() * 1000)
        return self


class CanonicalUser(BaseModel):
    """Canonical representation of a user profile across social platforms."""
    id: str = Field(..., description="Unique user identifier")
    platform: PlatformType = Field(default=PlatformType.TWITTER)
    screen_name: Optional[str] = Field(default=None, description="Unique username or screen handle")
    bio: Optional[str] = Field(default=None, description="Self-reported bio or description")
    location_raw: Optional[str] = Field(default=None, description="Unprocessed profile location string")
    location_coords: Optional[GeoCoordinates] = Field(default=None, description="Parsed geographic coordinates")
    indegree: Optional[int] = Field(default=None, ge=0, description="Network in-degree / follower count")
    outdegree: Optional[int] = Field(default=None, ge=0, description="Network out-degree / following count")
    metrics: UserMetrics = Field(default_factory=UserMetrics)
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def populate_coords_if_applicable(self) -> CanonicalUser:
        """Automatically parse GPS coordinates if present in location_raw and coords not set."""
        if self.location_coords is None and self.location_raw:
            parsed = GeoCoordinates.from_raw_string(self.location_raw)
            if parsed:
                self.location_coords = parsed
        return self


class CanonicalGraphEdge(BaseModel):
    """Canonical representation of an explicit relationship edge between two users."""
    source_id: str = Field(..., description="Originating node ID (e.g., the follower)")
    target_id: str = Field(..., description="Destination node ID (e.g., the followee)")
    relation_type: RelationType = Field(default=RelationType.FOLLOWS)
    timestamp: Optional[datetime] = Field(default=None, description="Timestamp of edge creation")
    weight: float = Field(default=1.0, ge=0.0)
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)


class CanonicalCascadeEvent(BaseModel):
    """A timestamped event in an information diffusion cascade (e.g., URL or topic share)."""
    cascade_id: str = Field(..., description="Cascade identifier (e.g. normalized URL or topic)")
    post_id: str = Field(..., description="ID of the post containing the cascade token")
    user_id: str = Field(..., description="User ID propagating the cascade")
    user_screen_name: Optional[str] = Field(default=None, description="User screen handle")
    timestamp: datetime = Field(..., description="Event timestamp (UTC normalized)")
    timestamp_ms: Optional[int] = Field(default=None, description="Epoch timestamp in milliseconds")
    adoption_order: int = Field(..., ge=0, description="Sequential order index of user adoption in cascade")
    parent_event_id: Optional[str] = Field(default=None, description="Parent event ID if direct transmission known")
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def synchronize_timestamps(self) -> CanonicalCascadeEvent:
        """Ensure UTC timezone and timestamp_ms consistency."""
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
        else:
            self.timestamp = self.timestamp.astimezone(timezone.utc)

        if self.timestamp_ms is None:
            self.timestamp_ms = int(self.timestamp.timestamp() * 1000)
        return self
