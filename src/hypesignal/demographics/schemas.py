"""Demographic profiling schemas for geographic, persona, and behavioral audience insights."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from hypesignal.models.canonical import GeoCoordinates


class GeoLocationProfile(BaseModel):
    """Normalized geographic profile of a user or post author."""
    raw_location: Optional[str] = None
    city: Optional[str] = None
    state_province: Optional[str] = None
    country: Optional[str] = None
    country_code: Optional[str] = None
    coordinates: Optional[GeoCoordinates] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = Field(default="unknown")  # 'explicit_gps', 'pattern_matched', 'ner_extracted', 'unknown'


class PersonaProfile(BaseModel):
    """Inferred persona, professional interests, language, and age bracket."""
    user_id: str
    primary_persona: str
    persona_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    top_interests: List[str] = Field(default_factory=list)
    age_bracket: str = Field(default="unknown")  # '18-24', '25-34', '35-49', '50+', 'unknown'
    language: str = Field(default="en")


class BehavioralProfile(BaseModel):
    """Circadian and posting activity profile."""
    user_id: str
    total_posts: int = Field(default=0, ge=0)
    active_hours_distribution: Dict[int, int] = Field(default_factory=dict)
    peak_hour: Optional[int] = None
    engagement_tier: str = Field(default="casual")  # 'casual', 'active', 'power_user', 'broadcaster'


class UserProfile(BaseModel):
    """Consolidated individual user demographic profile."""
    user_id: str
    screen_name: Optional[str] = None
    geo: GeoLocationProfile
    persona: PersonaProfile
    behavioral: Optional[BehavioralProfile] = None


class AggregateDemographics(BaseModel):
    """Population-level aggregated audience demographic insights."""
    total_users_profiled: int
    country_distribution: Dict[str, int] = Field(default_factory=dict)
    top_cities: Dict[str, int] = Field(default_factory=dict)
    language_distribution: Dict[str, int] = Field(default_factory=dict)
    persona_distribution: Dict[str, int] = Field(default_factory=dict)
    age_bracket_distribution: Dict[str, int] = Field(default_factory=dict)
    engagement_tier_distribution: Dict[str, int] = Field(default_factory=dict)
