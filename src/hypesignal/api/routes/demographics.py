"""Automated demographic and persona profiling routes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from hypesignal.api.deps import get_db, get_demographics
from hypesignal.api.schemas import UserDemographicProfileRequest
from hypesignal.demographics.demographics_engine import DemographicsEngine
from hypesignal.demographics.schemas import AggregateDemographics, UserProfile
from hypesignal.models.canonical import CanonicalPost, CanonicalUser
from hypesignal.models.enums import PlatformType
from hypesignal.storage.duckdb_manager import DuckDBManager

router = APIRouter(prefix="/demographics", tags=["Demographics & Persona Profiling"])


@router.post("/profile-user", response_model=UserProfile)
def profile_user(
    req: UserDemographicProfileRequest,
    demographics: DemographicsEngine = Depends(get_demographics),
    db: DuckDBManager = Depends(get_db),
) -> UserProfile:
    """Generate a demographic profile (geo, persona, behavior) for an ad-hoc or existing user."""
    user: Optional[CanonicalUser] = None

    if req.user_id:
        profiles = demographics.profile_users_from_duckdb(db=db, user_ids=[req.user_id], limit=1)
        if profiles:
            return profiles[0]

    # Create canonical user from input parameters
    user = CanonicalUser(
        id=req.user_id or "user_adhoc",
        platform=PlatformType.TWITTER,
        screen_name=req.screen_name or "adhoc_user",
        bio=req.bio,
        location_raw=req.location_raw,
    )

    sample_posts: Optional[List[CanonicalPost]] = None
    if req.sample_posts:
        now = datetime.now(timezone.utc)
        sample_posts = [
            CanonicalPost(
                id=f"p_{i}",
                author_id=user.id,
                text=t,
                timestamp=now,
            )
            for i, t in enumerate(req.sample_posts)
        ]

    return demographics.profile_user(user=user, sample_posts=sample_posts)


@router.get("/breakdown", response_model=AggregateDemographics)
def get_audience_breakdown(
    limit: int = Query(default=500, ge=1, le=5000, description="Max users to sample for breakdown"),
    demographics: DemographicsEngine = Depends(get_demographics),
    db: DuckDBManager = Depends(get_db),
) -> AggregateDemographics:
    """Retrieve population-level demographic breakdown across countries, cities, languages, and personas."""
    return demographics.get_audience_breakdown(db=db, limit=limit)


@router.get("/users", response_model=List[UserProfile])
def get_profiled_users(
    limit: int = Query(default=50, ge=1, le=500, description="Max users to profile"),
    demographics: DemographicsEngine = Depends(get_demographics),
    db: DuckDBManager = Depends(get_db),
) -> List[UserProfile]:
    """Retrieve demographic profiles for recent users stored in DuckDB."""
    return demographics.profile_users_from_duckdb(db=db, limit=limit)
