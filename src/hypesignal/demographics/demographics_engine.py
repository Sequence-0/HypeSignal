"""Unified Demographic Profiling Engine.

Orchestrates geographic entity extraction, zero-shot persona classification,
language detection, and circadian behavioral profiling for individual users
and audience-level aggregations.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

from hypesignal.demographics.behavioral_profiler import BehavioralProfiler
from hypesignal.demographics.geo_profiler import GeoProfiler
from hypesignal.demographics.language_detector import LanguageDetector
from hypesignal.demographics.persona_profiler import PersonaProfiler
from hypesignal.demographics.schemas import (
    AggregateDemographics,
    BehavioralProfile,
    GeoLocationProfile,
    PersonaProfile,
    UserProfile,
)
from hypesignal.models.canonical import CanonicalPost, CanonicalUser, GeoCoordinates
from hypesignal.models.enums import PlatformType
from hypesignal.storage.duckdb_manager import DuckDBManager
from hypesignal.storage.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class DemographicsEngine:
    """Consolidated demographic analytics engine for HypeSignal."""

    def __init__(
        self,
        geo_profiler: Optional[GeoProfiler] = None,
        persona_profiler: Optional[PersonaProfiler] = None,
        behavioral_profiler: Optional[BehavioralProfiler] = None,
        language_detector: Optional[LanguageDetector] = None,
        vector_store: Optional[VectorStoreManager] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize demographic profiler subcomponents.
        
        Args:
            geo_profiler: Custom GeoProfiler or defaults to standard gazetteer/spaCy.
            persona_profiler: Custom PersonaProfiler or defaults to SentenceTransformer.
            behavioral_profiler: Custom BehavioralProfiler.
            language_detector: Custom LanguageDetector.
            vector_store: Optional VectorStoreManager for Qdrant index.
            device: 'cuda', 'cpu', or None for auto-detection.
        """
        self.geo_profiler = geo_profiler or GeoProfiler()
        self.persona_profiler = persona_profiler or PersonaProfiler(
            vector_store=vector_store, device=device
        )
        self.behavioral_profiler = behavioral_profiler or BehavioralProfiler()
        self.language_detector = language_detector or LanguageDetector()

    def profile_user(
        self,
        user: CanonicalUser,
        sample_posts: Optional[List[CanonicalPost]] = None,
    ) -> UserProfile:
        """Generate a complete demographic profile for a single user.
        
        Args:
            user: CanonicalUser instance.
            sample_posts: Optional list of posts authored by this user.
            
        Returns:
            UserProfile containing geo, persona, and behavioral profiles.
        """
        # 1. Geographic profiling
        geo = self.geo_profiler.profile_user(user)

        # 2. Persona and interest profiling
        post_texts = [p.text for p in sample_posts] if sample_posts else None
        persona = self.persona_profiler.profile_persona(
            user_id=user.id,
            bio=user.bio,
            sample_posts=post_texts,
        )

        # 3. Behavioral profiling
        behavioral = None
        if sample_posts:
            behavioral = self.behavioral_profiler.profile_from_posts(
                user_id=user.id,
                posts=sample_posts,
            )

        return UserProfile(
            user_id=user.id,
            screen_name=user.screen_name,
            geo=geo,
            persona=persona,
            behavioral=behavioral,
        )

    def profile_users_batch(
        self,
        users: List[CanonicalUser],
        user_posts_map: Optional[Dict[str, List[CanonicalPost]]] = None,
    ) -> List[UserProfile]:
        """Batch profile multiple users efficiently.
        
        Args:
            users: List of CanonicalUser objects.
            user_posts_map: Optional mapping of user_id to lists of CanonicalPost objects.
            
        Returns:
            List of UserProfile objects.
        """
        if not users:
            return []

        user_posts_map = user_posts_map or {}

        # Prepare text map for persona batch profiling
        post_texts_map = {
            uid: [p.text for p in posts]
            for uid, posts in user_posts_map.items()
        }

        # 1. Batch persona profiling
        persona_profiles = self.persona_profiler.profile_users_batch(
            users=users,
            user_posts_map=post_texts_map,
        )
        persona_dict = {p.user_id: p for p in persona_profiles}

        results: List[UserProfile] = []
        for user in users:
            # Geo profile
            geo = self.geo_profiler.profile_user(user)

            # Persona profile
            persona = persona_dict.get(
                user.id,
                PersonaProfile(
                    user_id=user.id,
                    primary_persona="General & Everyday Lifestyle",
                    persona_confidence=0.0,
                    age_bracket="unknown",
                    language="en",
                ),
            )

            # Behavioral profile
            posts = user_posts_map.get(user.id)
            behavioral = None
            if posts:
                behavioral = self.behavioral_profiler.profile_from_posts(user.id, posts)

            results.append(
                UserProfile(
                    user_id=user.id,
                    screen_name=user.screen_name,
                    geo=geo,
                    persona=persona,
                    behavioral=behavioral,
                )
            )

        return results

    def profile_users_from_duckdb(
        self,
        db: DuckDBManager,
        user_ids: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[UserProfile]:
        """Load and profile users directly from DuckDB analytical store.
        
        Args:
            db: DuckDBManager instance.
            user_ids: Optional list of user IDs to filter.
            limit: Maximum number of users to profile.
            
        Returns:
            List of UserProfile instances.
        """
        if user_ids is not None:
            if not user_ids:
                return []
            placeholders = ", ".join(["?"] * len(user_ids[:limit]))
            query = f"""
                SELECT id, platform, screen_name, location_raw, latitude, longitude, indegree, outdegree, bio
                FROM users
                WHERE id IN ({placeholders})
                LIMIT {limit};
            """
            rows = db.con.execute(query, [str(u) for u in user_ids[:limit]]).fetchall()
        else:
            query = f"""
                SELECT id, platform, screen_name, location_raw, latitude, longitude, indegree, outdegree, bio
                FROM users
                LIMIT {limit};
            """
            rows = db.con.execute(query).fetchall()

        if not rows:
            return []

        users: List[CanonicalUser] = []
        fetched_ids: List[str] = []

        for row in rows:
            uid, platform, sname, loc_raw, lat, lon, indeg, outdeg, bio = row
            coords = GeoCoordinates(latitude=lat, longitude=lon) if lat is not None and lon is not None else None
            try:
                plat = PlatformType(platform)
            except ValueError:
                plat = PlatformType.TWITTER

            users.append(
                CanonicalUser(
                    id=str(uid),
                    platform=plat,
                    screen_name=sname,
                    bio=bio,
                    location_raw=loc_raw,
                    location_coords=coords,
                    indegree=indeg,
                    outdegree=outdeg,
                )
            )
            fetched_ids.append(str(uid))

        # Fetch recent posts for these users
        post_placeholders = ", ".join(["?"] * len(fetched_ids))
        posts_query = f"""
            SELECT id, platform, author_id, author_screen_name, text, timestamp
            FROM posts
            WHERE author_id IN ({post_placeholders})
            ORDER BY timestamp DESC;
        """
        post_rows = db.con.execute(posts_query, fetched_ids).fetchall()

        user_posts_map: Dict[str, List[CanonicalPost]] = {}
        for pid, pplat, author_id, asname, text, ts in post_rows:
            aid = str(author_id)
            if aid not in user_posts_map:
                user_posts_map[aid] = []
            try:
                plat_type = PlatformType(pplat)
            except ValueError:
                plat_type = PlatformType.TWITTER

            user_posts_map[aid].append(
                CanonicalPost(
                    id=str(pid),
                    platform=plat_type,
                    author_id=aid,
                    author_screen_name=asname,
                    text=text,
                    timestamp=ts,
                )
            )

        return self.profile_users_batch(users, user_posts_map)

    def aggregate_demographics(self, profiles: List[UserProfile]) -> AggregateDemographics:
        """Aggregate individual user profiles into population-level audience insights.
        
        Args:
            profiles: List of UserProfile instances.
            
        Returns:
            AggregateDemographics summary.
        """
        if not profiles:
            return AggregateDemographics(total_users_profiled=0)

        country_counts: Counter[str] = Counter()
        city_counts: Counter[str] = Counter()
        lang_counts: Counter[str] = Counter()
        persona_counts: Counter[str] = Counter()
        age_counts: Counter[str] = Counter()
        tier_counts: Counter[str] = Counter()

        for p in profiles:
            # Country
            if p.geo and p.geo.country:
                country_counts[p.geo.country] += 1

            # City
            if p.geo and p.geo.city:
                city_counts[p.geo.city] += 1

            # Language
            if p.persona and p.persona.language:
                lang_counts[p.persona.language] += 1

            # Persona
            if p.persona and p.persona.primary_persona:
                persona_counts[p.persona.primary_persona] += 1

            # Age bracket
            if p.persona and p.persona.age_bracket:
                age_counts[p.persona.age_bracket] += 1

            # Engagement tier
            if p.behavioral and p.behavioral.engagement_tier:
                tier_counts[p.behavioral.engagement_tier] += 1
            else:
                tier_counts["casual"] += 1

        return AggregateDemographics(
            total_users_profiled=len(profiles),
            country_distribution=dict(country_counts.most_common()),
            top_cities=dict(city_counts.most_common(20)),
            language_distribution=dict(lang_counts.most_common()),
            persona_distribution=dict(persona_counts.most_common()),
            age_bracket_distribution=dict(age_counts.most_common()),
            engagement_tier_distribution=dict(tier_counts.most_common()),
        )

    def get_audience_breakdown(
        self,
        db: DuckDBManager,
        limit: int = 500,
    ) -> AggregateDemographics:
        """High-level analytical endpoint computing full audience breakdown from DuckDB.
        
        Args:
            db: DuckDBManager instance.
            limit: Maximum users to profile from the database.
            
        Returns:
            AggregateDemographics summary.
        """
        profiles = self.profile_users_from_duckdb(db=db, limit=limit)
        return self.aggregate_demographics(profiles)
