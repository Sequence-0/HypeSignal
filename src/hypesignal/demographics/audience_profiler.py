"""Influencer Audience Demographic Profiler (Pillar C).

Aggregates follower demographic distributions (age cohorts, countries, personas,
languages, engagement tiers) across the social graph with a min_group_size
threshold filter to preserve audience privacy and suppress low-count noise.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from hypesignal.demographics.age_classifier import MultiStageAgeClassifier
from hypesignal.demographics.demographics_engine import DemographicsEngine
from hypesignal.demographics.schemas import UserProfile
from hypesignal.storage.duckdb_manager import DuckDBManager

logger = logging.getLogger(__name__)


class InfluencerAudienceProfile(BaseModel):
    """Aggregated demographic profile of an influencer's follower audience."""
    influencer_id: str
    total_followers_analyzed: int
    min_group_size_applied: int
    age_distribution: Dict[str, int] = Field(default_factory=dict)
    country_distribution: Dict[str, int] = Field(default_factory=dict)
    top_cities: Dict[str, int] = Field(default_factory=dict)
    persona_distribution: Dict[str, int] = Field(default_factory=dict)
    language_distribution: Dict[str, int] = Field(default_factory=dict)
    engagement_tier_distribution: Dict[str, int] = Field(default_factory=dict)
    follower_profiles: Optional[List[UserProfile]] = None


class InfluencerAudienceProfiler:
    """Demographic profiler for influencer follower networks with privacy filters."""

    def __init__(
        self,
        db: DuckDBManager,
        demographics_engine: Optional[DemographicsEngine] = None,
        age_classifier: Optional[MultiStageAgeClassifier] = None,
        min_group_size: int = 5,
    ) -> None:
        """Initialize audience profiler.
        
        Args:
            db: DuckDBManager instance containing graph_edges, users, and posts.
            demographics_engine: Optional DemographicsEngine instance (created if None).
            age_classifier: Optional MultiStageAgeClassifier for 4-stage age predictions.
            min_group_size: Threshold count below which demographic groups are suppressed into
                           'Other / Low Count' to protect follower privacy.
        """
        self.db = db
        self.engine = demographics_engine or DemographicsEngine()
        self.age_classifier = age_classifier
        self.min_group_size = max(1, min_group_size)

    @staticmethod
    def apply_min_group_size_filter(
        counts: Dict[str, int],
        min_size: int = 5,
        other_label: str = "Other / Low Count",
    ) -> Dict[str, int]:
        """Group buckets with count < min_size into 'Other / Low Count'."""
        if not counts or min_size <= 1:
            return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

        filtered: Dict[str, int] = {}
        other_sum = 0

        for key, count in counts.items():
            if count >= min_size:
                filtered[key] = count
            else:
                other_sum += count

        # Sort filtered items descending
        sorted_res = dict(sorted(filtered.items(), key=lambda item: item[1], reverse=True))

        if other_sum > 0:
            if other_label in sorted_res:
                sorted_res[other_label] += other_sum
            else:
                sorted_res[other_label] = other_sum

        return sorted_res

    def profile_influencer_audience(
        self,
        influencer_id: str,
        limit: int = 1000,
        include_profiles: bool = False,
    ) -> InfluencerAudienceProfile:
        """Profile demographic distribution of followers for a given influencer.
        
        Args:
            influencer_id: Unique author / user ID of target influencer.
            limit: Maximum follower nodes to profile.
            include_profiles: Whether to attach individual UserProfile objects in the result.
            
        Returns:
            InfluencerAudienceProfile with privacy-filtered demographic breakdowns.
        """
        # 1. Query follower IDs from graph_edges
        follower_ids = self.db.get_user_followers(influencer_id)
        if not follower_ids:
            # Fallback to any incoming directed edge (e.g. interaction, retweet, mention)
            rows = self.db.con.execute(
                "SELECT source_id FROM graph_edges WHERE target_id = ? LIMIT ?",
                [influencer_id, limit],
            ).fetchall()
            follower_ids = [r[0] for r in rows]

        if not follower_ids:
            logger.info("No followers or incoming graph edges found for user %s", influencer_id)
            return InfluencerAudienceProfile(
                influencer_id=influencer_id,
                total_followers_analyzed=0,
                min_group_size_applied=self.min_group_size,
            )

        # Truncate to limit
        target_follower_ids = follower_ids[:limit]

        # 2. Fetch and profile follower users from DuckDB
        profiles = self.engine.profile_users_from_duckdb(self.db, user_ids=target_follower_ids, limit=limit)
        if not profiles:
            return InfluencerAudienceProfile(
                influencer_id=influencer_id,
                total_followers_analyzed=0,
                min_group_size_applied=self.min_group_size,
            )

        # 3. If advanced age_classifier is present, refine age bracket predictions
        if self.age_classifier is not None and profiles:
            p_ids = [p.user_id for p in profiles]
            placeholders = ", ".join(["?"] * len(p_ids))

            # Batch fetch bios in a single query
            user_rows = self.db.con.execute(
                f"SELECT id, bio FROM users WHERE id IN ({placeholders})", p_ids
            ).fetchall()
            bio_map = {str(uid): (bio if bio else None) for uid, bio in user_rows}

            # Batch fetch recent posts in a single query
            post_rows = self.db.con.execute(
                f"SELECT author_id, text FROM posts WHERE author_id IN ({placeholders}) ORDER BY timestamp DESC",
                p_ids,
            ).fetchall()
            posts_map: Dict[str, List[str]] = {}
            for aid, text in post_rows:
                aid_str = str(aid)
                if aid_str not in posts_map:
                    posts_map[aid_str] = []
                if text and len(posts_map[aid_str]) < 5:
                    posts_map[aid_str].append(text)

            for p in profiles:
                bio = bio_map.get(p.user_id)
                sample_posts = posts_map.get(p.user_id, [])
                age_pred = self.age_classifier.classify_age(bio=bio, sample_posts=sample_posts)
                if p.persona:
                    p.persona.age_bracket = age_pred.bracket

        # 4. Aggregate follower distributions
        age_counts: Counter[str] = Counter()
        country_counts: Counter[str] = Counter()
        city_counts: Counter[str] = Counter()
        persona_counts: Counter[str] = Counter()
        lang_counts: Counter[str] = Counter()
        tier_counts: Counter[str] = Counter()

        for p in profiles:
            # Age
            if p.persona and p.persona.age_bracket:
                age_counts[p.persona.age_bracket] += 1

            # Country
            if p.geo and p.geo.country:
                country_counts[p.geo.country] += 1
            else:
                country_counts["Unknown"] += 1

            # City
            if p.geo and p.geo.city:
                city_counts[p.geo.city] += 1

            # Persona
            if p.persona and p.persona.primary_persona:
                persona_counts[p.persona.primary_persona] += 1

            # Language
            if p.persona and p.persona.language:
                lang_counts[p.persona.language] += 1

            # Engagement Tier
            if p.behavioral and p.behavioral.engagement_tier:
                tier_counts[p.behavioral.engagement_tier] += 1
            else:
                tier_counts["casual"] += 1

        # 5. Apply min_group_size threshold filter
        filtered_age = self.apply_min_group_size_filter(dict(age_counts), self.min_group_size)
        filtered_country = self.apply_min_group_size_filter(dict(country_counts), self.min_group_size)
        filtered_city = self.apply_min_group_size_filter(dict(city_counts), self.min_group_size)
        filtered_persona = self.apply_min_group_size_filter(dict(persona_counts), self.min_group_size)
        filtered_lang = self.apply_min_group_size_filter(dict(lang_counts), self.min_group_size)
        filtered_tier = self.apply_min_group_size_filter(dict(tier_counts), self.min_group_size)

        return InfluencerAudienceProfile(
            influencer_id=influencer_id,
            total_followers_analyzed=len(profiles),
            min_group_size_applied=self.min_group_size,
            age_distribution=filtered_age,
            country_distribution=filtered_country,
            top_cities=filtered_city,
            persona_distribution=filtered_persona,
            language_distribution=filtered_lang,
            engagement_tier_distribution=filtered_tier,
            follower_profiles=profiles if include_profiles else None,
        )
