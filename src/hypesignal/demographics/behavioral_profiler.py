"""Circadian and behavioral activity profiler.

Analyzes posting frequency, circadian hour-of-day distribution (0-23),
peak activity hours, and user engagement tiers.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Union

from hypesignal.demographics.schemas import BehavioralProfile
from hypesignal.models.canonical import CanonicalPost
from hypesignal.storage.duckdb_manager import DuckDBManager

logger = logging.getLogger(__name__)


def classify_engagement_tier(post_count: int) -> str:
    """Classify user engagement tier by total post count.
    
    - 0-5 posts: casual
    - 6-25 posts: active
    - 26-100 posts: power_user
    - >100 posts: broadcaster
    """
    if post_count <= 5:
        return "casual"
    elif post_count <= 25:
        return "active"
    elif post_count <= 100:
        return "power_user"
    else:
        return "broadcaster"


class BehavioralProfiler:
    """Profiles user posting behavior, circadian rhythms, and activity tiers."""

    def profile_from_timestamps(
        self,
        user_id: str,
        timestamps: Iterable[datetime],
    ) -> BehavioralProfile:
        """Compute behavioral profile from an iterable of UTC datetimes.
        
        Args:
            user_id: User identifier.
            timestamps: Iterable of post datetime objects.
            
        Returns:
            BehavioralProfile.
        """
        hour_counts: Counter[int] = Counter()
        total = 0

        for dt in timestamps:
            total += 1
            hour_counts[dt.hour] += 1

        dist = {h: hour_counts.get(h, 0) for h in range(24) if hour_counts.get(h, 0) > 0}
        peak = hour_counts.most_common(1)[0][0] if hour_counts else None
        tier = classify_engagement_tier(total)

        return BehavioralProfile(
            user_id=user_id,
            total_posts=total,
            active_hours_distribution=dist,
            peak_hour=peak,
            engagement_tier=tier,
        )

    def profile_from_posts(
        self,
        user_id: str,
        posts: List[CanonicalPost],
    ) -> BehavioralProfile:
        """Compute behavioral profile from a list of CanonicalPost objects."""
        user_posts = [p for p in posts if p.author_id == user_id]
        return self.profile_from_timestamps(
            user_id=user_id,
            timestamps=(p.timestamp for p in user_posts),
        )

    def profile_from_duckdb(
        self,
        user_id: str,
        db: DuckDBManager,
    ) -> BehavioralProfile:
        """Compute behavioral profile for a user directly via DuckDB analytical query.
        
        Args:
            user_id: User identifier.
            db: DuckDBManager instance.
            
        Returns:
            BehavioralProfile.
        """
        query = """
            SELECT 
                EXTRACT(hour FROM timezone('UTC', timestamp))::INTEGER AS hr,
                COUNT(*)::INTEGER AS cnt
            FROM posts
            WHERE author_id = ?
            GROUP BY hr
            ORDER BY cnt DESC, hr ASC;
        """
        rows = db.con.execute(query, [str(user_id)]).fetchall()
        if not rows:
            return BehavioralProfile(
                user_id=user_id,
                total_posts=0,
                active_hours_distribution={},
                peak_hour=None,
                engagement_tier="casual",
            )

        dist: Dict[int, int] = {}
        total = 0
        peak = rows[0][0]

        for hr, cnt in rows:
            dist[int(hr)] = int(cnt)
            total += int(cnt)

        # Sort distribution by hour
        sorted_dist = {h: dist[h] for h in sorted(dist.keys())}
        tier = classify_engagement_tier(total)

        return BehavioralProfile(
            user_id=user_id,
            total_posts=total,
            active_hours_distribution=sorted_dist,
            peak_hour=int(peak),
            engagement_tier=tier,
        )

    def profile_users_batch_from_duckdb(
        self,
        db: DuckDBManager,
        user_ids: Optional[List[str]] = None,
    ) -> Dict[str, BehavioralProfile]:
        """Batch compute behavioral profiles for users in DuckDB.
        
        Args:
            db: DuckDBManager instance.
            user_ids: Optional list of user IDs. If None, profiles all authors in posts table.
            
        Returns:
            Mapping of user_id to BehavioralProfile.
        """
        if user_ids is not None:
            if not user_ids:
                return {}
            placeholders = ", ".join(["?"] * len(user_ids))
            query = f"""
                SELECT 
                    author_id,
                    EXTRACT(hour FROM timezone('UTC', timestamp))::INTEGER AS hr,
                    COUNT(*)::INTEGER AS cnt
                FROM posts
                WHERE author_id IN ({placeholders})
                GROUP BY author_id, hr
                ORDER BY author_id, hr;
            """
            rows = db.con.execute(query, [str(u) for u in user_ids]).fetchall()
        else:
            query = """
                SELECT 
                    author_id,
                    EXTRACT(hour FROM timezone('UTC', timestamp))::INTEGER AS hr,
                    COUNT(*)::INTEGER AS cnt
                FROM posts
                GROUP BY author_id, hr
                ORDER BY author_id, hr;
            """
            rows = db.con.execute(query).fetchall()

        user_dists: Dict[str, Dict[int, int]] = {}
        for author_id, hr, cnt in rows:
            aid = str(author_id)
            if aid not in user_dists:
                user_dists[aid] = {}
            user_dists[aid][int(hr)] = int(cnt)

        results: Dict[str, BehavioralProfile] = {}
        target_ids = [str(u) for u in user_ids] if user_ids is not None else list(user_dists.keys())

        for uid in target_ids:
            dist = user_dists.get(uid, {})
            total = sum(dist.values())
            peak = max(dist.items(), key=lambda x: x[1])[0] if dist else None
            tier = classify_engagement_tier(total)

            results[uid] = BehavioralProfile(
                user_id=uid,
                total_posts=total,
                active_hours_distribution=dist,
                peak_hour=peak,
                engagement_tier=tier,
            )

        return results
