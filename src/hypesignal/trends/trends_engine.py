"""Unified Trend & Dynamic Topic Engine (Component D).

Orchestrates Tier-1 statistical burst detection, Tier-2 BERTopic dynamic
temporal topic modeling, and multi-factor trend ranking.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from hypesignal.models.canonical import CanonicalPost
from hypesignal.models.enums import PlatformType
from hypesignal.storage.duckdb_manager import DuckDBManager
from hypesignal.trends.burst_detector import BurstDetector
from hypesignal.trends.schemas import (
    BurstAlert,
    DynamicTopicTimeline,
    TopicRepresentation,
    TrendOverview,
    TrendRankingResult,
)
from hypesignal.trends.topic_modeler import DynamicTopicModeler
from hypesignal.trends.trend_ranker import TrendRanker

logger = logging.getLogger(__name__)


class TrendsEngine:
    """Consolidated Trend & Dynamic Topic Detection Engine."""

    def __init__(
        self,
        burst_detector: Optional[BurstDetector] = None,
        topic_modeler: Optional[DynamicTopicModeler] = None,
        trend_ranker: Optional[TrendRanker] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize TrendsEngine.
        
        Args:
            burst_detector: Custom BurstDetector or defaults to standard settings.
            topic_modeler: Custom DynamicTopicModeler or defaults to BERTopic all-MiniLM.
            trend_ranker: Custom TrendRanker or defaults to standard weights.
            device: 'cuda', 'cpu', or None for auto detection.
        """
        self.burst_detector = burst_detector or BurstDetector()
        self.topic_modeler = topic_modeler or DynamicTopicModeler(device=device)
        self.trend_ranker = trend_ranker or TrendRanker()

    def analyze_trends(
        self,
        db: DuckDBManager,
        current_window_end: Optional[datetime] = None,
        window_duration_minutes: int = 60,
        num_baseline_windows: int = 5,
        z_threshold: float = 2.5,
        min_count: int = 3,
        topic_doc_limit: int = 500,
        nr_topic_bins: int = 5,
    ) -> TrendOverview:
        """Perform end-to-end dual-tier trend detection and ranking from DuckDB.
        
        Args:
            db: DuckDBManager analytical instance.
            current_window_end: End of current observation interval.
            window_duration_minutes: Length of each analysis window in minutes.
            num_baseline_windows: Number of historical baseline intervals.
            z_threshold: Minimum Z-score to fire a burst alert.
            min_count: Minimum occurrence frequency in current window.
            topic_doc_limit: Maximum recent posts to cluster into dynamic topics.
            nr_topic_bins: Number of temporal bins for topic trajectory modeling.
            
        Returns:
            TrendOverview snapshot.
        """
        # Determine observation window
        if current_window_end is None:
            max_ts_row = db.con.execute("SELECT MAX(timestamp) FROM posts;").fetchone()
            if not max_ts_row or max_ts_row[0] is None:
                now = datetime.now(timezone.utc)
                return TrendOverview(window_start=now, window_end=now)
            current_window_end = max_ts_row[0]
            if current_window_end.tzinfo is None:
                current_window_end = current_window_end.replace(tzinfo=timezone.utc)

        win_delta = timedelta(minutes=window_duration_minutes)
        window_start = current_window_end - win_delta

        # 1. Tier-1 Statistical Burst Detection
        burst_alerts = self.burst_detector.detect_bursts_from_duckdb(
            db=db,
            current_window_end=current_window_end,
            window_duration_minutes=window_duration_minutes,
            num_baseline_windows=num_baseline_windows,
            z_threshold=z_threshold,
            min_count=min_count,
            target="all",
        )
        active_bursts = [b for b in burst_alerts if b.is_burst]

        # Fetch posts in current window for participant diversity calculation
        query = """
            SELECT id, platform, author_id, author_screen_name, text, timestamp
            FROM posts
            WHERE timestamp >= ? AND timestamp <= ?;
        """
        current_rows = db.con.execute(query, [window_start, current_window_end]).fetchall()
        current_posts: List[CanonicalPost] = []
        for pid, plat, aid, sname, text, ts in current_rows:
            try:
                ptype = PlatformType(plat)
            except ValueError:
                ptype = PlatformType.TWITTER
            current_posts.append(
                CanonicalPost(
                    id=str(pid),
                    platform=ptype,
                    author_id=str(aid),
                    author_screen_name=sname,
                    text=text,
                    timestamp=ts,
                )
            )

        # Group current posts by term
        posts_by_term: Dict[str, List[CanonicalPost]] = {}
        for post in current_posts:
            terms = self.burst_detector.extract_terms_from_post(post.text)
            for term, _ in terms:
                if term not in posts_by_term:
                    posts_by_term[term] = []
                posts_by_term[term].append(post)

        ranked_bursts = self.trend_ranker.rank_burst_alerts(
            alerts=active_bursts,
            posts_by_term=posts_by_term,
        )

        # 2. Tier-2 Dynamic Temporal Topic Modeling
        # Use posts across baseline + current window
        total_span = win_delta * (num_baseline_windows + 1)
        cluster_start = current_window_end - total_span
        topic_reps, timelines = self.topic_modeler.fit_from_duckdb(
            db=db,
            start_time=cluster_start,
            end_time=current_window_end,
            limit=topic_doc_limit,
            nr_bins=nr_topic_bins,
        )

        posts_by_topic = getattr(self.topic_modeler, "last_posts_by_topic", {})

        ranked_topics = self.trend_ranker.rank_dynamic_topics(
            topics=topic_reps,
            timelines=timelines,
            posts_by_topic=posts_by_topic,
        )

        # 3. Global Multi-Factor Trend Ranking
        all_ranked = self.trend_ranker.rank_all(
            burst_results=ranked_bursts,
            topic_results=ranked_topics,
        )

        return TrendOverview(
            window_start=window_start,
            window_end=current_window_end,
            active_bursts=active_bursts,
            dynamic_topics=topic_reps,
            ranked_trends=all_ranked,
        )
