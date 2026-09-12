"""Unified trend ranking engine combining burst acceleration, sentiment momentum, and participant diversity."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Set

from hypesignal.models.canonical import CanonicalPost
from hypesignal.trends.schemas import BurstAlert, DynamicTopicTimeline, TopicRepresentation, TrendRankingResult

logger = logging.getLogger(__name__)


class TrendRanker:
    """Combines burst acceleration, emotional momentum, and participant diversity into composite trend scores."""

    def __init__(
        self,
        weight_burst: float = 0.50,
        weight_sentiment: float = 0.25,
        weight_diversity: float = 0.25,
    ) -> None:
        """Initialize TrendRanker with relative feature weights.
        
        Args:
            weight_burst: Weight for Z-score burst acceleration.
            weight_sentiment: Weight for sentiment/emotional intensity.
            weight_diversity: Weight for participant decentralization.
        """
        total_w = weight_burst + weight_sentiment + weight_diversity
        if total_w <= 0:
            total_w = 1.0
        self.w_burst = weight_burst / total_w
        self.w_sentiment = weight_sentiment / total_w
        self.w_diversity = weight_diversity / total_w

    def _normalize_z_score(self, z: float) -> float:
        """Logarithmically compress Z-scores to [0.0, 1.0] scale."""
        if z <= 0.0:
            return 0.0
        # log(1 + z) / log(1 + 15) caps ~15 z-score to 1.0 smoothly
        return min(1.0, math.log1p(z) / math.log1p(15.0))

    def compute_participant_diversity(self, authors: List[str]) -> float:
        """Compute ratio of unique participants to total interactions.
        
        D = len(unique_authors) / len(authors).
        """
        if not authors:
            return 0.0
        return min(1.0, len(set(authors)) / len(authors))

    def rank_burst_alerts(
        self,
        alerts: List[BurstAlert],
        posts_by_term: Optional[Dict[str, List[CanonicalPost]]] = None,
        sentiment_scores: Optional[Dict[str, float]] = None,
    ) -> List[TrendRankingResult]:
        """Rank statistical burst alerts into TrendRankingResult items.
        
        Args:
            alerts: List of BurstAlerts.
            posts_by_term: Optional mapping of term -> CanonicalPosts in current window.
            sentiment_scores: Optional precomputed sentiment intensity [0.0, 1.0] per term.
            
        Returns:
            Ranked list of TrendRankingResults.
        """
        results: List[TrendRankingResult] = []
        posts_by_term = posts_by_term or {}
        sentiment_scores = sentiment_scores or {}

        for alert in alerts:
            norm_burst = self._normalize_z_score(alert.z_score)
            posts = posts_by_term.get(alert.term, [])

            # Diversity
            if posts:
                authors = [p.author_id for p in posts]
                diversity = self.compute_participant_diversity(authors)
                sample_texts = [p.text for p in posts[:3]]
                volume = len(posts)
                unique_authors = len(set(authors))
            else:
                diversity = 0.5  # Neutral assumption when posts not provided
                sample_texts = []
                volume = alert.current_count
                unique_authors = max(1, int(volume * 0.5))

            # Sentiment momentum
            sentiment = sentiment_scores.get(alert.term, 0.5)

            composite = (
                (self.w_burst * norm_burst)
                + (self.w_sentiment * sentiment)
                + (self.w_diversity * diversity)
            )

            results.append(
                TrendRankingResult(
                    trend_id=f"burst_{alert.term.lstrip('#')}",
                    name=alert.term,
                    trend_type="burst_term",
                    composite_score=round(composite, 4),
                    burst_acceleration=round(norm_burst, 4),
                    sentiment_momentum=round(sentiment, 4),
                    participant_diversity=round(diversity, 4),
                    total_volume=volume,
                    unique_authors=unique_authors,
                    top_keywords=[alert.term],
                    sample_texts=sample_texts,
                )
            )

        results.sort(key=lambda r: r.composite_score, reverse=True)
        return results

    def rank_dynamic_topics(
        self,
        topics: List[TopicRepresentation],
        timelines: Optional[List[DynamicTopicTimeline]] = None,
        posts_by_topic: Optional[Dict[int, List[CanonicalPost]]] = None,
        sentiment_scores: Optional[Dict[int, float]] = None,
    ) -> List[TrendRankingResult]:
        """Rank dynamic topics into TrendRankingResult items.
        
        Args:
            topics: List of TopicRepresentation objects.
            timelines: Optional list of DynamicTopicTimeline objects.
            posts_by_topic: Optional mapping of topic_id -> CanonicalPosts.
            sentiment_scores: Optional precomputed sentiment intensity per topic.
            
        Returns:
            Ranked list of TrendRankingResults.
        """
        results: List[TrendRankingResult] = []
        timelines_map = {t.topic_id: t for t in (timelines or [])}
        posts_by_topic = posts_by_topic or {}
        sentiment_scores = sentiment_scores or {}

        for topic in topics:
            timeline = timelines_map.get(topic.topic_id)
            # Velocity / acceleration across timeline bins
            if timeline and len(timeline.frequencies) >= 2:
                earliest = max(1, timeline.frequencies[0])
                latest = timeline.frequencies[-1]
                growth_ratio = (latest - earliest) / earliest
                norm_burst = min(1.0, max(0.0, growth_ratio / 3.0))
            else:
                norm_burst = min(1.0, topic.doc_count / 20.0)

            posts = posts_by_topic.get(topic.topic_id, [])
            if posts:
                authors = [p.author_id for p in posts]
                diversity = self.compute_participant_diversity(authors)
                sample_texts = [p.text for p in posts[:3]]
                unique_authors = len(set(authors))
                volume = len(posts)
            else:
                diversity = 0.6
                sample_texts = list(topic.representative_docs[:3])
                volume = topic.doc_count
                unique_authors = max(1, int(volume * 0.6))

            sentiment = sentiment_scores.get(topic.topic_id, 0.5)

            composite = (
                (self.w_burst * norm_burst)
                + (self.w_sentiment * sentiment)
                + (self.w_diversity * diversity)
            )

            top_kw = [word for word, _ in topic.top_words[:5]]

            results.append(
                TrendRankingResult(
                    trend_id=f"topic_{topic.topic_id}",
                    name=topic.name,
                    trend_type="dynamic_topic",
                    composite_score=round(composite, 4),
                    burst_acceleration=round(norm_burst, 4),
                    sentiment_momentum=round(sentiment, 4),
                    participant_diversity=round(diversity, 4),
                    total_volume=volume,
                    unique_authors=unique_authors,
                    top_keywords=top_kw,
                    sample_texts=sample_texts,
                )
            )

        results.sort(key=lambda r: r.composite_score, reverse=True)
        return results

    def rank_all(
        self,
        burst_results: List[TrendRankingResult],
        topic_results: List[TrendRankingResult],
    ) -> List[TrendRankingResult]:
        """Combine and globally rank burst terms and dynamic topics together."""
        combined = list(burst_results) + list(topic_results)
        combined.sort(key=lambda r: r.composite_score, reverse=True)
        return combined
