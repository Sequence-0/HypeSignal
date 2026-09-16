"""Temporal sentiment, sarcasm, and emotion fluctuation tracker.

Calculates rolling chronological sentiment polarity, sarcasm prevalence,
and nuanced emotional trajectories across time buckets in DuckDB and Polars.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import polars as pl

from hypesignal.models.enums import EmotionType, SentimentPolarity
from hypesignal.nlp.engine import MultiDimensionalSentimentEngine
from hypesignal.storage.duckdb_manager import DuckDBManager
from hypesignal.timeline.timeline_manager import RE_INTERVAL

logger = logging.getLogger(__name__)


class TemporalSentimentTracker:
    """Tracks chronological fluctuations of sentiment, sarcasm, and emotions over time."""

    def __init__(
        self,
        engine: MultiDimensionalSentimentEngine,
        db: Optional[DuckDBManager] = None,
    ) -> None:
        """Initialize tracker with NLP inference engine and optional DuckDB store."""
        self.engine = engine
        self.db = db
        self._mem_db: Optional[DuckDBManager] = None
        self._lock = threading.Lock()

    def _get_db(self) -> DuckDBManager:
        """Get shared DuckDB instance or cached in-memory store."""
        if self.db is not None:
            return self.db
        with self._lock:
            if self._mem_db is None:
                self._mem_db = DuckDBManager(":memory:")
            return self._mem_db

    def compute_timeline_fluctuations(
        self,
        posts_df: pl.DataFrame,
        interval: str = "1 hour",
        batch_size: int = 32,
    ) -> pl.DataFrame:
        """Enrich posts with multi-dimensional predictions and aggregate into temporal buckets.
        
        Args:
            posts_df: Polars DataFrame containing ['id', 'text', 'timestamp'] columns.
            interval: DuckDB time-bucket interval (e.g. '1 hour', '1 day').
            batch_size: Inference batch size.
            
        Returns:
            Polars DataFrame containing chronological sentiment and emotion metrics.
        """
        if posts_df.is_empty():
            return pl.DataFrame()

        interval_clean = interval.strip()
        if not RE_INTERVAL.match(interval_clean):
            raise ValueError(f"Invalid interval: {interval}")

        texts = posts_df["text"].to_list()
        results = self.engine.analyze_multidimensional(texts, batch_size=batch_size)

        # Build enrichment columns
        polarities = [r.effective_polarity.value for r in results]
        adjusted_scores = [r.adjusted_sentiment_score for r in results]
        is_sarcastic = [r.is_sarcasm_inverted or r.irony.is_ironic for r in results]
        irony_scores = [r.irony.irony_score for r in results]
        emotions = [r.emotion.primary_emotion.value for r in results]

        enriched_df = posts_df.with_columns([
            pl.Series("effective_polarity", polarities),
            pl.Series("adjusted_sentiment_score", adjusted_scores),
            pl.Series("is_sarcastic", is_sarcastic),
            pl.Series("irony_score", irony_scores),
            pl.Series("primary_emotion", emotions),
        ])

        # Use reused DuckDB instance to compute time-bucketed aggregations
        db = self._get_db()
        arrow_table = enriched_df.to_arrow()

        query = f"""
            SELECT 
                time_bucket(INTERVAL '{interval_clean}', timestamp) as bucket,
                COUNT(*) as post_count,
                AVG(adjusted_sentiment_score) as mean_sentiment_score,
                SUM(CASE WHEN effective_polarity = 'positive' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as positive_ratio,
                SUM(CASE WHEN effective_polarity = 'negative' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as negative_ratio,
                SUM(CASE WHEN effective_polarity = 'neutral' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as neutral_ratio,
                SUM(CASE WHEN is_sarcastic THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as sarcasm_rate,
                AVG(irony_score) as mean_irony_score,
                SUM(CASE WHEN primary_emotion = 'joy' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as joy_ratio,
                SUM(CASE WHEN primary_emotion IN ('fear', 'anxiety') THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as anxiety_ratio,
                SUM(CASE WHEN primary_emotion = 'anger' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as anger_ratio,
                SUM(CASE WHEN primary_emotion = 'sadness' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as sadness_ratio,
                MODE(primary_emotion) as dominant_emotion
            FROM tmp_enriched
            GROUP BY bucket
            ORDER BY bucket ASC
        """
        with self._lock:
            db.con.register("tmp_enriched", arrow_table)
            try:
                aggregated_df = db.con.execute(query).pl()
            finally:
                db.con.unregister("tmp_enriched")

        return aggregated_df

    def track_historical_window(
        self,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1 hour",
        limit: int = 1000,
    ) -> pl.DataFrame:
        """Extract and analyze chronological sentiment from the primary DuckDB database."""
        if self.db is None:
            raise ValueError("DuckDBManager must be provided to track historical window.")

        posts_df = self.db.get_posts_in_window(start_time, end_time, limit=limit)
        return self.compute_timeline_fluctuations(posts_df, interval=interval)
