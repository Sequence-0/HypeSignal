"""Dynamic narrative drift and sentiment inversion tracker (Pillar D).

Tracks semantic representation shifts using centroid cosine distance on
SentenceTransformer embeddings and detects sentiment polarity flips across
temporal windows using DuckDB post_analytics.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from hypesignal.storage.duckdb_manager import DuckDBManager

logger = logging.getLogger(__name__)


class NarrativeDriftAlert(BaseModel):
    """Alert for topic semantic shift and sentiment inversion across temporal windows."""
    topic: str
    cosine_drift: float = Field(..., ge=0.0, le=2.0)  # 1.0 - cosine_similarity
    is_semantic_drift: bool = False
    sentiment_before: float = Field(..., ge=-1.0, le=1.0)
    sentiment_after: float = Field(..., ge=-1.0, le=1.0)
    sentiment_delta: float = Field(..., ge=-2.0, le=2.0)  # after - before
    is_sentiment_inversion: bool = False
    alert_triggered: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


class NarrativeDriftTracker:
    """Tracks semantic centroid drift and sentiment inversion across consecutive time windows."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        sentence_transformer: Optional[SentenceTransformer] = None,
        drift_threshold: float = 0.25,
        inversion_threshold: float = 0.35,
    ) -> None:
        """Initialize tracker with embedding model and sensitivity thresholds.
        
        Args:
            model_name: HuggingFace sentence transformer identifier.
            device: Device target ('cpu', 'cuda', etc.).
            sentence_transformer: Optional shared SentenceTransformer instance.
            drift_threshold: Minimum cosine distance (1 - cos) to trigger semantic drift alert.
            inversion_threshold: Minimum change in sentiment score to consider significant.
        """
        self.device = device or "cpu"
        self.drift_threshold = float(drift_threshold)
        self.inversion_threshold = float(inversion_threshold)

        if sentence_transformer is not None:
            self.model = sentence_transformer
        else:
            logger.info("Loading SentenceTransformer %s on %s for narrative drift...", model_name, self.device)
            self.model = SentenceTransformer(model_name, device=self.device)

    def compute_centroid(self, texts: List[str]) -> np.ndarray:
        """Compute the normalized semantic centroid vector for a collection of documents."""
        clean_texts = [t.strip() for t in texts if t and t.strip()]
        if not clean_texts:
            dim = self.model.get_sentence_embedding_dimension() or 384
            return np.zeros(dim, dtype=np.float32)

        embeddings = self.model.encode(
            clean_texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        centroid = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-6:
            centroid = centroid / norm
        return centroid

    def compute_cosine_drift(self, texts_before: List[str], texts_after: List[str]) -> float:
        """Calculate semantic cosine distance (1 - cos_sim) between two window text corpora."""
        clean_before = [t.strip() for t in texts_before if t and t.strip()]
        clean_after = [t.strip() for t in texts_after if t and t.strip()]
        if not clean_before or not clean_after:
            return 0.0

        c1 = self.compute_centroid(clean_before)
        c2 = self.compute_centroid(clean_after)
        norm1 = np.linalg.norm(c1)
        norm2 = np.linalg.norm(c2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        cos_sim = float(np.dot(c1, c2) / (norm1 * norm2))
        cos_sim = max(-1.0, min(1.0, cos_sim))
        drift = 1.0 - cos_sim
        return round(float(drift), 4)

    def evaluate_sentiment_inversion(
        self,
        sentiments_before: List[float],
        sentiments_after: List[float],
    ) -> Tuple[float, float, float, bool]:
        """Evaluate net sentiment shift and check for sentiment inversion.
        
        Returns:
            Tuple of (mean_before, mean_after, delta, is_inversion).
        """
        if not sentiments_before or not sentiments_after:
            s1 = float(np.mean(sentiments_before)) if sentiments_before else 0.0
            s2 = float(np.mean(sentiments_after)) if sentiments_after else 0.0
            return round(s1, 4), round(s2, 4), round(s2 - s1, 4), False

        s1 = float(np.mean(sentiments_before))
        s2 = float(np.mean(sentiments_after))
        delta = s2 - s1

        # An inversion requires both windows to have opposite polarity with notable magnitude
        is_inversion = False
        if (s1 > 0.10 and s2 < -0.10) or (s1 < -0.10 and s2 > 0.10):
            is_inversion = True
        elif abs(delta) >= self.inversion_threshold and (s1 * s2 < 0):
            is_inversion = True

        return round(s1, 4), round(s2, 4), round(delta, 4), is_inversion

    def track_drift(
        self,
        topic: str,
        texts_before: List[str],
        texts_after: List[str],
        sentiments_before: Optional[List[float]] = None,
        sentiments_after: Optional[List[float]] = None,
    ) -> NarrativeDriftAlert:
        """Perform unified semantic drift and sentiment inversion evaluation across two corpora."""
        clean_before = [t.strip() for t in texts_before if t and t.strip()]
        clean_after = [t.strip() for t in texts_after if t and t.strip()]

        if not clean_before or not clean_after:
            # Handle empty window: insufficient data to evaluate semantic drift
            s_before, s_after, s_delta, is_inversion = self.evaluate_sentiment_inversion(
                sentiments_before=sentiments_before or [],
                sentiments_after=sentiments_after or [],
            )
            return NarrativeDriftAlert(
                topic=topic,
                cosine_drift=0.0,
                is_semantic_drift=False,
                sentiment_before=s_before,
                sentiment_after=s_after,
                sentiment_delta=s_delta,
                is_sentiment_inversion=is_inversion,
                alert_triggered=is_inversion,
                details={
                    "sample_count_before": len(clean_before),
                    "sample_count_after": len(clean_after),
                    "insufficient_data": True,
                    "drift_threshold": self.drift_threshold,
                    "inversion_threshold": self.inversion_threshold,
                },
            )

        # 1. Cosine semantic drift
        drift = self.compute_cosine_drift(clean_before, clean_after)
        is_semantic_drift = bool(drift >= self.drift_threshold)

        # 2. Sentiment inversion
        s_before, s_after, s_delta, is_inversion = self.evaluate_sentiment_inversion(
            sentiments_before=sentiments_before or [],
            sentiments_after=sentiments_after or [],
        )

        alert_triggered = is_semantic_drift or is_inversion

        return NarrativeDriftAlert(
            topic=topic,
            cosine_drift=drift,
            is_semantic_drift=is_semantic_drift,
            sentiment_before=s_before,
            sentiment_after=s_after,
            sentiment_delta=s_delta,
            is_sentiment_inversion=is_inversion,
            alert_triggered=alert_triggered,
            details={
                "sample_count_before": len(clean_before),
                "sample_count_after": len(clean_after),
                "insufficient_data": False,
                "drift_threshold": self.drift_threshold,
                "inversion_threshold": self.inversion_threshold,
            },
        )

    def track_drift_from_db(
        self,
        db: DuckDBManager,
        topic: str,
        window_1_start: datetime,
        window_1_end: datetime,
        window_2_start: datetime,
        window_2_end: datetime,
    ) -> NarrativeDriftAlert:
        """Query posts and analytics for topic from DuckDB across two windows and track drift.
        
        Args:
            db: DuckDBManager analytical store.
            topic: Target topic or keyword.
            window_1_start: Start timestamp of baseline window.
            window_1_end: End timestamp of baseline window.
            window_2_start: Start timestamp of active window.
            window_2_end: End timestamp of active window.
            
        Returns:
            NarrativeDriftAlert evaluating drift between window 1 and window 2.
        """
        def _fetch_window_data(t_start: datetime, t_end: datetime) -> Tuple[List[str], List[float]]:
            # Query posts joining post_analytics to fetch signed sentiment
            query = """
                SELECT p.text, a.effective_polarity, a.sentiment_score
                FROM posts p
                LEFT JOIN post_analytics a ON p.id = a.post_id
                WHERE p.timestamp >= ? AND p.timestamp <= ?
                  AND LOWER(p.text) LIKE LOWER(?)
                ORDER BY p.timestamp ASC;
            """
            topic_pat = f"%{topic.strip()}%"
            rows = db.con.execute(query, [t_start, t_end, topic_pat]).fetchall()
            texts: List[str] = []
            scores: List[float] = []

            for text, polarity, score in rows:
                if text:
                    texts.append(text)
                # Compute signed sentiment value in [-1.0, 1.0]
                if score is not None:
                    raw_score = float(score)
                    if polarity == "negative":
                        scores.append(-abs(raw_score))
                    elif polarity == "positive":
                        scores.append(abs(raw_score))
                    else:
                        scores.append(0.0)
                else:
                    scores.append(0.0)

            return texts, scores

        w1_texts, w1_scores = _fetch_window_data(window_1_start, window_1_end)
        w2_texts, w2_scores = _fetch_window_data(window_2_start, window_2_end)

        return self.track_drift(
            topic=topic,
            texts_before=w1_texts,
            texts_after=w2_texts,
            sentiments_before=w1_scores,
            sentiments_after=w2_scores,
        )
