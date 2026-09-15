"""Observed cross-segment information diffusion and sentiment trajectory tracker (Pillar E).

Tracks real chronological spread across Louvain communities and demographic
cohorts, measuring inter-segment transmission latencies and cross-segment
sentiment drift.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from hypesignal.storage.duckdb_manager import DuckDBManager

logger = logging.getLogger(__name__)


class SegmentAdoption(BaseModel):
    """A single user adoption event with associated segment metadata and sentiment."""
    user_id: str
    timestamp: datetime
    community_id: Optional[int] = None
    age_bracket: Optional[str] = None
    persona: Optional[str] = None
    sentiment_score: Optional[float] = None
    effective_polarity: Optional[str] = None


class SegmentTransition(BaseModel):
    """Inter-segment transmission hop from an upstream segment to a downstream segment."""
    from_segment: str
    to_segment: str
    segment_type: str  # 'community', 'age_bracket', 'persona'
    latency_seconds: float = Field(..., ge=0.0)
    source_user_id: str
    target_user_id: str
    sentiment_drift: float  # downstream sentiment - upstream sentiment


class CrossSegmentDiffusionReport(BaseModel):
    """Reconstructed cross-segment diffusion sequence and sentiment dynamics."""
    identifier: str  # cascade_id or topic
    segment_type: str  # 'community', 'age_bracket', 'persona'
    origin_segment: str
    origin_sentiment: float
    total_adoptions: int
    segments_reached: List[str]
    chronological_sequence: List[str]  # e.g. ['community_1', 'community_2', ...]
    transitions: List[SegmentTransition]
    segment_sentiments: Dict[str, float]  # mean sentiment per segment
    cross_segment_sentiment_drift: float  # final reached segment sentiment - origin segment sentiment


class CrossSegmentDiffusionTracker:
    """Tracks chronological diffusion across modular communities and demographic segments."""

    def __init__(self, db: Optional[DuckDBManager] = None) -> None:
        """Initialize tracker with optional DuckDBManager reference."""
        self.db = db

    def _resolve_segment_key(self, adoption: SegmentAdoption, segment_by: str) -> str:
        """Extract standardized segment identifier from adoption record."""
        segment_type = segment_by.lower().strip()
        if segment_type == "community":
            if adoption.community_id is not None:
                return f"community_{adoption.community_id}"
            return "community_unknown"
        elif segment_type == "age_bracket":
            return adoption.age_bracket or "unknown_age"
        elif segment_type == "persona":
            return adoption.persona or "unknown_persona"
        return f"{segment_type}_unknown"

    def track_diffusion_events(
        self,
        identifier: str,
        adoptions: List[SegmentAdoption],
        segment_by: str = "community",
    ) -> CrossSegmentDiffusionReport:
        """Reconstruct chronological cross-segment diffusion sequence from adoption events.
        
        Args:
            identifier: Unique cascade identifier or topic keyword.
            adoptions: List of timestamped SegmentAdoption instances.
            segment_by: Dimension to segment by ('community', 'age_bracket', 'persona').
            
        Returns:
            CrossSegmentDiffusionReport containing sequence, latencies, and sentiment drift.
        """
        if not adoptions:
            return CrossSegmentDiffusionReport(
                identifier=identifier,
                segment_type=segment_by,
                origin_segment="none",
                origin_sentiment=0.0,
                total_adoptions=0,
                segments_reached=[],
                chronological_sequence=[],
                transitions=[],
                segment_sentiments={},
                cross_segment_sentiment_drift=0.0,
            )

        # 1. Sort adoptions chronologically
        sorted_adoptions = sorted(adoptions, key=lambda a: a.timestamp)

        # 2. Collect sentiments and track first arrival per segment
        segment_scores: Dict[str, List[float]] = defaultdict(list)
        segment_first_seen: Dict[str, Tuple[datetime, str]] = {}
        chronological_sequence: List[str] = []

        for a in sorted_adoptions:
            seg_key = self._resolve_segment_key(a, segment_by)

            # Record sentiment if present
            if a.sentiment_score is not None:
                # Sign score according to polarity if available
                score = float(a.sentiment_score)
                if a.effective_polarity == "negative":
                    score = -abs(score)
                elif a.effective_polarity == "positive":
                    score = abs(score)
                segment_scores[seg_key].append(score)

            # Record first arrival in this segment
            if seg_key not in segment_first_seen:
                segment_first_seen[seg_key] = (a.timestamp, a.user_id)
                chronological_sequence.append(seg_key)

        # 3. Compute mean sentiment per segment
        segment_sentiments: Dict[str, float] = {}
        for seg, scores in segment_scores.items():
            segment_sentiments[seg] = round(float(np.mean(scores)), 4) if scores else 0.0

        # Fill missing segments with 0.0 sentiment if no sentiment scores
        for seg in chronological_sequence:
            if seg not in segment_sentiments:
                segment_sentiments[seg] = 0.0

        origin_segment = chronological_sequence[0]
        origin_sentiment = segment_sentiments.get(origin_segment, 0.0)

        # 4. Construct transitions between consecutively reached segments
        transitions: List[SegmentTransition] = []
        for i in range(len(chronological_sequence) - 1):
            from_seg = chronological_sequence[i]
            to_seg = chronological_sequence[i + 1]

            t_from, u_from = segment_first_seen[from_seg]
            t_to, u_to = segment_first_seen[to_seg]

            latency = max(0.0, (t_to - t_from).total_seconds())
            s_from = segment_sentiments.get(from_seg, 0.0)
            s_to = segment_sentiments.get(to_seg, 0.0)
            drift = round(s_to - s_from, 4)

            transitions.append(
                SegmentTransition(
                    from_segment=from_seg,
                    to_segment=to_seg,
                    segment_type=segment_by,
                    latency_seconds=latency,
                    source_user_id=u_from,
                    target_user_id=u_to,
                    sentiment_drift=drift,
                )
            )

        # Overall drift: final segment sentiment - origin segment sentiment
        final_segment = chronological_sequence[-1]
        overall_drift = round(segment_sentiments.get(final_segment, 0.0) - origin_sentiment, 4)

        return CrossSegmentDiffusionReport(
            identifier=identifier,
            segment_type=segment_by,
            origin_segment=origin_segment,
            origin_sentiment=origin_sentiment,
            total_adoptions=len(sorted_adoptions),
            segments_reached=list(segment_first_seen.keys()),
            chronological_sequence=chronological_sequence,
            transitions=transitions,
            segment_sentiments=segment_sentiments,
            cross_segment_sentiment_drift=overall_drift,
        )

    def track_cascade_from_db(
        self,
        db: DuckDBManager,
        cascade_id: str,
        segment_by: str = "community",
    ) -> CrossSegmentDiffusionReport:
        """Query cascade events and join with user communities, demographics, and analytics."""
        valid_segments = {"community", "age_bracket", "persona"}
        if segment_by.lower().strip() not in valid_segments:
            raise ValueError(f"Unsupported segment_by='{segment_by}'. Must be one of {valid_segments}")

        query = """
            SELECT 
                ce.user_id,
                ce.timestamp,
                uc.community_id,
                pa.effective_polarity,
                pa.sentiment_score,
                COALESCE(u.extra_metadata->>'$.age_bracket', u.extra_metadata->>'$.persona.age_bracket') AS age_bracket,
                COALESCE(u.extra_metadata->>'$.primary_persona', u.extra_metadata->>'$.persona.primary_persona', u.extra_metadata->>'$.persona.name', u.extra_metadata->>'$.persona') AS persona
            FROM cascade_events ce
            LEFT JOIN user_communities uc ON ce.user_id = uc.user_id
            LEFT JOIN post_analytics pa ON ce.post_id = pa.post_id
            LEFT JOIN users u ON ce.user_id = u.id
            WHERE ce.cascade_id = ?
            ORDER BY ce.timestamp ASC;
        """
        rows = db.con.execute(query, [cascade_id]).fetchall()
        adoptions: List[SegmentAdoption] = []

        for uid, ts, comm_id, pol, score, age, pers in rows:
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            adoptions.append(
                SegmentAdoption(
                    user_id=str(uid),
                    timestamp=ts,
                    community_id=int(comm_id) if comm_id is not None else None,
                    age_bracket=age,
                    persona=pers,
                    effective_polarity=pol,
                    sentiment_score=float(score) if score is not None else None,
                )
            )

        return self.track_diffusion_events(
            identifier=cascade_id,
            adoptions=adoptions,
            segment_by=segment_by,
        )

    def track_topic_diffusion_from_db(
        self,
        db: DuckDBManager,
        topic: str,
        start_time: datetime,
        end_time: datetime,
        segment_by: str = "community",
    ) -> CrossSegmentDiffusionReport:
        """Query posts mentioning topic and join with user communities, demographics, and analytics."""
        valid_segments = {"community", "age_bracket", "persona"}
        if segment_by.lower().strip() not in valid_segments:
            raise ValueError(f"Unsupported segment_by='{segment_by}'. Must be one of {valid_segments}")

        query = """
            SELECT 
                p.author_id AS user_id,
                p.timestamp,
                uc.community_id,
                pa.effective_polarity,
                pa.sentiment_score,
                COALESCE(u.extra_metadata->>'$.age_bracket', u.extra_metadata->>'$.persona.age_bracket') AS age_bracket,
                COALESCE(u.extra_metadata->>'$.primary_persona', u.extra_metadata->>'$.persona.primary_persona', u.extra_metadata->>'$.persona.name', u.extra_metadata->>'$.persona') AS persona
            FROM posts p
            LEFT JOIN user_communities uc ON p.author_id = uc.user_id
            LEFT JOIN post_analytics pa ON p.id = pa.post_id
            LEFT JOIN users u ON p.author_id = u.id
            WHERE p.timestamp >= ? AND p.timestamp <= ?
              AND LOWER(p.text) LIKE LOWER(?)
            ORDER BY p.timestamp ASC;
        """
        topic_pattern = f"%{topic.strip()}%"
        rows = db.con.execute(query, [start_time, end_time, topic_pattern]).fetchall()
        adoptions: List[SegmentAdoption] = []

        for uid, ts, comm_id, pol, score, age, pers in rows:
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            adoptions.append(
                SegmentAdoption(
                    user_id=str(uid),
                    timestamp=ts,
                    community_id=int(comm_id) if comm_id is not None else None,
                    age_bracket=age,
                    persona=pers,
                    effective_polarity=pol,
                    sentiment_score=float(score) if score is not None else None,
                )
            )

        return self.track_diffusion_events(
            identifier=topic,
            adoptions=adoptions,
            segment_by=segment_by,
        )
