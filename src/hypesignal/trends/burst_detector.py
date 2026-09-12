"""Statistical burstiness detection engine (Tier 1).

Computes term velocity and frequency acceleration using Z-score deviation over
historical moving average baseline windows to fire real-time trend alerts.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from hypesignal.storage.duckdb_manager import DuckDBManager
from hypesignal.trends.schemas import BurstAlert

logger = logging.getLogger(__name__)

RE_HASHTAG = re.compile(r"#(\w+)", re.UNICODE)
RE_MENTION = re.compile(r"@(\w+)", re.UNICODE)
RE_WORD = re.compile(r"\b[a-zA-Z]{3,}\b")

# Concise standard stopword filter for keyword burst extraction
DEFAULT_STOPWORDS: Set[str] = {
    "the", "and", "is", "in", "to", "of", "it", "that", "you", "for", "on",
    "with", "as", "at", "by", "from", "up", "about", "into", "over", "after",
    "this", "have", "from", "or", "one", "had", "by", "word", "but", "not",
    "what", "all", "were", "we", "when", "your", "can", "said", "there", "use",
    "an", "each", "which", "she", "do", "how", "their", "if", "will", "way",
    "about", "many", "then", "them", "write", "would", "like", "so", "these",
    "her", "long", "make", "thing", "see", "him", "two", "has", "look", "more",
    "day", "could", "go", "come", "did", "number", "sound", "no", "most",
    "people", "my", "over", "know", "water", "than", "call", "first", "who",
    "may", "down", "side", "been", "now", "find", "any", "new", "work", "part",
    "take", "get", "place", "made", "live", "where", "after", "back", "little",
    "only", "round", "man", "year", "came", "show", "every", "good", "me",
    "give", "our", "under", "name", "very", "through", "just", "form", "sentence",
    "great", "think", "say", "help", "low", "line", "differ", "turn", "cause",
    "much", "mean", "before", "move", "right", "boy", "old", "too", "same",
    "tell", "does", "set", "three", "want", "air", "well", "also", "play",
    "small", "end", "put", "home", "read", "hand", "port", "large", "spell",
    "add", "even", "land", "here", "must", "big", "high", "such", "follow",
    "act", "why", "ask", "men", "change", "went", "light", "kind", "off",
    "need", "house", "picture", "try", "us", "again", "animal", "point",
    "mother", "world", "near", "build", "self", "earth", "father", "http",
    "https", "com", "www", "rt", "amp",
}


class BurstDetector:
    """Tier-1 Statistical Burstiness Engine for real-time trend detection."""

    def __init__(
        self,
        default_z_threshold: float = 2.5,
        default_min_count: int = 3,
        stopwords: Optional[Set[str]] = None,
    ) -> None:
        """Initialize BurstDetector.
        
        Args:
            default_z_threshold: Standard deviations above baseline to flag as a burst.
            default_min_count: Minimum raw frequency in current window.
            stopwords: Set of terms to ignore during keyword extraction.
        """
        self.default_z_threshold = default_z_threshold
        self.default_min_count = default_min_count
        self.stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS

    def extract_terms_from_post(
        self,
        text: str,
        hashtags: Optional[List[str]] = None,
        target: str = "all",
    ) -> List[Tuple[str, str]]:
        """Extract terms and their category ('hashtag', 'keyword') from post content.
        
        Args:
            text: Post text.
            hashtags: Pre-extracted hashtags list if available.
            target: 'hashtags', 'keywords', or 'all'.
            
        Returns:
            List of (term, term_type) tuples.
        """
        terms: List[Tuple[str, str]] = []

        # 1. Hashtags
        if target in ("hashtags", "all"):
            if hashtags:
                for h in hashtags:
                    tag = h.lower().lstrip("#")
                    if tag:
                        terms.append((f"#{tag}", "hashtag"))
            else:
                for match in RE_HASHTAG.finditer(text):
                    tag = match.group(1).lower()
                    if tag:
                        terms.append((f"#{tag}", "hashtag"))

        # 2. Keywords
        if target in ("keywords", "all"):
            # Remove hashtags and URLs before extracting words
            cleaned_text = RE_HASHTAG.sub(" ", text)
            cleaned_text = re.sub(r"https?://\S+", " ", cleaned_text)
            words = RE_WORD.findall(cleaned_text.lower())
            for w in words:
                if w not in self.stopwords and len(w) >= 3:
                    terms.append((w, "keyword"))

        return terms

    def compute_burst_metrics(
        self,
        baseline_counts: List[int],
        current_count: int,
        window_duration_seconds: float,
    ) -> Tuple[float, float, float, float]:
        """Compute mean, std, Z-score, and velocity for a single term across windows.
        
        Args:
            baseline_counts: Historical observation counts across baseline windows.
            current_count: Raw count in current observation window.
            window_duration_seconds: Length of a single window in seconds.
            
        Returns:
            Tuple of (baseline_mean, baseline_std, z_score, velocity).
        """
        if not baseline_counts:
            mean = 0.0
            std = 0.0
        else:
            mean = float(np.mean(baseline_counts))
            std = float(np.std(baseline_counts, ddof=1)) if len(baseline_counts) > 1 else 0.0

        # Poisson variance lower bound to prevent division by zero / single-occurrence explosions
        effective_std = max(std, math.sqrt(max(mean, 1.0)))

        z_score = (current_count - mean) / effective_std
        window_hours = max(window_duration_seconds / 3600.0, 0.001)
        velocity = (current_count - mean) / window_hours

        return float(round(mean, 3)), float(round(std, 3)), float(round(z_score, 3)), float(round(velocity, 3))

    def detect_bursts_from_events(
        self,
        events: List[Tuple[datetime, str, str]],
        current_window_end: datetime,
        window_size: timedelta = timedelta(hours=1),
        num_baseline_windows: int = 5,
        z_threshold: Optional[float] = None,
        min_count: Optional[int] = None,
    ) -> List[BurstAlert]:
        """Detect bursts from an in-memory stream of (timestamp, term, term_type) tuples.
        
        Args:
            events: List of (timestamp, term, term_type).
            current_window_end: Upper timestamp bound for the current window.
            window_size: Duration of each analysis window.
            num_baseline_windows: Number of prior consecutive windows for baseline.
            z_threshold: Z-score threshold for burst alert.
            min_count: Minimum raw frequency in current window.
            
        Returns:
            List of BurstAlert instances sorted by Z-score descending.
        """
        z_thresh = z_threshold if z_threshold is not None else self.default_z_threshold
        min_cnt = min_count if min_count is not None else self.default_min_count

        total_windows = num_baseline_windows + 1
        window_sec = window_size.total_seconds()
        timeline_start = current_window_end - (window_size * total_windows)
        current_window_start = current_window_end - window_size

        # term -> array of length total_windows
        term_type_map: Dict[str, str] = {}
        term_counts: Dict[str, List[int]] = {}

        for ts, term, term_type in events:
            if ts < timeline_start or ts > current_window_end:
                continue

            delta_sec = (ts - timeline_start).total_seconds()
            win_idx = int(delta_sec // window_sec)
            # Bound within [0, total_windows - 1]
            win_idx = max(0, min(total_windows - 1, win_idx))

            if term not in term_counts:
                term_counts[term] = [0] * total_windows
                term_type_map[term] = term_type
            term_counts[term][win_idx] += 1

        alerts: List[BurstAlert] = []

        for term, counts in term_counts.items():
            baseline_counts = counts[:num_baseline_windows]
            curr_count = counts[num_baseline_windows]

            mean, std, z, vel = self.compute_burst_metrics(
                baseline_counts=baseline_counts,
                current_count=curr_count,
                window_duration_seconds=window_sec,
            )

            is_burst = bool(z >= z_thresh and curr_count >= min_cnt)

            alerts.append(
                BurstAlert(
                    term=term,
                    term_type=term_type_map.get(term, "keyword"),
                    current_count=curr_count,
                    baseline_mean=mean,
                    baseline_std=std,
                    z_score=z,
                    velocity=vel,
                    window_start=current_window_start,
                    window_end=current_window_end,
                    is_burst=is_burst,
                )
            )

        # Sort by Z-score descending
        alerts.sort(key=lambda a: a.z_score, reverse=True)
        return alerts

    def detect_bursts_from_duckdb(
        self,
        db: DuckDBManager,
        current_window_end: Optional[datetime] = None,
        window_duration_minutes: int = 60,
        num_baseline_windows: int = 5,
        z_threshold: Optional[float] = None,
        min_count: Optional[int] = None,
        target: str = "all",
    ) -> List[BurstAlert]:
        """Detect bursts directly from DuckDB analytical posts table.
        
        Args:
            db: DuckDBManager instance.
            current_window_end: End of current window (defaults to MAX(timestamp) in posts).
            window_duration_minutes: Duration of window bin in minutes.
            num_baseline_windows: Number of baseline intervals.
            z_threshold: Minimum Z-score to trigger burst alert.
            min_count: Minimum occurrence count in current window.
            target: 'hashtags', 'keywords', or 'all'.
            
        Returns:
            List of BurstAlert objects sorted by Z-score descending.
        """
        # Determine latest timestamp if not supplied
        if current_window_end is None:
            max_ts_row = db.con.execute("SELECT MAX(timestamp) FROM posts;").fetchone()
            if not max_ts_row or max_ts_row[0] is None:
                return []
            current_window_end = max_ts_row[0]
            if current_window_end.tzinfo is None:
                current_window_end = current_window_end.replace(tzinfo=timezone.utc)

        window_size = timedelta(minutes=window_duration_minutes)
        total_span = window_size * (num_baseline_windows + 1)
        start_time = current_window_end - total_span

        # Fetch posts within timeline bounds
        query = """
            SELECT text, hashtags, timestamp
            FROM posts
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC;
        """
        rows = db.con.execute(query, [start_time, current_window_end]).fetchall()
        if not rows:
            return []

        events: List[Tuple[datetime, str, str]] = []
        for text, hashtags, ts in rows:
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            extracted = self.extract_terms_from_post(text=text, hashtags=hashtags, target=target)
            for term, term_type in extracted:
                events.append((ts, term, term_type))

        return self.detect_bursts_from_events(
            events=events,
            current_window_end=current_window_end,
            window_size=window_size,
            num_baseline_windows=num_baseline_windows,
            z_threshold=z_threshold,
            min_count=min_count,
        )
