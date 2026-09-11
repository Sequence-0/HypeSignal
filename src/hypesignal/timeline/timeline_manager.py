"""Chronological Timeline Manager for HypeSignal.

Provides time-window slicing, historical conversation replay iterators,
time-series bucketed volume aggregations, and cascade diffusion chronology
over the DuckDB analytical store.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

import polars as pl

from hypesignal.models.canonical import CanonicalCascadeEvent, CanonicalPost
from hypesignal.storage.duckdb_manager import DuckDBManager

RE_INTERVAL = re.compile(r"^\d+\s+(?:second|minute|hour|day|week|month|year)s?$", re.IGNORECASE)


class TimelineManager:
    """Manages chronological timeline queries and replay on top of DuckDB."""

    def __init__(self, db: DuckDBManager) -> None:
        """Initialize TimelineManager with a connected DuckDBManager instance."""
        self.db = db

    def get_time_bounds(self) -> Dict[str, Optional[datetime]]:
        """Retrieve earliest and latest post timestamps in the database."""
        res = self.db.query("""
            SELECT 
                MIN(timestamp) as earliest,
                MAX(timestamp) as latest,
                COUNT(*) as total_posts
            FROM posts
        """)
        if res.is_empty() or res["earliest"][0] is None:
            return {"earliest": None, "latest": None, "total_posts": 0}
        return {
            "earliest": res["earliest"][0],
            "latest": res["latest"][0],
            "total_posts": res["total_posts"][0],
        }

    def get_timeline_slice(
        self,
        start_time: datetime,
        end_time: datetime,
        platform: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pl.DataFrame:
        """Extract a chronological slice of posts within [start_time, end_time]."""
        return self.db.get_posts_in_window(
            start_time=start_time,
            end_time=end_time,
            platform=platform,
            limit=limit,
        )

    def replay_timeline(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        batch_size: int = 500,
    ) -> Iterator[List[Dict[str, Any]]]:
        """Chronologically replay historical posts as micro-batches.
        
        Yields batches of posts ordered by timestamp ASC, simulating a live stream.
        """
        where_clauses: List[str] = []
        params: List[Any] = []

        if start_time:
            where_clauses.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            where_clauses.append("timestamp <= ?")
            params.append(end_time)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        offset = 0

        while True:
            query = f"""
                SELECT * FROM posts 
                {where_sql}
                ORDER BY timestamp_ms ASC
                LIMIT {batch_size} OFFSET {offset}
            """
            df_chunk = self.db.query(query, params)
            if df_chunk.is_empty():
                break

            records = df_chunk.to_dicts()
            yield records
            offset += len(records)
            if len(records) < batch_size:
                break

    def get_activity_timeseries(
        self,
        interval: str = "1 hour",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """Aggregate post volumes into regular time intervals.
        
        Args:
            interval: DuckDB time interval string (e.g., '10 minutes', '1 hour', '1 day').
            start_time: Optional start cutoff.
            end_time: Optional end cutoff.
            
        Returns:
            Polars DataFrame with columns ['bucket', 'post_count'].
        """
        interval_clean = interval.strip()
        if not RE_INTERVAL.match(interval_clean):
            raise ValueError(
                f"Invalid time interval '{interval}'. Must be of the form '<number> <unit>', "
                "e.g. '1 hour', '10 minutes', '1 day'."
            )

        where_clauses: List[str] = []
        params: List[Any] = []

        if start_time:
            where_clauses.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            where_clauses.append("timestamp <= ?")
            params.append(end_time)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        query = f"""
            SELECT 
                time_bucket(INTERVAL '{interval}', timestamp) as bucket,
                COUNT(*) as post_count
            FROM posts
            {where_sql}
            GROUP BY bucket
            ORDER BY bucket ASC
        """
        return self.db.query(query, params)

    def get_cascade_chronology(self, cascade_id: str) -> pl.DataFrame:
        """Get chronological propagation trace of a diffusion cascade with relative time lags."""
        df = self.db.get_cascade_series(cascade_id)
        if df.is_empty():
            return df

        # Calculate delta_seconds from the origin of the cascade
        min_ts = df["timestamp_ms"].min()
        df = df.with_columns(
            ((pl.col("timestamp_ms") - min_ts) / 1000.0).alias("seconds_since_origin")
        )
        return df

    def get_user_timeline(self, user_id: str, limit: int = 100) -> pl.DataFrame:
        """Retrieve chronological post history for an individual user."""
        query = """
            SELECT * FROM posts
            WHERE author_id = ?
            ORDER BY timestamp_ms ASC
            LIMIT ?
        """
        return self.db.query(query, [str(user_id), limit])
