"""DuckDB analytical storage manager for HypeSignal.

Handles high-throughput columnar storage, timeline indexing, and OLAP
analytical aggregations for posts, users, and diffusion cascades.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import duckdb
import polars as pl

from hypesignal.models.canonical import (
    CanonicalCascadeEvent,
    CanonicalPost,
    CanonicalUser,
)


class DuckDBManager:
    """Manages the DuckDB embedded analytical store."""

    def __init__(self, db_path: Optional[Union[str, Path]] = ":memory:") -> None:
        """Initialize connection to DuckDB.
        
        Args:
            db_path: File path to database or ':memory:' for transient in-memory mode.
        """
        if db_path is None or str(db_path) == ":memory:":
            self.db_path = ":memory:"
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(db_path)

        self.con = duckdb.connect(self.db_path)
        self._init_schemas()

    def _init_schemas(self) -> None:
        """Initialize relational analytical schemas and indexes."""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id VARCHAR PRIMARY KEY,
                platform VARCHAR NOT NULL,
                author_id VARCHAR NOT NULL,
                author_screen_name VARCHAR,
                text TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                timestamp_ms BIGINT NOT NULL,
                parent_id VARCHAR,
                reply_to_user_id VARCHAR,
                source_client VARCHAR,
                urls VARCHAR[],
                hashtags VARCHAR[],
                mentions VARCHAR[],
                metrics JSON,
                extra_metadata JSON
            );

            CREATE TABLE IF NOT EXISTS users (
                id VARCHAR PRIMARY KEY,
                platform VARCHAR NOT NULL,
                screen_name VARCHAR,
                location_raw VARCHAR,
                latitude DOUBLE,
                longitude DOUBLE,
                indegree INTEGER,
                outdegree INTEGER,
                bio TEXT,
                metrics JSON,
                extra_metadata JSON
            );

            CREATE TABLE IF NOT EXISTS cascade_events (
                cascade_id VARCHAR NOT NULL,
                post_id VARCHAR NOT NULL,
                user_id VARCHAR NOT NULL,
                user_screen_name VARCHAR,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                timestamp_ms BIGINT NOT NULL,
                adoption_order INTEGER NOT NULL,
                parent_event_id VARCHAR,
                extra_metadata JSON,
                PRIMARY KEY (cascade_id, post_id, adoption_order)
            );

            CREATE INDEX IF NOT EXISTS idx_posts_time ON posts (timestamp);
            CREATE INDEX IF NOT EXISTS idx_posts_time_ms ON posts (timestamp_ms);
            CREATE INDEX IF NOT EXISTS idx_posts_author ON posts (author_id);
            CREATE INDEX IF NOT EXISTS idx_cascades_id_time ON cascade_events (cascade_id, timestamp_ms);
            CREATE INDEX IF NOT EXISTS idx_users_platform ON users (platform, id);
        """)

    def insert_posts(self, posts: List[CanonicalPost]) -> None:
        """Insert a batch of CanonicalPost instances."""
        if not posts:
            return

        records = [
            {
                "id": p.id,
                "platform": p.platform.value,
                "author_id": p.author_id,
                "author_screen_name": p.author_screen_name,
                "text": p.text,
                "timestamp": p.timestamp,
                "timestamp_ms": p.timestamp_ms,
                "parent_id": p.parent_id,
                "reply_to_user_id": p.reply_to_user_id,
                "source_client": p.source_client,
                "urls": p.urls,
                "hashtags": p.hashtags,
                "mentions": p.mentions,
                "metrics": json.dumps(p.metrics.model_dump()),
                "extra_metadata": json.dumps(p.extra_metadata),
            }
            for p in posts
        ]
        df = pl.DataFrame(records)
        self.insert_posts_df(df)

    def insert_posts_df(self, df: pl.DataFrame) -> None:
        """Bulk insert posts using Polars DataFrame via zero-copy arrow buffer."""
        if df.is_empty():
            return
        arrow_table = df.to_arrow()
        self.con.register("tmp_posts_arrow", arrow_table)
        self.con.execute("""
            INSERT OR REPLACE INTO posts 
            SELECT 
                CAST(id AS VARCHAR),
                CAST(platform AS VARCHAR),
                CAST(author_id AS VARCHAR),
                CAST(author_screen_name AS VARCHAR),
                CAST(text AS TEXT),
                CAST(timestamp AS TIMESTAMP WITH TIME ZONE),
                CAST(timestamp_ms AS BIGINT),
                CAST(parent_id AS VARCHAR),
                CAST(reply_to_user_id AS VARCHAR),
                CAST(source_client AS VARCHAR),
                urls,
                hashtags,
                mentions,
                CAST(metrics AS JSON),
                CAST(extra_metadata AS JSON)
            FROM tmp_posts_arrow
        """)
        self.con.unregister("tmp_posts_arrow")

    def insert_users(self, users: List[CanonicalUser]) -> None:
        """Insert a batch of CanonicalUser instances."""
        if not users:
            return

        records = [
            {
                "id": u.id,
                "platform": u.platform.value,
                "screen_name": u.screen_name,
                "location_raw": u.location_raw,
                "latitude": u.location_coords.latitude if u.location_coords else None,
                "longitude": u.location_coords.longitude if u.location_coords else None,
                "indegree": u.indegree,
                "outdegree": u.outdegree,
                "bio": u.bio,
                "metrics": json.dumps(u.metrics.model_dump()),
                "extra_metadata": json.dumps(u.extra_metadata),
            }
            for u in users
        ]
        df = pl.DataFrame(records)
        self.insert_users_df(df)

    def insert_users_df(self, df: pl.DataFrame) -> None:
        """Bulk insert users using Polars DataFrame."""
        if df.is_empty():
            return
        arrow_table = df.to_arrow()
        self.con.register("tmp_users_arrow", arrow_table)
        self.con.execute("""
            INSERT OR REPLACE INTO users 
            SELECT 
                CAST(id AS VARCHAR),
                CAST(platform AS VARCHAR),
                CAST(screen_name AS VARCHAR),
                CAST(location_raw AS VARCHAR),
                CAST(latitude AS DOUBLE),
                CAST(longitude AS DOUBLE),
                CAST(indegree AS INTEGER),
                CAST(outdegree AS INTEGER),
                CAST(bio AS TEXT),
                CAST(metrics AS JSON),
                CAST(extra_metadata AS JSON)
            FROM tmp_users_arrow
        """)
        self.con.unregister("tmp_users_arrow")

    def insert_cascade_events(self, events: List[CanonicalCascadeEvent]) -> None:
        """Insert a batch of CanonicalCascadeEvent instances."""
        if not events:
            return

        records = [
            {
                "cascade_id": e.cascade_id,
                "post_id": e.post_id,
                "user_id": e.user_id,
                "user_screen_name": e.user_screen_name,
                "timestamp": e.timestamp,
                "timestamp_ms": e.timestamp_ms,
                "adoption_order": e.adoption_order,
                "parent_event_id": e.parent_event_id,
                "extra_metadata": json.dumps(e.extra_metadata),
            }
            for e in events
        ]
        df = pl.DataFrame(records)
        self.insert_cascade_events_df(df)

    def insert_cascade_events_df(self, df: pl.DataFrame) -> None:
        """Bulk insert cascade events using Polars DataFrame."""
        if df.is_empty():
            return
        arrow_table = df.to_arrow()
        self.con.register("tmp_cascades_arrow", arrow_table)
        self.con.execute("""
            INSERT OR REPLACE INTO cascade_events 
            SELECT 
                CAST(cascade_id AS VARCHAR),
                CAST(post_id AS VARCHAR),
                CAST(user_id AS VARCHAR),
                CAST(user_screen_name AS VARCHAR),
                CAST(timestamp AS TIMESTAMP WITH TIME ZONE),
                CAST(timestamp_ms AS BIGINT),
                CAST(adoption_order AS INTEGER),
                CAST(parent_event_id AS VARCHAR),
                CAST(extra_metadata AS JSON)
            FROM tmp_cascades_arrow
        """)
        self.con.unregister("tmp_cascades_arrow")

    def get_posts_count(self) -> int:
        """Get total number of posts stored."""
        return self.con.execute("SELECT COUNT(*) FROM posts").fetchone()[0]

    def get_users_count(self) -> int:
        """Get total number of users stored."""
        return self.con.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def get_cascade_events_count(self) -> int:
        """Get total number of cascade events stored."""
        return self.con.execute("SELECT COUNT(*) FROM cascade_events").fetchone()[0]

    def get_posts_in_window(
        self,
        start_time: datetime,
        end_time: datetime,
        platform: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pl.DataFrame:
        """Query posts within a specific time window as a Polars DataFrame."""
        query = """
            SELECT * FROM posts 
            WHERE timestamp >= ? AND timestamp <= ?
        """
        params: List[Any] = [start_time, end_time]

        if platform:
            query += " AND platform = ?"
            params.append(platform)

        query += " ORDER BY timestamp ASC"

        if limit:
            query += f" LIMIT {int(limit)}"

        res = self.con.execute(query, params).pl()
        return res

    def get_cascade_series(self, cascade_id: str) -> pl.DataFrame:
        """Retrieve chronological propagation sequence for a specific cascade/URL."""
        query = """
            SELECT * FROM cascade_events
            WHERE cascade_id = ?
            ORDER BY adoption_order ASC, timestamp_ms ASC
        """
        return self.con.execute(query, [cascade_id]).pl()

    def query(self, sql: str, params: Optional[List[Any]] = None) -> pl.DataFrame:
        """Execute arbitrary SQL and return results as Polars DataFrame."""
        if params:
            return self.con.execute(sql, params).pl()
        return self.con.execute(sql).pl()

    def close(self) -> None:
        """Close connection to DuckDB."""
        self.con.close()
