"""DuckDB analytical storage manager for HypeSignal.

Handles high-throughput columnar storage, timeline indexing, and OLAP
analytical aggregations for posts, users, and diffusion cascades.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import polars as pl

from hypesignal.models.canonical import (
    CanonicalCascadeEvent,
    CanonicalGraphEdge,
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
        self.con.execute("SET TimeZone='UTC';")
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

            CREATE TABLE IF NOT EXISTS graph_edges (
                source_id VARCHAR NOT NULL,
                target_id VARCHAR NOT NULL,
                relation_type VARCHAR NOT NULL DEFAULT 'FOLLOWS',
                weight DOUBLE DEFAULT 1.0,
                timestamp TIMESTAMP WITH TIME ZONE,
                extra_metadata JSON,
                PRIMARY KEY (source_id, target_id, relation_type)
            );

            CREATE TABLE IF NOT EXISTS post_analytics (
                post_id VARCHAR PRIMARY KEY,
                effective_polarity VARCHAR NOT NULL,
                sentiment_score DOUBLE NOT NULL,
                is_sarcastic BOOLEAN NOT NULL,
                irony_score DOUBLE NOT NULL,
                primary_emotion VARCHAR NOT NULL,
                emotion_score DOUBLE NOT NULL,
                joy DOUBLE DEFAULT 0.0,
                optimism DOUBLE DEFAULT 0.0,
                anger DOUBLE DEFAULT 0.0,
                sadness DOUBLE DEFAULT 0.0,
                fear DOUBLE DEFAULT 0.0,
                anxiety DOUBLE DEFAULT 0.0,
                excitement DOUBLE DEFAULT 0.0,
                surprise DOUBLE DEFAULT 0.0,
                disgust DOUBLE DEFAULT 0.0,
                neutral DOUBLE DEFAULT 0.0,
                stance VARCHAR DEFAULT 'neutral',
                stance_score DOUBLE DEFAULT 0.0,
                analyzed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS user_communities (
                user_id VARCHAR PRIMARY KEY,
                community_id INTEGER NOT NULL,
                modularity_score DOUBLE DEFAULT 0.0,
                assigned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_posts_time ON posts (timestamp);
            CREATE INDEX IF NOT EXISTS idx_posts_time_ms ON posts (timestamp_ms);
            CREATE INDEX IF NOT EXISTS idx_posts_author ON posts (author_id);
            CREATE INDEX IF NOT EXISTS idx_posts_parent_id ON posts (parent_id);
            CREATE INDEX IF NOT EXISTS idx_cascades_id_time ON cascade_events (cascade_id, timestamp_ms);
            CREATE INDEX IF NOT EXISTS idx_users_platform ON users (platform, id);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges (source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges (target_id);
            CREATE INDEX IF NOT EXISTS idx_user_communities_comm ON user_communities (community_id);
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

    def insert_edges(self, edges: List[CanonicalGraphEdge]) -> None:
        """Insert a batch of CanonicalGraphEdge instances."""
        if not edges:
            return

        records = [
            {
                "source_id": e.source_id,
                "target_id": e.target_id,
                "relation_type": e.relation_type.value if hasattr(e.relation_type, "value") else str(e.relation_type),
                "weight": e.weight,
                "timestamp": e.timestamp,
                "extra_metadata": json.dumps(e.extra_metadata),
            }
            for e in edges
        ]
        df = pl.DataFrame(records)
        self.insert_edges_df(df)

    def insert_edges_df(self, df: pl.DataFrame) -> None:
        """Bulk insert graph edges using Polars DataFrame."""
        if df.is_empty():
            return
        arrow_table = df.to_arrow()
        self.con.register("tmp_edges_arrow", arrow_table)
        self.con.execute("""
            INSERT OR REPLACE INTO graph_edges 
            SELECT 
                CAST(source_id AS VARCHAR),
                CAST(target_id AS VARCHAR),
                CAST(relation_type AS VARCHAR),
                CAST(weight AS DOUBLE),
                CAST(timestamp AS TIMESTAMP WITH TIME ZONE),
                CAST(extra_metadata AS JSON)
            FROM tmp_edges_arrow
        """)
        self.con.unregister("tmp_edges_arrow")

    # Alias for API compatibility
    insert_graph_edges = insert_edges

    def get_edges_count(self) -> int:
        """Get total number of graph edges stored."""
        return self.con.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]

    def get_user_followers(self, user_id: str) -> List[str]:
        """Get list of follower user IDs for a given target user."""
        rows = self.con.execute(
            "SELECT source_id FROM graph_edges WHERE target_id = ? AND relation_type = 'FOLLOWS'",
            [user_id],
        ).fetchall()
        return [r[0] for r in rows]

    def get_user_following(self, user_id: str) -> List[str]:
        """Get list of followee user IDs that a given user follows."""
        rows = self.con.execute(
            "SELECT target_id FROM graph_edges WHERE source_id = ? AND relation_type = 'FOLLOWS'",
            [user_id],
        ).fetchall()
        return [r[0] for r in rows]

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

    def get_conversation_thread(self, root_post_id: str) -> pl.DataFrame:
        """Retrieve the entire conversation tree starting from a root post using recursive CTE."""
        query = """
            WITH RECURSIVE thread_tree AS (
                SELECT * FROM posts WHERE id = ?
                UNION ALL
                SELECT p.* FROM posts p
                JOIN thread_tree t ON p.parent_id = t.id
            )
            SELECT * FROM thread_tree ORDER BY timestamp ASC;
        """
        return self.con.execute(query, [root_post_id]).pl()

    def get_comment_children(self, parent_id: str) -> pl.DataFrame:
        """Retrieve direct replies/comments for a parent post."""
        query = "SELECT * FROM posts WHERE parent_id = ? ORDER BY timestamp ASC;"
        return self.con.execute(query, [parent_id]).pl()

    def get_graph_edges_in_window(
        self,
        start_time: datetime,
        end_time: datetime,
        include_undated: bool = True,
    ) -> pl.DataFrame:
        """Query graph edges created or active within a time window."""
        if include_undated:
            query = """
                SELECT * FROM graph_edges
                WHERE timestamp IS NULL OR (timestamp >= ? AND timestamp <= ?)
                ORDER BY timestamp ASC NULLS FIRST;
            """
        else:
            query = """
                SELECT * FROM graph_edges
                WHERE timestamp IS NOT NULL AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC;
            """
        return self.con.execute(query, [start_time, end_time]).pl()

    def get_active_nodes_in_window(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[str]:
        """Get unique user IDs who authored posts within the specified time window."""
        query = """
            SELECT DISTINCT author_id FROM posts
            WHERE timestamp >= ? AND timestamp <= ?;
        """
        rows = self.con.execute(query, [start_time, end_time]).fetchall()
        return [r[0] for r in rows]

    def upsert_post_analytics(self, analytics_rows: List[Dict[str, Any]]) -> None:
        """Insert or replace analytics records in the post_analytics table."""
        if not analytics_rows:
            return

        default_row = {
            "effective_polarity": "neutral",
            "sentiment_score": 0.0,
            "is_sarcastic": False,
            "irony_score": 0.0,
            "primary_emotion": "neutral",
            "emotion_score": 0.0,
            "joy": 0.0,
            "optimism": 0.0,
            "anger": 0.0,
            "sadness": 0.0,
            "fear": 0.0,
            "anxiety": 0.0,
            "excitement": 0.0,
            "surprise": 0.0,
            "disgust": 0.0,
            "neutral": 0.0,
            "stance": "neutral",
            "stance_score": 0.0,
        }

        standardized = [
            {
                "post_id": str(r["post_id"]),
                **default_row,
                **{k: v for k, v in r.items() if k != "post_id"},
            }
            for r in analytics_rows
        ]

        df = pl.DataFrame(standardized)
        self.con.register("tmp_analytics_batch", df.to_arrow())
        self.con.execute("""
            INSERT OR REPLACE INTO post_analytics
            SELECT 
                CAST(post_id AS VARCHAR),
                CAST(effective_polarity AS VARCHAR),
                CAST(sentiment_score AS DOUBLE),
                CAST(is_sarcastic AS BOOLEAN),
                CAST(irony_score AS DOUBLE),
                CAST(primary_emotion AS VARCHAR),
                CAST(emotion_score AS DOUBLE),
                CAST(joy AS DOUBLE),
                CAST(optimism AS DOUBLE),
                CAST(anger AS DOUBLE),
                CAST(sadness AS DOUBLE),
                CAST(fear AS DOUBLE),
                CAST(anxiety AS DOUBLE),
                CAST(excitement AS DOUBLE),
                CAST(surprise AS DOUBLE),
                CAST(disgust AS DOUBLE),
                CAST(neutral AS DOUBLE),
                CAST(stance AS VARCHAR),
                CAST(stance_score AS DOUBLE),
                CURRENT_TIMESTAMP
            FROM tmp_analytics_batch;
        """)
        self.con.unregister("tmp_analytics_batch")

    def get_post_analytics(self, post_ids: Optional[List[str]] = None) -> pl.DataFrame:
        """Retrieve post analytics records optionally filtered by post IDs."""
        if post_ids is not None:
            if not post_ids:
                return pl.DataFrame()
            placeholders = ",".join(["?"] * len(post_ids))
            return self.con.execute(
                f"SELECT * FROM post_analytics WHERE post_id IN ({placeholders})",
                post_ids,
            ).pl()
        return self.con.execute("SELECT * FROM post_analytics ORDER BY analyzed_at DESC").pl()

    def get_post_analytics_count(self) -> int:
        """Get total number of analyzed posts in post_analytics table."""
        return self.con.execute("SELECT COUNT(*) FROM post_analytics").fetchone()[0]

    def upsert_user_communities(self, assignments: List[Tuple[str, int, float]]) -> None:
        """Insert or replace user community assignments.
        
        Args:
            assignments: List of (user_id, community_id, modularity_score) tuples.
        """
        if not assignments:
            return

        records = [
            {"user_id": u, "community_id": c, "modularity_score": float(m)}
            for u, c, m in assignments
        ]
        df = pl.DataFrame(records)
        self.con.register("tmp_user_comm_batch", df.to_arrow())
        self.con.execute("""
            INSERT OR REPLACE INTO user_communities
            SELECT user_id, community_id, modularity_score, CURRENT_TIMESTAMP
            FROM tmp_user_comm_batch;
        """)
        self.con.unregister("tmp_user_comm_batch")

    def get_user_communities(self) -> pl.DataFrame:
        """Retrieve all persisted user community assignments."""
        return self.con.execute("SELECT * FROM user_communities ORDER BY community_id ASC").pl()

    def get_user_communities_count(self) -> int:
        """Get total count of users assigned to communities."""
        return self.con.execute("SELECT COUNT(*) FROM user_communities").fetchone()[0]

    def query(self, sql: str, params: Optional[List[Any]] = None) -> pl.DataFrame:
        """Execute arbitrary SQL and return results as Polars DataFrame."""
        if params:
            return self.con.execute(sql, params).pl()
        return self.con.execute(sql).pl()

    def close(self) -> None:
        """Close connection to DuckDB."""
        self.con.close()

