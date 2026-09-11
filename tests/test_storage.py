"""Unit tests for DuckDB analytical store and Qdrant vector manager."""

from datetime import datetime, timezone, timedelta
import polars as pl
import pytest

from hypesignal.models import (
    CanonicalCascadeEvent,
    CanonicalPost,
    CanonicalUser,
    GeoCoordinates,
    PlatformType,
)
from hypesignal.storage import DuckDBManager, VectorStoreManager


def test_duckdb_posts_lifecycle():
    """Test inserting and querying posts in DuckDB."""
    db = DuckDBManager(":memory:")

    now = datetime.now(timezone.utc)
    posts = [
        CanonicalPost(
            id=f"p_{i}",
            platform=PlatformType.TWITTER,
            author_id=f"u_{i % 2}",
            author_screen_name=f"user_{i % 2}",
            text=f"Tweet message {i} #tech",
            timestamp=now + timedelta(minutes=i * 10),
            hashtags=["tech"],
        )
        for i in range(5)
    ]

    db.insert_posts(posts)
    assert db.get_posts_count() == 5

    # Query time window for the first 3 posts (0 to 25 mins)
    window_df = db.get_posts_in_window(
        start_time=now,
        end_time=now + timedelta(minutes=25),
    )
    assert len(window_df) == 3
    assert list(window_df["id"]) == ["p_0", "p_1", "p_2"]

    # Verify DuckDB SQL query execution
    df_agg = db.query("SELECT author_id, count(*) as cnt FROM posts GROUP BY author_id ORDER BY author_id")
    assert len(df_agg) == 2
    assert df_agg["cnt"][0] == 3  # u_0 has 3 posts (p_0, p_2, p_4)
    assert df_agg["cnt"][1] == 2  # u_1 has 2 posts (p_1, p_3)

    db.close()


def test_duckdb_users_and_cascades():
    """Test inserting users and cascade events in DuckDB."""
    db = DuckDBManager(":memory:")

    users = [
        CanonicalUser(
            id="u_100",
            screen_name="alpha",
            location_raw="San Francisco, CA",
            indegree=500,
            outdegree=200,
        ),
        CanonicalUser(
            id="u_200",
            screen_name="beta",
            location_raw="UT: 40.7128,-74.0060",
            indegree=120,
            outdegree=80,
        ),
    ]
    db.insert_users(users)
    assert db.get_users_count() == 2

    # Check GPS coordinates stored in DuckDB
    u2_res = db.query("SELECT latitude, longitude FROM users WHERE id = 'u_200'")
    assert pytest.approx(u2_res["latitude"][0], 1e-4) == 40.7128
    assert pytest.approx(u2_res["longitude"][0], 1e-4) == -74.0060

    # Insert cascade events
    cascade_url = "http://short.ly/cascade1"
    now = datetime.now(timezone.utc)
    events = [
        CanonicalCascadeEvent(
            cascade_id=cascade_url,
            post_id=f"post_{i}",
            user_id=f"u_{i}",
            timestamp=now + timedelta(seconds=i * 5),
            adoption_order=i,
        )
        for i in range(4)
    ]
    db.insert_cascade_events(events)
    assert db.get_cascade_events_count() == 4

    series_df = db.get_cascade_series(cascade_url)
    assert len(series_df) == 4
    assert list(series_df["adoption_order"]) == [0, 1, 2, 3]

    db.close()


def test_vector_store_manager():
    """Test Qdrant vector operations in embedded memory."""
    vec_mgr = VectorStoreManager(location=":memory:")
    coll_name = "test_personas"

    # Create collection
    assert vec_mgr.create_collection(coll_name, vector_size=4, distance="Cosine") is True
    # Idempotent creation
    assert vec_mgr.create_collection(coll_name, vector_size=4, distance="Cosine") is False

    # Upsert with arbitrary string IDs and payloads
    ids = ["persona_dev", "persona_finance", "persona_artist"]
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    payloads = [
        {"role": "Software Engineer", "category": "Tech"},
        {"role": "Financial Analyst", "category": "Finance"},
        {"role": "Designer", "category": "Arts"},
    ]
    vec_mgr.upsert_vectors(coll_name, ids=ids, vectors=vectors, payloads=payloads)
    assert vec_mgr.count(coll_name) == 3

    # Search for nearest to dev vector
    query_vec = [0.9, 0.1, 0.0, 0.0]
    hits = vec_mgr.search(coll_name, query_vector=query_vec, limit=2)
    assert len(hits) == 2
    assert hits[0]["id"] == "persona_dev"
    assert hits[0]["payload"]["role"] == "Software Engineer"
    assert hits[0]["score"] > 0.9

    # Delete collection
    assert vec_mgr.delete_collection(coll_name) is True


def test_duckdb_none_path_defaults_to_memory():
    """Verify DuckDBManager(None) gracefully initializes in-memory without error."""
    db = DuckDBManager(None)
    assert db.db_path == ":memory:"
    assert db.get_posts_count() == 0
    db.close()


def test_vector_store_snowflake_id_symmetry():
    """Verify 19-20 digit Twitter Snowflake IDs are symmetrically parsed as int or str."""
    snowflake_raw = 1445093874553315330
    snowflake_str = "1445093874553315330"

    norm_int, orig_int = VectorStoreManager._normalize_id(snowflake_raw)
    norm_str, orig_str = VectorStoreManager._normalize_id(snowflake_str)

    # Both must resolve to the identical integer representation
    assert norm_int == snowflake_raw
    assert norm_str == snowflake_raw
    assert norm_int == norm_str

    # Test idempotency in Qdrant (upserting as int then str updates the same point)
    vec_mgr = VectorStoreManager(location=":memory:")
    coll_name = "test_snowflakes"
    vec_mgr.create_collection(coll_name, vector_size=3)

    vec_mgr.upsert_vectors(coll_name, ids=[snowflake_raw], vectors=[[1.0, 0.0, 0.0]], payloads=[{"v": 1}])
    assert vec_mgr.count(coll_name) == 1

    vec_mgr.upsert_vectors(coll_name, ids=[snowflake_str], vectors=[[0.0, 1.0, 0.0]], payloads=[{"v": 2}])
    # Point count MUST remain 1 (no duplicate points created)
    assert vec_mgr.count(coll_name) == 1

    hits = vec_mgr.search(coll_name, query_vector=[0.0, 1.0, 0.0], limit=1)
    assert hits[0]["payload"]["v"] == 2
    assert hits[0]["id"] == snowflake_str

