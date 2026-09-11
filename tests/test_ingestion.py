"""Automated tests for dataset ingestion adapters and TweetEval evaluator."""

import pytest
from hypesignal.ingestion.geo_adapter import GeoDatasetAdapter
from hypesignal.ingestion.lerman_adapter import LermanDatasetAdapter
from hypesignal.models.enums import PlatformType, RelationType
from hypesignal.nlp.evaluation.tweeteval_loader import TweetEvalLoader
from hypesignal.storage.duckdb_manager import DuckDBManager


def test_geo_adapter_users_and_tweets():
    """Verify GeoDatasetAdapter parses tweets, user locations, and GPS coords into DuckDB."""
    db = DuckDBManager(":memory:")
    adapter = GeoDatasetAdapter()

    # Ingest 50 training users (textual locations)
    users_ingested = adapter.ingest_users(db, split="train", limit=50, batch_size=25)
    assert users_ingested == 50
    assert db.get_users_count() == 50

    # Ingest 50 test users (GPS coordinates 'UT: lat,lon')
    test_users_ingested = adapter.ingest_users(db, split="test", limit=50, batch_size=25)
    assert test_users_ingested == 50
    assert db.get_users_count() == 100

    # Verify that GPS coordinates were parsed from test set users
    gps_users = db.query("SELECT id, location_raw, latitude, longitude FROM users WHERE latitude IS NOT NULL")
    assert len(gps_users) > 0
    assert -90.0 <= gps_users["latitude"][0] <= 90.0

    # Ingest 50 tweets
    tweets_ingested = adapter.ingest_tweets(db, split="train", limit=50, batch_size=25)
    assert tweets_ingested == 50
    assert db.get_posts_count() == 50

    sample_posts = db.query("SELECT id, text, hashtags, mentions FROM posts LIMIT 5")
    assert len(sample_posts) == 5
    assert all(isinstance(t, str) for t in sample_posts["text"])

    db.close()


def test_lerman_adapter_cascades_and_edges():
    """Verify LermanDatasetAdapter parses cascades, degrees, and streams edges."""
    db = DuckDBManager(":memory:")
    adapter = LermanDatasetAdapter()

    # Ingest 50 users from distinct_users map
    users_count = adapter.ingest_users(db, limit=50, batch_size=25)
    assert users_count == 50
    assert db.get_users_count() == 50

    user_degrees = db.query("SELECT id, indegree, outdegree FROM users WHERE indegree > 0 LIMIT 1")
    assert len(user_degrees) == 1
    assert user_degrees["indegree"][0] > 0

    # Ingest 50 cascade events from zip
    cascades_count = adapter.ingest_cascades(db, limit=50, batch_size=25)
    assert cascades_count == 50
    assert db.get_cascade_events_count() == 50

    cascade_sample = db.query("SELECT cascade_id, adoption_order, timestamp_ms FROM cascade_events LIMIT 1")
    assert len(cascade_sample) == 1
    assert cascade_sample["cascade_id"][0].startswith("http")

    # Stream 10 follower graph edges directly from compressed SQL dump
    edges = list(adapter.stream_follower_edges(limit=10))
    assert len(edges) == 10
    assert edges[0].relation_type == RelationType.FOLLOWS
    assert edges[0].source_id.isdigit()
    assert edges[0].target_id.isdigit()

    db.close()


def test_tweeteval_isolated_loader():
    """Verify TweetEvalLoader loads mappings and benchmark splits without touching DuckDB."""
    loader = TweetEvalLoader()

    # Verify task mappings
    emotion_map = loader.load_mapping("emotion")
    assert emotion_map[0] == "anger"
    assert emotion_map[1] == "joy"

    irony_map = loader.load_mapping("irony")
    assert irony_map[0] == "non_irony"
    assert irony_map[1] == "irony"

    sentiment_map = loader.load_mapping("sentiment")
    assert sentiment_map[0] == "negative"
    assert sentiment_map[1] == "neutral"
    assert sentiment_map[2] == "positive"

    # Verify loading split samples
    test_samples = loader.load_split("emotion", split="test", limit=20)
    assert len(test_samples) == 20
    assert all(s.task == "emotion" for s in test_samples)
    assert all(s.label_name in ["anger", "joy", "optimism", "sadness"] for s in test_samples)

    # Verify hierarchical task (stance/climate)
    climate_samples = loader.load_split("stance/climate", split="test", limit=10)
    assert len(climate_samples) == 10
    assert all(s.label_name in ["none", "against", "favor"] for s in climate_samples)


def test_robust_decoding_preserves_utf8():
    """Verify robust decoding preserves genuine UTF-8 characters and handles latin-1 fallbacks."""
    from hypesignal.ingestion.base import robust_decode_line, robust_line_stream
    import io

    # Genuine UTF-8 with curly quote and em-dash
    utf8_bytes = "Ellis’s quote — “AI is great!”\n".encode("utf-8")
    decoded = robust_decode_line(utf8_bytes)
    assert decoded == "Ellis’s quote — “AI is great!”\n"
    # Ensure it didn't get mangled into â
    assert "â" not in decoded

    # Isolated byte that is invalid UTF-8 (e.g. 0x9c from test_set_users.txt)
    latin1_bytes = b"User location \x9c test\n"
    decoded_fallback = robust_decode_line(latin1_bytes)
    assert "User location" in decoded_fallback

    # Stream iterator
    stream = io.BytesIO(utf8_bytes + latin1_bytes)
    lines = list(robust_line_stream(stream))
    assert len(lines) == 2
    assert "Ellis’s" in lines[0]

