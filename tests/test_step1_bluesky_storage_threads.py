"""Unit and integration tests for Step 1: Bluesky connector, DuckDB analytics tables, and thread manager."""

import time
from datetime import datetime, timedelta, timezone
import pytest
from fastapi.testclient import TestClient

from hypesignal.api.app import create_app
from hypesignal.connectors.bluesky import BlueskyConnector
from hypesignal.connectors.schemas import ConnectorConfig
from hypesignal.models.canonical import CanonicalGraphEdge, CanonicalPost, PostMetrics
from hypesignal.models.enums import PlatformType, RelationType
from hypesignal.storage.duckdb_manager import DuckDBManager
from hypesignal.timeline.thread_manager import ConversationThreadManager


@pytest.fixture
def mem_db():
    """In-memory DuckDB manager fixture."""
    db = DuckDBManager(":memory:")
    yield db
    db.close()


def test_duckdb_analytics_tables_and_helpers(mem_db):
    """Test DuckDB post_analytics and user_communities schemas and helper methods."""
    # 1. Verify table counts initially zero
    assert mem_db.get_post_analytics_count() == 0
    assert mem_db.get_user_communities_count() == 0

    # 2. Test upsert_post_analytics
    rows = [
        {
            "post_id": "p_101",
            "effective_polarity": "positive",
            "sentiment_score": 0.92,
            "is_sarcastic": False,
            "irony_score": 0.05,
            "primary_emotion": "excitement",
            "emotion_score": 0.88,
            "joy": 0.1,
            "excitement": 0.88,
            "stance": "supportive",
            "stance_score": 0.95,
        },
        {
            "post_id": "p_102",
            "effective_polarity": "negative",
            "sentiment_score": 0.85,
            "is_sarcastic": True,
            "irony_score": 0.91,
            "primary_emotion": "anxiety",
            "emotion_score": 0.79,
            "fear": 0.2,
            "anxiety": 0.79,
            "stance": "against",
            "stance_score": 0.80,
        },
    ]
    mem_db.upsert_post_analytics(rows)
    assert mem_db.get_post_analytics_count() == 2

    # Query back
    analytics_df = mem_db.get_post_analytics(["p_101", "p_102"])
    assert len(analytics_df) == 2
    row101 = analytics_df.filter(analytics_df["post_id"] == "p_101").to_dicts()[0]
    assert row101["effective_polarity"] == "positive"
    assert row101["primary_emotion"] == "excitement"
    assert row101["excitement"] == pytest.approx(0.88)
    assert row101["stance"] == "supportive"
    assert row101["is_sarcastic"] is False

    row102 = analytics_df.filter(analytics_df["post_id"] == "p_102").to_dicts()[0]
    assert row102["is_sarcastic"] is True
    assert row102["anxiety"] == pytest.approx(0.79)

    # 3. Test upsert_user_communities
    communities = [
        ("user_alice", 1, 0.65),
        ("user_bob", 1, 0.65),
        ("user_carol", 2, 0.71),
    ]
    mem_db.upsert_user_communities(communities)
    assert mem_db.get_user_communities_count() == 3

    comm_df = mem_db.get_user_communities()
    assert len(comm_df) == 3
    alice_comm = comm_df.filter(comm_df["user_id"] == "user_alice").to_dicts()[0]
    assert alice_comm["community_id"] == 1
    assert alice_comm["modularity_score"] == pytest.approx(0.65)


def test_duckdb_windowed_graph_and_active_nodes(mem_db):
    """Test get_graph_edges_in_window and get_active_nodes_in_window."""
    t0 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 3, 1, 11, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Insert posts across window
    p1 = CanonicalPost(id="p1", author_id="u1", text="Post 1", timestamp=t0)
    p2 = CanonicalPost(id="p2", author_id="u2", text="Post 2", timestamp=t1)
    p3 = CanonicalPost(id="p3", author_id="u3", text="Post 3", timestamp=t2)
    mem_db.insert_posts([p1, p2, p3])

    # Insert edges
    e1 = CanonicalGraphEdge(source_id="u1", target_id="u2", timestamp=t0)
    e2 = CanonicalGraphEdge(source_id="u2", target_id="u3", timestamp=t1)
    e3 = CanonicalGraphEdge(source_id="u3", target_id="u1", timestamp=t2)
    mem_db.insert_graph_edges([e1, e2, e3])

    # Window from t0 to t1
    active_nodes = mem_db.get_active_nodes_in_window(t0, t1)
    assert sorted(active_nodes) == ["u1", "u2"]

    window_edges = mem_db.get_graph_edges_in_window(t0, t1)
    assert len(window_edges) == 2


def test_bluesky_connector_normalization():
    """Test Bluesky connector AT Protocol record parsing and entity extraction."""
    connector = BlueskyConnector()

    raw_at_post = {
        "uri": "at://did:plc:ragtjsm2j2vknwk2uhgahsob/app.bsky.feed.post/3kg2i37d2ls2w",
        "cid": "bafyreihyr2g6...",
        "author": {
            "did": "did:plc:ragtjsm2j2vknwk2uhgahsob",
            "handle": "alice.bsky.social",
            "displayName": "Alice Smith",
        },
        "record": {
            "$type": "app.bsky.feed.post",
            "text": "Deep dive into decentralized graphs! #atproto #analytics @bob.bsky.social https://hypesignal.ai",
            "createdAt": "2026-03-01T14:30:00.000Z",
            "reply": {
                "root": {"uri": "at://did:plc:root/app.bsky.feed.post/3kgroot"},
                "parent": {"uri": "at://did:plc:parent/app.bsky.feed.post/3kgparent123"},
            },
            "facets": [
                {
                    "features": [
                        {"$type": "app.bsky.richtext.facet#tag", "tag": "atproto"},
                        {"$type": "app.bsky.richtext.facet#tag", "tag": "analytics"},
                    ]
                },
                {
                    "features": [
                        {"$type": "app.bsky.richtext.facet#mention", "did": "did:plc:bob123"}
                    ]
                },
                {
                    "features": [
                        {"$type": "app.bsky.richtext.facet#link", "uri": "https://hypesignal.ai"}
                    ]
                },
            ],
        },
        "likeCount": 55,
        "repostCount": 18,
        "replyCount": 7,
        "quoteCount": 4,
    }

    post = connector.normalize_post(raw_at_post)
    assert post.id == "3kg2i37d2ls2w"
    assert post.platform == PlatformType.BLUESKY
    assert post.author_id == "did:plc:ragtjsm2j2vknwk2uhgahsob"
    assert post.author_screen_name == "alice.bsky.social"
    assert post.parent_id == "3kgparent123"
    assert post.metrics.likes == 55
    assert post.metrics.reposts == 18
    assert post.metrics.replies == 7
    assert post.metrics.shares == 4
    assert "atproto" in post.hashtags
    assert "analytics" in post.hashtags
    assert "did:plc:bob123" in post.mentions
    assert "https://hypesignal.ai" in post.urls
    assert post.extra_metadata["author_display_name"] == "Alice Smith"


def test_bluesky_connector_poll_mock_mode():
    """Test Bluesky connector synchronous poll method with mock generator."""
    config = ConnectorConfig(platform=PlatformType.BLUESKY, credentials={"mock": True})
    connector = BlueskyConnector(config=config)

    posts = connector.poll(query="machine learning", limit=2)
    assert len(posts) == 2
    root = posts[0]
    reply = posts[1]

    assert root.platform == PlatformType.BLUESKY
    assert reply.platform == PlatformType.BLUESKY
    assert root.parent_id is None
    # Reply points to root post
    assert reply.parent_id == root.id
    assert connector.stats.requests_made == 1
    assert connector.stats.messages_ingested == 2


def test_bluesky_connector_registered_in_fastapi_app():
    """Test that Bluesky connector is properly registered and accessible in FastAPI app state."""
    app = create_app()
    with TestClient(app) as client:
        # Check connector status endpoint
        resp = client.get("/api/v1/connectors/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "connectors" in data
        assert "bluesky" in data["connectors"]
        assert data["connectors"]["bluesky"]["platform"] == "bluesky"

        # Poll endpoint for bluesky
        poll_resp = client.post(
            "/api/v1/connectors/bluesky/poll",
            json={"query": "neural networks", "limit": 2, "persist": False},
        )
        assert poll_resp.status_code == 200
        poll_data = poll_resp.json()
        assert poll_data["platform"] == "bluesky"
        assert poll_data["count"] >= 1


def test_conversation_thread_manager_hierarchy(mem_db):
    """Test complete hierarchical conversation tree reconstruction and structural metrics."""
    t0 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=5)
    t2 = t0 + timedelta(minutes=15)
    t3 = t0 + timedelta(minutes=25)
    t4 = t0 + timedelta(minutes=40)

    # Construct conversation tree:
    # root (p0)
    # ├── reply 1 (p1, parent=p0)
    # │   └── reply 1.1 (p3, parent=p1)
    # └── reply 2 (p2, parent=p0)
    #     └── reply 2.1 (p4, parent=p2)
    posts = [
        CanonicalPost(id="p0", author_id="u0", text="Root question", timestamp=t0, metrics=PostMetrics(replies=2)),
        CanonicalPost(id="p1", author_id="u1", parent_id="p0", text="Reply 1", timestamp=t1),
        CanonicalPost(id="p2", author_id="u2", parent_id="p0", text="Reply 2", timestamp=t2),
        CanonicalPost(id="p3", author_id="u3", parent_id="p1", text="Reply 1.1", timestamp=t3),
        CanonicalPost(id="p4", author_id="u1", parent_id="p2", text="Reply 2.1", timestamp=t4),
    ]
    mem_db.insert_posts(posts)

    manager = ConversationThreadManager(db=mem_db)

    # 1. Direct comment children
    children_p0 = manager.db.get_comment_children("p0")
    assert len(children_p0) == 2
    assert sorted(children_p0["id"].to_list()) == ["p1", "p2"]

    # 2. Full thread reconstruction
    thread = manager.reconstruct_thread("p0")
    assert thread is not None
    assert thread.root_post_id == "p0"
    assert thread.total_posts == 5
    assert thread.total_replies == 4
    assert thread.max_depth == 2
    # Branching factor: parents p0 (2 children), p1 (1 child), p2 (1 child) -> (2+1+1)/3 = 1.33
    assert thread.avg_branching_factor == pytest.approx(1.33, abs=0.05)
    assert sorted(thread.participant_ids) == ["u0", "u1", "u2", "u3"]
    assert thread.duration_seconds == 40 * 60  # 40 minutes

    # Depth distribution
    assert thread.depth_distribution == {0: 1, 1: 2, 2: 2}

    # Verify root node children
    assert len(thread.root_node.children) == 2
    assert thread.root_node.children[0].post_id == "p1"
    assert thread.root_node.children[1].post_id == "p2"
    assert len(thread.root_node.children[0].children) == 1
    assert thread.root_node.children[0].children[0].post_id == "p3"

    # 3. Chronological reply stream (excluding root)
    replies_stream = manager.get_chronological_reply_stream("p0")
    assert len(replies_stream) == 4
    assert [p.id for p in replies_stream] == ["p1", "p2", "p3", "p4"]

    # 4. Non-existent root returns None
    assert manager.reconstruct_thread("non_existent_id") is None
