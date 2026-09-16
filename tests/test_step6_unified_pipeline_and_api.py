"""Step 6 Tests: Consolidated Pipeline Orchestrator, Analytics API & Resilient SSE Stream (Pillar F).

Verifies:
1. EventBroadcaster bounded queue backpressure (drop-oldest), fan-out, and disconnect cleanup.
2. AnalyticsPipelineOrchestrator stage execution, non-blocking asyncio.to_thread delegation, error isolation, and start/stop lifecycle.
3. Unified REST endpoints under /api/v1/analytics/* (threads, audience, trend forecast, narrative drift, cross-segment diffusion, bridge KOLs).
4. Real-time SSE feed under /api/v1/streaming/events.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hypesignal.api.app import create_app
from hypesignal.api.streaming import EventBroadcaster
from hypesignal.connectors.base import PlatformConnector
from hypesignal.models.canonical import (
    CanonicalPost,
    PlatformType,
    PostMetrics,
)
from hypesignal.network.graph_store import NetworkXGraphStore
from hypesignal.network.network_engine import NetworkEngine
from hypesignal.orchestration.pipeline_orchestrator import AnalyticsPipelineOrchestrator
from hypesignal.storage.duckdb_manager import DuckDBManager
from hypesignal.trends.trend_forecaster import TrendForecaster


class DummyConnector(PlatformConnector):
    """Simple synchronous connector for testing pipeline ingestion."""

    def __init__(self, platform_name: str = "bluesky", num_posts: int = 2) -> None:
        super().__init__()
        self._platform_name = platform_name
        self.num_posts = num_posts
        self.poll_called_count = 0

    @property
    def platform(self) -> PlatformType:
        return PlatformType.BLUESKY

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def normalize_post(self, raw_post: Any) -> CanonicalPost:
        if isinstance(raw_post, CanonicalPost):
            return raw_post
        return CanonicalPost(
            id=str(getattr(raw_post, "id", "dummy")),
            author_id="author",
            text=str(getattr(raw_post, "text", "")),
            timestamp=datetime.now(timezone.utc),
        )

    def poll(self, query: str, limit: int = 10, **kwargs: Any) -> List[CanonicalPost]:
        self.poll_called_count += 1
        now = datetime.now(timezone.utc)
        posts = []
        for i in range(min(self.num_posts, limit)):
            p = CanonicalPost(
                id=f"sync_p_{self.poll_called_count}_{i}",
                platform=PlatformType.BLUESKY,
                author_id=f"author_{i}",
                author_screen_name=f"author_{i}",
                text=f"Breaking update {i}: technology hype is rising #tech",
                timestamp=now,
                metrics=PostMetrics(),
                hashtags=["tech"],
            )
            posts.append(p)
        return posts


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_step6.duckdb"
        manager = DuckDBManager(db_path=db_path)
        yield manager
        manager.close()


@pytest.mark.anyio
async def test_event_broadcaster_bounded_backpressure():
    """Verify that EventBroadcaster drops the oldest unconsumed item when queue capacity is reached."""
    broadcaster = EventBroadcaster(max_queue_size=3)
    queue = await broadcaster.subscribe()

    assert broadcaster.subscriber_count == 1

    # Broadcast 5 events into a queue of capacity 3
    for i in range(1, 6):
        delivered = broadcaster.broadcast({"msg": f"event_{i}"}, event_type="test_event")
        assert delivered == 1

    assert queue.full()
    assert queue.qsize() == 3

    # The oldest events (1 and 2) should have been dropped.
    # The queue should now contain events 3, 4, 5.
    item1 = queue.get_nowait()
    assert item1["data"]["msg"] == "event_3"

    item2 = queue.get_nowait()
    assert item2["data"]["msg"] == "event_4"

    item3 = queue.get_nowait()
    assert item3["data"]["msg"] == "event_5"

    assert queue.empty()

    # Unsubscribe cleanup
    await broadcaster.unsubscribe(queue)
    assert broadcaster.subscriber_count == 0


@pytest.mark.anyio
async def test_analytics_pipeline_orchestrator_execution(temp_db):
    """Test AnalyticsPipelineOrchestrator single tick and stages (ingestion, enrichment, trend)."""
    broadcaster = EventBroadcaster(max_queue_size=50)
    b_queue = await broadcaster.subscribe()

    connector = DummyConnector(num_posts=3)
    connector.connect()

    # Mock NLP engine that returns valid flattened post_analytics dictionaries
    mock_nlp = MagicMock()
    mock_nlp.analyze_and_flatten.side_effect = lambda texts, ids: [
        {
            "post_id": pid,
            "effective_polarity": "positive",
            "sentiment_score": 0.85,
            "is_sarcastic": False,
            "irony_score": 0.0,
            "primary_emotion": "excitement",
            "emotion_score": 0.85,
            "joy": 0.5,
            "optimism": 0.5,
            "anger": 0.0,
            "sadness": 0.0,
            "fear": 0.0,
            "anxiety": 0.0,
            "excitement": 0.8,
            "surprise": 0.0,
            "disgust": 0.0,
            "neutral": 0.0,
            "stance": "supportive",
            "stance_score": 0.8,
        }
        for pid in ids
    ]

    sensitive_forecaster = TrendForecaster(velocity_threshold=0.01, acceleration_threshold=0.0001)
    orchestrator = AnalyticsPipelineOrchestrator(
        db=temp_db,
        nlp=mock_nlp,
        connectors={"bluesky": connector},
        broadcaster=broadcaster,
        trend_forecaster=sensitive_forecaster,
        tick_interval_seconds=0.1,
    )

    # 1. Run single tick
    summary = await orchestrator.run_single_tick()
    assert summary["status"] == "ok"
    assert summary["ingested_count"] == 3
    assert summary["enriched_count"] == 3

    # Check that posts were persisted to DuckDB
    post_count = temp_db.con.execute("SELECT count(*) FROM posts;").fetchone()[0]
    assert post_count == 3

    analytics_count = temp_db.con.execute("SELECT count(*) FROM post_analytics;").fetchone()[0]
    assert analytics_count == 3

    # Verify broadcaster received events for ingestion, enrichment, and trend_alert
    received_events = []
    while not b_queue.empty():
        received_events.append(b_queue.get_nowait())

    event_types = [e["event"] for e in received_events]
    assert "ingestion" in event_types
    assert "enrichment" in event_types
    assert "trend_alert" in event_types

    alert_events = [e for e in received_events if e["event"] == "trend_alert"]
    assert len(alert_events) >= 1
    assert "current_velocity" in alert_events[0]["data"]
    assert "velocity" in alert_events[0]["data"]
    assert "virality_potential_score" in alert_events[0]["data"]

    # 2. Test lifecycle start / stop
    orchestrator.start()
    assert orchestrator.is_running is True
    await asyncio.sleep(0.2)
    await orchestrator.stop()
    assert orchestrator.is_running is False


@pytest.mark.anyio
async def test_orchestrator_stage_error_isolation(temp_db):
    """Verify that an exception in one stage does not crash the orchestrator tick loop."""
    failing_connector = MagicMock()
    failing_connector.config.enabled = True
    failing_connector.poll.side_effect = RuntimeError("Network timeout simulation")

    orchestrator = AnalyticsPipelineOrchestrator(
        db=temp_db,
        connectors={"faulty": failing_connector},
        tick_interval_seconds=0.1,
    )

    # Single tick should complete gracefully despite the failing connector
    summary = await orchestrator.run_single_tick()
    assert summary["status"] == "ok"
    assert summary["ingested_count"] == 0


def test_unified_analytics_api_endpoints(temp_db):
    """Test all /api/v1/analytics/* endpoints via FastAPI TestClient."""
    # 1. Populate database with test fixtures
    t0 = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=5)

    # Users
    u1_meta = json.dumps({"age_bracket": "18-24", "persona": "crypto_trader"})
    u2_meta = json.dumps({"age_bracket": "25-34", "persona": "tech_dev"})
    temp_db.con.execute(
        "INSERT INTO users (id, platform, screen_name, extra_metadata) VALUES "
        "('root_author', 'bluesky', 'RootAlice', ?), "
        "('reply_author1', 'bluesky', 'ReplyBob', ?);",
        [u1_meta, u2_meta],
    )

    # Conversation thread: root -> reply1
    temp_db.con.execute(
        "INSERT INTO posts (id, platform, author_id, text, timestamp, timestamp_ms, parent_id, hashtags) VALUES "
        "('conv_root', 'bluesky', 'root_author', 'AI revolution is accelerating! #ai', ?, ?, NULL, ['ai']), "
        "('reply_1', 'bluesky', 'reply_author1', 'Totally agree, groundbreaking stuff!', ?, ?, 'conv_root', ['ai']);",
        [t0, int(t0.timestamp() * 1000), t1, int(t1.timestamp() * 1000)],
    )

    # Analytics for conversation
    temp_db.con.execute(
        "INSERT INTO post_analytics (post_id, effective_polarity, sentiment_score, is_sarcastic, irony_score, primary_emotion, emotion_score) VALUES "
        "('conv_root', 'positive', 0.8, false, 0.0, 'excitement', 0.8), "
        "('reply_1', 'positive', 0.9, false, 0.0, 'joy', 0.9);"
    )

    # Communities & cascade
    temp_db.con.execute(
        "INSERT INTO user_communities (user_id, community_id, modularity_score) VALUES "
        "('root_author', 1, 0.4), "
        "('reply_author1', 2, 0.4);"
    )

    temp_db.con.execute(
        "INSERT INTO cascade_events (cascade_id, post_id, user_id, timestamp, timestamp_ms, adoption_order) VALUES "
        "('casc_ai_step6', 'conv_root', 'root_author', ?, ?, 0), "
        "('casc_ai_step6', 'reply_1', 'reply_author1', ?, ?, 1);",
        [t0, int(t0.timestamp() * 1000), t1, int(t1.timestamp() * 1000)],
    )

    # Follower graph for bridge KOL & audience testing
    temp_db.con.execute(
        "INSERT INTO graph_edges (source_id, target_id, relation_type) VALUES "
        "('reply_author1', 'root_author', 'follows'), "
        "('root_author', 'reply_author1', 'follows');"
    )
    graph_store = NetworkXGraphStore()
    graph_store.add_edge("reply_author1", "root_author", relation_type="follows")
    graph_store.add_edge("root_author", "reply_author1", relation_type="follows")

    network_engine = NetworkEngine(graph_store=graph_store, db=temp_db)

    # 2. Instantiate app
    app = create_app(
        db=temp_db,
        network=network_engine,
        auto_connect=False,
    )

    with TestClient(app) as client:
        # Healthcheck
        health_res = client.get("/api/v1/health")
        assert health_res.status_code == 200
        health_data = health_res.json()
        assert health_data["engines"]["duckdb"] == "online"
        assert health_data["engines"]["orchestrator"] == "online"
        assert health_data["engines"]["broadcaster"] == "online"

        # 1. /api/v1/analytics/conversations/{conversation_id}
        conv_res = client.get("/api/v1/analytics/conversations/conv_root")
        assert conv_res.status_code == 200
        conv_data = conv_res.json()
        assert conv_data["conversation_id"] == "conv_root"
        assert conv_data["metrics"]["total_replies"] == 1
        assert "sentiment_dynamics" in conv_data
        assert conv_data["sentiment_dynamics"]["root_post_id"] == "conv_root"
        assert conv_data["sentiment_dynamics"]["total_comments_analyzed"] == 1
        assert conv_data["thread_tree"]["post_id"] == "conv_root"

        # 404 on nonexistent conversation
        conv_404 = client.get("/api/v1/analytics/conversations/missing_thread")
        assert conv_404.status_code == 404

        # 2. /api/v1/analytics/audience/{user_id}
        aud_res = client.get("/api/v1/analytics/audience/root_author?min_group_size=1")
        assert aud_res.status_code == 200
        aud_data = aud_res.json()
        assert aud_data["influencer_id"] == "root_author"
        assert "age_distribution" in aud_data

        # 3. /api/v1/analytics/trends/forecast
        fc_res = client.get("/api/v1/analytics/trends/forecast?term=ai&window_minutes=60")
        assert fc_res.status_code == 200
        fc_data = fc_res.json()
        assert isinstance(fc_data, list)
        assert len(fc_data) == 1
        assert fc_data[0]["term"] == "ai"
        assert "current_velocity" in fc_data[0]
        assert "acceleration" in fc_data[0]

        # 3b. /api/v1/analytics/trends/forecast without term (UNNEST discovery, verifies C2)
        fc_auto = client.get("/api/v1/analytics/trends/forecast?window_minutes=60")
        assert fc_auto.status_code == 200
        fc_auto_data = fc_auto.json()
        assert len(fc_auto_data) >= 1
        assert fc_auto_data[0]["term"] == "ai"

        # 4. /api/v1/analytics/trends/narrative-drift
        drift_res = client.get("/api/v1/analytics/trends/narrative-drift?term=ai&window_minutes=60")
        assert drift_res.status_code == 200
        drift_data = drift_res.json()
        assert drift_data["topic"] == "ai"
        assert "cosine_drift" in drift_data

        # 5. /api/v1/analytics/diffusion/cross-segment/{cascade_id}
        diff_res = client.get("/api/v1/analytics/diffusion/cross-segment/casc_ai_step6?segment_by=community")
        assert diff_res.status_code == 200
        diff_data = diff_res.json()
        assert diff_data["identifier"] == "casc_ai_step6"
        assert diff_data["origin_segment"] == "community_1"
        assert diff_data["chronological_sequence"] == ["community_1", "community_2"]

        # Invalid segment_by rejected with 400
        invalid_diff = client.get("/api/v1/analytics/diffusion/cross-segment/casc_ai_step6?segment_by=unsupported_dim")
        assert invalid_diff.status_code == 400

        # 6. /api/v1/analytics/influencers/bridge-kols
        kols_res = client.get("/api/v1/analytics/influencers/bridge-kols?top_k=5&min_neighbors=1")
        assert kols_res.status_code == 200
        kols_data = kols_res.json()
        assert "top_bridges" in kols_data


@pytest.mark.anyio
async def test_streaming_sse_generator():
    """Test EventBroadcaster.sse_generator streaming output and event formatting."""
    broadcaster = EventBroadcaster(max_queue_size=10)

    # Mock Request
    request = MagicMock()
    request.is_disconnected.return_value = False

    async def _test_stream():
        events_collected = []
        gen = broadcaster.sse_generator(request, heartbeat_interval=5.0)

        async def _delayed_broadcast():
            await asyncio.sleep(0.05)
            broadcaster.broadcast({"status": "active", "item": 42}, event_type="system_event")

        b_task = asyncio.create_task(_delayed_broadcast())

        async for event in gen:
            if event["event"] != "ping":
                events_collected.append(event)
                break  # Exit after first non-ping event

        await b_task
        return events_collected

    events = await _test_stream()
    assert len(events) == 1
    assert events[0]["event"] == "system_event"
    payload = json.loads(events[0]["data"])
    assert payload["status"] == "active"
    assert payload["item"] == 42
