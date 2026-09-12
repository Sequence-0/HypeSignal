"""Integration and unit tests for the unified HypeSignal FastAPI application."""

from datetime import datetime, timedelta, timezone
from typing import Dict

import pytest
from fastapi.testclient import TestClient

from hypesignal.api.app import create_app
from hypesignal.connectors.reddit import RedditConnector
from hypesignal.connectors.telegram import TelegramConnector
from hypesignal.connectors.twitter import TwitterConnector
from hypesignal.connectors.youtube import YouTubeConnector
from hypesignal.demographics.demographics_engine import DemographicsEngine
from hypesignal.demographics.geo_profiler import GeoProfiler
from hypesignal.demographics.persona_profiler import PersonaProfiler
from hypesignal.models.canonical import (
    CanonicalCascadeEvent,
    CanonicalGraphEdge,
    CanonicalPost,
    CanonicalUser,
    PostMetrics,
)
from hypesignal.models.enums import PlatformType, RelationType
from hypesignal.network.graph_store import NetworkXGraphStore
from hypesignal.network.network_engine import NetworkEngine
from hypesignal.nlp.engine import MultiDimensionalSentimentEngine
from hypesignal.nlp.temporal_sentiment import TemporalSentimentTracker
from hypesignal.storage.duckdb_manager import DuckDBManager
from hypesignal.timeline.timeline_manager import TimelineManager
from hypesignal.trends.burst_detector import BurstDetector
from hypesignal.trends.topic_modeler import DynamicTopicModeler
from hypesignal.trends.trend_ranker import TrendRanker
from hypesignal.trends.trends_engine import TrendsEngine


@pytest.fixture(scope="module")
def shared_nlp_engine():
    """Module-level NLP engine to avoid re-loading model weights repeatedly."""
    return MultiDimensionalSentimentEngine()


@pytest.fixture(scope="module")
def api_test_data(shared_nlp_engine):
    """Seed test database, network, and engines for API tests."""
    db = DuckDBManager(":memory:")

    # 1. Seed historical posts
    base_time = datetime(2024, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
    sample_posts = []
    for i in range(25):
        ts = base_time + timedelta(hours=i)
        sample_posts.append(
            CanonicalPost(
                id=f"p_{i}",
                platform=PlatformType.TWITTER,
                author_id="user_alice" if i % 2 == 0 else f"user_{i}",
                author_screen_name="alice_crypto" if i % 2 == 0 else f"user_{i}",
                text=f"AI and cryptocurrency technology update #{i % 3} #tech https://example.com/post/{i}",
                timestamp=ts,
                metrics=PostMetrics(likes=10 + i, reposts=2 + i, replies=1),
            )
        )
    db.insert_posts(sample_posts)

    # 2. Seed users
    users = [
        CanonicalUser(
            id="user_alice",
            platform=PlatformType.TWITTER,
            screen_name="alice_crypto",
            bio="AI researcher and decentralization enthusiast in San Francisco, CA",
            location_raw="San Francisco, CA",
        ),
        CanonicalUser(
            id="user_bob",
            platform=PlatformType.TWITTER,
            screen_name="bob_engineer",
            bio="Software engineer in New York, NY",
            location_raw="New York, NY",
        ),
    ]
    db.insert_users(users)

    # 3. Seed cascade events
    cascade_events = [
        CanonicalCascadeEvent(
            cascade_id="cascade_ai_news",
            post_id="p_0",
            user_id="user_alice",
            user_screen_name="alice_crypto",
            timestamp=base_time,
            adoption_order=1,
        ),
        CanonicalCascadeEvent(
            cascade_id="cascade_ai_news",
            post_id="p_1",
            user_id="user_bob",
            user_screen_name="bob_engineer",
            timestamp=base_time + timedelta(minutes=15),
            adoption_order=2,
        ),
    ]
    db.insert_cascade_events(cascade_events)

    # 4. Network Graph Store
    graph_store = NetworkXGraphStore()
    graph_edges = [
        ("user_bob", "user_alice"),
        ("user_carol", "user_alice"),
        ("user_dan", "user_alice"),
        ("user_erin", "user_alice"),
        ("user_alice", "user_bob"),
    ]
    for s, t in graph_edges:
        graph_store.add_edge(s, t, relation_type="FOLLOWS", weight=1.0)

    # 5. Build sub-engines
    timeline = TimelineManager(db=db)
    temporal_sentiment = TemporalSentimentTracker(engine=shared_nlp_engine, db=db)
    demographics = DemographicsEngine(
        persona_profiler=PersonaProfiler(device="cpu"),
        device="cpu",
    )
    trends = TrendsEngine(
        burst_detector=BurstDetector(default_z_threshold=1.5, default_min_count=2),
        topic_modeler=DynamicTopicModeler(device="cpu", min_topic_size=2),
        trend_ranker=TrendRanker(),
        device="cpu",
    )
    network = NetworkEngine(graph_store=graph_store)

    connectors = {
        "twitter": TwitterConnector(),
        "reddit": RedditConnector(),
        "youtube": YouTubeConnector(),
        "telegram": TelegramConnector(),
    }

    # 6. Create unified FastAPI app
    app = create_app(
        db=db,
        timeline=timeline,
        nlp=shared_nlp_engine,
        temporal_sentiment=temporal_sentiment,
        demographics=demographics,
        trends=trends,
        network=network,
        connectors=connectors,
        auto_connect=True,
    )

    return {
        "app": app,
        "db": db,
        "base_time": base_time,
        "timeline": timeline,
        "network": network,
    }


def test_healthcheck_endpoints(api_test_data):
    """Test /health and /api/v1/health endpoints."""
    app = api_test_data["app"]
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "duckdb" in data["engines"]
        assert data["engines"]["duckdb"] == "online"
        assert data["engines"]["nlp"] == "online"

        res_v1 = client.get("/api/v1/health")
        assert res_v1.status_code == 200
        assert res_v1.json()["status"] == "ok"


def test_timeline_endpoints(api_test_data):
    """Test /api/v1/timeline bounds, slice, timeseries, cascade chronology, and user posts."""
    app = api_test_data["app"]
    base_time = api_test_data["base_time"]

    with TestClient(app) as client:
        # Bounds
        res = client.get("/api/v1/timeline/bounds")
        assert res.status_code == 200
        bounds = res.json()
        assert bounds["total_posts"] == 25
        assert bounds["earliest"] is not None
        assert bounds["latest"] is not None

        # Slice
        start_str = base_time.isoformat()
        end_str = (base_time + timedelta(hours=5)).isoformat()
        res = client.get(
            "/api/v1/timeline/slice",
            params={"start_time": start_str, "end_time": end_str, "limit": 10},
        )
        assert res.status_code == 200
        posts = res.json()
        assert len(posts) > 0
        assert posts[0]["platform"] == "twitter"

        # Timeseries
        res = client.get("/api/v1/timeline/timeseries?interval=6 hours")
        assert res.status_code == 200
        ts_data = res.json()
        assert ts_data["interval"] == "6 hours"
        assert ts_data["total_buckets"] > 0
        assert len(ts_data["points"]) > 0

        # Cascade chronology
        res = client.get("/api/v1/timeline/cascade/cascade_ai_news")
        assert res.status_code == 200
        casc = res.json()
        assert casc["cascade_id"] == "cascade_ai_news"
        assert casc["total_events"] == 2

        # Non-existent cascade returns 404
        res_404 = client.get("/api/v1/timeline/cascade/non_existent_cascade")
        assert res_404.status_code == 404

        # User timeline
        res = client.get("/api/v1/timeline/user/user_alice?limit=10")
        assert res.status_code == 200
        user_posts = res.json()
        assert len(user_posts) > 0
        assert user_posts[0]["author_id"] == "user_alice"


def test_sentiment_endpoints(api_test_data):
    """Test /api/v1/sentiment analyze and temporal trajectory."""
    app = api_test_data["app"]

    with TestClient(app) as client:
        # Single text
        res = client.post(
            "/api/v1/sentiment/analyze",
            json={"text": "I am thrilled and ecstatic about this wonderful news!"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["count"] == 1
        assert data["results"][0]["effective_polarity"] == "positive"

        # Batch texts
        res = client.post(
            "/api/v1/sentiment/analyze",
            json={"texts": ["Great achievement!", "Terrible disaster and total failure."]},
        )
        assert res.status_code == 200
        batch_data = res.json()
        assert batch_data["count"] == 2

        # Empty body validation
        res_bad = client.post("/api/v1/sentiment/analyze", json={})
        assert res_bad.status_code == 400

        # Temporal trajectory over seeded posts
        res = client.get("/api/v1/sentiment/temporal?interval=12 hours&limit=25")
        assert res.status_code == 200
        temp_data = res.json()
        assert temp_data["interval"] == "12 hours"
        assert temp_data["total_buckets"] > 0


def test_demographics_endpoints(api_test_data):
    """Test /api/v1/demographics profile-user, breakdown, and users list."""
    app = api_test_data["app"]

    with TestClient(app) as client:
        # Profile ad-hoc user
        res = client.post(
            "/api/v1/demographics/profile-user",
            json={
                "screen_name": "silicon_dev",
                "bio": "Senior AI Architect building neural models in San Francisco, CA",
                "location_raw": "San Francisco, CA",
                "sample_posts": ["Just trained a 7B parameter LLM!", "Deploying on Kubernetes"],
            },
        )
        assert res.status_code == 200
        prof = res.json()
        assert prof["geo"]["city"] == "San Francisco"
        assert prof["persona"]["primary_persona"] is not None

        # Profile existing user from DuckDB
        res_db = client.post(
            "/api/v1/demographics/profile-user",
            json={"user_id": "user_alice"},
        )
        assert res_db.status_code == 200
        assert res_db.json()["user_id"] == "user_alice"

        # Audience breakdown
        res = client.get("/api/v1/demographics/breakdown?limit=10")
        assert res.status_code == 200
        breakdown = res.json()
        assert breakdown["total_users_profiled"] >= 2

        # List profiled users
        res = client.get("/api/v1/demographics/users?limit=5")
        assert res.status_code == 200
        users_list = res.json()
        assert len(users_list) >= 2


def test_trends_endpoints(api_test_data):
    """Test /api/v1/trends bursts, overview, and topics."""
    app = api_test_data["app"]

    with TestClient(app) as client:
        # Bursts
        res = client.get("/api/v1/trends/bursts?only_active=false")
        assert res.status_code == 200
        assert isinstance(res.json(), list)

        # Overview
        res = client.get("/api/v1/trends/overview?window_duration_minutes=1440&z_threshold=1.0&min_count=2")
        assert res.status_code == 200
        overview = res.json()
        assert "active_bursts" in overview
        assert "ranked_trends" in overview

        # Topics
        res = client.get("/api/v1/trends/topics?hours_back=48&limit=25&nr_bins=3")
        assert res.status_code == 200
        topics_data = res.json()
        assert "topics" in topics_data
        assert "timelines" in topics_data


def test_network_endpoints(api_test_data):
    """Test /api/v1/network overview, kols, user profile, subgraph, and cascade analysis."""
    app = api_test_data["app"]

    with TestClient(app) as client:
        # Overview
        res = client.get("/api/v1/network/overview")
        assert res.status_code == 200
        overview = res.json()
        assert overview["total_nodes"] == 5
        assert overview["total_edges"] == 5
        assert len(overview["top_kols"]) > 0

        # Top KOLs
        res = client.get("/api/v1/network/kols?top_k=5")
        assert res.status_code == 200
        kols = res.json()
        assert len(kols) > 0
        # Alice has the highest in-degree (4 incoming edges)
        assert kols[0]["user_id"] == "user_alice"
        assert kols[0]["community_id"] is not None
        overview_alice = next(k for k in overview["top_kols"] if k["user_id"] == "user_alice")
        assert kols[0]["community_id"] == overview_alice["community_id"]

        # User network profile
        res = client.get("/api/v1/network/user/user_alice")
        assert res.status_code == 200
        alice_prof = res.json()
        assert alice_prof["user_id"] == "user_alice"
        assert alice_prof["in_degree"] == 4

        # Non-existent user
        res_404 = client.get("/api/v1/network/user/ghost_user")
        assert res_404.status_code == 404

        # Subgraph export
        res = client.get("/api/v1/network/subgraph/user_alice?radius=1")
        assert res.status_code == 200
        subgraph = res.json()
        assert "nodes" in subgraph
        assert "links" in subgraph
        assert len(subgraph["nodes"]) > 0

        # Cascade analysis
        res = client.get("/api/v1/network/cascade/cascade_ai_news")
        assert res.status_code == 200
        casc_tree = res.json()
        assert casc_tree["cascade_id"] == "cascade_ai_news"
        assert len(casc_tree["nodes"]) == 2
        assert casc_tree["metrics"]["total_adoptions"] == 2

        # Non-existent cascade
        res = client.get("/api/v1/network/cascade/non_existent_cascade")
        assert res.status_code == 404


def test_connectors_endpoints(api_test_data):
    """Test /api/v1/connectors status and polling across platforms."""
    app = api_test_data["app"]

    with TestClient(app) as client:
        # Status
        res = client.get("/api/v1/connectors/status")
        assert res.status_code == 200
        status_data = res.json()
        connectors = status_data["connectors"]
        assert "twitter" in connectors
        assert "reddit" in connectors
        assert "youtube" in connectors
        assert "telegram" in connectors
        assert connectors["twitter"]["status"] == "connected"

        # Poll Twitter
        res = client.post(
            "/api/v1/connectors/twitter/poll",
            json={"query": "AI research", "limit": 2},
        )
        assert res.status_code == 200
        tw_poll = res.json()
        assert tw_poll["platform"] == "twitter"
        assert tw_poll["count"] > 0
        assert len(tw_poll["posts"]) > 0

        # Poll Reddit
        res = client.post(
            "/api/v1/connectors/reddit/poll",
            json={"query": "Machine Learning", "limit": 2},
        )
        assert res.status_code == 200
        rd_poll = res.json()
        assert rd_poll["platform"] == "reddit"
        assert rd_poll["count"] > 0

        # Poll YouTube
        res = client.post(
            "/api/v1/connectors/youtube/poll",
            json={"query": "Deep Learning", "limit": 2},
        )
        assert res.status_code == 200
        yt_poll = res.json()
        assert yt_poll["platform"] == "youtube"
        assert yt_poll["count"] > 0

        # Poll Telegram
        res = client.post(
            "/api/v1/connectors/telegram/poll",
            json={"query": "Signals", "limit": 2},
        )
        assert res.status_code == 200
        tg_poll = res.json()
        assert tg_poll["platform"] == "telegram"
        assert tg_poll["count"] > 0

        # Invalid platform
        res = client.post(
            "/api/v1/connectors/myspace/poll",
            json={"query": "hello"},
        )
        assert res.status_code == 404


def test_empty_data_trends_topics_does_not_crash(shared_nlp_engine):
    """Test that GET /api/v1/trends/topics returns clean 200 with empty list on empty database."""
    empty_db = DuckDBManager(":memory:")
    app = create_app(
        db=empty_db,
        nlp=shared_nlp_engine,
        trends=TrendsEngine(
            burst_detector=BurstDetector(),
            topic_modeler=DynamicTopicModeler(device="cpu", min_topic_size=2),
            device="cpu",
        ),
    )

    with TestClient(app) as client:
        res = client.get("/api/v1/trends/topics?hours_back=24")
        assert res.status_code == 200
        data = res.json()
        assert data["total_topics"] == 0
        assert data["topics"] == []
        assert data["timelines"] == []


def test_cors_security_not_echoing_arbitrary_origins(api_test_data):
    """Test that arbitrary untrusted origins are NOT echoed back with allow-origin and allow-credentials."""
    app = api_test_data["app"]

    with TestClient(app) as client:
        # Untrusted external origin
        res_evil = client.get(
            "/api/v1/health",
            headers={"Origin": "https://evil.example.com"},
        )
        assert res_evil.status_code == 200
        # Untrusted origin must not be echoed in Access-Control-Allow-Origin
        assert res_evil.headers.get("access-control-allow-origin") != "https://evil.example.com"
        assert res_evil.headers.get("access-control-allow-credentials") is None

        # Trusted allowed origin
        res_good = client.get(
            "/api/v1/health",
            headers={"Origin": "http://localhost:3000"},
        )
        assert res_good.status_code == 200
        assert res_good.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_sentiment_batch_size_cap(api_test_data):
    """Test that oversized batches (>100 items) are rejected with 422 Unprocessable Entity."""
    app = api_test_data["app"]

    with TestClient(app) as client:
        oversized = [f"Text snippet {i}" for i in range(101)]
        res = client.post(
            "/api/v1/sentiment/analyze",
            json={"texts": oversized},
        )
        assert res.status_code == 422
