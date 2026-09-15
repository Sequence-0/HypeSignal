"""Step 5 Tests: Real Cross-Segment Diffusion & Bridge KOL Analysis (Pillar E).

Verifies:
1. Automatic Louvain community persistence to DuckDB user_communities via NetworkEngine.
2. CrossSegmentDiffusionTracker chronological sequence reconstruction, inter-segment transmission latencies, and sentiment drift.
3. BridgeKOLAnalyzer calculation of cross-community edge ratio C(u), betweenness centrality, and boundary spanner ranking.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import networkx as nx
import pytest

from hypesignal.network.bridge_kols import BridgeKOLAnalyzer, BridgeKOLLeaderboard, BridgeKOLProfile
from hypesignal.network.cross_segment_diffusion import (
    CrossSegmentDiffusionReport,
    CrossSegmentDiffusionTracker,
    SegmentAdoption,
    SegmentTransition,
)
from hypesignal.network.graph_store import NetworkXGraphStore
from hypesignal.network.network_engine import NetworkEngine
from hypesignal.storage.duckdb_manager import DuckDBManager


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_step5.duckdb"
        manager = DuckDBManager(db_path=db_path)
        yield manager
        manager.close()


def test_network_engine_community_persistence(temp_db):
    """Test that NetworkEngine.detect_communities automatically persists partitions to DuckDB."""
    graph_store = NetworkXGraphStore()
    
    # Create two disjoint triangles
    # Triangle 1: nodes 1, 2, 3
    graph_store.add_edge("user_1", "user_2", relation_type="follows")
    graph_store.add_edge("user_2", "user_3", relation_type="follows")
    graph_store.add_edge("user_3", "user_1", relation_type="follows")
    
    # Triangle 2: nodes 4, 5, 6
    graph_store.add_edge("user_4", "user_5", relation_type="follows")
    graph_store.add_edge("user_5", "user_6", relation_type="follows")
    graph_store.add_edge("user_6", "user_4", relation_type="follows")

    engine = NetworkEngine(graph_store=graph_store, db=temp_db)
    comm_res = engine.detect_communities(min_community_size=2)

    assert comm_res.num_communities >= 2
    assert len(comm_res.partition) == 6

    # Verify DuckDB persistence
    stored_rows = temp_db.con.execute(
        "SELECT user_id, community_id, modularity_score FROM user_communities ORDER BY user_id;"
    ).fetchall()
    
    assert len(stored_rows) == 6
    user_ids = [r[0] for r in stored_rows]
    assert set(user_ids) == {"user_1", "user_2", "user_3", "user_4", "user_5", "user_6"}
    
    # Verify communities assigned correctly
    comm_1 = [r[1] for r in stored_rows if r[0] in {"user_1", "user_2", "user_3"}]
    comm_2 = [r[1] for r in stored_rows if r[0] in {"user_4", "user_5", "user_6"}]
    assert len(set(comm_1)) == 1
    assert len(set(comm_2)) == 1
    assert comm_1[0] != comm_2[0]


def test_cross_segment_diffusion_tracker_synthetic():
    """Test chronological adoption reconstruction, transition latencies, and sentiment drift."""
    t0 = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)

    adoptions = [
        # Segment A (Origin): starts positive
        SegmentAdoption(
            user_id="u_a1",
            timestamp=t0,
            community_id=1,
            sentiment_score=0.8,
            effective_polarity="positive",
        ),
        SegmentAdoption(
            user_id="u_a2",
            timestamp=t0 + timedelta(seconds=15),
            community_id=1,
            sentiment_score=0.6,
            effective_polarity="positive",
        ),
        # Segment B (Transmitted downstream 60s later): neutral
        SegmentAdoption(
            user_id="u_b1",
            timestamp=t0 + timedelta(seconds=60),
            community_id=2,
            sentiment_score=0.1,
            effective_polarity="neutral",
        ),
        # Segment C (Transmitted further downstream 180s later): strongly negative
        SegmentAdoption(
            user_id="u_c1",
            timestamp=t0 + timedelta(seconds=180),
            community_id=3,
            sentiment_score=0.75,
            effective_polarity="negative",
        ),
    ]

    tracker = CrossSegmentDiffusionTracker()
    report = tracker.track_diffusion_events(
        identifier="cascade_101",
        adoptions=adoptions,
        segment_by="community",
    )

    assert report.identifier == "cascade_101"
    assert report.origin_segment == "community_1"
    assert pytest.approx(report.origin_sentiment, rel=1e-2) == 0.70  # mean(0.8, 0.6)
    assert report.total_adoptions == 4
    assert report.chronological_sequence == ["community_1", "community_2", "community_3"]

    assert len(report.transitions) == 2
    
    # Transition 1: A -> B
    t1 = report.transitions[0]
    assert t1.from_segment == "community_1"
    assert t1.to_segment == "community_2"
    assert t1.latency_seconds == 60.0
    assert pytest.approx(t1.sentiment_drift, rel=1e-2) == -0.60  # 0.1 - 0.7

    # Transition 2: B -> C
    t2 = report.transitions[1]
    assert t2.from_segment == "community_2"
    assert t2.to_segment == "community_3"
    assert t2.latency_seconds == 120.0  # 180 - 60
    assert pytest.approx(t2.sentiment_drift, rel=1e-2) == -0.85  # -0.75 - 0.1

    # Overall cross-segment sentiment drift: C - A
    assert pytest.approx(report.cross_segment_sentiment_drift, rel=1e-2) == -1.45  # -0.75 - 0.70


def test_cross_segment_diffusion_demographic_segmentation():
    """Test diffusion tracking by demographic age brackets."""
    t0 = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)

    adoptions = [
        SegmentAdoption(
            user_id="u1",
            timestamp=t0,
            age_bracket="18-24",
            sentiment_score=0.5,
            effective_polarity="positive",
        ),
        SegmentAdoption(
            user_id="u2",
            timestamp=t0 + timedelta(minutes=5),
            age_bracket="25-34",
            sentiment_score=0.3,
            effective_polarity="positive",
        ),
        SegmentAdoption(
            user_id="u3",
            timestamp=t0 + timedelta(minutes=15),
            age_bracket="35-49",
            sentiment_score=0.2,
            effective_polarity="negative",
        ),
    ]

    tracker = CrossSegmentDiffusionTracker()
    report = tracker.track_diffusion_events(
        identifier="topic_ai",
        adoptions=adoptions,
        segment_by="age_bracket",
    )

    assert report.origin_segment == "18-24"
    assert report.chronological_sequence == ["18-24", "25-34", "35-49"]
    assert len(report.transitions) == 2
    assert report.transitions[0].latency_seconds == 300.0  # 5 minutes
    assert report.transitions[1].latency_seconds == 600.0  # 10 minutes


def test_cross_segment_diffusion_empty_cases():
    """Test graceful handling of empty or singleton adoption lists."""
    tracker = CrossSegmentDiffusionTracker()
    empty_report = tracker.track_diffusion_events(
        identifier="empty_test",
        adoptions=[],
        segment_by="community",
    )
    assert empty_report.total_adoptions == 0
    assert empty_report.origin_segment == "none"
    assert empty_report.transitions == []
    assert empty_report.cross_segment_sentiment_drift == 0.0


def test_bridge_kol_analyzer_synthetic_boundary_spanner():
    """Test that a node bridging two distinct communities receives a high cross-community edge ratio."""
    graph_store = NetworkXGraphStore()

    # Community 0: Clique A (nodes A1, A2, A3)
    # Directed follows edges
    for u in ["A1", "A2", "A3"]:
        for v in ["A1", "A2", "A3"]:
            if u != v:
                graph_store.add_edge(u, v, relation_type="follows")

    # Community 1: Clique B (nodes B1, B2, B3)
    for u in ["B1", "B2", "B3"]:
        for v in ["B1", "B2", "B3"]:
            if u != v:
                graph_store.add_edge(u, v, relation_type="follows")

    # Bridge Node: X is connected to 2 members of Clique A and 2 members of Clique B
    graph_store.add_edge("X", "A1", relation_type="follows")
    graph_store.add_edge("A2", "X", relation_type="follows")
    graph_store.add_edge("X", "B1", relation_type="follows")
    graph_store.add_edge("B2", "X", relation_type="follows")

    # Explicit partition: A's in 0, B's in 1, X in 0
    partition = {
        "A1": 0, "A2": 0, "A3": 0,
        "B1": 1, "B2": 1, "B3": 1,
        "X": 0,
    }

    analyzer = BridgeKOLAnalyzer()
    leaderboard = analyzer.analyze_bridges(
        graph_store=graph_store,
        community_partition=partition,
        top_k=5,
        min_neighbors=2,
    )

    assert leaderboard.total_nodes_analyzed == 7
    assert len(leaderboard.top_bridges) >= 1

    top_bridge = leaderboard.top_bridges[0]
    assert top_bridge.user_id == "X"
    # X has 4 neighbors: A1, A2 (community 0) and B1, B2 (community 1).
    # Cross community ratio = 2 / 4 = 0.50
    assert top_bridge.cross_community_edge_ratio == 0.50
    assert top_bridge.cross_community_neighbors == 2
    assert top_bridge.total_neighbors == 4
    assert top_bridge.connected_communities == [1]
    assert top_bridge.bridge_kol_score > 0.3


def test_network_engine_get_bridge_kols_integration(temp_db):
    """Test NetworkEngine.get_bridge_kols end-to-end with community auto-detection and DuckDB user resolution."""
    # Insert user names into DuckDB
    temp_db.con.execute(
        "INSERT INTO users (id, platform, screen_name) VALUES "
        "('u_bridge', 'bluesky', 'BridgeMaster'), "
        "('u_a1', 'bluesky', 'Alpha1'), "
        "('u_b1', 'bluesky', 'Beta1');"
    )

    graph_store = NetworkXGraphStore()
    
    # Community A: u_a1, u_a2, u_a3
    for u in ["u_a1", "u_a2", "u_a3"]:
        for v in ["u_a1", "u_a2", "u_a3"]:
            if u != v:
                graph_store.add_edge(u, v, relation_type="follows")

    # Community B: u_b1, u_b2, u_b3
    for u in ["u_b1", "u_b2", "u_b3"]:
        for v in ["u_b1", "u_b2", "u_b3"]:
            if u != v:
                graph_store.add_edge(u, v, relation_type="follows")

    # u_bridge connects to both
    graph_store.add_edge("u_bridge", "u_a1", relation_type="follows")
    graph_store.add_edge("u_bridge", "u_a2", relation_type="follows")
    graph_store.add_edge("u_bridge", "u_b1", relation_type="follows")
    graph_store.add_edge("u_bridge", "u_b2", relation_type="follows")

    engine = NetworkEngine(graph_store=graph_store, db=temp_db)
    leaderboard = engine.get_bridge_kols(top_k=5, min_neighbors=2)

    assert len(leaderboard.top_bridges) >= 1
    bridge = leaderboard.top_bridges[0]
    assert bridge.user_id == "u_bridge"
    assert bridge.screen_name == "BridgeMaster"
    assert bridge.cross_community_edge_ratio >= 0.50
    assert bridge.bridge_kol_score > 0.0


def test_cross_segment_diffusion_from_db(temp_db):
    """Test querying and reconstructing cross-segment diffusion directly from DuckDB tables."""
    # 1. Populate users and user_communities
    temp_db.con.execute(
        "INSERT INTO user_communities (user_id, community_id, modularity_score) VALUES "
        "('user_1', 10, 0.45), "
        "('user_2', 20, 0.45);"
    )

    # 2. Populate posts
    t0 = datetime(2026, 3, 15, 8, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=10)
    temp_db.con.execute(
        "INSERT INTO posts (id, platform, author_id, text, timestamp, timestamp_ms) VALUES "
        "('p1', 'bluesky', 'user_1', 'AI breakthrough is phenomenal!', ?, ?), "
        "('p2', 'bluesky', 'user_2', 'AI breakthrough is disastrous and terrible.', ?, ?);",
        [t0, int(t0.timestamp() * 1000), t1, int(t1.timestamp() * 1000)],
    )

    # 3. Populate post_analytics
    temp_db.con.execute(
        "INSERT INTO post_analytics (post_id, effective_polarity, sentiment_score, is_sarcastic, irony_score, primary_emotion, emotion_score) VALUES "
        "('p1', 'positive', 0.85, false, 0.0, 'joy', 0.85), "
        "('p2', 'negative', 0.90, false, 0.0, 'fear', 0.90);"
    )

    # 4. Populate cascade_events
    temp_db.con.execute(
        "INSERT INTO cascade_events (cascade_id, post_id, user_id, timestamp, timestamp_ms, adoption_order) VALUES "
        "('casc_ai', 'p1', 'user_1', ?, ?, 0), "
        "('casc_ai', 'p2', 'user_2', ?, ?, 1);",
        [t0, int(t0.timestamp() * 1000), t1, int(t1.timestamp() * 1000)],
    )

    tracker = CrossSegmentDiffusionTracker(db=temp_db)

    # Test cascade tracking from DB
    cascade_report = tracker.track_cascade_from_db(
        db=temp_db,
        cascade_id="casc_ai",
        segment_by="community",
    )
    assert cascade_report.identifier == "casc_ai"
    assert cascade_report.origin_segment == "community_10"
    assert cascade_report.chronological_sequence == ["community_10", "community_20"]
    assert len(cascade_report.transitions) == 1
    assert cascade_report.transitions[0].latency_seconds == 600.0  # 10 minutes
    assert pytest.approx(cascade_report.transitions[0].sentiment_drift, rel=1e-2) == -1.75  # -0.90 - 0.85

    # Test topic diffusion from DB
    topic_report = tracker.track_topic_diffusion_from_db(
        db=temp_db,
        topic="breakthrough",
        start_time=t0 - timedelta(hours=1),
        end_time=t0 + timedelta(hours=1),
        segment_by="community",
    )
    assert topic_report.identifier == "breakthrough"
    assert topic_report.origin_segment == "community_10"
    assert topic_report.chronological_sequence == ["community_10", "community_20"]


def test_bridge_kol_analyzer_noise_and_unassigned_rejection():
    """Verify that unassigned nodes (None) and noise bucket nodes (-1) do not inflate C(u) or become ghost bridges."""
    graph_store = NetworkXGraphStore()
    analyzer = BridgeKOLAnalyzer()

    # Node U1 belongs to noise cluster (-1)
    # Node U2 is not assigned at all (None)
    # Node U3 is in community 0, but only connects to noise nodes (-1)
    # Node U4 is in community 0, connects to 2 members of community 0, 1 member of community 1, and 1 noise node (-1)
    graph_store.add_edge("U1", "N1", relation_type="follows")
    graph_store.add_edge("U1", "N2", relation_type="follows")

    graph_store.add_edge("U2", "N1", relation_type="follows")
    graph_store.add_edge("U2", "N2", relation_type="follows")

    graph_store.add_edge("U3", "N1", relation_type="follows")
    graph_store.add_edge("U3", "N2", relation_type="follows")

    graph_store.add_edge("U4", "A1", relation_type="follows")
    graph_store.add_edge("U4", "A2", relation_type="follows")
    graph_store.add_edge("U4", "B1", relation_type="follows")
    graph_store.add_edge("U4", "N1", relation_type="follows")

    # Connect internal cliques so degree >= 2
    graph_store.add_edge("A1", "A2", relation_type="follows")
    graph_store.add_edge("A2", "A1", relation_type="follows")
    graph_store.add_edge("B1", "B2", relation_type="follows")
    graph_store.add_edge("B2", "B1", relation_type="follows")

    partition = {
        "U1": -1,  # Noise bucket
        # U2 omitted -> None
        "U3": 0,
        "U4": 0,
        "A1": 0,
        "A2": 0,
        "B1": 1,
        "B2": 1,
        "N1": -1,  # Noise
        "N2": -1,  # Noise
    }

    # 1. Direct ratio check
    ratio_u1, cross_u1, foreign_u1 = analyzer.compute_cross_community_ratio("U1", {"N1", "N2"}, partition)
    assert ratio_u1 == 0.0
    assert cross_u1 == 0
    assert foreign_u1 == []

    ratio_u2, cross_u2, foreign_u2 = analyzer.compute_cross_community_ratio("U2", {"N1", "N2"}, partition)
    assert ratio_u2 == 0.0
    assert cross_u2 == 0
    assert foreign_u2 == []

    ratio_u3, cross_u3, foreign_u3 = analyzer.compute_cross_community_ratio("U3", {"N1", "N2"}, partition)
    assert ratio_u3 == 0.0
    assert cross_u3 == 0
    assert foreign_u3 == []

    # U4 has neighbors: A1(0), A2(0), B1(1), N1(-1).
    # Total neighbors = 4. Only B1 is valid foreign community.
    # Cross community ratio = 1 / 4 = 0.25
    ratio_u4, cross_u4, foreign_u4 = analyzer.compute_cross_community_ratio("U4", {"A1", "A2", "B1", "N1"}, partition)
    assert ratio_u4 == 0.25
    assert cross_u4 == 1
    assert foreign_u4 == [1]

    # 2. Leaderboard check: U1, U2, U3 must NOT be present on the leaderboard
    leaderboard = analyzer.analyze_bridges(graph_store, community_partition=partition, min_neighbors=2)
    bridge_ids = [b.user_id for b in leaderboard.top_bridges]
    assert "U1" not in bridge_ids
    assert "U2" not in bridge_ids
    assert "U3" not in bridge_ids


def test_cross_segment_diffusion_from_db_demographics(temp_db):
    """Test DuckDB diffusion queries with age_bracket and persona extracted from users table."""
    import json

    t0 = datetime(2026, 3, 15, 14, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=5)
    t2 = t0 + timedelta(minutes=15)

    # 1. Populate users with demographics in extra_metadata
    u1_meta = json.dumps({"age_bracket": "18-24", "persona": "crypto_trader"})
    u2_meta = json.dumps({"age_bracket": "25-34", "persona": "tech_developer"})
    u3_meta = json.dumps({"persona": {"age_bracket": "35-49", "primary_persona": "finance_exec"}})

    temp_db.con.execute(
        "INSERT INTO users (id, platform, screen_name, extra_metadata) VALUES "
        "('demo_u1', 'bluesky', 'GenZ_Trader', ?), "
        "('demo_u2', 'bluesky', 'Dev_Coder', ?), "
        "('demo_u3', 'bluesky', 'Exec_Boss', ?);",
        [u1_meta, u2_meta, u3_meta],
    )

    # 2. Populate posts
    temp_db.con.execute(
        "INSERT INTO posts (id, platform, author_id, text, timestamp, timestamp_ms) VALUES "
        "('dp1', 'bluesky', 'demo_u1', 'Bullish sentiment!', ?, ?), "
        "('dp2', 'bluesky', 'demo_u2', 'Skeptical about code quality.', ?, ?), "
        "('dp3', 'bluesky', 'demo_u3', 'Bearish on macro.', ?, ?);",
        [
            t0, int(t0.timestamp() * 1000),
            t1, int(t1.timestamp() * 1000),
            t2, int(t2.timestamp() * 1000),
        ],
    )

    # 3. Populate post_analytics
    temp_db.con.execute(
        "INSERT INTO post_analytics (post_id, effective_polarity, sentiment_score, is_sarcastic, irony_score, primary_emotion, emotion_score) VALUES "
        "('dp1', 'positive', 0.8, false, 0.0, 'excitement', 0.8), "
        "('dp2', 'neutral', 0.0, false, 0.0, 'neutral', 0.5), "
        "('dp3', 'negative', 0.7, false, 0.0, 'fear', 0.7);"
    )

    # 4. Populate cascade_events
    temp_db.con.execute(
        "INSERT INTO cascade_events (cascade_id, post_id, user_id, timestamp, timestamp_ms, adoption_order) VALUES "
        "('casc_demo', 'dp1', 'demo_u1', ?, ?, 0), "
        "('casc_demo', 'dp2', 'demo_u2', ?, ?, 1), "
        "('casc_demo', 'dp3', 'demo_u3', ?, ?, 2);",
        [
            t0, int(t0.timestamp() * 1000),
            t1, int(t1.timestamp() * 1000),
            t2, int(t2.timestamp() * 1000),
        ],
    )

    tracker = CrossSegmentDiffusionTracker(db=temp_db)

    # Age bracket tracking from DB
    age_report = tracker.track_cascade_from_db(temp_db, "casc_demo", segment_by="age_bracket")
    assert age_report.segment_type == "age_bracket"
    assert age_report.origin_segment == "18-24"
    assert age_report.chronological_sequence == ["18-24", "25-34", "35-49"]
    assert len(age_report.transitions) == 2
    assert age_report.transitions[0].latency_seconds == 300.0  # 5 min
    assert age_report.transitions[1].latency_seconds == 600.0  # 10 min

    # Persona tracking from DB
    persona_report = tracker.track_cascade_from_db(temp_db, "casc_demo", segment_by="persona")
    assert persona_report.segment_type == "persona"
    assert persona_report.origin_segment == "crypto_trader"
    assert persona_report.chronological_sequence == ["crypto_trader", "tech_developer", "finance_exec"]

    # Invalid dimension validation
    with pytest.raises(ValueError, match="Unsupported segment_by='invalid_dim'"):
        tracker.track_cascade_from_db(temp_db, "casc_demo", segment_by="invalid_dim")
