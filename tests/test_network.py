"""Comprehensive test suite for Link Analysis & Network Topology Engine (Component E)."""

from datetime import datetime, timedelta, timezone
import pytest

from hypesignal.ingestion.lerman_adapter import LermanDatasetAdapter
from hypesignal.models.canonical import CanonicalCascadeEvent, CanonicalGraphEdge
from hypesignal.models.enums import RelationType
from hypesignal.network.cascade_tracer import CascadeTracer
from hypesignal.network.community_detector import CommunityDetector
from hypesignal.network.graph_store import MemgraphStore, NetworkXGraphStore, validate_relation_type
from hypesignal.network.kol_analyzer import KOLAnalyzer
from hypesignal.network.network_engine import NetworkEngine
from hypesignal.network.schemas import (
    CascadeTree,
    CommunityDetectionResult,
    KOLRankingResult,
    NetworkOverview,
)
from hypesignal.storage.duckdb_manager import DuckDBManager


@pytest.fixture
def graph_store():
    return NetworkXGraphStore()


@pytest.fixture
def kol_analyzer():
    return KOLAnalyzer(weight_pagerank=0.5, weight_indegree=0.35, weight_betweenness=0.15)


@pytest.fixture
def community_detector():
    return CommunityDetector(default_algorithm="louvain", random_seed=42)


@pytest.fixture
def cascade_tracer():
    return CascadeTracer()


@pytest.fixture
def network_engine(graph_store, kol_analyzer, community_detector, cascade_tracer):
    return NetworkEngine(
        graph_store=graph_store,
        kol_analyzer=kol_analyzer,
        community_detector=community_detector,
        cascade_tracer=cascade_tracer,
    )


# ---------------------------------------------------------------------------
# 1. Graph Store Tests
# ---------------------------------------------------------------------------

def test_graph_store_lifecycle(graph_store):
    # Add nodes and edges
    graph_store.add_node("alice", role="user")
    graph_store.add_node("bob", role="influencer")
    graph_store.add_edge(source_id="alice", target_id="bob", relation_type="FOLLOWS", weight=1.0)

    assert graph_store.has_node("alice")
    assert graph_store.has_node("bob")
    assert not graph_store.has_node("charlie")

    assert graph_store.has_edge("alice", "bob")
    assert not graph_store.has_edge("bob", "alice")

    assert graph_store.get_node_count() == 2
    assert graph_store.get_edge_count() == 1

    # In-degree / followers of bob
    assert graph_store.get_in_degree("bob") == 1
    assert graph_store.get_out_degree("bob") == 0
    assert graph_store.get_followers("bob") == ["alice"]
    assert graph_store.get_following("bob") == []

    # Out-degree / following of alice
    assert graph_store.get_in_degree("alice") == 0
    assert graph_store.get_out_degree("alice") == 1
    assert graph_store.get_following("alice") == ["bob"]

    # Bulk CanonicalGraphEdge addition
    edges = [
        CanonicalGraphEdge(source_id="charlie", target_id="bob", relation_type=RelationType.FOLLOWS),
        CanonicalGraphEdge(source_id="dave", target_id="bob", relation_type=RelationType.FOLLOWS),
    ]
    graph_store.add_edges_from(edges)
    assert graph_store.get_node_count() == 4
    assert graph_store.get_edge_count() == 3
    assert graph_store.get_in_degree("bob") == 3

    # Subgraph
    sub = graph_store.get_subgraph(["alice", "bob"])
    assert sub.get_node_count() == 2
    assert sub.get_edge_count() == 1

    # Clear
    graph_store.clear()
    assert graph_store.get_node_count() == 0
    assert graph_store.get_edge_count() == 0


def test_graph_store_duckdb_roundtrip():
    db = DuckDBManager(":memory:")
    edges = [
        CanonicalGraphEdge(source_id="u1", target_id="u2", relation_type=RelationType.FOLLOWS),
        CanonicalGraphEdge(source_id="u3", target_id="u2", relation_type=RelationType.FOLLOWS),
        CanonicalGraphEdge(source_id="u2", target_id="u4", relation_type=RelationType.FOLLOWS),
    ]
    db.insert_edges(edges)
    assert db.get_edges_count() == 3

    store = NetworkXGraphStore()
    loaded = store.load_from_duckdb(db)
    assert loaded == 3
    assert store.get_node_count() == 4
    assert store.get_in_degree("u2") == 2
    assert store.get_following("u2") == ["u4"]


# ---------------------------------------------------------------------------
# 2. KOL Analyzer Tests
# ---------------------------------------------------------------------------

def test_kol_analyzer_hierarchy(graph_store, kol_analyzer):
    # Star graph: KOL_1 followed by 10 users, KOL_2 followed by 4 users
    for i in range(10):
        uid = f"user_{i}"
        graph_store.add_edge(source_id=uid, target_id="kol_1")
    for i in range(4):
        uid = f"user_{i}"
        graph_store.add_edge(source_id=uid, target_id="kol_2")

    # Bridge edge between KOL_2 and KOL_1
    graph_store.add_edge(source_id="kol_2", target_id="kol_1")

    res = kol_analyzer.rank_kols(graph_store, top_k=5)

    assert isinstance(res, KOLRankingResult)
    assert len(res.top_kols) >= 2
    top_1 = res.top_kols[0]
    top_2 = res.top_kols[1]

    # kol_1 should rank #1 with highest in-degree and PageRank
    assert top_1.user_id == "kol_1"
    assert top_1.rank == 1
    assert top_1.in_degree == 11
    assert top_1.pagerank > top_2.pagerank
    assert top_1.influence_score > top_2.influence_score

    assert top_2.user_id == "kol_2"
    assert top_2.rank == 2
    assert top_2.in_degree == 4


def test_kol_analyzer_spearman_correlation(kol_analyzer):
    predicted = {"a": 0.95, "b": 0.80, "c": 0.60, "d": 0.30, "e": 0.10}
    ground_truth = {"a": 1000.0, "b": 850.0, "c": 500.0, "d": 200.0, "e": 50.0}

    rho = kol_analyzer.compute_spearman_correlation(predicted, ground_truth)
    assert rho == 1.0

    # Inverted correlation test
    inverted_gt = {"a": 10.0, "b": 20.0, "c": 30.0, "d": 40.0, "e": 50.0}
    rho_neg = kol_analyzer.compute_spearman_correlation(predicted, inverted_gt)
    assert rho_neg == -1.0


# ---------------------------------------------------------------------------
# 3. Community Detector Tests
# ---------------------------------------------------------------------------

def test_community_detector_louvain_and_lpa(graph_store, community_detector):
    # Cluster 1: nodes 0..4 fully connected
    for i in range(5):
        for j in range(5):
            if i != j:
                graph_store.add_edge(source_id=f"c1_{i}", target_id=f"c1_{j}")

    # Cluster 2: nodes 0..4 fully connected
    for i in range(5):
        for j in range(5):
            if i != j:
                graph_store.add_edge(source_id=f"c2_{i}", target_id=f"c2_{j}")

    # Weak bridge between clusters
    graph_store.add_edge(source_id="c1_4", target_id="c2_0")

    # 1. Louvain
    louvain_res = community_detector.detect_communities(graph_store, algorithm="louvain")
    assert isinstance(louvain_res, CommunityDetectionResult)
    assert louvain_res.num_communities == 2
    assert louvain_res.modularity is not None
    assert louvain_res.modularity > 0.35
    for comm in louvain_res.communities:
        assert comm.member_count == 5
        assert comm.density > 0.8

    # 2. Label Propagation
    lpa_res = community_detector.detect_communities(graph_store, algorithm="label_propagation")
    assert isinstance(lpa_res, CommunityDetectionResult)
    assert lpa_res.num_communities == 2


# ---------------------------------------------------------------------------
# 4. Cascade Tracer Tests
# ---------------------------------------------------------------------------

def test_cascade_tracer_star_vs_chain_virality(cascade_tracer):
    now = datetime(2026, 9, 12, 10, 0, tzinfo=timezone.utc)

    # 1. Broadcast / Star Cascade: root connects to 10 adopters directly
    star_events = [
        CanonicalCascadeEvent(
            cascade_id="http://broadcast.com/news",
            post_id=f"p_star_{i}",
            user_id=f"u_star_{i}",
            timestamp=now + timedelta(minutes=i),
            adoption_order=i,
        )
        for i in range(11)
    ]
    star_tree = cascade_tracer.reconstruct_cascade_tree(star_events, graph_store=None)

    assert isinstance(star_tree, CascadeTree)
    assert star_tree.root_user_id == "u_star_0"
    assert star_tree.metrics.total_adoptions == 11
    assert star_tree.metrics.max_depth == 1  # All direct children of root
    assert star_tree.metrics.max_breadth == 10
    # Star virality: 2 - 2/11 = ~1.818
    assert 1.7 < star_tree.metrics.structural_virality < 2.0

    # 2. Viral / Chain Cascade: relay u_0 -> u_1 -> u_2 -> ... -> u_10
    chain_events = []
    for i in range(11):
        parent_pid = f"p_chain_{i-1}" if i > 0 else None
        chain_events.append(
            CanonicalCascadeEvent(
                cascade_id="http://viral.com/chain",
                post_id=f"p_chain_{i}",
                user_id=f"u_chain_{i}",
                timestamp=now + timedelta(minutes=i * 5),
                adoption_order=i,
                parent_event_id=parent_pid,
            )
        )
    chain_tree = cascade_tracer.reconstruct_cascade_tree(chain_events, graph_store=None)

    assert chain_tree.metrics.total_adoptions == 11
    assert chain_tree.metrics.max_depth == 10  # Deep chain
    assert chain_tree.metrics.max_breadth == 1
    # Chain virality: (11 + 1) / 3 = 4.0
    assert round(chain_tree.metrics.structural_virality, 1) == 4.0
    # Chain virality should be strictly higher than star virality
    assert chain_tree.metrics.structural_virality > star_tree.metrics.structural_virality


def test_cascade_tracer_with_social_graph(cascade_tracer, graph_store):
    now = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)

    # Topology: B follows A, C follows B (follower -> followee)
    # A posts -> B reads & posts -> C reads & posts
    graph_store.add_edge(source_id="B", target_id="A")
    graph_store.add_edge(source_id="C", target_id="B")

    events = [
        CanonicalCascadeEvent(
            cascade_id="http://topic.com/1",
            post_id="p1",
            user_id="A",
            timestamp=now,
            adoption_order=0,
        ),
        CanonicalCascadeEvent(
            cascade_id="http://topic.com/1",
            post_id="p2",
            user_id="B",
            timestamp=now + timedelta(minutes=10),
            adoption_order=1,
        ),
        CanonicalCascadeEvent(
            cascade_id="http://topic.com/1",
            post_id="p3",
            user_id="C",
            timestamp=now + timedelta(minutes=25),
            adoption_order=2,
        ),
    ]

    tree = cascade_tracer.reconstruct_cascade_tree(events, graph_store=graph_store)

    assert tree.metrics.total_adoptions == 3
    # B was infected by A (depth 1), C was infected by B (depth 2)
    assert tree.nodes["A"].depth == 0
    assert tree.nodes["B"].depth == 1
    assert tree.nodes["B"].parent_user_id == "A"
    assert tree.nodes["C"].depth == 2
    assert tree.nodes["C"].parent_user_id == "B"
    assert tree.metrics.max_depth == 2
    assert tree.nodes["C"].time_delay_seconds == 15 * 60.0


def test_cascade_tracer_from_duckdb(cascade_tracer):
    db = DuckDBManager(":memory:")
    now = datetime(2026, 9, 12, 14, 0, tzinfo=timezone.utc)

    events = [
        CanonicalCascadeEvent(
            cascade_id="http://duckdb.org/post",
            post_id=f"post_{i}",
            user_id=f"user_{i}",
            timestamp=now + timedelta(minutes=i * 2),
            adoption_order=i,
        )
        for i in range(6)
    ]
    db.insert_cascade_events(events)

    tree = cascade_tracer.trace_from_duckdb(db, cascade_id="http://duckdb.org/post")
    assert tree is not None
    assert tree.cascade_id == "http://duckdb.org/post"
    assert tree.metrics.total_adoptions == 6

    top_cascades = cascade_tracer.trace_top_cascades_from_duckdb(db, min_size=5, limit=5)
    assert len(top_cascades) == 1
    assert top_cascades[0].cascade_id == "http://duckdb.org/post"


# ---------------------------------------------------------------------------
# 5. Network Engine End-to-End Tests
# ---------------------------------------------------------------------------

def test_network_engine_end_to_end(network_engine):
    db = DuckDBManager(":memory:")
    now = datetime(2026, 9, 12, 16, 0, tzinfo=timezone.utc)

    # Insert cascade events
    events = [
        CanonicalCascadeEvent(
            cascade_id="http://hypesignal.ai/launch",
            post_id=f"p_{i}",
            user_id=f"u_{i}",
            timestamp=now + timedelta(minutes=i * 3),
            adoption_order=i,
        )
        for i in range(8)
    ]
    db.insert_cascade_events(events)

    # Populate graph
    for i in range(1, 8):
        network_engine.graph_store.add_edge(source_id=f"u_{i}", target_id="u_0")

    overview = network_engine.analyze_network(kol_top_k=5, cascade_limit=5, db=db)

    assert isinstance(overview, NetworkOverview)
    assert overview.total_nodes == 8
    assert overview.total_edges == 7
    assert overview.density > 0.0
    assert len(overview.top_kols) > 0
    assert overview.top_kols[0].user_id == "u_0"
    assert overview.top_kols[0].in_degree == 7

    assert len(overview.recent_cascades) == 1
    assert overview.recent_cascades[0].cascade_id == "http://hypesignal.ai/launch"

    # Profile lookup
    u0_prof = network_engine.get_user_network_profile("u_0", db=db)
    assert u0_prof is not None
    assert u0_prof.rank == 1

    # Ego subgraph export
    subgraph_data = network_engine.export_subgraph(center_user_id="u_0", radius=1)
    assert "nodes" in subgraph_data
    assert "links" in subgraph_data
    assert len(subgraph_data["nodes"]) == 8
    assert len(subgraph_data["links"]) == 7


def test_lerman_real_edge_stream_and_kol():
    adapter = LermanDatasetAdapter()
    store = NetworkXGraphStore()
    analyzer = KOLAnalyzer()

    # Stream real follower edges from compressed sql archive
    count = store.load_from_edge_stream(adapter.stream_follower_edges(limit=150), limit=150)
    assert count == 150
    assert store.get_edge_count() == 150
    assert store.get_node_count() > 10

    # Rank KOLs on real dataset edges
    kols = analyzer.rank_kols(store, top_k=5)
    assert len(kols.top_kols) == 5
    assert kols.top_kols[0].in_degree > 0
    assert kols.top_kols[0].influence_score > 0.0


def test_memgraph_store_cypher_injection_prevention():
    from unittest.mock import MagicMock
    store = MemgraphStore()

    # 1. Reproduce reviewer exploit: invalid syntax tokens / clauses must raise ValueError
    malicious_type = "FOLLOWS]->(x) DETACH DELETE x MERGE (s)-[r:FOLLOWS"
    with pytest.raises(ValueError, match="Invalid relation_type"):
        store.add_edge("alice", "bob", relation_type=malicious_type)

    with pytest.raises(ValueError):
        store.add_edges_from([
            CanonicalGraphEdge.model_construct(source_id="alice", target_id="bob", relation_type=malicious_type)
        ])

    # 2. Non-alphanumeric injections
    with pytest.raises(ValueError):
        store.add_edge("alice", "bob", relation_type="FOLLOWS; DROP TABLE users;")

    # 3. Test Cypher query construction when live driver is present (mocked)
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    store._driver = mock_driver

    store.add_edge("alice", "bob", relation_type="follows", weight=2.5)
    call_args = mock_session.run.call_args
    query_str = call_args[0][0]
    kwargs = call_args[1]

    # Relation type must be sanitized to uppercase identifier in Cypher
    assert "MERGE (s)-[r:FOLLOWS]->(t)" in query_str
    assert kwargs["source_id"] == "alice"
    assert kwargs["target_id"] == "bob"
    assert kwargs["weight"] == 2.5


def test_network_engine_profile_caching(network_engine):
    # Setup graph
    for i in range(1, 5):
        network_engine.graph_store.add_edge(source_id=f"user_{i}", target_id="target_star")

    # Initial analyze_network caches rankings
    overview = network_engine.analyze_network(kol_top_k=5)
    assert len(overview.top_kols) > 0
    assert network_engine._cached_kol_profiles is not None

    # Cached lookup should not re-run full PageRank/betweenness
    cached_prof = network_engine.get_user_network_profile("target_star", use_cache=True)
    assert cached_prof is not None
    assert cached_prof.user_id == "target_star"
    assert cached_prof.in_degree == 4

    # Now directly mutate the underlying graph store (reviewer scenario)
    network_engine.graph_store.add_edge(source_id="user_new", target_id="target_star")
    assert network_engine.graph_store.revision > network_engine._cached_revision

    # Lookup automatically detects revision mismatch, invalidates cache, and recomputes in_degree to 5
    updated_prof = network_engine.get_user_network_profile("target_star", use_cache=True)
    assert updated_prof is not None
    assert updated_prof.in_degree == 5
    assert network_engine._cached_revision == network_engine.graph_store.revision

    # Explicit invalidation
    network_engine.invalidate_cache()
    assert network_engine._cached_kol_profiles is None

    # Non-existent node
    assert network_engine.get_user_network_profile("unknown_user") is None
