"""Consolidated Link Analysis & Network Topology Engine (Component E).

Orchestrates social graph storage, Key Opinion Leader (KOL) ranking, community clustering,
and information diffusion cascade tracking.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import networkx as nx

from hypesignal.ingestion.lerman_adapter import LermanDatasetAdapter
from hypesignal.network.bridge_kols import BridgeKOLAnalyzer, BridgeKOLLeaderboard, BridgeKOLProfile
from hypesignal.network.cascade_tracer import CascadeTracer
from hypesignal.network.community_detector import CommunityDetector
from hypesignal.network.graph_store import BaseGraphStore, NetworkXGraphStore
from hypesignal.network.kol_analyzer import KOLAnalyzer
from hypesignal.network.schemas import (
    CascadeTree,
    CommunityCluster,
    CommunityDetectionResult,
    KOLProfile,
    KOLRankingResult,
    NetworkOverview,
)
from hypesignal.storage.duckdb_manager import DuckDBManager

logger = logging.getLogger(__name__)


class NetworkEngine:
    """Consolidated Link Analysis & Network Topology Coordinator."""

    def __init__(
        self,
        graph_store: Optional[BaseGraphStore] = None,
        kol_analyzer: Optional[KOLAnalyzer] = None,
        community_detector: Optional[CommunityDetector] = None,
        cascade_tracer: Optional[CascadeTracer] = None,
        bridge_kol_analyzer: Optional[BridgeKOLAnalyzer] = None,
        db: Optional[DuckDBManager] = None,
    ) -> None:
        """Initialize NetworkEngine with modular sub-components.
        
        Args:
            graph_store: Graph storage implementation (defaults to NetworkXGraphStore).
            kol_analyzer: Custom KOLAnalyzer or standard weighted centralities.
            community_detector: Custom CommunityDetector (Louvain / LPA).
            cascade_tracer: Custom CascadeTracer.
            bridge_kol_analyzer: Custom BridgeKOLAnalyzer for boundary spanners.
            db: Optional DuckDBManager instance for community persistence.
        """
        self.graph_store = graph_store or NetworkXGraphStore()
        self.kol_analyzer = kol_analyzer or KOLAnalyzer()
        self.community_detector = community_detector or CommunityDetector()
        self.cascade_tracer = cascade_tracer or CascadeTracer()
        self.bridge_kol_analyzer = bridge_kol_analyzer or BridgeKOLAnalyzer()
        self.db = db
        self._cached_kol_profiles: Optional[Dict[str, KOLProfile]] = None
        self._cached_revision: int = -1

    def detect_communities(
        self,
        min_community_size: int = 2,
        db: Optional[DuckDBManager] = None,
    ) -> CommunityDetectionResult:
        """Detect Louvain communities and automatically persist partition to DuckDB user_communities."""
        comm_res = self.community_detector.detect_communities(
            graph_store=self.graph_store,
            min_community_size=min_community_size,
        )
        target_db = db or self.db
        if target_db is not None and comm_res.partition:
            mod_score = float(comm_res.modularity) if comm_res.modularity is not None else 0.0
            assignments = [
                (str(node_id), int(comm_id), mod_score)
                for node_id, comm_id in comm_res.partition.items()
            ]
            try:
                target_db.upsert_user_communities(assignments)
            except Exception as e:
                logger.warning("Failed to persist user communities to DuckDB: %s", e)
        return comm_res

    def invalidate_cache(self) -> None:
        """Explicitly invalidate cached influencer rankings."""
        self._cached_kol_profiles = None
        self._cached_revision = -1

    def load_from_lerman_adapter(
        self,
        lerman_adapter: LermanDatasetAdapter,
        limit: Optional[int] = None,
    ) -> int:
        """Stream follower edges from Lerman dataset directly into graph store."""
        if hasattr(self.graph_store, "load_from_edge_stream"):
            return self.graph_store.load_from_edge_stream(
                lerman_adapter.stream_follower_edges(limit=limit),
                limit=limit,
            )

        count = 0
        for edge in lerman_adapter.stream_follower_edges(limit=limit):
            self.graph_store.add_edge(
                source_id=edge.source_id,
                target_id=edge.target_id,
                relation_type=edge.relation_type.value if hasattr(edge.relation_type, "value") else str(edge.relation_type),
                weight=edge.weight,
                **edge.extra_metadata,
            )
            count += 1
            if limit and count >= limit:
                break
        self._cached_kol_profiles = None
        return count

    def load_from_duckdb(
        self,
        db: DuckDBManager,
        limit: Optional[int] = None,
    ) -> int:
        """Load stored follower edges from DuckDB into graph store."""
        self._cached_kol_profiles = None
        if hasattr(self.graph_store, "load_from_duckdb"):
            return self.graph_store.load_from_duckdb(db, limit=limit)

        query = "SELECT source_id, target_id, relation_type, weight FROM graph_edges"
        params = []
        if limit:
            query += " LIMIT ?"
            params.append(int(limit))

        rows = db.con.execute(query, params).fetchall()
        for s, t, r, w in rows:
            self.graph_store.add_edge(s, t, relation_type=r, weight=w or 1.0)
        return len(rows)

    def analyze_network(
        self,
        min_community_size: int = 2,
        kol_top_k: int = 20,
        cascade_limit: int = 5,
        db: Optional[DuckDBManager] = None,
    ) -> NetworkOverview:
        """Run comprehensive link analysis: topology, communities, KOLs, and cascades.
        
        Args:
            min_community_size: Minimum members per community cluster.
            kol_top_k: Number of top KOL influencers to extract.
            cascade_limit: Number of top diffusion cascades to reconstruct from DuckDB.
            db: Optional DuckDBManager instance to extract cascade history and user metadata.
            
        Returns:
            NetworkOverview summarizing network health, clusters, KOLs, and cascades.
        """
        nx_digraph = self.graph_store.to_networkx()
        total_nodes = nx_digraph.number_of_nodes()
        total_edges = nx_digraph.number_of_edges()

        if total_nodes == 0:
            return NetworkOverview(
                total_nodes=0,
                total_edges=0,
                density=0.0,
                is_connected=False,
                num_connected_components=0,
                top_kols=[],
                communities=[],
                recent_cascades=[],
            )

        # 1. Topology density & connectivity
        undir = nx_digraph.to_undirected()
        is_connected = nx.is_connected(undir) if total_nodes > 0 else False
        num_components = nx.number_connected_components(undir) if total_nodes > 0 else 0

        # Directed density: E / (V * (V - 1))
        if total_nodes > 1:
            density = round(total_edges / (total_nodes * (total_nodes - 1)), 6)
        else:
            density = 0.0

        # 2. Community Detection
        comm_res = self.detect_communities(
            min_community_size=min_community_size,
            db=db,
        )

        # 3. User metadata lookup from DuckDB (if available)
        user_meta_map: Dict[str, Dict[str, Any]] = {}
        if db is not None:
            try:
                user_rows = db.con.execute("SELECT id, screen_name FROM users;").fetchall()
                for uid, sname in user_rows:
                    user_meta_map[str(uid)] = {"screen_name": sname}
            except Exception as e:
                logger.debug("User metadata lookup skipped: %s", e)

        # 4. KOL Ranking
        kol_res = self.kol_analyzer.rank_kols(
            graph_store=self.graph_store,
            top_k=kol_top_k,
            user_metadata=user_meta_map,
            community_partition=comm_res.partition,
        )
        self._cached_kol_profiles = {p.user_id: p for p in kol_res.top_kols}
        self._cached_revision = getattr(self.graph_store, "revision", 0)

        # 5. Cascades (if DuckDB provided)
        cascades: List[CascadeTree] = []
        if db is not None:
            try:
                cascades = self.cascade_tracer.trace_top_cascades_from_duckdb(
                    db=db,
                    min_size=3,
                    limit=cascade_limit,
                    graph_store=self.graph_store,
                )
            except Exception as e:
                logger.warning("Cascade tracing from DuckDB encountered an issue: %s", e)

        return NetworkOverview(
            total_nodes=total_nodes,
            total_edges=total_edges,
            density=density,
            is_connected=is_connected,
            num_connected_components=num_components,
            top_kols=kol_res.top_kols,
            communities=comm_res.communities,
            recent_cascades=cascades,
        )

    def get_top_kols(
        self,
        top_k: int = 20,
        db: Optional[DuckDBManager] = None,
        min_community_size: int = 2,
        use_cache: bool = True,
    ) -> List[KOLProfile]:
        """Retrieve ranked Key Opinion Leaders (KOLs) with caching and community partition.
        
        Args:
            top_k: Number of top KOLs to return.
            db: Optional DuckDBManager instance for user screen_name resolution.
            min_community_size: Minimum community size for partition detection.
            use_cache: Whether to return cached results if graph revision is unchanged.
            
        Returns:
            List of KOLProfile objects sorted by rank.
        """
        current_rev = getattr(self.graph_store, "revision", 0)

        # Check if cached profiles exist, match current revision, and cover the requested top_k
        if (
            use_cache
            and self._cached_kol_profiles is not None
            and self._cached_revision == current_rev
        ):
            cached_list = sorted(self._cached_kol_profiles.values(), key=lambda p: p.rank)
            if len(cached_list) >= top_k or len(cached_list) == self.graph_store.get_node_count():
                return cached_list[:top_k]

        total_nodes = self.graph_store.get_node_count()
        if total_nodes == 0:
            return []

        # 1. Detect communities so community_id is always consistently populated
        comm_res = self.detect_communities(
            min_community_size=min_community_size,
            db=db,
        )

        # 2. Resolve screen names from DuckDB
        user_meta_map: Dict[str, Dict[str, Any]] = {}
        if db is not None:
            try:
                user_rows = db.con.execute("SELECT id, screen_name FROM users;").fetchall()
                for uid, sname in user_rows:
                    user_meta_map[str(uid)] = {"screen_name": sname}
            except Exception as e:
                logger.debug("User metadata lookup skipped in get_top_kols: %s", e)

        # 3. Compute KOL rankings for all nodes (or at least top_k)
        fetch_k = max(top_k, total_nodes)
        kol_res = self.kol_analyzer.rank_kols(
            graph_store=self.graph_store,
            top_k=fetch_k,
            user_metadata=user_meta_map,
            community_partition=comm_res.partition,
        )

        self._cached_kol_profiles = {p.user_id: p for p in kol_res.top_kols}
        self._cached_revision = current_rev
        return kol_res.top_kols[:top_k]

    def get_user_network_profile(
        self,
        user_id: str,
        db: Optional[DuckDBManager] = None,
        use_cache: bool = True,
    ) -> Optional[KOLProfile]:
        """Fetch local network metrics and ranking for an individual user."""
        s_uid = str(user_id)
        if not self.graph_store.has_node(s_uid):
            return None

        current_rev = getattr(self.graph_store, "revision", 0)

        # Return cached profile if available and underlying graph has not been mutated
        if (
            use_cache
            and self._cached_kol_profiles is not None
            and self._cached_revision == current_rev
            and s_uid in self._cached_kol_profiles
        ):
            return self._cached_kol_profiles[s_uid]

        # Compute full KOL rankings with community partition and cache them
        self.get_top_kols(
            top_k=self.graph_store.get_node_count(),
            db=db,
            use_cache=False,
        )
        return self._cached_kol_profiles.get(s_uid) if self._cached_kol_profiles else None

    def get_cascade_analysis(
        self,
        cascade_id: str,
        db: DuckDBManager,
    ) -> Optional[CascadeTree]:
        """Analyze a specific information diffusion cascade using the current graph topology."""
        return self.cascade_tracer.trace_from_duckdb(
            db=db,
            cascade_id=cascade_id,
            graph_store=self.graph_store,
        )

    def get_bridge_kols(
        self,
        top_k: int = 20,
        min_neighbors: int = 2,
        min_cross_ratio: float = 0.0,
        db: Optional[DuckDBManager] = None,
        min_community_size: int = 2,
    ) -> BridgeKOLLeaderboard:
        """Identify and rank boundary spanner nodes bridging distinct communities.
        
        Args:
            top_k: Maximum number of bridge nodes to return.
            min_neighbors: Minimum degree required to qualify as bridge candidate.
            min_cross_ratio: Minimum C(u) required to qualify.
            db: Optional DuckDBManager instance for user screen_name resolution.
            min_community_size: Minimum size of communities to partition into.
            
        Returns:
            BridgeKOLLeaderboard containing ranked BridgeKOLProfile entries.
        """
        target_db = db or self.db
        comm_res = self.detect_communities(min_community_size=min_community_size, db=target_db)
        return self.bridge_kol_analyzer.analyze_bridges(
            graph_store=self.graph_store,
            community_partition=comm_res.partition,
            top_k=top_k,
            min_neighbors=min_neighbors,
            min_cross_ratio=min_cross_ratio,
            db=target_db,
        )

    def export_subgraph(
        self,
        center_user_id: str,
        radius: int = 1,
        max_nodes: int = 100,
    ) -> Dict[str, Any]:
        """Export egocentric neighborhood subgraph around a center node for frontend visualizers.
        
        Args:
            center_user_id: Target user ID.
            radius: Graph hop radius (1 = direct neighbors, 2 = 2nd degree).
            max_nodes: Maximum nodes to return.
            
        Returns:
            JSON-serializable dict with 'nodes' and 'links'.
        """
        nx_g = self.graph_store.to_networkx()
        s_center = str(center_user_id)
        if not nx_g.has_node(s_center):
            return {"nodes": [], "links": []}

        # BFS ego graph
        ego_nodes = set([s_center])
        curr_frontier = set([s_center])

        for _ in range(radius):
            next_frontier = set()
            for u in curr_frontier:
                next_frontier.update(nx_g.predecessors(u))  # followers
                next_frontier.update(nx_g.successors(u))    # following
            ego_nodes.update(next_frontier)
            curr_frontier = next_frontier
            if len(ego_nodes) >= max_nodes:
                break

        selected_nodes = list(ego_nodes)[:max_nodes]
        sub = nx_g.subgraph(selected_nodes)

        nodes_list = [
            {
                "id": n,
                "in_degree": int(sub.in_degree(n)),
                "out_degree": int(sub.out_degree(n)),
                "is_center": (n == s_center),
            }
            for n in sub.nodes()
        ]
        links_list = [
            {
                "source": u,
                "target": v,
                "relation_type": data.get("relation_type", "FOLLOWS"),
                "weight": float(data.get("weight", 1.0)),
            }
            for u, v, data in sub.edges(data=True)
        ]

        return {"nodes": nodes_list, "links": links_list}
