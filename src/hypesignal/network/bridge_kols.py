"""Inter-Community Bridge Key Opinion Leader (KOL) Analyzer (Component E).

Identifies boundary spanner nodes that bridge distinct Louvain communities,
calculating Cross-Community Edge Ratio C(u) and betweenness centrality.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import networkx as nx
from pydantic import BaseModel, Field

from hypesignal.network.community_detector import CommunityDetector
from hypesignal.network.graph_store import BaseGraphStore
from hypesignal.storage.duckdb_manager import DuckDBManager

logger = logging.getLogger(__name__)


class BridgeKOLProfile(BaseModel):
    """Profile of a node bridging multiple network communities."""
    user_id: str = Field(..., description="Unique user identifier")
    community_id: Optional[int] = Field(default=None, description="Primary community assignment")
    screen_name: Optional[str] = Field(default=None, description="User screen handle if known")
    cross_community_edge_ratio: float = Field(..., ge=0.0, le=1.0, description="C(u): fraction of neighbors in external communities")
    betweenness_centrality: float = Field(default=0.0, ge=0.0, description="Normalized betweenness centrality")
    total_neighbors: int = Field(default=0, ge=0, description="Total unique undirected neighbors")
    cross_community_neighbors: int = Field(default=0, ge=0, description="Neighbors residing in different communities")
    connected_communities: List[int] = Field(default_factory=list, description="List of foreign community IDs connected to")
    bridge_kol_score: float = Field(..., ge=0.0, le=1.0, description="Composite boundary spanner score")
    rank: int = Field(default=1, ge=1, description="Rank in bridge leaderboard")


class BridgeKOLLeaderboard(BaseModel):
    """Ranked leaderboard of inter-community bridge influencers."""
    total_nodes_analyzed: int = Field(default=0, ge=0)
    top_bridges: List[BridgeKOLProfile] = Field(default_factory=list)
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


BridgeKOLAnalysisResult = BridgeKOLLeaderboard


class BridgeKOLAnalyzer:
    """Identifies and ranks boundary spanners bridging distinct network communities."""

    def __init__(
        self,
        weight_cross_ratio: float = 0.60,
        weight_betweenness: float = 0.25,
        weight_communities: float = 0.15,
        betweenness_max_samples: int = 500,
    ) -> None:
        """Initialize BridgeKOLAnalyzer with scoring weights.
        
        Args:
            weight_cross_ratio: Weight for cross-community edge ratio C(u).
            weight_betweenness: Weight for betweenness centrality.
            weight_communities: Weight for count of distinct foreign communities bridged.
            betweenness_max_samples: Maximum pivot samples for betweenness centrality calculation.
        """
        tot = weight_cross_ratio + weight_betweenness + weight_communities
        if tot <= 0:
            tot = 1.0
        self.w_cross = weight_cross_ratio / tot
        self.w_betw = weight_betweenness / tot
        self.w_comm = weight_communities / tot
        self.betweenness_max_samples = betweenness_max_samples

    def compute_cross_community_ratio(
        self,
        node_id: str,
        neighbors: Set[str],
        community_partition: Dict[str, int],
    ) -> tuple[float, int, List[int]]:
        """Compute C(u) = |{v in N(u) | Comm(v) != Comm(u)}| / |N(u)|.
        
        Returns:
            Tuple of (cross_community_ratio, cross_community_neighbors_count, connected_foreign_communities).
        """
        if not neighbors:
            return 0.0, 0, []

        c_u = community_partition.get(str(node_id), community_partition.get(node_id))
        if c_u is None or c_u < 0:
            return 0.0, 0, []

        cross_neighbors = 0
        foreign_communities: Set[int] = set()

        for v in neighbors:
            c_v = community_partition.get(str(v), community_partition.get(v))
            # Must strictly connect to another valid, distinct community
            if c_v is not None and c_v >= 0 and c_v != c_u:
                cross_neighbors += 1
                foreign_communities.add(c_v)

        ratio = round(cross_neighbors / len(neighbors), 4)
        return ratio, cross_neighbors, sorted(foreign_communities)

    def analyze_bridges(
        self,
        graph_store: BaseGraphStore,
        community_partition: Optional[Dict[str, int]] = None,
        top_k: int = 20,
        min_neighbors: int = 2,
        min_cross_ratio: float = 0.0,
        user_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        db: Optional[DuckDBManager] = None,
    ) -> BridgeKOLLeaderboard:
        """Identify boundary spanner nodes bridging distinct communities.
        
        Args:
            graph_store: BaseGraphStore containing the social graph.
            community_partition: Optional node_id -> community_id mapping.
                If not provided, Louvain community detection will be run automatically.
            top_k: Maximum number of bridge nodes to return.
            min_neighbors: Minimum degree required to qualify as bridge candidate.
            min_cross_ratio: Minimum C(u) required to qualify.
            user_metadata: Optional mapping of user_id -> metadata (e.g. screen_name).
            db: Optional DuckDBManager instance to look up user screen names.
            
        Returns:
            BridgeKOLLeaderboard containing ranked BridgeKOLProfile entries.
        """
        nx_digraph = graph_store.to_networkx()
        total_nodes = nx_digraph.number_of_nodes()

        if total_nodes == 0:
            return BridgeKOLLeaderboard(total_nodes_analyzed=0, top_bridges=[])

        # 1. Resolve community partition if not provided
        partition = community_partition
        if partition is None:
            detector = CommunityDetector()
            comm_res = detector.detect_communities(graph_store)
            partition = comm_res.partition

        # Total valid communities count
        valid_comm_ids = set(c for c in partition.values() if c >= 0)
        num_communities = len(valid_comm_ids)

        # 2. Resolve user metadata (screen names)
        meta_map: Dict[str, Dict[str, Any]] = dict(user_metadata or {})
        if db is not None and not meta_map:
            try:
                rows = db.con.execute("SELECT id, screen_name FROM users;").fetchall()
                for uid, sname in rows:
                    meta_map[str(uid)] = {"screen_name": sname}
            except Exception as e:
                logger.debug("BridgeKOLAnalyzer user metadata query skipped: %s", e)

        # 3. Compute betweenness centrality
        k_sample = (
            min(self.betweenness_max_samples, total_nodes)
            if total_nodes > self.betweenness_max_samples
            else None
        )
        try:
            undir_g = nx_digraph.to_undirected()
            raw_betw = nx.betweenness_centrality(
                undir_g,
                k=k_sample,
                normalized=True,
                weight="weight",
                seed=42,
            )
        except Exception as e:
            logger.warning("Betweenness centrality calculation in BridgeKOLAnalyzer failed: %s", e)
            raw_betw = {node: 0.0 for node in nx_digraph.nodes()}

        max_betw = max(raw_betw.values()) if raw_betw else 0.0

        # 4. Evaluate each candidate node
        candidates: List[BridgeKOLProfile] = []

        for node_id in nx_digraph.nodes():
            s_nid = str(node_id)
            # Undirected neighbors (both followers and followees)
            nbrs = set(nx_digraph.predecessors(node_id)).union(nx_digraph.successors(node_id))
            nbrs.discard(node_id)
            deg = len(nbrs)

            if deg < min_neighbors:
                continue

            ratio, cross_count, foreign_comms = self.compute_cross_community_ratio(
                node_id=s_nid,
                neighbors=nbrs,
                community_partition=partition,
            )

            if not foreign_comms or cross_count == 0 or ratio < min_cross_ratio:
                continue

            # Normalized betweenness: [0.0, 1.0]
            betw_val = raw_betw.get(node_id, 0.0)
            norm_betw = (betw_val / max_betw) if max_betw > 1e-9 else 0.0

            # Normalized foreign communities bridged: [0.0, 1.0]
            if num_communities > 1:
                norm_comm = min(1.0, len(foreign_comms) / max(1, num_communities - 1))
            else:
                norm_comm = min(1.0, len(foreign_comms) / 3.0)

            # Composite Bridge Score
            bridge_score = (
                (self.w_cross * ratio)
                + (self.w_betw * norm_betw)
                + (self.w_comm * norm_comm)
            )
            bridge_score = round(max(0.0, min(1.0, bridge_score)), 4)

            u_meta = meta_map.get(s_nid, {})
            screen_name = u_meta.get("screen_name")
            c_u = partition.get(s_nid)

            candidates.append(
                BridgeKOLProfile(
                    user_id=s_nid,
                    community_id=c_u if (c_u is not None and c_u >= 0) else None,
                    screen_name=screen_name,
                    cross_community_edge_ratio=ratio,
                    betweenness_centrality=round(betw_val, 6),
                    total_neighbors=deg,
                    cross_community_neighbors=cross_count,
                    connected_communities=foreign_comms,
                    bridge_kol_score=bridge_score,
                    rank=1,
                )
            )

        # 5. Sort descending by bridge_kol_score, then cross_community_edge_ratio, then betweenness
        candidates.sort(
            key=lambda p: (p.bridge_kol_score, p.cross_community_edge_ratio, p.betweenness_centrality),
            reverse=True,
        )

        for idx, prof in enumerate(candidates, start=1):
            prof.rank = idx

        top_bridges = candidates[:top_k]

        return BridgeKOLLeaderboard(
            total_nodes_analyzed=total_nodes,
            top_bridges=top_bridges,
        )
