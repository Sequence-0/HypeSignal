"""Community detection and follower clustering engine (Component E).

Segments follower graphs into modular audience clusters using Louvain modularity optimization
and Asynchronous Label Propagation, identifying internal community influencers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import networkx as nx
from networkx.algorithms.community import asyn_lpa_communities, louvain_communities, modularity

from hypesignal.network.graph_store import BaseGraphStore
from hypesignal.network.schemas import CommunityCluster, CommunityDetectionResult

logger = logging.getLogger(__name__)


class CommunityDetector:
    """Detects community structures and audience sub-networks in social graphs."""

    def __init__(
        self,
        default_algorithm: str = "louvain",
        resolution: float = 1.0,
        random_seed: int = 42,
    ) -> None:
        """Initialize CommunityDetector.
        
        Args:
            default_algorithm: 'louvain' or 'label_propagation'.
            resolution: Louvain modularity resolution parameter (higher = smaller clusters).
            random_seed: Random seed for stochastic community partition reproducibility.
        """
        self.default_algorithm = default_algorithm.lower()
        self.resolution = resolution
        self.random_seed = random_seed

    def _to_undirected_weighted(self, directed_g: nx.DiGraph) -> nx.Graph:
        """Convert directed follower graph to undirected weighted graph.
        
        Bidirectional following edges are given weight 2.0; unidirectional edges get 1.0.
        """
        undirected = nx.Graph()
        for u in directed_g.nodes():
            undirected.add_node(u)

        for u, v in directed_g.edges():
            if undirected.has_edge(u, v):
                # Existing reciprocal edge
                undirected[u][v]["weight"] += 1.0
            else:
                undirected.add_edge(u, v, weight=1.0)

        return undirected

    def detect_communities(
        self,
        graph_store: BaseGraphStore,
        algorithm: Optional[str] = None,
        min_community_size: int = 2,
    ) -> CommunityDetectionResult:
        """Detect audience communities and compute modularity.
        
        Args:
            graph_store: BaseGraphStore instance.
            algorithm: 'louvain' or 'label_propagation' (defaults to instance setting).
            min_community_size: Minimum number of members to form an independent cluster.
            
        Returns:
            CommunityDetectionResult containing clusters, modularity, and node partition.
        """
        algo = (algorithm or self.default_algorithm).lower()
        nx_digraph = graph_store.to_networkx()

        if nx_digraph.number_of_nodes() == 0:
            return CommunityDetectionResult(
                algorithm=algo,
                num_communities=0,
                modularity=0.0,
                communities=[],
                partition={},
            )

        undir_g = self._to_undirected_weighted(nx_digraph)

        # Execute community partitioning
        raw_communities: List[Set[str]] = []
        if algo == "label_propagation":
            try:
                raw_communities = [
                    set(c) for c in asyn_lpa_communities(undir_g, weight="weight", seed=self.random_seed)
                ]
            except Exception as e:
                logger.warning("Label propagation community detection failed (%s), falling back to Louvain", e)
                algo = "louvain"

        if algo == "louvain":
            try:
                raw_communities = [
                    set(c)
                    for c in louvain_communities(
                        undir_g,
                        weight="weight",
                        resolution=self.resolution,
                        seed=self.random_seed,
                    )
                ]
            except Exception as e:
                logger.warning("Louvain community detection encountered an issue: %s", e)
                # Fallback to connected components
                raw_communities = [set(c) for c in nx.connected_components(undir_g)]

        # Calculate modularity score Q on the undirected representation
        modularity_score: Optional[float] = None
        if undir_g.number_of_edges() > 0 and len(raw_communities) > 1:
            try:
                modularity_score = round(float(modularity(undir_g, raw_communities, weight="weight")), 4)
            except Exception as e:
                logger.debug("Modularity calculation skipped: %s", e)
                modularity_score = None

        # Build community clusters, filtering by min size
        clusters: List[CommunityCluster] = []
        partition: Dict[str, int] = {}
        cluster_id_counter = 0

        # Sort raw communities descending by size
        sorted_raw = sorted(raw_communities, key=lambda c: len(c), reverse=True)

        for member_set in sorted_raw:
            if len(member_set) < min_community_size:
                # Assign singletons / micro-clusters to miscellaneous bucket -1
                for u in member_set:
                    partition[u] = -1
                continue

            members = list(member_set)
            cid = cluster_id_counter
            cluster_id_counter += 1

            for u in members:
                partition[u] = cid

            # Subgraph edge metrics inside this community in original directed graph
            subgraph = nx_digraph.subgraph(members)
            internal_edges = subgraph.number_of_edges()
            n_members = len(members)

            # Directed density = E / (V * (V - 1))
            if n_members > 1:
                density = round(internal_edges / (n_members * (n_members - 1)), 4)
            else:
                density = 0.0

            # Find top local influencers inside this community by in-degree
            top_influencers = sorted(
                members,
                key=lambda u: nx_digraph.in_degree(u),
                reverse=True,
            )[:5]

            clusters.append(
                CommunityCluster(
                    community_id=cid,
                    member_count=n_members,
                    members=members,
                    top_influencers=top_influencers,
                    internal_edge_count=internal_edges,
                    density=density,
                )
            )

        return CommunityDetectionResult(
            algorithm=algo,
            num_communities=len(clusters),
            modularity=modularity_score,
            communities=clusters,
            partition=partition,
        )
