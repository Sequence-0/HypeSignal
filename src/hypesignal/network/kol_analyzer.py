"""Key Opinion Leader (KOL) and Influencer Ranking Engine (Component E).

Identifies authoritative network nodes combining PageRank, in-degree centrality (followers),
and betweenness centrality into calibrated multi-factor influence rankings.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import networkx as nx

from hypesignal.network.graph_store import BaseGraphStore
from hypesignal.network.schemas import KOLProfile, KOLRankingResult

logger = logging.getLogger(__name__)


class KOLAnalyzer:
    """Computes node centralities and ranks Key Opinion Leaders in social graphs."""

    def __init__(
        self,
        weight_pagerank: float = 0.50,
        weight_indegree: float = 0.35,
        weight_betweenness: float = 0.15,
        pagerank_alpha: float = 0.85,
        pagerank_max_iter: int = 100,
    ) -> None:
        """Initialize KOLAnalyzer with feature weights.
        
        Args:
            weight_pagerank: Relative weight for PageRank authority.
            weight_indegree: Relative weight for follower count (in-degree).
            weight_betweenness: Relative weight for network bridge centrality.
            pagerank_alpha: PageRank damping factor (standard 0.85).
            pagerank_max_iter: Maximum power iterations for PageRank convergence.
        """
        tot = weight_pagerank + weight_indegree + weight_betweenness
        if tot <= 0:
            tot = 1.0
        self.w_pr = weight_pagerank / tot
        self.w_indeg = weight_indegree / tot
        self.w_betw = weight_betweenness / tot
        self.pr_alpha = pagerank_alpha
        self.pr_max_iter = pagerank_max_iter

    def compute_pagerank(self, graph_store: BaseGraphStore) -> Dict[str, float]:
        """Compute PageRank scores across all graph nodes."""
        nx_graph = graph_store.to_networkx()
        if nx_graph.number_of_nodes() == 0:
            return {}
        try:
            return nx.pagerank(
                nx_graph,
                alpha=self.pr_alpha,
                max_iter=self.pr_max_iter,
                weight="weight",
            )
        except Exception as e:
            logger.warning("PageRank calculation fallback due to convergence: %s", e)
            n = nx_graph.number_of_nodes()
            return {node: 1.0 / n for node in nx_graph.nodes()}

    def compute_indegree_centrality(self, graph_store: BaseGraphStore) -> Dict[str, float]:
        """Compute normalized in-degree centrality."""
        nx_graph = graph_store.to_networkx()
        if nx_graph.number_of_nodes() <= 1:
            return {n: 0.0 for n in nx_graph.nodes()}
        return nx.in_degree_centrality(nx_graph)

    def compute_betweenness_centrality(
        self,
        graph_store: BaseGraphStore,
        max_samples: int = 500,
    ) -> Dict[str, float]:
        """Compute betweenness centrality with node sampling for large graphs."""
        nx_graph = graph_store.to_networkx()
        n = nx_graph.number_of_nodes()
        if n <= 2:
            return {node: 0.0 for node in nx_graph.nodes()}

        # Sample k pivot nodes if graph is large to avoid O(V^3) computational blowup
        k_sample = min(max_samples, n) if n > max_samples else None
        try:
            return nx.betweenness_centrality(
                nx_graph,
                k=k_sample,
                normalized=True,
                weight="weight",
                seed=42,
            )
        except Exception as e:
            logger.warning("Betweenness centrality calculation failed: %s", e)
            return {node: 0.0 for node in nx_graph.nodes()}

    def _normalize_dict(self, values: Dict[str, float]) -> Dict[str, float]:
        """Logarithmically compress and normalize scores to [0.0, 1.0]."""
        if not values:
            return {}
        # Apply log1p for heavy-tailed social metrics
        log_vals = {k: math.log1p(max(0.0, v)) for k, v in values.items()}
        max_val = max(log_vals.values()) if log_vals else 0.0
        min_val = min(log_vals.values()) if log_vals else 0.0
        rng = max_val - min_val
        if rng <= 1e-9:
            return {k: 1.0 / len(values) for k in values}
        return {k: (v - min_val) / rng for k, v in log_vals.items()}

    def rank_kols(
        self,
        graph_store: BaseGraphStore,
        top_k: int = 20,
        min_indegree: int = 0,
        user_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        community_partition: Optional[Dict[str, int]] = None,
    ) -> KOLRankingResult:
        """Calculate composite influence scores and rank top Key Opinion Leaders.
        
        Args:
            graph_store: BaseGraphStore instance.
            top_k: Number of top KOLs to return.
            min_indegree: Minimum follower count required to qualify as KOL candidate.
            user_metadata: Optional user_id -> metadata dict (e.g. screen_name).
            community_partition: Optional user_id -> community_id mapping.
            
        Returns:
            KOLRankingResult with ranked KOLProfile list and graph stats.
        """
        nx_graph = graph_store.to_networkx()
        total_nodes = nx_graph.number_of_nodes()
        total_edges = nx_graph.number_of_edges()

        if total_nodes == 0:
            return KOLRankingResult(
                top_kols=[],
                total_nodes=0,
                total_edges=0,
                algorithm_weights={
                    "pagerank": self.w_pr,
                    "indegree": self.w_indeg,
                    "betweenness": self.w_betw,
                },
            )

        user_metadata = user_metadata or {}
        community_partition = community_partition or {}

        # 1. Compute individual centrality metrics
        raw_pr = self.compute_pagerank(graph_store)
        raw_indeg_centrality = self.compute_indegree_centrality(graph_store)
        raw_betweenness = self.compute_betweenness_centrality(graph_store)

        # 2. Normalize components
        norm_pr = self._normalize_dict(raw_pr)
        norm_indeg = self._normalize_dict(raw_indeg_centrality)
        norm_betw = self._normalize_dict(raw_betweenness)

        # 3. Compute composite influence score per node
        profiles: List[KOLProfile] = []
        for node_id in nx_graph.nodes():
            in_deg = int(nx_graph.in_degree(node_id))
            if in_deg < min_indegree:
                continue

            out_deg = int(nx_graph.out_degree(node_id))
            pr_val = raw_pr.get(node_id, 0.0)
            betw_val = raw_betweenness.get(node_id, 0.0)

            score = (
                (self.w_pr * norm_pr.get(node_id, 0.0))
                + (self.w_indeg * norm_indeg.get(node_id, 0.0))
                + (self.w_betw * norm_betw.get(node_id, 0.0))
            )
            score = max(0.0, min(1.0, score))

            u_meta = user_metadata.get(node_id, {})
            screen_name = u_meta.get("screen_name")
            comm_id = community_partition.get(node_id)

            profiles.append(
                KOLProfile(
                    user_id=node_id,
                    screen_name=screen_name,
                    pagerank=round(pr_val, 6),
                    in_degree=in_deg,
                    outdegree=out_deg,
                    betweenness=round(betw_val, 6),
                    influence_score=round(score, 4),
                    rank=1,  # Placeholder, assigned after sorting
                    community_id=comm_id,
                    is_kol=True,
                    extra_metadata=u_meta,
                )
            )

        # 4. Sort descending by composite influence score
        profiles.sort(key=lambda p: (p.influence_score, p.in_degree, p.pagerank), reverse=True)

        for rank_idx, prof in enumerate(profiles, start=1):
            prof.rank = rank_idx

        top_kols = profiles[:top_k]

        return KOLRankingResult(
            top_kols=top_kols,
            total_nodes=total_nodes,
            total_edges=total_edges,
            algorithm_weights={
                "pagerank": round(self.w_pr, 4),
                "indegree": round(self.w_indeg, 4),
                "betweenness": round(self.w_betw, 4),
            },
        )

    def compute_spearman_correlation(
        self,
        predicted_ranks: Dict[str, float],
        ground_truth_values: Dict[str, float],
    ) -> float:
        """Compute Spearman rank-order correlation coefficient between predicted and ground truth.
        
        Args:
            predicted_ranks: node_id -> predicted score/rank
            ground_truth_values: node_id -> ground truth metric (e.g. Lerman indegree)
            
        Returns:
            Spearman rho in [-1.0, 1.0].
        """
        common_keys = [k for k in predicted_ranks if k in ground_truth_values]
        n = len(common_keys)
        if n < 2:
            return 1.0

        # Sort and assign ranks
        pred_sorted = sorted(common_keys, key=lambda k: predicted_ranks[k], reverse=True)
        pred_rank_map = {k: r for r, k in enumerate(pred_sorted, start=1)}

        gt_sorted = sorted(common_keys, key=lambda k: ground_truth_values[k], reverse=True)
        gt_rank_map = {k: r for r, k in enumerate(gt_sorted, start=1)}

        # d_i^2
        d_sq = sum((pred_rank_map[k] - gt_rank_map[k]) ** 2 for k in common_keys)
        rho = 1.0 - (6.0 * d_sq) / (n * (n**2 - 1))
        return round(float(rho), 4)
