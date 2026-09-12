"""Information diffusion cascade tracer and structural virality engine (Component E).

Reconstructs multi-generational propagation trees from timestamped adoption sequences and
underlying follower topology, computing structural virality (Goel et al. 2015), branching factors,
transmission lag, and diffusion velocities.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from hypesignal.models.canonical import CanonicalCascadeEvent
from hypesignal.network.graph_store import BaseGraphStore
from hypesignal.network.schemas import CascadeNode, CascadeTree, DiffusionMetrics
from hypesignal.storage.duckdb_manager import DuckDBManager

logger = logging.getLogger(__name__)


class CascadeTracer:
    """Reconstructs and analyzes information diffusion cascades across social graphs."""

    def reconstruct_cascade_tree(
        self,
        events: List[CanonicalCascadeEvent],
        graph_store: Optional[BaseGraphStore] = None,
    ) -> Optional[CascadeTree]:
        """Reconstruct a hierarchical diffusion tree from chronological adoption events.
        
        Args:
            events: List of CanonicalCascadeEvent objects.
            graph_store: Optional follower graph to identify parent transmission edges.
            
        Returns:
            CascadeTree object containing nodes, tree depth, and virality metrics.
        """
        if not events:
            return None

        # 1. Deduplicate by user_id, retaining earliest adoption
        sorted_events = sorted(events, key=lambda e: (e.adoption_order, e.timestamp_ms or 0))
        unique_events: List[CanonicalCascadeEvent] = []
        seen_users: Set[str] = set()

        for ev in sorted_events:
            if ev.user_id not in seen_users:
                seen_users.add(ev.user_id)
                unique_events.append(ev)

        if not unique_events:
            return None

        cascade_id = unique_events[0].cascade_id
        root_event = unique_events[0]
        root_user_id = root_event.user_id

        # 2. Build nodes dictionary
        nodes: Dict[str, CascadeNode] = {}
        # Root node
        nodes[root_user_id] = CascadeNode(
            user_id=root_user_id,
            user_screen_name=root_event.user_screen_name,
            post_id=root_event.post_id,
            timestamp=root_event.timestamp,
            adoption_order=0,
            depth=0,
            parent_user_id=None,
            children_user_ids=[],
            time_delay_seconds=0.0,
        )

        prior_adopters: List[str] = [root_user_id]

        # 3. Process subsequent adopters
        for order_idx, ev in enumerate(unique_events[1:], start=1):
            curr_user = ev.user_id
            curr_time = ev.timestamp

            inferred_parent: Optional[str] = None

            # If explicit parent event recorded
            if ev.parent_event_id:
                for prior in prior_adopters:
                    if nodes[prior].post_id == ev.parent_event_id:
                        inferred_parent = prior
                        break

            # If social graph available, find prior adopters that curr_user follows
            if inferred_parent is None and graph_store is not None:
                # In Twitter semantics: curr_user follows candidate => curr_user -> candidate edge exists
                followed_by_curr = graph_store.get_following(curr_user)
                followed_prior = [p for p in prior_adopters if p in followed_by_curr]

                if followed_prior:
                    # Choose the most recent earlier adopter followed by curr_user
                    inferred_parent = max(followed_prior, key=lambda p: nodes[p].timestamp)

            # Baseline fallback: if not connected to prior adopters, attach to root (broadcast exposure)
            if inferred_parent is None or inferred_parent not in nodes:
                inferred_parent = root_user_id

            parent_node = nodes[inferred_parent]
            depth = parent_node.depth + 1
            delay_sec = max(0.0, (curr_time - parent_node.timestamp).total_seconds())

            node = CascadeNode(
                user_id=curr_user,
                user_screen_name=ev.user_screen_name,
                post_id=ev.post_id,
                timestamp=curr_time,
                adoption_order=order_idx,
                depth=depth,
                parent_user_id=inferred_parent,
                children_user_ids=[],
                time_delay_seconds=round(delay_sec, 2),
            )
            nodes[curr_user] = node
            parent_node.children_user_ids.append(curr_user)
            prior_adopters.append(curr_user)

        # 4. Compute metrics and structural virality
        metrics = self.compute_diffusion_metrics(cascade_id=cascade_id, nodes=nodes)

        return CascadeTree(
            cascade_id=cascade_id,
            root_user_id=root_user_id,
            metrics=metrics,
            nodes=nodes,
        )

    def compute_structural_virality(self, nodes: Dict[str, CascadeNode]) -> float:
        """Compute the structural virality (Wiener index) of the diffusion tree.
        
        Defined as the average shortest path distance between all pairs of adopters:
            v(T) = 1 / (n * (n - 1)) * sum_{i != j} dist(i, j)
            
        - Pure star / broadcast cascade: v ~ 2.0
        - Pure multi-generational chain / viral cascade: v = (n + 1) / 3 > 3.0
        """
        n = len(nodes)
        if n <= 1:
            return 0.0

        # Construct undirected tree representation
        tree_graph = nx.Graph()
        for uid, node in nodes.items():
            tree_graph.add_node(uid)
            if node.parent_user_id and node.parent_user_id in nodes:
                tree_graph.add_edge(node.parent_user_id, uid)

        # Compute all-pairs shortest paths
        total_dist = 0
        pair_count = 0

        # Run BFS from each node
        for source_node in tree_graph.nodes():
            lengths = nx.single_source_shortest_path_length(tree_graph, source_node)
            for target_node, dist in lengths.items():
                if source_node != target_node:
                    total_dist += dist
                    pair_count += 1

        if pair_count == 0:
            return 0.0

        virality = total_dist / pair_count
        return round(float(virality), 4)

    def compute_diffusion_metrics(
        self,
        cascade_id: str,
        nodes: Dict[str, CascadeNode],
    ) -> DiffusionMetrics:
        """Calculate diffusion speed, half-life, depth-to-breadth, and virality."""
        n = len(nodes)
        if n == 0:
            return DiffusionMetrics(
                cascade_id=cascade_id,
                total_adoptions=0,
                max_depth=0,
                max_breadth=0,
                structural_virality=0.0,
            )

        max_depth = max(node.depth for node in nodes.values())

        # Count nodes at each depth level to find max breadth
        depth_counts: Dict[int, int] = {}
        for node in nodes.values():
            depth_counts[node.depth] = depth_counts.get(node.depth, 0) + 1
        max_breadth = max(depth_counts.values()) if depth_counts else 1

        # Branching factor (average children for non-leaf adopters)
        internal_nodes = [node for node in nodes.values() if len(node.children_user_ids) > 0]
        if internal_nodes:
            branching_factor = round(sum(len(n.children_user_ids) for n in internal_nodes) / len(internal_nodes), 3)
        else:
            branching_factor = 0.0

        # Structural virality
        virality = self.compute_structural_virality(nodes)

        # Chronological timestamps
        sorted_times = sorted(node.timestamp for node in nodes.values())
        t_start = sorted_times[0]
        t_end = sorted_times[-1]
        total_duration_sec = max(0.0, (t_end - t_start).total_seconds())

        # Half-life: time to reach 50% adoptions
        half_idx = max(0, (n // 2) - 1)
        half_time = sorted_times[half_idx]
        half_life_sec = max(0.0, (half_time - t_start).total_seconds())

        # Adoptions velocity per hour
        duration_hours = max(total_duration_sec / 3600.0, 1.0 / 60.0)  # at least 1 min
        velocity = round(n / duration_hours, 2)

        depth_to_breadth = round(max_depth / max(1, max_breadth), 4)

        return DiffusionMetrics(
            cascade_id=cascade_id,
            total_adoptions=n,
            max_depth=max_depth,
            max_breadth=max_breadth,
            structural_virality=virality,
            branching_factor=branching_factor,
            half_life_seconds=round(half_life_sec, 2),
            total_duration_seconds=round(total_duration_sec, 2),
            velocity_adoptions_per_hour=velocity,
            depth_to_breadth_ratio=depth_to_breadth,
        )

    def trace_from_duckdb(
        self,
        db: DuckDBManager,
        cascade_id: str,
        graph_store: Optional[BaseGraphStore] = None,
    ) -> Optional[CascadeTree]:
        """Retrieve cascade events for a given cascade_id from DuckDB and reconstruct its tree."""
        series_df = db.get_cascade_series(cascade_id)
        if series_df.is_empty():
            return None

        events: List[CanonicalCascadeEvent] = []
        for row in series_df.iter_rows(named=True):
            events.append(
                CanonicalCascadeEvent(
                    cascade_id=str(row["cascade_id"]),
                    post_id=str(row["post_id"]),
                    user_id=str(row["user_id"]),
                    user_screen_name=row.get("user_screen_name"),
                    timestamp=row["timestamp"],
                    timestamp_ms=row.get("timestamp_ms"),
                    adoption_order=int(row.get("adoption_order", 0)),
                    parent_event_id=row.get("parent_event_id"),
                )
            )

        return self.reconstruct_cascade_tree(events=events, graph_store=graph_store)

    def trace_top_cascades_from_duckdb(
        self,
        db: DuckDBManager,
        min_size: int = 5,
        limit: int = 10,
        graph_store: Optional[BaseGraphStore] = None,
    ) -> List[CascadeTree]:
        """Find largest cascades in DuckDB and reconstruct their diffusion trees."""
        query = """
            SELECT cascade_id, COUNT(*) as cnt
            FROM cascade_events
            GROUP BY cascade_id
            HAVING COUNT(*) >= ?
            ORDER BY cnt DESC
            LIMIT ?;
        """
        rows = db.con.execute(query, [int(min_size), int(limit)]).fetchall()
        trees: List[CascadeTree] = []

        for cascade_id, _ in rows:
            tree = self.trace_from_duckdb(db, cascade_id=cascade_id, graph_store=graph_store)
            if tree is not None:
                trees.append(tree)

        return trees
