"""Conversation and comment thread chronology manager (Component A).

Reconstructs hierarchical conversation trees from root posts and replies,
computing tree depth, branching factors, participant diversity, and reply latency.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field

from hypesignal.models.canonical import CanonicalPost, PostMetrics
from hypesignal.models.enums import PlatformType
from hypesignal.storage.duckdb_manager import DuckDBManager

logger = logging.getLogger(__name__)


class ThreadNode(BaseModel):
    """A single node (post or comment) in a conversation thread."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    post_id: str
    author_id: str
    author_screen_name: Optional[str] = None
    text: str
    timestamp: datetime
    parent_id: Optional[str] = None
    depth: int = 0
    children: List[ThreadNode] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class ConversationThread(BaseModel):
    """Complete hierarchical representation and metrics of a social conversation thread."""
    root_post_id: str
    root_node: ThreadNode
    total_posts: int
    total_replies: int
    max_depth: int
    avg_branching_factor: float
    participant_ids: List[str]
    timeline_start: datetime
    timeline_end: datetime
    duration_seconds: float
    mean_reply_latency_seconds: float
    depth_distribution: Dict[int, int] = Field(default_factory=dict)


class ConversationThreadManager:
    """Manages thread extraction, hierarchical tree building, and conversation chronology."""

    def __init__(self, db: DuckDBManager) -> None:
        """Initialize thread manager with DuckDB store."""
        self.db = db

    def reconstruct_thread(self, root_post_id: str) -> Optional[ConversationThread]:
        """Extract and reconstruct the hierarchical comment tree for a root post.
        
        Args:
            root_post_id: Identifier of the initiating post.
            
        Returns:
            ConversationThread object or None if root post not found.
        """
        thread_df = self.db.get_conversation_thread(root_post_id)
        if thread_df.is_empty():
            return None

        records = thread_df.to_dicts()

        # Build map of raw nodes
        nodes_map: Dict[str, ThreadNode] = {}
        parent_map: Dict[str, Optional[str]] = {}

        for r in records:
            p_id = str(r["id"])
            p_metrics = r.get("metrics")
            if isinstance(p_metrics, str):
                try:
                    p_metrics = json.loads(p_metrics)
                except Exception:
                    p_metrics = {}
            elif not isinstance(p_metrics, dict):
                p_metrics = {}

            ts = r["timestamp"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            elif ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            node = ThreadNode(
                post_id=p_id,
                author_id=str(r.get("author_id", "unknown")),
                author_screen_name=r.get("author_screen_name"),
                text=r.get("text", ""),
                timestamp=ts,
                parent_id=str(r["parent_id"]) if r.get("parent_id") else None,
                depth=0,
                children=[],
                metrics=p_metrics,
            )
            nodes_map[p_id] = node
            parent_map[p_id] = node.parent_id

        if root_post_id not in nodes_map:
            return None

        # Build tree connections and calculate depths via BFS
        children_by_parent: Dict[str, List[str]] = defaultdict(list)
        for child_id, p_id in parent_map.items():
            if p_id and p_id in nodes_map and child_id != p_id:
                children_by_parent[p_id].append(child_id)

        # BFS from root
        root_node = nodes_map[root_post_id]
        root_node.depth = 0

        queue: List[str] = [root_post_id]
        max_depth = 0
        depth_counts: Dict[int, int] = defaultdict(int)
        depth_counts[0] = 1

        reply_latencies: List[float] = []

        while queue:
            curr_id = queue.pop(0)
            curr_node = nodes_map[curr_id]
            curr_depth = curr_node.depth
            max_depth = max(max_depth, curr_depth)

            for child_id in children_by_parent.get(curr_id, []):
                child_node = nodes_map[child_id]
                child_node.depth = curr_depth + 1
                curr_node.children.append(child_node)
                depth_counts[child_node.depth] += 1

                # Calculate reply latency
                latency = (child_node.timestamp - curr_node.timestamp).total_seconds()
                reply_latencies.append(max(0.0, latency))

                queue.append(child_id)

        # Sort children of each node chronologically
        def sort_children(node: ThreadNode) -> None:
            node.children.sort(key=lambda c: c.timestamp)
            for c in node.children:
                sort_children(c)

        sort_children(root_node)

        # Thread-wide metrics
        all_nodes = list(nodes_map.values())
        all_timestamps = [n.timestamp for n in all_nodes]
        t_start = min(all_timestamps)
        t_end = max(all_timestamps)
        duration = max(0.0, (t_end - t_start).total_seconds())

        total_posts = len(all_nodes)
        total_replies = total_posts - 1

        # Branching factor (mean number of replies for parents that have at least 1 reply)
        branching_parents = [len(n.children) for n in all_nodes if len(n.children) > 0]
        avg_branching = float(sum(branching_parents) / len(branching_parents)) if branching_parents else 0.0

        participant_ids = sorted(list({n.author_id for n in all_nodes}))
        mean_latency = float(sum(reply_latencies) / len(reply_latencies)) if reply_latencies else 0.0

        return ConversationThread(
            root_post_id=root_post_id,
            root_node=root_node,
            total_posts=total_posts,
            total_replies=total_replies,
            max_depth=max_depth,
            avg_branching_factor=round(avg_branching, 2),
            participant_ids=participant_ids,
            timeline_start=t_start,
            timeline_end=t_end,
            duration_seconds=duration,
            mean_reply_latency_seconds=round(mean_latency, 2),
            depth_distribution=dict(depth_counts),
        )

    def get_chronological_reply_stream(self, root_post_id: str) -> List[CanonicalPost]:
        """Retrieve all reply comments in a conversation thread in chronological order.
        
        Args:
            root_post_id: Root post ID.
            
        Returns:
            List of CanonicalPost instances (excluding root post).
        """
        thread_df = self.db.get_conversation_thread(root_post_id)
        if thread_df.is_empty():
            return []

        replies_df = thread_df.filter(thread_df["id"] != root_post_id)
        if replies_df.is_empty():
            return []

        posts: List[CanonicalPost] = []
        for r in replies_df.iter_rows(named=True):
            p_metrics = r.get("metrics")
            if isinstance(p_metrics, str):
                try:
                    p_metrics = json.loads(p_metrics)
                except Exception:
                    p_metrics = {}
            elif not isinstance(p_metrics, dict):
                p_metrics = {}

            ts = r["timestamp"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            elif ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            post = CanonicalPost(
                id=str(r["id"]),
                platform=PlatformType(r["platform"]),
                author_id=str(r["author_id"]),
                author_screen_name=r.get("author_screen_name"),
                text=r.get("text", ""),
                timestamp=ts,
                parent_id=str(r["parent_id"]) if r.get("parent_id") else None,
                reply_to_user_id=str(r["reply_to_user_id"]) if r.get("reply_to_user_id") else None,
                source_client=r.get("source_client"),
                urls=r.get("urls") or [],
                hashtags=r.get("hashtags") or [],
                mentions=r.get("mentions") or [],
                metrics=PostMetrics(**p_metrics),
                extra_metadata=json.loads(r.get("extra_metadata")) if isinstance(r.get("extra_metadata"), str) else (r.get("extra_metadata") or {}),
            )
            posts.append(post)

        posts.sort(key=lambda p: p.timestamp)
        return posts
