"""Pydantic data schemas for link analysis and network topology (Component E).

Defines structured models for Key Opinion Leader (KOL) ranking, community clusters,
cascade trees, and network topology overviews.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class KOLProfile(BaseModel):
    """Profile of an authoritative Key Opinion Leader (influencer) in the graph."""
    user_id: str = Field(..., description="Unique user identifier")
    screen_name: Optional[str] = Field(default=None, description="User screen handle if known")
    pagerank: float = Field(..., ge=0.0, description="PageRank centrality score")
    in_degree: int = Field(..., ge=0, description="Follower count (incoming edges)")
    outdegree: int = Field(default=0, ge=0, description="Following count (outgoing edges)")
    betweenness: float = Field(default=0.0, ge=0.0, description="Betweenness centrality score")
    influence_score: float = Field(..., ge=0.0, le=1.0, description="Normalized composite influence score")
    rank: int = Field(..., ge=1, description="Rank position (1 = top influencer)")
    community_id: Optional[int] = Field(default=None, description="Community cluster ID user belongs to")
    is_kol: bool = Field(default=True, description="Whether node passes KOL threshold")
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)


class KOLRankingResult(BaseModel):
    """Result of KOL identification across the network."""
    top_kols: List[KOLProfile] = Field(default_factory=list)
    total_nodes: int = Field(..., ge=0)
    total_edges: int = Field(..., ge=0)
    algorithm_weights: Dict[str, float] = Field(default_factory=dict)
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CommunityCluster(BaseModel):
    """A cohesive community partition in the network."""
    community_id: int = Field(..., description="Cluster identifier")
    member_count: int = Field(..., ge=0, description="Number of users in community")
    members: List[str] = Field(default_factory=list, description="User IDs of members")
    top_influencers: List[str] = Field(default_factory=list, description="Top central nodes in cluster")
    internal_edge_count: int = Field(default=0, ge=0)
    density: float = Field(default=0.0, ge=0.0, le=1.0, description="Internal subgraph edge density")


class CommunityDetectionResult(BaseModel):
    """Result of network community detection."""
    algorithm: str = Field(..., description="Detection algorithm used (e.g. 'louvain', 'label_propagation')")
    num_communities: int = Field(..., ge=0)
    modularity: Optional[float] = Field(default=None, description="Network modularity score Q")
    communities: List[CommunityCluster] = Field(default_factory=list)
    partition: Dict[str, int] = Field(default_factory=dict, description="user_id -> community_id mapping")


class CascadeNode(BaseModel):
    """Node representing a user adoption in an information diffusion cascade."""
    user_id: str = Field(..., description="User ID who adopted/shared the information")
    user_screen_name: Optional[str] = Field(default=None)
    post_id: str = Field(..., description="Post ID carrying the cascade token")
    timestamp: datetime = Field(..., description="Adoption timestamp")
    adoption_order: int = Field(..., ge=0, description="Chronological adoption index")
    depth: int = Field(default=0, ge=0, description="Hops from cascade initiator (root depth = 0)")
    parent_user_id: Optional[str] = Field(default=None, description="Inferred or explicit parent adopter")
    children_user_ids: List[str] = Field(default_factory=list, description="Direct child adopters infected")
    time_delay_seconds: float = Field(default=0.0, ge=0.0, description="Delay in seconds since parent adoption")


class DiffusionMetrics(BaseModel):
    """Diffusion dynamics and structural virality metrics for an information cascade."""
    cascade_id: str = Field(..., description="Identifier of the cascade (e.g. URL or topic)")
    total_adoptions: int = Field(..., ge=1)
    max_depth: int = Field(..., ge=0, description="Maximum propagation depth (tree height)")
    max_breadth: int = Field(..., ge=1, description="Maximum adoptions at any single depth level")
    structural_virality: float = Field(
        ...,
        ge=0.0,
        description="Wiener index: average pairwise shortest path distance between all adopters",
    )
    branching_factor: float = Field(
        default=0.0,
        ge=0.0,
        description="Average number of children for non-leaf adopters",
    )
    half_life_seconds: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Time in seconds taken to achieve 50% of total cascade adoptions",
    )
    total_duration_seconds: float = Field(default=0.0, ge=0.0)
    velocity_adoptions_per_hour: float = Field(default=0.0, ge=0.0)
    depth_to_breadth_ratio: float = Field(default=0.0, ge=0.0)


class CascadeTree(BaseModel):
    """Reconstructed diffusion tree for an information cascade."""
    cascade_id: str = Field(..., description="Cascade identifier (e.g., URL or topic)")
    root_user_id: str = Field(..., description="Initiator user ID who started the cascade")
    metrics: DiffusionMetrics
    nodes: Dict[str, CascadeNode] = Field(default_factory=dict, description="user_id -> CascadeNode")


class NetworkOverview(BaseModel):
    """Comprehensive network topology and link analysis overview."""
    total_nodes: int = Field(..., ge=0)
    total_edges: int = Field(..., ge=0)
    density: float = Field(..., ge=0.0, le=1.0)
    is_connected: bool = Field(default=False)
    num_connected_components: int = Field(default=1, ge=0)
    top_kols: List[KOLProfile] = Field(default_factory=list)
    communities: List[CommunityCluster] = Field(default_factory=list)
    recent_cascades: List[CascadeTree] = Field(default_factory=list)
