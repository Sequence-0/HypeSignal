"""Link analysis, influencer (KOL) ranking, and diffusion cascade routes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from hypesignal.api.deps import get_db, get_network
from hypesignal.network.network_engine import NetworkEngine
from hypesignal.network.schemas import CascadeTree, KOLProfile, NetworkOverview
from hypesignal.storage.duckdb_manager import DuckDBManager

router = APIRouter(prefix="/network", tags=["Link Analysis & Network Topology"])


@router.get("/overview", response_model=NetworkOverview)
def get_network_overview(
    min_community_size: int = Query(default=2, ge=1, le=50),
    kol_top_k: int = Query(default=20, ge=1, le=200),
    cascade_limit: int = Query(default=5, ge=1, le=50),
    network: NetworkEngine = Depends(get_network),
    db: DuckDBManager = Depends(get_db),
) -> NetworkOverview:
    """Run comprehensive link analysis: density, communities, top KOLs, and cascade trees."""
    return network.analyze_network(
        min_community_size=min_community_size,
        kol_top_k=kol_top_k,
        cascade_limit=cascade_limit,
        db=db,
    )


@router.get("/kols", response_model=List[KOLProfile])
def get_top_kols(
    top_k: int = Query(default=20, ge=1, le=500),
    network: NetworkEngine = Depends(get_network),
    db: DuckDBManager = Depends(get_db),
) -> List[KOLProfile]:
    """Retrieve ranked Key Opinion Leaders (KOLs) using multi-factor centrality."""
    user_meta_map: Dict[str, Dict[str, Any]] = {}
    try:
        user_rows = db.con.execute("SELECT id, screen_name FROM users;").fetchall()
        for uid, sname in user_rows:
            user_meta_map[str(uid)] = {"screen_name": sname}
    except Exception:
        pass

    result = network.kol_analyzer.rank_kols(
        graph_store=network.graph_store,
        top_k=top_k,
        user_metadata=user_meta_map,
    )
    return result.top_kols


@router.get("/user/{user_id}", response_model=KOLProfile)
def get_user_network_profile(
    user_id: str,
    network: NetworkEngine = Depends(get_network),
    db: DuckDBManager = Depends(get_db),
) -> KOLProfile:
    """Retrieve network influence profile and centrality scores for an individual user."""
    profile = network.get_user_network_profile(user_id=user_id, db=db)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' not found in network topology store.",
        )
    return profile


@router.get("/subgraph/{user_id}")
def get_user_subgraph(
    user_id: str,
    radius: int = Query(default=1, ge=1, le=3, description="Neighborhood hop distance"),
    max_nodes: int = Query(default=100, ge=5, le=500, description="Max nodes to include in export"),
    network: NetworkEngine = Depends(get_network),
) -> Dict[str, Any]:
    """Export egocentric neighborhood subgraph around a center node for graph visualizers."""
    return network.export_subgraph(
        center_user_id=user_id,
        radius=radius,
        max_nodes=max_nodes,
    )


@router.get("/cascade/{cascade_id:path}", response_model=CascadeTree)
def get_cascade_analysis(
    cascade_id: str,
    network: NetworkEngine = Depends(get_network),
    db: DuckDBManager = Depends(get_db),
) -> CascadeTree:
    """Reconstruct diffusion cascade tree, parent-child transmissions, and structural virality."""
    cascade_tree = network.get_cascade_analysis(cascade_id=cascade_id, db=db)
    if cascade_tree is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cascade '{cascade_id}' not found or has no recorded events.",
        )
    return cascade_tree
