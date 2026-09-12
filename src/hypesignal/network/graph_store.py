"""Pluggable graph storage layer for social network topologies (Component E).

Provides an abstract BaseGraphStore interface with:
1. NetworkXGraphStore: High-speed in-memory directed graph for local analytics and tests.
2. MemgraphStore: Bolt-protocol driver for high-scale Memgraph/Neo4j graph database deployment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import re
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import networkx as nx

from hypesignal.models.canonical import CanonicalGraphEdge
from hypesignal.models.enums import RelationType
from hypesignal.storage.duckdb_manager import DuckDBManager

logger = logging.getLogger(__name__)

ALLOWED_RELATION_TYPES: Set[str] = {
    e.value.upper() for e in RelationType
} | {
    "FOLLOWS",
    "RETWEETS",
    "REPLIES",
    "MENTIONS",
    "QUOTES",
    "FRIENDS",
    "SUBSCRIBES",
    "INTERACTS",
}
RE_RELATION_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


def validate_relation_type(relation_type: Any) -> str:
    """Validate and sanitize graph relationship type to prevent Cypher injection.
    
    Args:
        relation_type: String or RelationType enum value.
        
    Returns:
        Sanitized uppercase relationship identifier string.
        
    Raises:
        ValueError: If relation_type is empty, non-string, or contains disallowed syntax tokens.
    """
    if hasattr(relation_type, "value"):
        relation_type = relation_type.value

    if not isinstance(relation_type, str):
        raise ValueError(
            f"Relationship type must be a string or RelationType, got {type(relation_type).__name__}"
        )

    clean = relation_type.strip().upper()
    if not clean:
        raise ValueError("Relationship type cannot be empty")

    if clean in ALLOWED_RELATION_TYPES:
        return clean

    # Strict identifier check: alphanumeric letters/digits/underscores only
    if not RE_RELATION_IDENTIFIER.match(clean):
        raise ValueError(
            f"Invalid relation_type '{relation_type}'. Relationship types must be alphanumeric identifiers "
            f"matching regex '^[A-Za-z][A-Za-z0-9_]{{0,63}}$' to prevent Cypher injection."
        )

    return clean


class BaseGraphStore(ABC):
    """Abstract interface for social graph storage engines."""
    revision: int = 0

    @abstractmethod
    def add_node(self, node_id: str, **attrs: Any) -> None:
        """Add a single node to the graph."""
        pass

    @abstractmethod
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str = "FOLLOWS",
        weight: float = 1.0,
        **attrs: Any,
    ) -> None:
        """Add a directed edge (source -> target) to the graph."""
        pass

    @abstractmethod
    def add_edges_from(self, edges: List[CanonicalGraphEdge]) -> None:
        """Add multiple CanonicalGraphEdge instances in bulk."""
        pass

    @abstractmethod
    def has_node(self, node_id: str) -> bool:
        """Check if node exists in graph."""
        pass

    @abstractmethod
    def has_edge(self, source_id: str, target_id: str) -> bool:
        """Check if directed edge exists from source to target."""
        pass

    @abstractmethod
    def get_nodes(self) -> List[str]:
        """Return all node IDs in the graph."""
        pass

    @abstractmethod
    def get_node_count(self) -> int:
        """Get total number of unique nodes."""
        pass

    @abstractmethod
    def get_edge_count(self) -> int:
        """Get total number of directed edges."""
        pass

    @abstractmethod
    def get_followers(self, user_id: str) -> List[str]:
        """Return nodes that follow user_id (incoming edges: s -> user_id)."""
        pass

    @abstractmethod
    def get_following(self, user_id: str) -> List[str]:
        """Return nodes that user_id follows (outgoing edges: user_id -> t)."""
        pass

    @abstractmethod
    def get_in_degree(self, user_id: str) -> int:
        """Return number of incoming edges (followers)."""
        pass

    @abstractmethod
    def get_out_degree(self, user_id: str) -> int:
        """Return number of outgoing edges (following)."""
        pass

    @abstractmethod
    def to_networkx(self) -> nx.DiGraph:
        """Export as NetworkX DiGraph representation."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all nodes and edges."""
        pass


class NetworkXGraphStore(BaseGraphStore):
    """In-memory graph store backed by NetworkX DiGraph."""

    def __init__(self, graph: Optional[nx.DiGraph] = None) -> None:
        """Initialize NetworkXGraphStore."""
        self.graph: nx.DiGraph = graph if graph is not None else nx.DiGraph()
        self.revision: int = 0

    def add_node(self, node_id: str, **attrs: Any) -> None:
        """Add node with attributes."""
        self.graph.add_node(str(node_id), **attrs)
        self.revision += 1

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str = "FOLLOWS",
        weight: float = 1.0,
        **attrs: Any,
    ) -> None:
        """Add directed edge: follower (source) -> followee (target)."""
        rel = validate_relation_type(relation_type)
        self.graph.add_edge(
            str(source_id),
            str(target_id),
            relation_type=rel,
            weight=float(weight),
            **attrs,
        )
        self.revision += 1

    def add_edges_from(self, edges: List[CanonicalGraphEdge]) -> None:
        """Bulk add CanonicalGraphEdge instances."""
        for e in edges:
            rel = validate_relation_type(e.relation_type)
            self.graph.add_edge(
                str(e.source_id),
                str(e.target_id),
                relation_type=rel,
                weight=float(e.weight),
                **e.extra_metadata,
            )
        self.revision += 1

    def has_node(self, node_id: str) -> bool:
        """Check if node exists."""
        return self.graph.has_node(str(node_id))

    def has_edge(self, source_id: str, target_id: str) -> bool:
        """Check if edge exists."""
        return self.graph.has_edge(str(source_id), str(target_id))

    def get_nodes(self) -> List[str]:
        """Return list of node IDs."""
        return list(self.graph.nodes())

    def get_node_count(self) -> int:
        """Total node count."""
        return self.graph.number_of_nodes()

    def get_edge_count(self) -> int:
        """Total edge count."""
        return self.graph.number_of_edges()

    def get_followers(self, user_id: str) -> List[str]:
        """Incoming edges: users who follow user_id."""
        s_id = str(user_id)
        if not self.graph.has_node(s_id):
            return []
        return list(self.graph.predecessors(s_id))

    def get_following(self, user_id: str) -> List[str]:
        """Outgoing edges: users followed by user_id."""
        s_id = str(user_id)
        if not self.graph.has_node(s_id):
            return []
        return list(self.graph.successors(s_id))

    def get_in_degree(self, user_id: str) -> int:
        """Number of followers (in-degree)."""
        s_id = str(user_id)
        if not self.graph.has_node(s_id):
            return 0
        return int(self.graph.in_degree(s_id))

    def get_out_degree(self, user_id: str) -> int:
        """Number of users followed (out-degree)."""
        s_id = str(user_id)
        if not self.graph.has_node(s_id):
            return 0
        return int(self.graph.out_degree(s_id))

    def get_subgraph(self, node_ids: List[str]) -> NetworkXGraphStore:
        """Extract induced subgraph over specified node IDs."""
        sub = self.graph.subgraph([str(n) for n in node_ids]).copy()
        return NetworkXGraphStore(graph=sub)

    def to_networkx(self) -> nx.DiGraph:
        """Return the underlying NetworkX graph."""
        return self.graph

    def clear(self) -> None:
        """Clear the graph."""
        self.graph.clear()
        self.revision += 1

    def load_from_duckdb(self, db: DuckDBManager, limit: Optional[int] = None) -> int:
        """Load edges from DuckDB graph_edges table into graph.
        
        Args:
            db: DuckDBManager instance.
            limit: Maximum edges to load.
            
        Returns:
            Number of edges loaded.
        """
        query = "SELECT source_id, target_id, relation_type, weight FROM graph_edges"
        params: List[Any] = []
        if limit:
            query += " LIMIT ?"
            params.append(int(limit))
        rows = db.con.execute(query, params).fetchall()
        count = 0
        for s, t, r, w in rows:
            self.add_edge(source_id=s, target_id=t, relation_type=r, weight=w or 1.0)
            count += 1
        return count

    def load_from_edge_stream(
        self,
        edge_stream: Iterator[CanonicalGraphEdge],
        limit: Optional[int] = None,
    ) -> int:
        """Consume edge stream (e.g. from LermanDatasetAdapter) into in-memory graph."""
        count = 0
        for edge in edge_stream:
            self.add_edge(
                source_id=edge.source_id,
                target_id=edge.target_id,
                relation_type=edge.relation_type.value if hasattr(edge.relation_type, "value") else str(edge.relation_type),
                weight=edge.weight,
                **edge.extra_metadata,
            )
            count += 1
            if limit and count >= limit:
                break
        return count


class MemgraphStore(BaseGraphStore):
    """Bolt protocol graph store driver for Memgraph / Neo4j deployment."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "",
        password: str = "",
    ) -> None:
        """Initialize Memgraph Bolt connection.
        
        Falls back gracefully if the optional 'neo4j' driver package is not installed.
        """
        self.uri = uri
        self.username = username
        self.password = password
        self._driver = None
        self._fallback_store = NetworkXGraphStore()
        self.revision: int = 0

        try:
            from neo4j import GraphDatabase
            auth = (username, password) if username or password else None
            self._driver = GraphDatabase.driver(uri, auth=auth)
            # Verify connectivity
            with self._driver.session() as session:
                session.run("RETURN 1;")
            logger.info("Connected successfully to Memgraph at %s", uri)
        except Exception as e:
            logger.info(
                "Memgraph at %s unavailable (%s); using in-memory NetworkX store as fallback.",
                uri,
                e,
            )
            self._driver = None

    @property
    def is_connected(self) -> bool:
        """Check if live Bolt connection is active."""
        return self._driver is not None

    def add_node(self, node_id: str, **attrs: Any) -> None:
        """Add node to Memgraph or fallback."""
        if self._driver:
            with self._driver.session() as session:
                session.run("MERGE (u:User {id: $id}) SET u += $attrs", id=str(node_id), attrs=attrs)
        else:
            self._fallback_store.add_node(node_id, **attrs)
        self.revision += 1

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str = "FOLLOWS",
        weight: float = 1.0,
        **attrs: Any,
    ) -> None:
        """Add edge to Memgraph or fallback with Cypher injection prevention."""
        rel = validate_relation_type(relation_type)
        if self._driver:
            query = f"""
                MERGE (s:User {{id: $source_id}})
                MERGE (t:User {{id: $target_id}})
                MERGE (s)-[r:{rel}]->(t)
                SET r.weight = $weight, r += $attrs
            """
            with self._driver.session() as session:
                session.run(query, source_id=str(source_id), target_id=str(target_id), weight=float(weight), attrs=attrs)
        else:
            self._fallback_store.add_edge(source_id, target_id, rel, weight, **attrs)
        self.revision += 1

    def add_edges_from(self, edges: List[CanonicalGraphEdge]) -> None:
        """Add bulk edges to Memgraph or fallback."""
        for e in edges:
            rel = validate_relation_type(e.relation_type)
            self.add_edge(e.source_id, e.target_id, relation_type=rel, weight=e.weight, **e.extra_metadata)
        self.revision += 1

    def has_node(self, node_id: str) -> bool:
        if self._driver:
            with self._driver.session() as session:
                res = session.run("MATCH (u:User {id: $id}) RETURN count(u) > 0 AS exists", id=str(node_id)).single()
                return bool(res["exists"]) if res else False
        return self._fallback_store.has_node(node_id)

    def has_edge(self, source_id: str, target_id: str) -> bool:
        if self._driver:
            with self._driver.session() as session:
                res = session.run(
                    "MATCH (s:User {id: $s})-[r]->(t:User {id: $t}) RETURN count(r) > 0 AS exists",
                    s=str(source_id),
                    t=str(target_id),
                ).single()
                return bool(res["exists"]) if res else False
        return self._fallback_store.has_edge(source_id, target_id)

    def get_nodes(self) -> List[str]:
        if self._driver:
            with self._driver.session() as session:
                res = session.run("MATCH (u:User) RETURN u.id AS id")
                return [r["id"] for r in res]
        return self._fallback_store.get_nodes()

    def get_node_count(self) -> int:
        if self._driver:
            with self._driver.session() as session:
                res = session.run("MATCH (u:User) RETURN count(u) AS count").single()
                return int(res["count"]) if res else 0
        return self._fallback_store.get_node_count()

    def get_edge_count(self) -> int:
        if self._driver:
            with self._driver.session() as session:
                res = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()
                return int(res["count"]) if res else 0
        return self._fallback_store.get_edge_count()

    def get_followers(self, user_id: str) -> List[str]:
        if self._driver:
            with self._driver.session() as session:
                res = session.run("MATCH (s:User)-[:FOLLOWS]->(t:User {id: $id}) RETURN s.id AS id", id=str(user_id))
                return [r["id"] for r in res]
        return self._fallback_store.get_followers(user_id)

    def get_following(self, user_id: str) -> List[str]:
        if self._driver:
            with self._driver.session() as session:
                res = session.run("MATCH (s:User {id: $id})-[:FOLLOWS]->(t:User) RETURN t.id AS id", id=str(user_id))
                return [r["id"] for r in res]
        return self._fallback_store.get_following(user_id)

    def get_in_degree(self, user_id: str) -> int:
        if self._driver:
            with self._driver.session() as session:
                res = session.run("MATCH ()-[r]->(t:User {id: $id}) RETURN count(r) AS count", id=str(user_id)).single()
                return int(res["count"]) if res else 0
        return self._fallback_store.get_in_degree(user_id)

    def get_out_degree(self, user_id: str) -> int:
        if self._driver:
            with self._driver.session() as session:
                res = session.run("MATCH (s:User {id: $id})-[r]->() RETURN count(r) AS count", id=str(user_id)).single()
                return int(res["count"]) if res else 0
        return self._fallback_store.get_out_degree(user_id)

    def to_networkx(self) -> nx.DiGraph:
        if self._driver:
            G = nx.DiGraph()
            with self._driver.session() as session:
                res = session.run("MATCH (s:User)-[r]->(t:User) RETURN s.id AS s, t.id AS t, type(r) AS rel, r.weight AS w")
                for record in res:
                    G.add_edge(record["s"], record["t"], relation_type=record["rel"], weight=record["w"] or 1.0)
            return G
        return self._fallback_store.to_networkx()

    def clear(self) -> None:
        if self._driver:
            with self._driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n;")
        else:
            self._fallback_store.clear()
        self.revision += 1
