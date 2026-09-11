"""Qdrant vector store manager for HypeSignal.

Provides HNSW vector indexing and similarity retrieval for user bio personas,
semantic post embeddings, and interest clustering. Supports embedded mode
(:memory: or local disk path) for fast offline operation.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Union

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models


class VectorStoreManager:
    """Manages Qdrant vector database client, collection lifecycle, and vector indexing."""

    def __init__(
        self,
        location: Optional[str] = ":memory:",
        path: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize Qdrant client in embedded or remote client/server mode."""
        if path:
            self.client = QdrantClient(path=path)
        elif url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(location=location or ":memory:")

    @staticmethod
    def _normalize_id(point_id: Union[str, int]) -> tuple[Union[int, str], str]:
        """Normalize arbitrary string/int IDs into Qdrant-compatible IDs.
        
        Qdrant accepts unsigned 64-bit integers [0, 2^64 - 1] or valid UUID strings.
        If point_id represents a non-negative integer (including 19-20 digit Twitter/X
        Snowflake IDs), parses it to int consistently regardless of whether it arrived
        as int or str. Otherwise, checks for valid UUID or generates a deterministic UUIDv5.
        """
        orig_str = str(point_id)
        
        # 1. Check if input is a valid non-negative integer within uint64 range
        try:
            val_int = int(point_id)
            if 0 <= val_int <= 18446744073709551615:
                return val_int, orig_str
        except (ValueError, TypeError):
            pass

        # 2. Check if already a valid UUID string
        try:
            val_uuid = uuid.UUID(orig_str)
            return str(val_uuid), orig_str
        except (ValueError, TypeError, AttributeError):
            pass

        # 3. Fallback: generate deterministic UUIDv5 for non-integer strings
        gen_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, orig_str))
        return gen_uuid, orig_str

    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 384,
        distance: str = "Cosine",
    ) -> bool:
        """Create a collection if it does not already exist.
        
        Args:
            collection_name: Name of collection (e.g. 'bio_personas').
            vector_size: Dimensionality of embeddings (default: 384 for all-MiniLM-L6-v2).
            distance: Distance metric ('Cosine', 'Euclid', 'Dot').
        """
        dist_enum = getattr(rest_models.Distance, distance.upper(), rest_models.Distance.COSINE)
        
        collections = [c.name for c in self.client.get_collections().collections]
        if collection_name in collections:
            return False

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=rest_models.VectorParams(
                size=vector_size,
                distance=dist_enum,
            ),
        )
        return True

    def upsert_vectors(
        self,
        collection_name: str,
        ids: List[Union[str, int]],
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Batch upsert points (embeddings + metadata payloads)."""
        if not ids:
            return

        points = []
        for i, (point_id, vector) in enumerate(zip(ids, vectors)):
            payload = dict(payloads[i]) if payloads and i < len(payloads) else {}
            qdrant_id, orig_id = self._normalize_id(point_id)
            if "_original_id" not in payload:
                payload["_original_id"] = orig_id

            points.append(
                rest_models.PointStruct(
                    id=qdrant_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=collection_name,
            points=points,
        )

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search nearest neighbors for a query vector."""
        search_result = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        ).points

        results = []
        for hit in search_result:
            payload = hit.payload or {}
            orig_id = payload.get("_original_id", str(hit.id))
            results.append({
                "id": orig_id,
                "qdrant_id": hit.id,
                "score": hit.score,
                "payload": payload,
            })
        return results

    def count(self, collection_name: str) -> int:
        """Return total number of points in collection."""
        res = self.client.count(collection_name=collection_name, exact=True)
        return res.count

    def delete_collection(self, collection_name: str) -> bool:
        """Delete an existing collection."""
        return self.client.delete_collection(collection_name=collection_name)
