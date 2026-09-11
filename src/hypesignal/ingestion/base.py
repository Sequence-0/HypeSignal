"""Abstract base class for dataset and platform ingestion adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, BinaryIO, Dict, Iterator, Optional

from hypesignal.storage.duckdb_manager import DuckDBManager


def robust_decode_line(raw_bytes: bytes) -> str:
    """Decode raw bytes with UTF-8 first, falling back to Latin-1 on decode error.
    
    Prevents silent corruption of UTF-8 multi-byte characters (e.g. curly quotes,
    emojis, special punctuation) that occur when blindly reading with latin-1.
    """
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1", errors="replace")


def robust_line_stream(binary_stream: BinaryIO) -> Iterator[str]:
    """Iterate over a binary stream line by line with UTF-8 and Latin-1 fallback."""
    for raw_line in binary_stream:
        yield robust_decode_line(raw_line)


class BaseDatasetAdapter(ABC):
    """Base class for all dataset ingestion adapters."""

    @abstractmethod
    def ingest(self, db: DuckDBManager, limit: Optional[int] = None, batch_size: int = 10000) -> Dict[str, int]:
        """Ingest raw dataset records into the analytical DuckDB database.
        
        Args:
            db: DuckDBManager instance.
            limit: Optional maximum number of records to ingest.
            batch_size: Number of records per batch.
            
        Returns:
            Dictionary containing counts of ingested entities (e.g. {'posts': 100, 'users': 50}).
        """
        pass

