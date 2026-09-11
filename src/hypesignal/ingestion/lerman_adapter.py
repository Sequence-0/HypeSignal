"""Ingestion adapter for the Lerman Twitter 2010 cascade and follower dataset.

Handles streaming extraction of URL diffusion cascades, degree mappings,
and follower network edge lists directly from compressed zip archives.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import polars as pl

from hypesignal.ingestion.base import BaseDatasetAdapter, robust_line_stream
from hypesignal.models.canonical import CanonicalGraphEdge
from hypesignal.models.enums import PlatformType, RelationType
from hypesignal.storage.duckdb_manager import DuckDBManager

logger = logging.getLogger(__name__)

RE_SQL_TUPLE = re.compile(r"\((\d+),(\d+)\)")


class LermanDatasetAdapter(BaseDatasetAdapter):
    """Adapter for Lerman 2010 Twitter cascade and follower graph dataset."""

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "Dataset/twitter/Lerman(Main+Graph)",
    ) -> None:
        """Initialize adapter with dataset directory."""
        self.dataset_dir = Path(dataset_dir)
        self.users_file = self.dataset_dir / "distinct_users_from_search_table_real_map.csv"
        self.cascades_zip = self.dataset_dir / "link_status_search_with_ordering_real_csv.zip"
        self.follower_zip = self.dataset_dir / "active_follower_real_sql.zip"

    def ingest_users(
        self,
        db: DuckDBManager,
        limit: Optional[int] = None,
        batch_size: int = 10000,
    ) -> int:
        """Ingest distinct users and their indegree/outdegree metrics into DuckDB."""
        if not self.users_file.exists():
            raise FileNotFoundError(f"Lerman users file not found: {self.users_file}")

        total_ingested = 0
        batch_records: List[Dict[str, Any]] = []

        with open(self.users_file, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if limit and total_ingested >= limit:
                    break

                user_id = row.get("user_id", "").strip()
                screen_name = row.get("user_screen_name", "").strip()
                try:
                    indegree = int(row.get("indegree", 0))
                except (ValueError, TypeError):
                    indegree = 0

                try:
                    outdegree = int(row.get("outdegree", 0))
                except (ValueError, TypeError):
                    outdegree = 0

                batch_records.append({
                    "id": user_id,
                    "platform": PlatformType.TWITTER.value,
                    "screen_name": screen_name or None,
                    "location_raw": None,
                    "latitude": None,
                    "longitude": None,
                    "indegree": indegree,
                    "outdegree": outdegree,
                    "bio": None,
                    "metrics": json.dumps({"followers_count": indegree, "following_count": outdegree}),
                    "extra_metadata": json.dumps({"source": "lerman_2010", "bad_user_id": row.get("bad_user_id")}),
                })
                total_ingested += 1

                if len(batch_records) >= batch_size:
                    df = pl.DataFrame(batch_records)
                    db.insert_users_df(df)
                    batch_records.clear()

            if batch_records:
                df = pl.DataFrame(batch_records)
                db.insert_users_df(df)
                batch_records.clear()

        logger.info("Ingested %d users from %s", total_ingested, self.users_file.name)
        return total_ingested

    def ingest_cascades(
        self,
        db: DuckDBManager,
        limit: Optional[int] = None,
        batch_size: int = 10000,
    ) -> int:
        """Stream and ingest URL cascade occurrences from compressed zip archive."""
        if not self.cascades_zip.exists():
            raise FileNotFoundError(f"Lerman cascades zip not found: {self.cascades_zip}")

        total_ingested = 0
        batch_records: List[Dict[str, Any]] = []

        with zipfile.ZipFile(self.cascades_zip) as z:
            with z.open("link_status_search_with_ordering_real.csv") as f:
                reader = csv.DictReader(robust_line_stream(f))
                for row in reader:
                    if limit and total_ingested >= limit:
                        break

                    cascade_id = row.get("link", "").strip()
                    post_id = row.get("id", "").strip()
                    user_id = row.get("user_id", "").strip()
                    user_screen_name = row.get("user_screen_name", "").strip()
                    order_str = row.get("order_of_users", "0").strip()
                    ts_long_str = row.get("create_at_long", "").strip()

                    try:
                        order_idx = int(order_str)
                    except ValueError:
                        order_idx = 0

                    try:
                        ts_ms = int(ts_long_str)
                        dt_utc = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
                    except (ValueError, TypeError, OSError):
                        dt_utc = datetime.now(timezone.utc)
                        ts_ms = int(dt_utc.timestamp() * 1000)

                    batch_records.append({
                        "cascade_id": cascade_id,
                        "post_id": post_id,
                        "user_id": user_id,
                        "user_screen_name": user_screen_name or None,
                        "timestamp": dt_utc,
                        "timestamp_ms": ts_ms,
                        "adoption_order": order_idx,
                        "parent_event_id": None,
                        "extra_metadata": json.dumps({
                            "source_client": row.get("source"),
                            "inreplyto_screen_name": row.get("inreplyto_screen_name"),
                            "inreplyto_user_id": row.get("inreplyto_user_id"),
                        }),
                    })
                    total_ingested += 1

                    if len(batch_records) >= batch_size:
                        df = pl.DataFrame(batch_records)
                        db.insert_cascade_events_df(df)
                        batch_records.clear()

                if batch_records:
                    df = pl.DataFrame(batch_records)
                    db.insert_cascade_events_df(df)
                    batch_records.clear()

        logger.info("Ingested %d cascade events from %s", total_ingested, self.cascades_zip.name)
        return total_ingested

    def stream_follower_edges(
        self,
        limit: Optional[int] = None,
    ) -> Iterator[CanonicalGraphEdge]:
        """Stream follower relationships directly from SQL archive as CanonicalGraphEdge objects.
        
        Yields edges where source_id = follower_id and target_id = followee user_id.
        """
        if not self.follower_zip.exists():
            raise FileNotFoundError(f"Lerman follower zip not found: {self.follower_zip}")

        yielded = 0
        with zipfile.ZipFile(self.follower_zip) as z:
            with z.open("active_follower_real.sql") as f:
                reader = io.TextIOWrapper(f, encoding="latin-1", errors="replace")
                for line in reader:
                    if "insert" not in line.lower():
                        continue

                    matches = RE_SQL_TUPLE.findall(line)
                    for user_id, follower_id in matches:
                        yield CanonicalGraphEdge(
                            source_id=follower_id,
                            target_id=user_id,
                            relation_type=RelationType.FOLLOWS,
                            weight=1.0,
                            extra_metadata={"source": "lerman_2010"},
                        )
                        yielded += 1
                        if limit and yielded >= limit:
                            return

    def ingest(
        self,
        db: DuckDBManager,
        limit: Optional[int] = None,
        batch_size: int = 10000,
    ) -> Dict[str, int]:
        """Ingest users and cascade events into DuckDB."""
        users_count = self.ingest_users(db, limit=limit, batch_size=batch_size)
        cascades_count = self.ingest_cascades(db, limit=limit, batch_size=batch_size)
        return {"users": users_count, "cascade_events": cascades_count}
