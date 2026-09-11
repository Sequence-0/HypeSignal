"""Ingestion adapter for the Cheng-Caverlee-Lee (Geo) Twitter dataset.

Streams GPS-tagged and profile-located tweets and user profiles into DuckDB
without excessive memory consumption using chunked Polars batching.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from zoneinfo import ZoneInfo

import polars as pl

from hypesignal.ingestion.base import BaseDatasetAdapter, robust_line_stream
from hypesignal.models.canonical import GeoCoordinates
from hypesignal.models.enums import PlatformType
from hypesignal.storage.duckdb_manager import DuckDBManager

logger = logging.getLogger(__name__)

# Precompiled regex patterns for fast entity extraction
RE_HASHTAG = re.compile(r"#(\w+)")
RE_MENTION = re.compile(r"@(\w+)")
RE_URL = re.compile(r"https?://\S+")


class GeoDatasetAdapter(BaseDatasetAdapter):
    """Adapter for reading Cheng-Caverlee-Lee CIKM 2010 geo-spatial Twitter corpus."""

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "Dataset/twitter/Cheng-Caver-Lee(Geo)/twitter_cikm_2010",
        source_tz: str = "UTC",
    ) -> None:
        """Initialize adapter with dataset directory and timezone hint.
        
        Args:
            dataset_dir: Root directory containing training_set_* and test_set_* files.
            source_tz: Assumed timezone for naive timestamps in the corpus (default: 'UTC').
        """
        self.dataset_dir = Path(dataset_dir)
        self.source_tz = ZoneInfo(source_tz)

    def ingest_users(
        self,
        db: DuckDBManager,
        split: str = "train",
        limit: Optional[int] = None,
        batch_size: int = 10000,
    ) -> int:
        """Stream and ingest user locations into DuckDB users table.
        
        Args:
            db: DuckDBManager instance.
            split: 'train' (training_set_users.txt) or 'test' (test_set_users.txt).
            limit: Maximum users to ingest.
            batch_size: Chunk size for Polars batch inserts.
            
        Returns:
            Total number of ingested users.
        """
        prefix = "training" if split in ("train", "training") else "test"
        filename = f"{prefix}_set_users.txt"
        filepath = self.dataset_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"User dataset file not found: {filepath}")

        total_ingested = 0
        batch_records: List[Dict[str, Any]] = []

        with open(filepath, "rb") as f:
            for line in robust_line_stream(f):
                if limit and total_ingested >= limit:
                    break

                line = line.strip("\r\n")
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 2:
                    continue

                user_id, location_raw = parts[0].strip(), parts[1].strip()
                coords = GeoCoordinates.from_raw_string(location_raw)

                batch_records.append({
                    "id": user_id,
                    "platform": PlatformType.TWITTER.value,
                    "screen_name": None,
                    "location_raw": location_raw,
                    "latitude": coords.latitude if coords else None,
                    "longitude": coords.longitude if coords else None,
                    "indegree": None,
                    "outdegree": None,
                    "bio": None,
                    "metrics": json.dumps({}),
                    "extra_metadata": json.dumps({"corpus_split": split, "source": "cheng_caverlee_lee"}),
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

        logger.info("Ingested %d users from %s", total_ingested, filename)
        return total_ingested

    def ingest_tweets(
        self,
        db: DuckDBManager,
        split: str = "train",
        limit: Optional[int] = None,
        batch_size: int = 10000,
    ) -> int:
        """Stream and ingest tweets into DuckDB posts table.
        
        Args:
            db: DuckDBManager instance.
            split: 'train' (training_set_tweets.txt) or 'test' (test_set_tweets.txt).
            limit: Maximum tweets to ingest.
            batch_size: Chunk size for Polars batch inserts.
            
        Returns:
            Total number of ingested tweets.
        """
        prefix = "training" if split in ("train", "training") else "test"
        filename = f"{prefix}_set_tweets.txt"
        filepath = self.dataset_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Tweet dataset file not found: {filepath}")

        total_ingested = 0
        batch_records: List[Dict[str, Any]] = []

        with open(filepath, "rb") as f:
            for line in robust_line_stream(f):
                if limit and total_ingested >= limit:
                    break

                line = line.strip("\r\n")
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 4:
                    continue

                user_id, tweet_id, text, ts_raw = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()

                try:
                    naive_dt = datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S")
                    dt_utc = naive_dt.replace(tzinfo=self.source_tz).astimezone(timezone.utc)
                    ts_ms = int(dt_utc.timestamp() * 1000)
                except ValueError:
                    # Skip or fallback if malformed timestamp
                    continue

                hashtags = RE_HASHTAG.findall(text)
                mentions = RE_MENTION.findall(text)
                urls = RE_URL.findall(text)

                batch_records.append({
                    "id": tweet_id,
                    "platform": PlatformType.TWITTER.value,
                    "author_id": user_id,
                    "author_screen_name": None,
                    "text": text,
                    "timestamp": dt_utc,
                    "timestamp_ms": ts_ms,
                    "parent_id": None,
                    "reply_to_user_id": None,
                    "source_client": None,
                    "urls": urls,
                    "hashtags": hashtags,
                    "mentions": mentions,
                    "metrics": json.dumps({}),
                    "extra_metadata": json.dumps({"corpus_split": split, "source": "cheng_caverlee_lee"}),
                })
                total_ingested += 1

                if len(batch_records) >= batch_size:
                    df = pl.DataFrame(batch_records)
                    db.insert_posts_df(df)
                    batch_records.clear()

            if batch_records:
                df = pl.DataFrame(batch_records)
                db.insert_posts_df(df)
                batch_records.clear()

        logger.info("Ingested %d tweets from %s", total_ingested, filename)
        return total_ingested

    def ingest(
        self,
        db: DuckDBManager,
        limit: Optional[int] = None,
        batch_size: int = 10000,
    ) -> Dict[str, int]:
        """Ingest both training users and training tweets."""
        users_count = self.ingest_users(db, split="train", limit=limit, batch_size=batch_size)
        posts_count = self.ingest_tweets(db, split="train", limit=limit, batch_size=batch_size)
        return {"users": users_count, "posts": posts_count}
