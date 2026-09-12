#!/usr/bin/env python3
"""Ingestion script for populating a persistent DuckDB database with offline benchmark datasets.

Ingests:
1. Cheng-Caverlee-Lee Geo Tweets and User Locations (training + test splits)
2. Lerman 2010 Twitter distinct users (with in/out degrees)
3. Lerman 2010 URL information cascades (with millisecond timestamps & adoption orders)
4. Lerman 2010 Follower network topology edge list into graph_edges table
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from hypesignal.ingestion.geo_adapter import GeoDatasetAdapter
from hypesignal.ingestion.lerman_adapter import LermanDatasetAdapter
from hypesignal.storage.duckdb_manager import DuckDBManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hypesignal.ingest")


def ingest_data(
    db_path: str = "data/hypesignal.duckdb",
    tweets_limit: Optional[int] = 5000,
    users_limit: Optional[int] = 5000,
    cascades_limit: Optional[int] = 5000,
    edges_limit: Optional[int] = 10000,
    batch_size: int = 1000,
) -> None:
    """Ingest local datasets into the target DuckDB database file."""
    start_time = time.time()
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to DuckDB database at: %s", db_path)
    db = DuckDBManager(db_path=db_path)

    # 1. Cheng-Caverlee-Lee Geo Dataset (Tweets & User Locations)
    logger.info("--- Phase 1: Ingesting Cheng-Caverlee-Lee Geo Corpora ---")
    geo_adapter = GeoDatasetAdapter()

    try:
        # Ingest training set users (textual locations)
        train_users = geo_adapter.ingest_users(
            db, split="train", limit=users_limit, batch_size=batch_size
        )
        logger.info("Ingested %d training users from Cheng-Caverlee-Lee.", train_users)

        # Ingest test set users (GPS coordinates)
        test_users = geo_adapter.ingest_users(
            db, split="test", limit=users_limit, batch_size=batch_size
        )
        logger.info("Ingested %d test users with parsed GPS coordinates.", test_users)

        # Ingest tweets
        tweets_count = geo_adapter.ingest_tweets(
            db, split="train", limit=tweets_limit, batch_size=batch_size
        )
        logger.info("Ingested %d tweets from Cheng-Caverlee-Lee.", tweets_count)
    except Exception as e:
        logger.error("Error during Cheng-Caverlee-Lee ingestion: %s", e)

    # 2. Lerman 2010 Dataset (Users, Cascades, Follower Graph)
    logger.info("--- Phase 2: Ingesting Lerman 2010 Network & Diffusion Dataset ---")
    lerman_adapter = LermanDatasetAdapter()

    try:
        # Ingest distinct users with indegree / outdegree metrics
        lerman_users = lerman_adapter.ingest_users(
            db, limit=users_limit, batch_size=batch_size
        )
        logger.info("Ingested %d users with degree metrics from Lerman map.", lerman_users)

        # Ingest information diffusion cascade events
        cascades = lerman_adapter.ingest_cascades(
            db, limit=cascades_limit, batch_size=batch_size
        )
        logger.info("Ingested %d cascade propagation events from Lerman.", cascades)

        # Ingest follower network edges into graph_edges table
        logger.info("Streaming follower topology edges into DuckDB graph_edges...")
        edge_batch = []
        total_edges_ingested = 0
        for edge in lerman_adapter.stream_follower_edges(limit=edges_limit):
            edge_batch.append(edge)
            if len(edge_batch) >= batch_size:
                db.insert_edges(edge_batch)
                total_edges_ingested += len(edge_batch)
                edge_batch = []
                logger.info("  ...ingested %d edges so far", total_edges_ingested)

        if edge_batch:
            db.insert_edges(edge_batch)
            total_edges_ingested += len(edge_batch)

        logger.info("Ingested %d follower graph edges into DuckDB.", total_edges_ingested)
    except Exception as e:
        logger.error("Error during Lerman dataset ingestion: %s", e)

    elapsed = time.time() - start_time

    # Summary Report
    logger.info("--- Ingestion Complete in %.2f seconds ---", elapsed)
    logger.info("DuckDB Database Summary for '%s':", db_path)
    logger.info("  Total Posts:          %d", db.get_posts_count())
    logger.info("  Total Users:          %d", db.get_users_count())
    logger.info("  Total Cascade Events: %d", db.get_cascade_events_count())
    logger.info("  Total Graph Edges:    %d", db.get_edges_count())

    db.close()
    logger.info("Database closed successfully.")


def main() -> None:
    """Parse CLI arguments and run dataset ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest offline social media datasets into persistent DuckDB."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/hypesignal.duckdb",
        help="Path to persistent DuckDB file (default: data/hypesignal.duckdb)",
    )
    parser.add_argument(
        "--tweets-limit",
        type=int,
        default=5000,
        help="Max tweets to ingest from Cheng-Caverlee-Lee (default: 5000)",
    )
    parser.add_argument(
        "--users-limit",
        type=int,
        default=5000,
        help="Max users to ingest per source (default: 5000)",
    )
    parser.add_argument(
        "--cascades-limit",
        type=int,
        default=5000,
        help="Max cascade events to ingest (default: 5000)",
    )
    parser.add_argument(
        "--edges-limit",
        type=int,
        default=10000,
        help="Max follower graph edges to stream (default: 10000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for bulk insertions (default: 1000)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Ingest all available data without row limits",
    )

    args = parser.parse_args()

    if args.full:
        logger.info("Full ingestion mode enabled (limits disabled).")
        ingest_data(
            db_path=args.db_path,
            tweets_limit=None,
            users_limit=None,
            cascades_limit=None,
            edges_limit=None,
            batch_size=args.batch_size,
        )
    else:
        ingest_data(
            db_path=args.db_path,
            tweets_limit=args.tweets_limit,
            users_limit=args.users_limit,
            cascades_limit=args.cascades_limit,
            edges_limit=args.edges_limit,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
