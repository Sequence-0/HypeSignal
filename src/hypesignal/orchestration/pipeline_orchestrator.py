"""Consolidated Analytics Pipeline Orchestrator (Pillar F).

Runs an asynchronous tick loop coordinating ingestion, NLP enrichment,
trend kinematics forecasting, and SSE alert broadcasting with non-blocking
asyncio.to_thread execution and strict stage error isolation.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from hypesignal.api.streaming import EventBroadcaster
from hypesignal.connectors.base import PlatformConnector
from hypesignal.nlp.engine import MultiDimensionalSentimentEngine
from hypesignal.storage.duckdb_manager import DuckDBManager
from hypesignal.trends.narrative_drift import NarrativeDriftTracker
from hypesignal.trends.trend_forecaster import TrendForecaster, TrendLifecycleState
from hypesignal.trends.trends_engine import TrendsEngine

logger = logging.getLogger(__name__)


class AnalyticsPipelineOrchestrator:
    """Consolidated background orchestrator coordinating ingestion, enrichment, and trend tracking."""

    def __init__(
        self,
        db: DuckDBManager,
        nlp: Optional[MultiDimensionalSentimentEngine] = None,
        trends: Optional[TrendsEngine] = None,
        trend_forecaster: Optional[TrendForecaster] = None,
        narrative_drift: Optional[NarrativeDriftTracker] = None,
        connectors: Optional[Dict[str, PlatformConnector]] = None,
        broadcaster: Optional[EventBroadcaster] = None,
        tick_interval_seconds: float = 5.0,
        enrichment_batch_size: int = 50,
    ) -> None:
        """Initialize pipeline orchestrator with modular analytical subcomponents.
        
        Args:
            db: DuckDBManager instance for data persistence.
            nlp: Optional MultiDimensionalSentimentEngine for emotion/stance analysis.
            trends: Optional TrendsEngine for topic clustering.
            trend_forecaster: Optional TrendForecaster for velocity/acceleration kinematics.
            narrative_drift: Optional NarrativeDriftTracker for semantic drift/inversion.
            connectors: Optional dictionary of registered platform connectors.
            broadcaster: Optional EventBroadcaster for real-time SSE event dispatch.
            tick_interval_seconds: Polling sleep interval between loop iterations.
            enrichment_batch_size: Maximum posts analyzed per enrichment tick.
        """
        self.db = db
        self.nlp = nlp
        self.trends = trends
        self.trend_forecaster = trend_forecaster or TrendForecaster()
        self.narrative_drift = narrative_drift
        self.connectors = connectors or {}
        self.broadcaster = broadcaster
        self.tick_interval_seconds = tick_interval_seconds
        self.enrichment_batch_size = enrichment_batch_size

        self.is_running: bool = False
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def run_ingestion_stage(self, query: str = "news", limit: int = 10) -> int:
        """Stage 1: Poll enabled connectors and store raw posts to DuckDB."""
        total_ingested = 0
        for name, connector in list(self.connectors.items()):
            if not getattr(connector.config, "enabled", True):
                continue

            try:
                # Wrap synchronous connector poll in asyncio.to_thread
                posts = await asyncio.to_thread(connector.poll, query=query, limit=limit)
                if posts:
                    await asyncio.to_thread(self.db.insert_posts, posts)
                    total_ingested += len(posts)
                    if self.broadcaster:
                        self.broadcaster.broadcast(
                            {"connector": name, "ingested_count": len(posts)},
                            event_type="ingestion",
                        )
            except Exception as e:
                logger.warning("Ingestion stage error for connector '%s': %s", name, e)

        return total_ingested

    async def run_enrichment_stage(self, limit: Optional[int] = None) -> int:
        """Stage 2: Query unanalyzed posts, run PyTorch inference in thread, and persist to post_analytics."""
        if self.nlp is None:
            return 0

        batch_lim = limit or self.enrichment_batch_size

        def _fetch_unanalyzed():
            query = """
                SELECT p.id, p.text 
                FROM posts p 
                LEFT JOIN post_analytics pa ON p.id = pa.post_id 
                WHERE pa.post_id IS NULL 
                ORDER BY p.timestamp ASC 
                LIMIT ?;
            """
            return self.db.con.execute(query, [batch_lim]).fetchall()

        try:
            unanalyzed_rows = await asyncio.to_thread(_fetch_unanalyzed)
            if not unanalyzed_rows:
                return 0

            ids = [str(r[0]) for r in unanalyzed_rows]
            texts = [str(r[1]) for r in unanalyzed_rows]

            # Execute PyTorch batch inference in background worker thread
            analytics_rows = await asyncio.to_thread(self.nlp.analyze_and_flatten, texts, ids)

            # Persist analytics
            await asyncio.to_thread(self.db.upsert_post_analytics, analytics_rows)

            if self.broadcaster:
                self.broadcaster.broadcast(
                    {"enriched_count": len(analytics_rows)},
                    event_type="enrichment",
                )

            return len(analytics_rows)
        except Exception as e:
            logger.warning("Enrichment stage error: %s", e)
            return 0

    async def run_trend_stage(self, window_minutes: float = 60.0) -> List[Dict[str, Any]]:
        """Stage 3: Evaluate trend kinematics and broadcast viral surge alerts."""
        if self.trend_forecaster is None:
            return []

        def _find_recent_terms():
            # Query top recent hashtags or frequent keywords in window
            query = """
                SELECT tag, count(*) as cnt 
                FROM (
                    SELECT unnest(hashtags) as tag 
                    FROM posts 
                    WHERE timestamp >= (SELECT COALESCE(MAX(timestamp), CURRENT_TIMESTAMP) FROM posts) - INTERVAL '12 hours'
                      AND hashtags IS NOT NULL
                )
                GROUP BY tag 
                ORDER BY cnt DESC 
                LIMIT 5;
            """
            try:
                rows = self.db.con.execute(query).fetchall()
                return [r[0] for r in rows if r[0]]
            except Exception as e:
                logger.debug("Recent terms extraction skipped: %s", e)
                return []

        alerts: List[Dict[str, Any]] = []
        try:
            terms = await asyncio.to_thread(_find_recent_terms)
            for term in terms:
                forecast = await asyncio.to_thread(
                    self.trend_forecaster.forecast_trend_kinematics,
                    db=self.db,
                    term=term,
                    window_duration_minutes=window_minutes,
                )
                if forecast.lifecycle_state in (
                    TrendLifecycleState.VIRAL_SURGE,
                    TrendLifecycleState.EMERGING,
                ):
                    alert_data = {
                        "term": term,
                        "lifecycle_state": forecast.lifecycle_state.value,
                        "velocity": forecast.current_velocity,
                        "current_velocity": forecast.current_velocity,
                        "acceleration": forecast.acceleration,
                        "virality_score": forecast.virality_potential_score,
                        "virality_potential_score": forecast.virality_potential_score,
                    }
                    alerts.append(alert_data)
                    if self.broadcaster:
                        self.broadcaster.broadcast(alert_data, event_type="trend_alert")
        except Exception as e:
            logger.warning("Trend forecasting stage error: %s", e)

        return alerts

    async def run_single_tick(self) -> Dict[str, Any]:
        """Execute a single complete pipeline tick across all analytical stages with error isolation."""
        ingested = 0
        enriched = 0
        alerts: List[Dict[str, Any]] = []

        try:
            ingested = await self.run_ingestion_stage()
        except Exception as e:
            logger.error("Ingestion stage crashed: %s", e)

        try:
            enriched = await self.run_enrichment_stage()
        except Exception as e:
            logger.error("Enrichment stage crashed: %s", e)

        try:
            alerts = await self.run_trend_stage()
        except Exception as e:
            logger.error("Trend stage crashed: %s", e)

        return {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ingested_count": ingested,
            "enriched_count": enriched,
            "alerts_count": len(alerts),
            "alerts": alerts,
        }

    async def _run_loop(self) -> None:
        """Background asynchronous tick loop."""
        logger.info("Starting AnalyticsPipelineOrchestrator background loop...")
        while self.is_running and not self._stop_event.is_set():
            try:
                await self.run_single_tick()
            except Exception as e:
                logger.error("Unexpected failure in pipeline tick loop: %s", e)

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.tick_interval_seconds)
                # If stop_event was set, exit cleanly
                break
            except asyncio.TimeoutError:
                # Normal tick interval elapsed
                continue

        logger.info("AnalyticsPipelineOrchestrator background loop stopped.")

    def start(self) -> None:
        """Start background asynchronous tick worker."""
        if self.is_running:
            return
        self.is_running = True
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())
        logger.info("AnalyticsPipelineOrchestrator started.")

    async def stop(self) -> None:
        """Gracefully stop background asynchronous tick worker."""
        if not self.is_running:
            return
        self.is_running = False
        self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("AnalyticsPipelineOrchestrator cleanly shut down.")
