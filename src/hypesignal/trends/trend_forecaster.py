"""Predictive trend velocity and acceleration forecaster (Pillar D).

Computes 1st derivative (velocity) and 2nd derivative (acceleration) of term
frequency across temporal sliding windows, classifies lifecycle states
(EMERGING, VIRAL_SURGE, PEAKING, DECELERATING, STABLE, DORMANT), and estimates
virality potential.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from hypesignal.storage.duckdb_manager import DuckDBManager

logger = logging.getLogger(__name__)


class TrendLifecycleState(str, Enum):
    """Lifecycle trajectory states of an evolving trend."""
    EMERGING = "emerging"
    VIRAL_SURGE = "viral_surge"
    PEAKING = "peaking"
    DECELERATING = "decelerating"
    STABLE = "stable"
    DORMANT = "dormant"


class TrendForecast(BaseModel):
    """Kinematic forecast and lifecycle classification of a trending topic."""
    term: str
    current_velocity: float  # v(t) in posts per unit time
    previous_velocity: float  # v(t-1)
    acceleration: float  # a(t) = Delta v / Delta t
    lifecycle_state: TrendLifecycleState
    virality_potential_score: float = Field(..., ge=0.0, le=1.0)
    user_diversity_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    total_volume: int = Field(..., ge=0)
    window_volumes: List[int] = Field(default_factory=list)

    @property
    def velocity(self) -> float:
        """Alias for current_velocity."""
        return self.current_velocity


class TrendForecaster:
    """1st and 2nd derivative kinematic trend forecaster and lifecycle classifier."""

    def __init__(
        self,
        velocity_threshold: float = 10.0,
        acceleration_threshold: float = 5.0,
        viral_surge_velocity_multiplier: float = 2.5,
        viral_surge_accel_multiplier: float = 2.0,
        stable_tolerance: float = 0.15,
    ) -> None:
        """Initialize trend forecaster with lifecycle classification thresholds.
        
        Args:
            velocity_threshold: Minimum velocity to be considered emerging.
            acceleration_threshold: Minimum positive acceleration for strong growth.
            viral_surge_velocity_multiplier: Multiplier on velocity_threshold for viral surge.
            viral_surge_accel_multiplier: Multiplier on acceleration_threshold for viral surge.
            stable_tolerance: Maximum relative change in velocity considered stable.
        """
        self.velocity_threshold = float(velocity_threshold)
        self.acceleration_threshold = float(acceleration_threshold)
        self.viral_surge_velocity = self.velocity_threshold * float(viral_surge_velocity_multiplier)
        self.viral_surge_accel = self.acceleration_threshold * float(viral_surge_accel_multiplier)
        self.stable_tolerance = float(stable_tolerance)

    def forecast_from_volumes(
        self,
        term: str,
        volumes: List[int],
        window_duration_minutes: float = 60.0,
        unique_authors: Optional[int] = None,
    ) -> TrendForecast:
        """Compute velocity, acceleration, and lifecycle state from chronological window volumes.
        
        Args:
            term: The keyword, hashtag, or topic name.
            volumes: List of chronological window counts (at least 2, ideally 3+).
            window_duration_minutes: Duration of each window in minutes (default 60 min).
            unique_authors: Total unique users authoring posts for diversity calculation.
            
        Returns:
            TrendForecast model with kinematic metrics and lifecycle classification.
        """
        if not volumes:
            return TrendForecast(
                term=term,
                current_velocity=0.0,
                previous_velocity=0.0,
                acceleration=0.0,
                lifecycle_state=TrendLifecycleState.DORMANT,
                virality_potential_score=0.0,
                user_diversity_ratio=1.0,
                total_volume=0,
                window_volumes=[],
            )

        dt = max(1.0, float(window_duration_minutes))
        total_vol = sum(volumes)

        if len(volumes) == 1:
            # Single window: no acceleration can be computed
            v_curr = round(volumes[0] / dt, 4)
            state = TrendLifecycleState.EMERGING if v_curr >= self.velocity_threshold else TrendLifecycleState.STABLE
            diversity = min(1.0, (unique_authors / max(1, total_vol))) if unique_authors is not None else 1.0
            return TrendForecast(
                term=term,
                current_velocity=v_curr,
                previous_velocity=0.0,
                acceleration=0.0,
                lifecycle_state=state,
                virality_potential_score=round(min(1.0, v_curr / max(1.0, self.viral_surge_velocity)), 4),
                user_diversity_ratio=round(diversity, 4),
                total_volume=total_vol,
                window_volumes=volumes,
            )

        # Velocities: rate of posts in each window
        # v(t) = Volume(t) / dt
        velocities = [vol / dt for vol in volumes]
        v_curr = velocities[-1]
        v_prev = velocities[-2]

        # 2nd Derivative Acceleration: Delta v / Delta t
        # (v_curr - v_prev) / dt
        accel = (v_curr - v_prev) / dt
        eff_accel = v_curr - v_prev  # Velocity delta: accel * dt

        # Lifecycle Classification
        state = self._classify_lifecycle(
            v_curr=v_curr,
            v_prev=v_prev,
            accel=accel,
            eff_accel=eff_accel,
            volumes=volumes,
        )

        # Author diversity ratio and virality potential score
        if total_vol == 0:
            diversity = 0.0
            virality_potential = 0.0
        else:
            diversity = min(1.0, (unique_authors / total_vol)) if unique_authors is not None else 1.0
            v_norm = min(1.0, v_curr / max(1.0, self.viral_surge_velocity))
            a_norm = max(0.0, min(1.0, eff_accel / max(1.0, self.viral_surge_accel)))
            virality_potential = round(0.45 * v_norm + 0.40 * a_norm + 0.15 * diversity, 4)

        return TrendForecast(
            term=term,
            current_velocity=round(v_curr, 4),
            previous_velocity=round(v_prev, 4),
            acceleration=round(accel, 4),
            lifecycle_state=state,
            virality_potential_score=virality_potential,
            user_diversity_ratio=round(diversity, 4),
            total_volume=total_vol,
            window_volumes=volumes,
        )

    def _classify_lifecycle(
        self,
        v_curr: float,
        v_prev: float,
        accel: float,
        eff_accel: float,
        volumes: List[int],
    ) -> TrendLifecycleState:
        """Classify trend lifecycle stage based on kinematic metrics and configured thresholds."""
        # 1. Check if consistently stable across all available windows
        if len(volumes) >= 2:
            is_consistent_stable = True
            for i in range(len(volumes) - 1):
                prev_v = max(0.1, float(volumes[i]))
                diff_ratio = abs(float(volumes[i + 1]) - float(volumes[i])) / prev_v
                if diff_ratio > self.stable_tolerance:
                    is_consistent_stable = False
                    break
            if is_consistent_stable:
                if v_curr < (self.velocity_threshold * 0.25):
                    return TrendLifecycleState.DORMANT
                return TrendLifecycleState.STABLE

        # 2. Dormant: very low volume and flat velocity
        if v_curr < (self.velocity_threshold * 0.25) and abs(eff_accel) < (self.acceleration_threshold * 0.2):
            return TrendLifecycleState.DORMANT

        # 3. Viral Surge: steep upward trajectory with velocity >= viral_surge_velocity and eff_accel >= viral_surge_accel
        if v_curr >= self.viral_surge_velocity and eff_accel >= self.viral_surge_accel:
            return TrendLifecycleState.VIRAL_SURGE

        # 4. Emerging: above threshold velocity and positive acceleration
        if v_curr >= self.velocity_threshold and eff_accel > 0.0:
            return TrendLifecycleState.EMERGING

        # 5. Decelerating: clear downward drop in velocity with negative acceleration
        if eff_accel < 0.0 and v_curr < (v_prev * (1.0 - self.stable_tolerance)):
            return TrendLifecycleState.DECELERATING

        # 6. Peaking: was at high volume following a rise, acceleration has flattened/turned slightly negative
        if v_curr >= self.velocity_threshold and (eff_accel <= (self.stable_tolerance * v_prev) or eff_accel <= 0.0):
            max_vol = max(volumes)
            if volumes[-1] >= (max_vol * 0.85):
                return TrendLifecycleState.PEAKING

        # 7. Stable fallback
        if abs(eff_accel) <= (max(0.1, v_prev) * self.stable_tolerance):
            return TrendLifecycleState.STABLE

        if eff_accel < 0.0:
            return TrendLifecycleState.DECELERATING

        return TrendLifecycleState.EMERGING

    def forecast_from_db(
        self,
        db: DuckDBManager,
        term: str,
        window_minutes: int = 60,
        num_windows: int = 3,
        reference_time: Optional[datetime] = None,
    ) -> TrendForecast:
        """Extract sliding window volumes from DuckDB posts and compute forecast.
        
        Args:
            db: DuckDBManager instance.
            term: Keyword or hashtag to search in post texts.
            window_minutes: Duration of each sliding window in minutes.
            num_windows: Number of historical windows to evaluate (typically 3: [t-2, t-1, t]).
            reference_time: Evaluation end time (defaults to latest post timestamp or now).
            
        Returns:
            TrendForecast computed from database records.
        """
        ref_time = reference_time
        if ref_time is None:
            # Query max timestamp from posts
            max_ts_row = db.con.execute("SELECT MAX(timestamp) FROM posts").fetchone()
            if max_ts_row and max_ts_row[0]:
                ref_time = max_ts_row[0]
            else:
                ref_time = datetime.now(timezone.utc)

        if ref_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=timezone.utc)

        total_span = timedelta(minutes=window_minutes * num_windows)
        start_time = ref_time - total_span

        # Query posts matching term in total span with word boundary regex
        clean_term = term.strip().lstrip("#")
        escaped_term = re.escape(clean_term)
        pattern = f"(?i)(?:^|[^a-zA-Z0-9_])#?{escaped_term}(?:[^a-zA-Z0-9_]|$)"
        query = """
            SELECT timestamp, author_id FROM posts
            WHERE timestamp >= ? AND timestamp <= ?
              AND REGEXP_MATCHES(text, ?)
            ORDER BY timestamp ASC;
        """
        rows = db.con.execute(query, [start_time, ref_time, pattern]).fetchall()

        # Bucket into windows
        volumes = [0] * num_windows
        unique_authors = set()

        for ts, author_id in rows:
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            elapsed = (ts - start_time).total_seconds()
            window_idx = min(num_windows - 1, max(0, int(elapsed // (window_minutes * 60))))
            volumes[window_idx] += 1
            if author_id:
                unique_authors.add(str(author_id))

        return self.forecast_from_volumes(
            term=term,
            volumes=volumes,
            window_duration_minutes=float(window_minutes),
            unique_authors=len(unique_authors),
        )

    def forecast_trend_kinematics(
        self,
        db: DuckDBManager,
        term: str,
        window_duration_minutes: float = 60.0,
        **kwargs: Any,
    ) -> TrendForecast:
        """Forecast trend velocity, acceleration, and lifecycle state from DuckDB."""
        return self.forecast_from_db(
            db=db,
            term=term,
            window_minutes=int(window_duration_minutes),
            **kwargs,
        )
