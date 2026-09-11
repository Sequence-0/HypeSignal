"""Automated tests for TimelineManager chronological analytics and replay."""

from datetime import datetime, timezone, timedelta
import pytest
from hypesignal.ingestion.geo_adapter import GeoDatasetAdapter
from hypesignal.ingestion.lerman_adapter import LermanDatasetAdapter
from hypesignal.storage.duckdb_manager import DuckDBManager
from hypesignal.timeline.timeline_manager import TimelineManager


@pytest.fixture
def populated_timeline():
    """Fixture providing an in-memory DuckDB populated with Geo and Lerman slices."""
    db = DuckDBManager(":memory:")
    
    geo = GeoDatasetAdapter()
    geo.ingest(db, limit=100, batch_size=50)

    lerman = LermanDatasetAdapter()
    lerman.ingest_cascades(db, limit=100, batch_size=50)

    tl = TimelineManager(db)
    yield tl
    db.close()


def test_timeline_bounds_and_slicing(populated_timeline: TimelineManager):
    """Verify time bounds and chronological slicing."""
    bounds = populated_timeline.get_time_bounds()
    assert bounds["total_posts"] == 100
    assert bounds["earliest"] is not None
    assert bounds["latest"] is not None
    assert bounds["earliest"] <= bounds["latest"]

    # Slice the entire window
    slice_df = populated_timeline.get_timeline_slice(
        start_time=bounds["earliest"],
        end_time=bounds["latest"],
    )
    assert len(slice_df) == 100

    # Slice a narrower window
    mid_time = bounds["earliest"] + (bounds["latest"] - bounds["earliest"]) / 2
    narrow_slice = populated_timeline.get_timeline_slice(
        start_time=bounds["earliest"],
        end_time=mid_time,
    )
    assert 0 < len(narrow_slice) <= 100


def test_timeline_replay(populated_timeline: TimelineManager):
    """Verify micro-batch replay generator yields strictly chronological posts."""
    batches = list(populated_timeline.replay_timeline(batch_size=30))
    assert len(batches) >= 4  # 100 posts / 30 = 4 batches (30, 30, 30, 10)

    # Verify timestamps are strictly monotonic non-decreasing across all batches
    all_ts = [p["timestamp_ms"] for batch in batches for p in batch]
    assert len(all_ts) == 100
    for i in range(1, len(all_ts)):
        assert all_ts[i] >= all_ts[i - 1]


def test_activity_timeseries(populated_timeline: TimelineManager):
    """Verify time-series volume aggregation."""
    ts_df = populated_timeline.get_activity_timeseries(interval="1 day")
    assert len(ts_df) > 0
    assert "bucket" in ts_df.columns
    assert "post_count" in ts_df.columns
    assert ts_df["post_count"].sum() == 100


def test_cascade_chronology(populated_timeline: TimelineManager):
    """Verify cascade diffusion sequence and seconds_since_origin."""
    sample_c = populated_timeline.db.query("SELECT cascade_id FROM cascade_events LIMIT 1")
    cascade_id = sample_c["cascade_id"][0]

    casc_df = populated_timeline.get_cascade_chronology(cascade_id)
    assert len(casc_df) > 0
    assert "seconds_since_origin" in casc_df.columns
    assert casc_df["seconds_since_origin"][0] == 0.0
    assert casc_df["seconds_since_origin"].is_sorted()


def test_activity_timeseries_interval_validation(populated_timeline: TimelineManager):
    """Verify get_activity_timeseries validates the interval format and rejects invalid inputs."""
    # Valid interval formats
    df_10m = populated_timeline.get_activity_timeseries(interval="10 minutes")
    assert len(df_10m) > 0

    df_2h = populated_timeline.get_activity_timeseries(interval="2 hours")
    assert len(df_2h) > 0

    # Malicious or malformed inputs must raise ValueError
    with pytest.raises(ValueError, match="Invalid time interval"):
        populated_timeline.get_activity_timeseries(interval="1 hour; DROP TABLE posts;--")

    with pytest.raises(ValueError, match="Invalid time interval"):
        populated_timeline.get_activity_timeseries(interval="invalid")

