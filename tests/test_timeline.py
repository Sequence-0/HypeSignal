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


def test_timeline_filtering_keyword_platform_video(populated_timeline: TimelineManager):
    """Verify dynamic keyword, platform, and parent_id filters on timeline slice and timeseries."""
    from hypesignal.models.canonical import CanonicalPost
    from hypesignal.models.enums import PlatformType

    # Insert test posts with specific keywords and video parent IDs
    now = datetime.now(timezone.utc)
    custom_posts = [
        CanonicalPost(
            id="yt_post_001",
            platform=PlatformType.YOUTUBE,
            author_id="user_yt_1",
            author_screen_name="CreatorYT",
            text="Breaking benchmark: neural network speedup 10x! #deeplearning",
            timestamp=now - timedelta(minutes=10),
            parent_id="video_dQw4w9WgXcQ",
        ),
        CanonicalPost(
            id="tw_post_002",
            platform=PlatformType.TWITTER,
            author_id="user_tw_2",
            author_screen_name="TweeterAI",
            text="Discussion on latest AI papers #deeplearning",
            timestamp=now - timedelta(minutes=5),
        ),
        CanonicalPost(
            id="rd_post_003",
            platform=PlatformType.REDDIT,
            author_id="user_rd_3",
            author_screen_name="RedditorX",
            text="Why fast databases matter for realtime streaming",
            timestamp=now - timedelta(minutes=2),
        ),
    ]
    populated_timeline.db.insert_posts(custom_posts)

    # 1. Filter by keyword: "neural network"
    df_kw = populated_timeline.get_timeline_slice(keyword="neural network")
    assert len(df_kw) == 1
    assert df_kw["id"][0] == "yt_post_001"

    # 2. Filter by hashtag keyword: "deeplearning"
    df_tag = populated_timeline.get_timeline_slice(keyword="deeplearning")
    assert len(df_tag) == 2

    # 3. Filter by platform: "youtube"
    df_plat = populated_timeline.get_timeline_slice(platform="youtube")
    assert len(df_plat) == 1
    assert df_plat["platform"][0] == "youtube"

    # 4. Filter by video / parent_id: "dQw4w9WgXcQ"
    df_vid = populated_timeline.get_timeline_slice(parent_id="dQw4w9WgXcQ")
    assert len(df_vid) == 1
    assert df_vid["id"][0] == "yt_post_001"

    # 5. Combined filter: keyword + platform
    df_both = populated_timeline.get_timeline_slice(keyword="deeplearning", platform="twitter")
    assert len(df_both) == 1
    assert df_both["id"][0] == "tw_post_002"

    # 6. Timeseries with keyword and platform filter
    ts_yt = populated_timeline.get_activity_timeseries(interval="1 hour", platform="youtube", keyword="neural")
    assert len(ts_yt) >= 1
    assert ts_yt["post_count"].sum() == 1

    # 7. Case-insensitive platform filter (Issue 4 fix)
    df_plat_case = populated_timeline.get_timeline_slice(platform="YouTube")
    assert len(df_plat_case) == 1
    assert df_plat_case["platform"][0] == "youtube"

    df_tw_case = populated_timeline.get_timeline_slice(platform="TWITTER")
    assert len(df_tw_case) >= 1


def test_timeline_slice_no_data_loss_on_json_deserialization():
    """Verify Issue 2 fix: JSON strings from DuckDB do not cause ValidationError or data loss."""
    import json
    from hypesignal.api.routes.timeline import _record_to_canonical_post
    from hypesignal.models.canonical import PostMetrics

    raw_duckdb_row = {
        "id": "post_full_123",
        "platform": "youtube",
        "author_id": "author_yt_99",
        "author_screen_name": "TechChannel",
        "text": "Complete guide to modern AI #ai https://youtube.com/watch?v=123",
        "timestamp": datetime.now(timezone.utc),
        "timestamp_ms": 1700000000000,
        "parent_id": "video_123",
        "reply_to_user_id": None,
        "source_client": "YouTube Data API v3",
        "urls": json.dumps(["https://youtube.com/watch?v=123"]),
        "hashtags": json.dumps(["ai", "tech"]),
        "mentions": json.dumps(["@google"]),
        "media_urls": json.dumps([]),
        "metrics": json.dumps({"likes": 42, "replies": 5, "reposts": 0, "views": 1000}),
        "extra_metadata": json.dumps({"video_id": "123", "verified": True}),
    }

    post = _record_to_canonical_post(raw_duckdb_row)
    assert post.id == "post_full_123"
    assert post.platform.value == "youtube"
    assert isinstance(post.metrics, PostMetrics)
    assert post.metrics.likes == 42
    assert post.metrics.replies == 5
    assert post.metrics.views == 1000
    assert post.hashtags == ["ai", "tech"]
    assert post.urls == ["https://youtube.com/watch?v=123"]
    assert post.mentions == ["@google"]
    assert post.extra_metadata.get("video_id") == "123"
    assert post.extra_metadata.get("verified") is True



