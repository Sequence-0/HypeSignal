"""Unit tests for HypeSignal canonical data models."""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from hypesignal.models import (
    CanonicalCascadeEvent,
    CanonicalGraphEdge,
    CanonicalPost,
    CanonicalUser,
    EmotionType,
    GeoCoordinates,
    PlatformType,
    PostMetrics,
    RelationType,
    SentimentPolarity,
    StanceType,
    UserMetrics,
)


def test_geo_coordinates_parsing():
    """Test parsing GPS coordinates from raw strings."""
    # Standard format with UT prefix (Cheng-Caverlee-Lee test set)
    c1 = GeoCoordinates.from_raw_string("UT: 43.009815,-83.710408")
    assert c1 is not None
    assert pytest.approx(c1.latitude, 1e-5) == 43.009815
    assert pytest.approx(c1.longitude, 1e-5) == -83.710408

    # Comma with space
    c2 = GeoCoordinates.from_raw_string("37.7749, -122.4194")
    assert c2 is not None
    assert pytest.approx(c2.latitude, 1e-4) == 37.7749
    assert pytest.approx(c2.longitude, 1e-4) == -122.4194

    # Southern & Eastern hemisphere (e.g. Sydney)
    c3 = GeoCoordinates.from_raw_string("UT: -33.8688,151.2093")
    assert c3 is not None
    assert pytest.approx(c3.latitude, 1e-4) == -33.8688
    assert pytest.approx(c3.longitude, 1e-4) == 151.2093

    # Non-coordinate text, whitespace, and empty strings should return None
    assert GeoCoordinates.from_raw_string("San Francisco, CA") is None
    assert GeoCoordinates.from_raw_string("") is None
    assert GeoCoordinates.from_raw_string("   ") is None
    assert GeoCoordinates.from_raw_string(None) is None

    # Invalid latitude/longitude boundary violations
    assert GeoCoordinates.from_raw_string("150.0, -120.0") is None
    assert GeoCoordinates.from_raw_string("-91.0, 50.0") is None
    assert GeoCoordinates.from_raw_string("45.0, 185.0") is None


def test_canonical_post_timestamp_sync():
    """Test CanonicalPost automatic timestamp timezone and millisecond synchronization."""
    naive_dt = datetime(2010, 3, 15, 12, 0, 0)
    post = CanonicalPost(
        id="tweet_123",
        author_id="user_456",
        author_screen_name="test_author",
        text="Testing canonical post normalization #AI",
        timestamp=naive_dt,
        hashtags=["AI"],
        metrics=PostMetrics(likes=10, reposts=2),
    )

    assert post.timestamp.tzinfo is not None
    assert post.timestamp.tzinfo == timezone.utc
    assert post.timestamp_ms == int(post.timestamp.timestamp() * 1000)
    assert post.platform == PlatformType.TWITTER
    assert post.metrics.likes == 10


def test_canonical_user_auto_coords():
    """Test CanonicalUser automatic coordinate population from raw location string."""
    # User with GPS coordinates in location string
    user_with_gps = CanonicalUser(
        id="user_gps",
        screen_name="gps_user",
        location_raw="UT: 34.0522,-118.2437",
        indegree=150,
        outdegree=50,
    )
    assert user_with_gps.location_coords is not None
    assert pytest.approx(user_with_gps.location_coords.latitude, 1e-4) == 34.0522
    assert pytest.approx(user_with_gps.location_coords.longitude, 1e-4) == -118.2437

    # User with textual city location
    user_with_city = CanonicalUser(
        id="user_city",
        screen_name="city_user",
        location_raw="Chicago, IL",
        indegree=200,
        outdegree=180,
    )
    assert user_with_city.location_coords is None
    assert user_with_city.location_raw == "Chicago, IL"


def test_canonical_graph_edge():
    """Test graph edge definition."""
    edge = CanonicalGraphEdge(
        source_id="follower_1",
        target_id="influencer_99",
        relation_type=RelationType.FOLLOWS,
        weight=1.0,
    )
    assert edge.source_id == "follower_1"
    assert edge.target_id == "influencer_99"
    assert edge.relation_type == RelationType.FOLLOWS


def test_canonical_cascade_event():
    """Test diffusion cascade event with epoch timestamp."""
    event = CanonicalCascadeEvent(
        cascade_id="http://is.gd/f4wDA",
        post_id="post_999",
        user_id="user_147240385",
        user_screen_name="barra_fake",
        timestamp=datetime.fromtimestamp(1285367458, tz=timezone.utc),
        timestamp_ms=1285367458000,
        adoption_order=3009,
    )
    assert event.cascade_id == "http://is.gd/f4wDA"
    assert event.adoption_order == 3009
    assert event.timestamp_ms == 1285367458000
