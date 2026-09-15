"""Unit and integration tests for Step 4: Predictive trend velocity & narrative drift."""

from datetime import datetime, timedelta, timezone
import pytest

from hypesignal.models.canonical import CanonicalPost
from hypesignal.storage.duckdb_manager import DuckDBManager
from hypesignal.trends.narrative_drift import NarrativeDriftAlert, NarrativeDriftTracker
from hypesignal.trends.trend_forecaster import (
    TrendForecast,
    TrendForecaster,
    TrendLifecycleState,
)


@pytest.fixture
def mem_db():
    """In-memory DuckDB manager fixture."""
    db = DuckDBManager(":memory:")
    yield db
    db.close()


@pytest.fixture(scope="module")
def drift_tracker():
    """Shared NarrativeDriftTracker on CPU."""
    return NarrativeDriftTracker(device="cpu", drift_threshold=0.25, inversion_threshold=0.35)


def test_trend_forecaster_velocity_and_acceleration():
    """Test 1st derivative (velocity) and 2nd derivative (acceleration) calculations."""
    forecaster = TrendForecaster(velocity_threshold=10.0, acceleration_threshold=5.0)

    # 3 hourly windows: 10, 30, 90 posts
    volumes = [10, 30, 90]
    forecast = forecaster.forecast_from_volumes("ai_agent", volumes=volumes, window_duration_minutes=60.0)

    assert forecast.term == "ai_agent"
    assert forecast.total_volume == 130
    assert forecast.window_volumes == [10, 30, 90]

    # Velocity: v(t-1) = 30/60 = 0.5, v(t) = 90/60 = 1.5
    assert forecast.previous_velocity == pytest.approx(0.5, abs=0.01)
    assert forecast.current_velocity == pytest.approx(1.5, abs=0.01)

    # Acceleration: a(t) = (1.5 - 0.5) / 60 = 0.0167 posts/min^2
    assert forecast.acceleration == pytest.approx(0.0167, abs=0.005)
    assert forecast.virality_potential_score > 0.0
    assert forecast.user_diversity_ratio == 1.0


def test_trend_lifecycle_state_transitions():
    """Test lifecycle classification across all trajectory regimes."""
    # Using window_duration_minutes=1.0 so volume equals velocity directly for intuitive testing
    forecaster = TrendForecaster(
        velocity_threshold=10.0,
        acceleration_threshold=5.0,
        viral_surge_velocity_multiplier=2.5,  # 25.0
        viral_surge_accel_multiplier=2.0,     # 10.0
    )

    # 1. Dormant
    fc_dormant = forecaster.forecast_from_volumes("dormant_topic", [0, 1, 1], window_duration_minutes=1.0)
    assert fc_dormant.lifecycle_state == TrendLifecycleState.DORMANT

    # 2. Emerging: velocity above 10, positive acceleration
    fc_emerging = forecaster.forecast_from_volumes("emerging_topic", [5, 12, 20], window_duration_minutes=1.0)
    assert fc_emerging.lifecycle_state == TrendLifecycleState.EMERGING

    # 3. Viral Surge: velocity >= 25, steep acceleration > 10
    fc_surge = forecaster.forecast_from_volumes("viral_topic", [10, 40, 95], window_duration_minutes=1.0)
    assert fc_surge.lifecycle_state == TrendLifecycleState.VIRAL_SURGE
    assert fc_surge.virality_potential_score > 0.70

    # 4. Peaking: was at high volume, acceleration has flattened/turned slightly negative
    fc_peaking = forecaster.forecast_from_volumes("peaking_topic", [40, 95, 93], window_duration_minutes=1.0)
    assert fc_peaking.lifecycle_state == TrendLifecycleState.PEAKING

    # 5. Decelerating: sharp drop in volume and negative acceleration
    fc_decel = forecaster.forecast_from_volumes("dying_topic", [95, 60, 20], window_duration_minutes=1.0)
    assert fc_decel.lifecycle_state == TrendLifecycleState.DECELERATING

    # 6. Stable: relatively constant velocity
    fc_stable = forecaster.forecast_from_volumes("stable_topic", [50, 52, 51], window_duration_minutes=1.0)
    assert fc_stable.lifecycle_state == TrendLifecycleState.STABLE


def test_trend_forecaster_zero_volume_and_hourly_scale():
    """Test zero volume virality score (m2) and multi-minute hourly window scaling (m1)."""
    forecaster = TrendForecaster(
        velocity_threshold=1.0,      # 1 post/min = 60 posts/hr
        acceleration_threshold=0.5,  # 0.5 post/min delta = 30 posts/hr delta
        viral_surge_velocity_multiplier=2.5,  # 2.5 posts/min = 150 posts/hr
        viral_surge_accel_multiplier=2.0,     # 1.0 post/min delta = 60 posts/hr delta
    )

    # 1. Zero volume edge case (m2): virality potential score must be exactly 0.0
    fc_zero = forecaster.forecast_from_volumes("zero_topic", [0, 0, 0], window_duration_minutes=60.0)
    assert fc_zero.total_volume == 0
    assert fc_zero.virality_potential_score == 0.0
    assert fc_zero.user_diversity_ratio == 0.0
    assert fc_zero.lifecycle_state == TrendLifecycleState.DORMANT

    # 2. Realistic 60-minute windows (m1):
    # Hour 1: 30 posts (v = 0.5 posts/min)
    # Hour 2: 90 posts (v = 1.5 posts/min, delta_v = +1.0)
    # Hour 3: 240 posts (v = 4.0 posts/min, delta_v = +2.5 >= viral_surge_accel)
    fc_hourly_surge = forecaster.forecast_from_volumes("surge_hourly", [30, 90, 240], window_duration_minutes=60.0)
    assert fc_hourly_surge.lifecycle_state == TrendLifecycleState.VIRAL_SURGE
    assert fc_hourly_surge.virality_potential_score > 0.60

    # Decelerating hourly: 240 -> 180 -> 60
    fc_hourly_decel = forecaster.forecast_from_volumes("decel_hourly", [240, 180, 60], window_duration_minutes=60.0)
    assert fc_hourly_decel.lifecycle_state == TrendLifecycleState.DECELERATING


def test_trend_forecaster_from_duckdb(mem_db):
    """Test DuckDB sliding window querying and forecasting."""
    t0 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=60)
    t2 = t1 + timedelta(minutes=60)
    t3 = t2 + timedelta(minutes=60)

    posts = []
    # Window 1 (10:00 - 11:00): 5 posts
    for i in range(5):
        posts.append(CanonicalPost(id=f"w1_{i}", author_id=f"u_{i}", text="Studying quantum algorithms.", timestamp=t0 + timedelta(minutes=i*10)))

    # Window 2 (11:00 - 12:00): 15 posts
    for j in range(15):
        posts.append(CanonicalPost(id=f"w2_{j}", author_id=f"u_{j+10}", text="Major quantum supremacy milestone!", timestamp=t1 + timedelta(minutes=j*3)))

    # Window 3 (12:00 - 13:00): 45 posts
    for k in range(45):
        posts.append(CanonicalPost(id=f"w3_{k}", author_id=f"u_{k+30}", text="Quantum computing breakthrough is insane! #quantum", timestamp=t2 + timedelta(minutes=k)))

    mem_db.insert_posts(posts)

    forecaster = TrendForecaster(velocity_threshold=0.1, acceleration_threshold=0.01)
    forecast = forecaster.forecast_from_db(
        db=mem_db,
        term="quantum",
        window_minutes=60,
        num_windows=3,
        reference_time=t3,
    )

    assert forecast.term == "quantum"
    assert forecast.total_volume == 65
    assert forecast.window_volumes == [5, 15, 45]
    assert forecast.current_velocity > forecast.previous_velocity
    assert forecast.acceleration > 0.0
    assert forecast.lifecycle_state in (TrendLifecycleState.VIRAL_SURGE, TrendLifecycleState.EMERGING)


def test_narrative_drift_semantic_cosine_shift(drift_tracker):
    """Test cosine semantic distance calculation on topic evolution."""
    # Cohesive baseline
    texts_baseline = [
        "Decentralized storage nodes and cryptographic file encryption.",
        "Peer-to-peer data distribution and IPFS pinning architecture.",
    ]

    # Semantically aligned continuation
    texts_aligned = [
        "P2P decentralized node hosting and content-addressed storage proofs.",
        "Distributed file systems and blockchain verified data retrieval.",
    ]

    # Radical semantic drift (drifted from tech storage to celebrity red carpet)
    texts_drifted = [
        "Celebrity red carpet gossip and Hollywood movie awards ceremony drama.",
        "Fashion gala designer dresses and red carpet celebrity interviews.",
    ]

    # 1. Aligned corpus: low drift
    alert_aligned = drift_tracker.track_drift("storage", texts_baseline, texts_aligned)
    assert alert_aligned.cosine_drift < 0.25
    assert alert_aligned.is_semantic_drift is False
    assert alert_aligned.alert_triggered is False

    # 2. Drifted corpus: high drift triggering alert
    alert_drifted = drift_tracker.track_drift("storage", texts_baseline, texts_drifted)
    assert alert_drifted.cosine_drift >= 0.25
    assert alert_drifted.is_semantic_drift is True
    assert alert_drifted.alert_triggered is True


def test_narrative_drift_sentiment_inversion(drift_tracker):
    """Test sentiment inversion detection on polarity flips."""
    # 1. Net positive flipping to net negative
    s_before = [0.85, 0.70, 0.65]  # mean ~ 0.73
    s_after = [-0.75, -0.80, -0.60]  # mean ~ -0.72

    alert_inversion = drift_tracker.track_drift(
        topic="OpenProtocol",
        texts_before=["OpenProtocol is revolutionary and awesome!"],
        texts_after=["OpenProtocol is buggy, broken, and completely crashed!"],
        sentiments_before=s_before,
        sentiments_after=s_after,
    )

    assert alert_inversion.is_sentiment_inversion is True
    assert alert_inversion.sentiment_delta < -1.0
    assert alert_inversion.alert_triggered is True

    # 2. Stable positive sentiment: no inversion
    s_steady = [0.70, 0.65, 0.60]
    alert_steady = drift_tracker.track_drift(
        topic="OpenProtocol",
        texts_before=["Great progress today."],
        texts_after=["Continued solid engineering work."],
        sentiments_before=s_before,
        sentiments_after=s_steady,
    )
    assert alert_steady.is_sentiment_inversion is False


def test_narrative_drift_cold_start_and_empty_windows(drift_tracker):
    """Test cold-start baseline (M2) and empty window data (M3) do not trigger false alerts."""
    # 1. Cold start (M2): topic has no posts in window 1, but emerges with positive sentiment in window 2
    alert_cold_start = drift_tracker.track_drift(
        topic="BrandNewTopic",
        texts_before=[],
        texts_after=["BrandNewTopic is gaining rapid traction and community support!"],
        sentiments_before=[],
        sentiments_after=[0.65],
    )
    # Must NOT trigger sentiment inversion or semantic drift
    assert alert_cold_start.is_sentiment_inversion is False
    assert alert_cold_start.is_semantic_drift is False
    assert alert_cold_start.alert_triggered is False
    assert alert_cold_start.details.get("insufficient_data") is True

    # 2. Vanishing topic (M3): topic existed in window 1 but 0 posts in window 2
    alert_empty_after = drift_tracker.track_drift(
        topic="VanishingTopic",
        texts_before=["VanishingTopic was active yesterday."],
        texts_after=[],
        sentiments_before=[0.40],
        sentiments_after=[],
    )
    assert alert_empty_after.cosine_drift == 0.0
    assert alert_empty_after.is_semantic_drift is False
    assert alert_empty_after.is_sentiment_inversion is False
    assert alert_empty_after.alert_triggered is False
    assert alert_empty_after.details.get("insufficient_data") is True


def test_narrative_drift_from_duckdb(drift_tracker, mem_db):
    """Test integration: DuckDB posts & post_analytics -> NarrativeDriftTracker."""
    t0 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=1)
    t2 = t1 + timedelta(hours=1)

    # Window 1: enthusiastic launch
    w1_posts = [
        CanonicalPost(id="p_w1_1", author_id="u_a", text="Huge launch of ProductX! Absolutely thrilling!", timestamp=t0 + timedelta(minutes=10)),
        CanonicalPost(id="p_w1_2", author_id="u_b", text="ProductX is fantastic, loving the smooth UI.", timestamp=t0 + timedelta(minutes=20)),
    ]
    w1_analytics = [
        {"post_id": "p_w1_1", "effective_polarity": "positive", "sentiment_score": 0.85},
        {"post_id": "p_w1_2", "effective_polarity": "positive", "sentiment_score": 0.90},
    ]

    # Window 2: critical flaw backlash
    w2_posts = [
        CanonicalPost(id="p_w2_1", author_id="u_c", text="ProductX just leaked private user credentials! Severe exploit!", timestamp=t1 + timedelta(minutes=10)),
        CanonicalPost(id="p_w2_2", author_id="u_d", text="Boycott ProductX immediately, total security disaster.", timestamp=t1 + timedelta(minutes=20)),
    ]
    w2_analytics = [
        {"post_id": "p_w2_1", "effective_polarity": "negative", "sentiment_score": 0.90},
        {"post_id": "p_w2_2", "effective_polarity": "negative", "sentiment_score": 0.95},
    ]

    mem_db.insert_posts(w1_posts + w2_posts)
    mem_db.upsert_post_analytics(w1_analytics + w2_analytics)

    alert = drift_tracker.track_drift_from_db(
        db=mem_db,
        topic="ProductX",
        window_1_start=t0,
        window_1_end=t1,
        window_2_start=t1,
        window_2_end=t2,
    )

    assert alert.topic == "ProductX"
    assert alert.sentiment_before > 0.5
    assert alert.sentiment_after < -0.5
    assert alert.is_sentiment_inversion is True
    assert alert.alert_triggered is True
