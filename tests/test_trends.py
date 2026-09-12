"""Comprehensive test suite for trend and dynamic topic detection engine (Component D)."""

from datetime import datetime, timedelta, timezone
import pytest

from hypesignal.models.canonical import CanonicalPost
from hypesignal.models.enums import PlatformType
from hypesignal.storage.duckdb_manager import DuckDBManager
from hypesignal.trends.burst_detector import BurstDetector
from hypesignal.trends.schemas import (
    BurstAlert,
    DynamicTopicTimeline,
    TopicRepresentation,
    TrendOverview,
    TrendRankingResult,
)
from hypesignal.trends.topic_modeler import DynamicTopicModeler
from hypesignal.trends.trend_ranker import TrendRanker
from hypesignal.trends.trends_engine import TrendsEngine


@pytest.fixture(scope="module")
def burst_detector():
    return BurstDetector(default_z_threshold=2.5, default_min_count=3)


@pytest.fixture(scope="module")
def topic_modeler():
    return DynamicTopicModeler(device="cpu", min_topic_size=2)


@pytest.fixture(scope="module")
def trend_ranker():
    return TrendRanker(weight_burst=0.5, weight_sentiment=0.25, weight_diversity=0.25)


@pytest.fixture(scope="module")
def trends_engine(burst_detector, topic_modeler, trend_ranker):
    return TrendsEngine(
        burst_detector=burst_detector,
        topic_modeler=topic_modeler,
        trend_ranker=trend_ranker,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# 1. Burst Detector Tests
# ---------------------------------------------------------------------------

def test_burst_metrics_computation(burst_detector):
    # Steady baseline: [10, 10, 10, 10, 10], current count: 10 -> Z = 0
    mean, std, z, vel = burst_detector.compute_burst_metrics(
        baseline_counts=[10, 10, 10, 10, 10],
        current_count=10,
        window_duration_seconds=3600,
    )
    assert mean == 10.0
    assert std == 0.0
    assert z == 0.0
    assert vel == 0.0

    # Low baseline: [1, 2, 1, 2, 1], current count: 20 -> Huge burst
    mean_b, std_b, z_b, vel_b = burst_detector.compute_burst_metrics(
        baseline_counts=[1, 2, 1, 2, 1],
        current_count=20,
        window_duration_seconds=3600,
    )
    assert round(mean_b, 1) == 1.4
    assert z_b > 10.0
    assert vel_b > 15.0


def test_term_extraction_from_post(burst_detector):
    text = "Breaking announcement: #OpenAI announces new #AI model for developers! @sama"
    terms = burst_detector.extract_terms_from_post(text, target="all")
    term_dict = dict(terms)

    assert "#openai" in term_dict
    assert term_dict["#openai"] == "hashtag"
    assert "#ai" in term_dict
    assert "breaking" in term_dict
    assert term_dict["breaking"] == "keyword"
    assert "announcement" in term_dict
    # Stopwords should be omitted
    assert "for" not in term_dict
    assert "new" not in term_dict


def test_detect_bursts_from_events(burst_detector):
    now = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
    events = []

    # 5 baseline intervals (each 1 hour)
    for h in range(1, 6):
        ts = now - timedelta(hours=h, minutes=30)
        events.extend([(ts, "#breaking", "hashtag") for _ in range(2)])
        events.extend([(ts, "#steady", "hashtag") for _ in range(15)])

    # Current window (last 1 hour): surge for #breaking, steady for #steady
    events.extend([(now - timedelta(minutes=15), "#breaking", "hashtag") for _ in range(35)])
    events.extend([(now - timedelta(minutes=20), "#steady", "hashtag") for _ in range(15)])

    # Rare single word (below min_count=3)
    events.append((now - timedelta(minutes=10), "#rare", "hashtag"))

    alerts = burst_detector.detect_bursts_from_events(
        events=events,
        current_window_end=now,
        window_size=timedelta(hours=1),
        num_baseline_windows=5,
    )

    alert_map = {a.term: a for a in alerts}
    assert "#breaking" in alert_map
    assert alert_map["#breaking"].is_burst is True
    assert alert_map["#breaking"].z_score > 15.0

    assert "#steady" in alert_map
    assert alert_map["#steady"].is_burst is False
    assert alert_map["#steady"].z_score < 1.0

    assert "#rare" in alert_map
    assert alert_map["#rare"].is_burst is False  # filtered out by min_count=3


def test_detect_bursts_from_duckdb(burst_detector):
    db = DuckDBManager(":memory:")
    now = datetime(2026, 9, 12, 18, 0, tzinfo=timezone.utc)
    posts = []

    # Historical baseline posts
    for h in range(1, 6):
        ts = now - timedelta(hours=h, minutes=30)
        for i in range(2):
            posts.append(
                CanonicalPost(
                    id=f"p_base_{h}_{i}",
                    platform=PlatformType.TWITTER,
                    author_id=f"u_{i}",
                    text="Just testing things #surge",
                    hashtags=["surge"],
                    timestamp=ts,
                )
            )

    # Current window surge
    for i in range(25):
        posts.append(
            CanonicalPost(
                id=f"p_curr_{i}",
                platform=PlatformType.TWITTER,
                author_id=f"u_{i}",
                text="Massive news happening now #surge",
                hashtags=["surge"],
                timestamp=now - timedelta(minutes=20),
            )
        )

    db.insert_posts(posts)

    alerts = burst_detector.detect_bursts_from_duckdb(
        db=db,
        current_window_end=now,
        window_duration_minutes=60,
        num_baseline_windows=5,
    )

    assert len(alerts) > 0
    surge_alert = next((a for a in alerts if a.term == "#surge"), None)
    assert surge_alert is not None
    assert surge_alert.is_burst is True
    assert surge_alert.current_count == 25
    assert surge_alert.baseline_mean == 2.0


# ---------------------------------------------------------------------------
# 2. Dynamic Topic Modeler Tests
# ---------------------------------------------------------------------------

def test_topic_modeler_clustering(topic_modeler):
    docs = [
        "Bitcoin surges to all-time high as cryptocurrency volume rallies.",
        "Ethereum and BTC lead the crypto market gainers today.",
        "Bitcoin ETF inflows hit record volume on institutional exchanges.",
        "Crypto traders celebrate massive price breakouts.",
        "Federal Reserve cuts interest rates amid slowing inflation data.",
        "Central bank announces economic policy and interest rate decisions.",
        "Inflation eases as economy shows steady job market growth.",
        "Federal reserve chairman speaks on fiscal policy and inflation.",
        "NASA announces new mission to explore Mars craters with rovers.",
        "Astronauts prepare for space station mission with rocket technology.",
        "James Webb space telescope captures stunning distant galaxies.",
        "Space rocket launch succeeds in placing satellites into orbit.",
    ]

    timestamps = [
        datetime(2026, 9, 1, tzinfo=timezone.utc) + timedelta(days=i)
        for i in range(len(docs))
    ]

    topics, timelines = topic_modeler.fit_topics(docs=docs, timestamps=timestamps, nr_bins=3)

    assert len(topics) >= 2
    # Check that top words exist
    for top in topics:
        assert top.doc_count >= 2
        assert len(top.top_words) > 0
        assert len(top.representative_docs) > 0

    assert timelines is not None
    assert len(timelines) >= 1
    for timeline in timelines:
        assert len(timeline.timestamps) > 0
        assert len(timeline.frequencies) > 0


def test_topic_modeler_micro_corpus_fallback(topic_modeler):
    tiny_docs = ["Single post about tech", "Another tiny post"]
    topics, timelines = topic_modeler.fit_topics(tiny_docs)

    assert len(topics) == 1
    assert topics[0].topic_id == 0
    assert timelines is None


def test_topic_modeler_from_duckdb(topic_modeler):
    db = DuckDBManager(":memory:")
    now = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
    docs = [
        "Bitcoin and crypto trading",
        "Ethereum cryptocurrency blockchain",
        "BTC crypto wallet transfer",
        "Deep space telescope mission",
        "Astronaut rocket orbit launch",
        "NASA space satellite exploration",
    ]
    posts = [
        CanonicalPost(
            id=f"tp_{i}",
            platform=PlatformType.TWITTER,
            author_id=f"u_{i}",
            text=d,
            timestamp=now - timedelta(hours=i),
        )
        for i, d in enumerate(docs)
    ]
    db.insert_posts(posts)

    topics, _ = topic_modeler.fit_from_duckdb(db, limit=100)
    assert len(topics) >= 1
    assert hasattr(topic_modeler, "last_posts_by_topic")
    assert len(topic_modeler.last_posts_by_topic) >= 1


def test_topic_modeler_small_corpus_no_empty_topics():
    """Reproduce reviewer case: 6-post corpus with min_topic_size=3 must not yield empty topics."""
    modeler = DynamicTopicModeler(device="cpu", min_topic_size=3)
    docs = [
        "Bitcoin rallies to all-time high amid huge crypto trading volume.",
        "Ethereum breaks resistance level as cryptocurrency market booms.",
        "Crypto investors celebrate massive gains in BTC and ETH.",
        "Federal Reserve decides to lower benchmark interest rates.",
        "Central bank policy meeting addresses inflation and employment data.",
        "Fed interest rate cut expected to stimulate economic growth.",
    ]
    topics, _ = modeler.fit_topics(docs)
    assert len(topics) >= 1
    assert all(t.doc_count > 0 for t in topics)
    assert all(len(t.top_words) > 0 for t in topics)


# ---------------------------------------------------------------------------
# 3. Trend Ranker Tests
# ---------------------------------------------------------------------------

def test_participant_diversity(trend_ranker):
    # 10 posts from 10 distinct authors -> 1.0 (organic)
    div_high = trend_ranker.compute_participant_diversity([f"user_{i}" for i in range(10)])
    assert div_high == 1.0

    # 10 posts from 1 author -> 0.1 (spam/bot)
    div_low = trend_ranker.compute_participant_diversity(["spammer_bot"] * 10)
    assert round(div_low, 2) == 0.1

    # Empty list
    assert trend_ranker.compute_participant_diversity([]) == 0.0


def test_rank_burst_alerts(trend_ranker):
    now = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
    b1 = BurstAlert(
        term="#breaking",
        term_type="hashtag",
        current_count=50,
        baseline_mean=2.0,
        baseline_std=1.0,
        z_score=15.0,
        velocity=48.0,
        window_start=now - timedelta(hours=1),
        window_end=now,
        is_burst=True,
    )
    b2 = BurstAlert(
        term="#mild",
        term_type="hashtag",
        current_count=10,
        baseline_mean=2.0,
        baseline_std=1.0,
        z_score=3.0,
        velocity=8.0,
        window_start=now - timedelta(hours=1),
        window_end=now,
        is_burst=True,
    )

    # Posts with high diversity for #breaking (50 unique authors)
    posts_b1 = [
        CanonicalPost(id=f"p1_{i}", platform=PlatformType.TWITTER, author_id=f"auth_{i}", text="#breaking news", timestamp=now)
        for i in range(50)
    ]
    # Posts with low diversity for #mild (1 bot)
    posts_b2 = [
        CanonicalPost(id=f"p2_{i}", platform=PlatformType.TWITTER, author_id="solo_bot", text="#mild news", timestamp=now)
        for i in range(10)
    ]

    ranked = trend_ranker.rank_burst_alerts(
        alerts=[b1, b2],
        posts_by_term={"#breaking": posts_b1, "#mild": posts_b2},
        sentiment_scores={"#breaking": 0.8, "#mild": 0.4},
    )

    assert len(ranked) == 2
    # #breaking should rank significantly higher than #mild
    assert ranked[0].name == "#breaking"
    assert ranked[0].composite_score > ranked[1].composite_score
    assert ranked[0].participant_diversity == 1.0
    assert ranked[1].participant_diversity == 0.1


def test_rank_all_combination(trend_ranker):
    r1 = TrendRankingResult(
        trend_id="1", name="#surge", composite_score=0.92,
        total_volume=100, unique_authors=90,
    )
    r2 = TrendRankingResult(
        trend_id="2", name="crypto_market", composite_score=0.75,
        total_volume=80, unique_authors=70,
    )
    combined = trend_ranker.rank_all([r1], [r2])
    assert combined[0].composite_score == 0.92
    assert combined[1].composite_score == 0.75


# ---------------------------------------------------------------------------
# 4. Trends Engine End-to-End Tests
# ---------------------------------------------------------------------------

def test_trends_engine_end_to_end(trends_engine):
    db = DuckDBManager(":memory:")
    now = datetime(2026, 9, 12, 15, 0, tzinfo=timezone.utc)
    posts = []

    # Insert baseline posts (5 hours)
    for h in range(1, 6):
        ts = now - timedelta(hours=h, minutes=30)
        for i in range(3):
            posts.append(
                CanonicalPost(
                    id=f"base_{h}_{i}",
                    platform=PlatformType.TWITTER,
                    author_id=f"user_{i}",
                    text="Regular daily updates on technology",
                    hashtags=["tech"],
                    timestamp=ts,
                )
            )

    # Insert current surge in window [14:00, 15:00]
    for i in range(30):
        posts.append(
            CanonicalPost(
                id=f"surge_{i}",
                platform=PlatformType.TWITTER,
                author_id=f"participant_{i}",
                text="Massive AI breakthrough announced today! #AIBreakthrough #innovation",
                hashtags=["AIBreakthrough", "innovation"],
                timestamp=now - timedelta(minutes=20),
            )
        )

    db.insert_posts(posts)

    overview = trends_engine.analyze_trends(
        db=db,
        current_window_end=now,
        window_duration_minutes=60,
        num_baseline_windows=5,
    )

    assert isinstance(overview, TrendOverview)
    assert len(overview.active_bursts) >= 1
    # Verify #aibreakthrough was caught as an active burst
    burst_terms = [b.term.lower() for b in overview.active_bursts]
    assert any("aibreakthrough" in t for t in burst_terms)

    assert len(overview.ranked_trends) >= 1
    top_trend = overview.ranked_trends[0]
    assert top_trend.composite_score > 0.0
    assert top_trend.participant_diversity > 0.0
