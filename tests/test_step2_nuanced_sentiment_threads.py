"""Unit and integration tests for Step 2: Nuanced emotion calibration, stance towards target, and thread sentiment dynamics."""

from datetime import datetime, timedelta, timezone
import pytest

from hypesignal.models.canonical import CanonicalPost, PostMetrics
from hypesignal.models.enums import EmotionType, PlatformType, SentimentPolarity, StanceType
from hypesignal.nlp.engine import MultiDimensionalSentimentEngine
from hypesignal.nlp.thread_sentiment import ThreadSentimentAnalyzer
from hypesignal.storage.duckdb_manager import DuckDBManager
from hypesignal.timeline.thread_manager import ConversationThreadManager


@pytest.fixture(scope="module")
def nlp_engine():
    """Shared NLP inference engine on CPU."""
    return MultiDimensionalSentimentEngine(device="cpu")


@pytest.fixture
def mem_db():
    """In-memory DuckDB manager fixture."""
    db = DuckDBManager(":memory:")
    yield db
    db.close()


def test_nuanced_emotion_anxiety_and_excitement(nlp_engine):
    """Test that lexical-arousal calibration cleanly isolates anxiety from fear and excitement from joy."""
    texts = [
        "I am panicking, terrified, and deeply anxious about the severe risks! Dread is setting in.",
        "I am so hyped, ecstatic, and thrilled about this unbelievable breakthrough! Absolutely epic! 🚀",
        "A venomous snake crawled into the dark room, terrifying danger!",
        "What a peaceful, pleasant, and lovely gentle sunny morning.",
    ]

    predictions = nlp_engine.analyze_nuanced_emotions_batch(texts)
    assert len(predictions) == 4

    # 1. Anxious text
    anx_pred = predictions[0]
    assert anx_pred.primary_emotion == EmotionType.ANXIETY
    assert anx_pred.probabilities["anxiety"] > 0.35
    assert sum(anx_pred.probabilities.values()) == pytest.approx(1.0, abs=0.01)

    # 2. Excitement text
    exc_pred = predictions[1]
    assert exc_pred.primary_emotion == EmotionType.EXCITEMENT
    assert exc_pred.probabilities["excitement"] > 0.35
    assert sum(exc_pred.probabilities.values()) == pytest.approx(1.0, abs=0.01)

    # 3. Fear text
    fear_pred = predictions[2]
    assert fear_pred.probabilities["fear"] > 0.05

    # 4. Joy text
    joy_pred = predictions[3]
    assert joy_pred.primary_emotion in (EmotionType.JOY, EmotionType.OPTIMISM, EmotionType.NEUTRAL)


def test_stance_towards_arbitrary_target(nlp_engine):
    """Test domain-agnostic stance evaluation towards explicit targets."""
    texts = [
        "I wholeheartedly support and believe in open-source decentralized protocols! Fantastic work.",
        "Centralized censorship and arbitrary account bans are completely terrible and harmful.",
        "The conference will take place on Wednesday at 10 AM UTC.",
    ]

    stances = nlp_engine.analyze_stance_towards_target(texts, target="open-source")
    assert len(stances) == 3

    # Positive text -> supportive
    assert stances[0].stance == StanceType.SUPPORTIVE
    assert stances[0].score > 0.5

    # Negative text -> against
    assert stances[1].stance == StanceType.AGAINST
    assert stances[1].score > 0.5

    # Neutral text -> neutral
    assert stances[2].stance == StanceType.NEUTRAL


def test_analyze_and_flatten_to_duckdb(nlp_engine, mem_db):
    """Test analyze_and_flatten output formatting and DuckDB post_analytics persistence."""
    texts = [
        "We are ecstatic and hyped to launch our new product today! #launch",
        "Honestly what a terrible, buggy and frustrating disaster.",
    ]
    post_ids = ["post_alpha", "post_beta"]

    flattened = nlp_engine.analyze_and_flatten(texts, post_ids=post_ids, target="product launch")
    assert len(flattened) == 2

    row_a = flattened[0]
    assert row_a["post_id"] == "post_alpha"
    assert row_a["effective_polarity"] == "positive"
    assert row_a["primary_emotion"] == "excitement"
    assert row_a["excitement"] > 0.0
    assert row_a["stance"] == "supportive"

    row_b = flattened[1]
    assert row_b["post_id"] == "post_beta"
    assert row_b["effective_polarity"] == "negative"
    assert row_b["primary_emotion"] in ("anger", "sadness", "disgust", "fear", "anxiety")
    assert row_b["stance"] == "against"

    # Insert into DuckDB post_analytics table
    mem_db.upsert_post_analytics(flattened)
    assert mem_db.get_post_analytics_count() == 2

    # Query back and verify
    stored_df = mem_db.get_post_analytics(["post_alpha", "post_beta"])
    assert len(stored_df) == 2
    row_alpha_stored = stored_df.filter(stored_df["post_id"] == "post_alpha").to_dicts()[0]
    assert row_alpha_stored["effective_polarity"] == "positive"
    assert row_alpha_stored["primary_emotion"] == "excitement"


def test_thread_sentiment_analyzer_dynamics(nlp_engine, mem_db):
    """Test conversation thread emotional arc, polarity drift, controversy index, and hostility velocity."""
    t0 = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=5)
    t2 = t0 + timedelta(minutes=10)
    t3 = t0 + timedelta(minutes=15)

    root = CanonicalPost(
        id="root_1",
        author_id="founder_1",
        text="We are ecstatic to unveil our new open autonomous agent architecture! 🚀",
        timestamp=t0,
    )
    replies = [
        CanonicalPost(
            id="rep_1",
            parent_id="root_1",
            author_id="fan_1",
            text="Incredible work! Absolutely hyped to test this out! 🔥",
            timestamp=t1,
        ),
        CanonicalPost(
            id="rep_2",
            parent_id="root_1",
            author_id="critic_1",
            text="I am deeply worried and anxious about the critical security flaws here.",
            timestamp=t2,
        ),
        CanonicalPost(
            id="rep_3",
            parent_id="root_1",
            author_id="troll_1",
            text="This is an absolute fraud and utter garbage! Horrible!",
            timestamp=t3,
        ),
    ]

    analyzer = ThreadSentimentAnalyzer(engine=nlp_engine)
    analysis = analyzer.analyze_thread(root, replies, target_topic="autonomous agent")

    # 1. Root assertions
    assert analysis.root_post_id == "root_1"
    assert analysis.root_polarity == "positive"
    assert analysis.root_sentiment_score > 0.5
    assert analysis.root_primary_emotion in ("excitement", "joy")

    # 2. Dynamics assertions
    # Comments include positive, anxious, and negative -> lower mean than enthusiastic root
    assert analysis.polarity_drift < 0.0  # Drifted downward
    assert analysis.controversy_index > 0.3  # Polarized/divergent comment sentiments
    assert analysis.hostility_velocity > 0.0  # Anger/anxiety comments present
    assert analysis.total_comments_analyzed == 3

    # Stance distribution
    assert analysis.supportive_ratio > 0.0
    assert analysis.against_ratio > 0.0

    # 3. Trajectory
    assert len(analysis.emotional_trajectory) == 4
    assert analysis.emotional_trajectory[0]["is_root"] is True
    assert analysis.emotional_trajectory[1]["is_root"] is False
    assert analysis.emotional_trajectory[1]["post_id"] == "rep_1"


def test_thread_sentiment_analyzer_from_db(nlp_engine, mem_db):
    """Test full integration: DuckDB posts -> ConversationThreadManager -> ThreadSentimentAnalyzer."""
    t0 = datetime(2026, 3, 1, 14, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=10)

    posts = [
        CanonicalPost(id="db_root", author_id="u0", text="Great achievement for our team!", timestamp=t0),
        CanonicalPost(id="db_rep1", parent_id="db_root", author_id="u1", text="Congratulations, well deserved!", timestamp=t1),
    ]
    mem_db.insert_posts(posts)

    thread_manager = ConversationThreadManager(db=mem_db)
    analyzer = ThreadSentimentAnalyzer(engine=nlp_engine)

    analysis = analyzer.analyze_thread_from_db("db_root", thread_manager=thread_manager)
    assert analysis is not None
    assert analysis.root_post_id == "db_root"
    assert analysis.total_comments_analyzed == 1
    assert analysis.supportive_ratio == 1.0
    assert analysis.polarity_drift >= -0.5

    # Non-existent thread returns None
    assert analyzer.analyze_thread_from_db("non_existent", thread_manager=thread_manager) is None
