"""Automated tests for Multi-Dimensional Sentiment, Emotion, Sarcasm Inversion, and Temporal Tracking."""

from datetime import datetime, timezone, timedelta
import polars as pl
import pytest

from hypesignal.models.enums import EmotionType, SentimentPolarity, StanceType
from hypesignal.nlp.engine import MultiDimensionalSentimentEngine
from hypesignal.nlp.evaluation.evaluator import ModelBenchmarkEvaluator
from hypesignal.nlp.temporal_sentiment import TemporalSentimentTracker


@pytest.fixture(scope="module")
def nlp_engine():
    """Module-scoped shared NLP engine instance for test suite."""
    return MultiDimensionalSentimentEngine()


def test_genuine_sentiment_and_emotions(nlp_engine: MultiDimensionalSentimentEngine):
    """Verify standard sentiment polarity and nuanced emotion detection."""
    texts = [
        "I am so thrilled and ecstatic about the breakthrough research!",
        "The terrible service and rude staff made me completely furious.",
        "The package arrived on Tuesday as scheduled.",
    ]
    results = nlp_engine.analyze_multidimensional(texts)
    assert len(results) == 3

    # Positive sample
    assert results[0].effective_polarity == SentimentPolarity.POSITIVE
    assert results[0].emotion.primary_emotion in [EmotionType.JOY, EmotionType.SURPRISE]
    assert results[0].is_sarcasm_inverted is False

    # Negative sample
    assert results[1].effective_polarity == SentimentPolarity.NEGATIVE
    assert results[1].emotion.primary_emotion in [EmotionType.ANGER, EmotionType.DISGUST, EmotionType.SADNESS]

    # Neutral sample
    assert results[2].sentiment.polarity == SentimentPolarity.NEUTRAL


def test_sarcasm_inversion_detection(nlp_engine: MultiDimensionalSentimentEngine):
    """Verify sarcasm inversion: positive surface sentiment with high irony inverts to negative."""
    sarcastic_tweet = "Oh wonderful, my flight is delayed by another 6 hours! Just what I needed!"
    result = nlp_engine.analyze_single(sarcastic_tweet, irony_threshold=0.85)

    # Surface sentiment was positive
    assert result.sentiment.polarity == SentimentPolarity.POSITIVE
    # Irony detected
    assert result.irony.irony_score >= 0.85
    # Effective polarity successfully inverted to NEGATIVE
    assert result.effective_polarity == SentimentPolarity.NEGATIVE
    assert result.is_sarcasm_inverted is True
    assert result.adjusted_sentiment_score < 0.0
    assert result.explanation is not None
    assert "Inverted to NEGATIVE" in result.explanation


def test_temporal_sentiment_tracker(nlp_engine: MultiDimensionalSentimentEngine):
    """Verify temporal aggregation of sentiment, emotions, and sarcasm over time buckets."""
    base_time = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    
    # Bucket 1: 10:00 - 11:00 (Mostly happy)
    # Bucket 2: 12:00 - 13:00 (Mostly frustrated/sarcastic)
    records = [
        {"id": "p1", "text": "Incredible achievements by the team today! So happy!", "timestamp": base_time + timedelta(minutes=10)},
        {"id": "p2", "text": "Delighted to announce our new launch!", "timestamp": base_time + timedelta(minutes=20)},
        {"id": "p3", "text": "Oh fantastic, the server crashed during peak hours. Brilliant!", "timestamp": base_time + timedelta(hours=2, minutes=5)},
        {"id": "p4", "text": "Total outage and data loss, unacceptable disaster.", "timestamp": base_time + timedelta(hours=2, minutes=15)},
    ]
    df = pl.DataFrame(records)

    tracker = TemporalSentimentTracker(engine=nlp_engine)
    timeline_df = tracker.compute_timeline_fluctuations(df, interval="1 hour")

    assert len(timeline_df) == 2
    assert "bucket" in timeline_df.columns
    assert "mean_sentiment_score" in timeline_df.columns
    assert "sarcasm_rate" in timeline_df.columns
    assert "joy_ratio" in timeline_df.columns

    # Bucket 1 should have positive sentiment score
    assert timeline_df["mean_sentiment_score"][0] > 0.0
    assert timeline_df["positive_ratio"][0] > 0.5

    # Bucket 2 should have negative sentiment score and higher sarcasm rate
    assert timeline_df["mean_sentiment_score"][1] < 0.0
    assert timeline_df["negative_ratio"][1] > 0.5


def test_benchmark_evaluator_on_tweeteval(nlp_engine: MultiDimensionalSentimentEngine):
    """Verify model evaluator calculates valid metrics against TweetEval test splits."""
    evaluator = ModelBenchmarkEvaluator(engine=nlp_engine)

    # Evaluate sentiment on first 15 test samples
    sent_metrics = evaluator.evaluate_sentiment(split="test", limit=15)
    assert sent_metrics["task"] == "sentiment"
    assert sent_metrics["total_samples"] == 15
    assert 0.0 <= sent_metrics["accuracy"] <= 1.0
    assert 0.0 <= sent_metrics["macro_f1"] <= 1.0

    # Evaluate irony on first 15 test samples
    irony_metrics = evaluator.evaluate_irony(split="test", limit=15)
    assert irony_metrics["task"] == "irony"
    assert irony_metrics["total_samples"] == 15
    assert 0.0 <= irony_metrics["accuracy"] <= 1.0


def test_stance_allowlist_and_fallback(nlp_engine: MultiDimensionalSentimentEngine):
    """Verify stance classification with allowlisted target and safe fallback."""
    texts = [
        "A woman has the constitutional right to choose and control her own body!",
        "Abortion is murder and must be banned completely.",
    ]

    # Supported target: 'abortion'
    abortion_res = nlp_engine.analyze_stance_batch(texts, target="abortion")
    assert len(abortion_res) == 2
    assert abortion_res[0].stance == StanceType.SUPPORTIVE
    assert abortion_res[1].stance == StanceType.AGAINST

    # Unknown target: should safely fallback to default without raising unhandled exceptions
    fallback_res = nlp_engine.analyze_stance_batch(texts, target="quantum_computing")
    assert len(fallback_res) == 2
    assert fallback_res[0].stance in [StanceType.SUPPORTIVE, StanceType.AGAINST, StanceType.NONE]

