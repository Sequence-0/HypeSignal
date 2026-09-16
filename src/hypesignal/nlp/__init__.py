"""Multi-dimensional sentiment, emotion, and temporal fluctuation engine."""

from hypesignal.nlp.engine import MultiDimensionalSentimentEngine
from hypesignal.nlp.evaluation.evaluator import ModelBenchmarkEvaluator
from hypesignal.nlp.schemas import (
    EmotionPrediction,
    IronyPrediction,
    MultiDimensionalResult,
    SentimentPrediction,
    StancePrediction,
)
from hypesignal.nlp.temporal_sentiment import TemporalSentimentTracker
from hypesignal.nlp.thread_sentiment import (
    ThreadSentimentAnalysis,
    ThreadSentimentAnalyzer,
)

__all__ = [
    "MultiDimensionalSentimentEngine",
    "TemporalSentimentTracker",
    "ThreadSentimentAnalysis",
    "ThreadSentimentAnalyzer",
    "ModelBenchmarkEvaluator",
    "SentimentPrediction",
    "IronyPrediction",
    "EmotionPrediction",
    "StancePrediction",
    "MultiDimensionalResult",
]
