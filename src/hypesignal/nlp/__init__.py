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

__all__ = [
    "MultiDimensionalSentimentEngine",
    "TemporalSentimentTracker",
    "ModelBenchmarkEvaluator",
    "SentimentPrediction",
    "IronyPrediction",
    "EmotionPrediction",
    "StancePrediction",
    "MultiDimensionalResult",
]
