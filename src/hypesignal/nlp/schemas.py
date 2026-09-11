"""Data structures and prediction schemas for Multi-Dimensional Sentiment and Emotion."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from hypesignal.models.enums import EmotionType, SentimentPolarity, StanceType


class SentimentPrediction(BaseModel):
    """Output from the primary sentiment polarity classifier."""
    polarity: SentimentPolarity
    score: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float] = Field(default_factory=dict)


class IronyPrediction(BaseModel):
    """Output from sarcasm/irony detection."""
    is_ironic: bool
    irony_score: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float] = Field(default_factory=dict)


class EmotionPrediction(BaseModel):
    """Output from nuanced multi-class emotion classifier."""
    primary_emotion: EmotionType
    score: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float] = Field(default_factory=dict)


class StancePrediction(BaseModel):
    """Output from target-oriented stance detection."""
    stance: StanceType
    score: float = Field(..., ge=0.0, le=1.0)
    target: Optional[str] = None
    probabilities: Dict[str, float] = Field(default_factory=dict)


class MultiDimensionalResult(BaseModel):
    """Consolidated multi-dimensional sentiment and emotion inference output."""
    text: str
    sentiment: SentimentPrediction
    irony: IronyPrediction
    emotion: EmotionPrediction
    stance: Optional[StancePrediction] = None
    effective_polarity: SentimentPolarity
    adjusted_sentiment_score: float
    is_sarcasm_inverted: bool = False
    explanation: Optional[str] = None
