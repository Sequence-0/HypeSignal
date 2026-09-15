"""Enumerations for the HypeSignal framework."""

from enum import Enum


class PlatformType(str, Enum):
    """Supported social media and data source platforms."""
    TWITTER = "twitter"
    TELEGRAM = "telegram"
    REDDIT = "reddit"
    YOUTUBE = "youtube"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    BLUESKY = "bluesky"
    DATASET = "dataset"
    OTHER = "other"


class RelationType(str, Enum):
    """Types of graph relationships between social network entities."""
    FOLLOWS = "follows"
    RETWEETS = "retweets"
    REPLIES = "replies"
    MENTIONS = "mentions"
    QUOTES = "quotes"


class SentimentPolarity(str, Enum):
    """Overall sentiment polarity classes."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class EmotionType(str, Enum):
    """Nuanced emotion categories."""
    JOY = "joy"
    OPTIMISM = "optimism"
    ANGER = "anger"
    SADNESS = "sadness"
    FEAR = "fear"
    ANXIETY = "anxiety"
    EXCITEMENT = "excitement"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"


class StanceType(str, Enum):
    """Target-oriented stance classification."""
    SUPPORTIVE = "supportive"
    AGAINST = "against"
    NEUTRAL = "neutral"
    NONE = "none"
