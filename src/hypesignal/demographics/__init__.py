"""Demographic profiling package for HypeSignal.

Provides geographic entity extraction, language identification, zero-shot persona
interest classification, circadian behavioral profiling, and audience aggregations.
"""

from hypesignal.demographics.age_classifier import (
    AGE_BRACKETS,
    AgePrediction,
    MultiStageAgeClassifier,
)
from hypesignal.demographics.audience_profiler import (
    InfluencerAudienceProfile,
    InfluencerAudienceProfiler,
)
from hypesignal.demographics.behavioral_profiler import (
    BehavioralProfiler,
    classify_engagement_tier,
)
from hypesignal.demographics.demographics_engine import DemographicsEngine
from hypesignal.demographics.geo_profiler import (
    GeoProfiler,
    MAJOR_CITIES,
    US_STATES,
)
from hypesignal.demographics.language_detector import LanguageDetector
from hypesignal.demographics.persona_profiler import (
    PERSONA_TAXONOMY,
    PersonaProfiler,
)
from hypesignal.demographics.schemas import (
    AggregateDemographics,
    BehavioralProfile,
    GeoLocationProfile,
    PersonaProfile,
    UserProfile,
)

__all__ = [
    "AGE_BRACKETS",
    "AgePrediction",
    "AggregateDemographics",
    "BehavioralProfile",
    "BehavioralProfiler",
    "DemographicsEngine",
    "GeoLocationProfile",
    "GeoProfiler",
    "InfluencerAudienceProfile",
    "InfluencerAudienceProfiler",
    "LanguageDetector",
    "MAJOR_CITIES",
    "MultiStageAgeClassifier",
    "PERSONA_TAXONOMY",
    "PersonaProfile",
    "PersonaProfiler",
    "US_STATES",
    "UserProfile",
    "classify_engagement_tier",
]
