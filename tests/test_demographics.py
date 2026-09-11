"""Comprehensive test suite for demographic profiling engine (Component C)."""

from datetime import datetime, timezone
from pathlib import Path
import pytest

from hypesignal.demographics.behavioral_profiler import (
    BehavioralProfiler,
    classify_engagement_tier,
)
from hypesignal.demographics.demographics_engine import DemographicsEngine
from hypesignal.demographics.geo_profiler import GeoProfiler
from hypesignal.demographics.language_detector import LanguageDetector
from hypesignal.demographics.persona_profiler import PersonaProfiler
from hypesignal.demographics.schemas import (
    AggregateDemographics,
    BehavioralProfile,
    GeoLocationProfile,
    PersonaProfile,
    UserProfile,
)
from hypesignal.models.canonical import (
    CanonicalPost,
    CanonicalUser,
    GeoCoordinates,
)
from hypesignal.models.enums import PlatformType
from hypesignal.storage.duckdb_manager import DuckDBManager
from hypesignal.storage.vector_store import VectorStoreManager


@pytest.fixture(scope="module")
def lang_detector():
    return LanguageDetector()


@pytest.fixture(scope="module")
def geo_profiler():
    return GeoProfiler()


@pytest.fixture(scope="module")
def persona_profiler():
    return PersonaProfiler(device="cpu")


@pytest.fixture(scope="module")
def behavioral_profiler():
    return BehavioralProfiler()


@pytest.fixture(scope="module")
def demographics_engine(geo_profiler, persona_profiler, behavioral_profiler, lang_detector):
    return DemographicsEngine(
        geo_profiler=geo_profiler,
        persona_profiler=persona_profiler,
        behavioral_profiler=behavioral_profiler,
        language_detector=lang_detector,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# 1. Language Detector Tests
# ---------------------------------------------------------------------------

def test_language_detection_multilingual(lang_detector):
    en_lang, en_prob = lang_detector.detect_language("This is a lovely sunny morning in London!")
    assert en_lang == "en"
    assert en_prob > 0.8

    fr_lang, fr_prob = lang_detector.detect_language("Bonjour à tous, bienvenue sur notre nouvelle plateforme.")
    assert fr_lang == "fr"
    assert fr_prob > 0.8

    es_lang, es_prob = lang_detector.detect_language("Hola a todos, cómo están en este hermoso día?")
    assert es_lang == "es"
    assert es_prob > 0.8


def test_language_detection_edge_cases(lang_detector):
    # Empty string
    assert lang_detector.detect_language("") == ("en", 0.0)
    assert lang_detector.detect_language(None) == ("en", 0.0)

    # Whitespace and punctuation only
    assert lang_detector.detect_language("   ... ???   ") == ("en", 0.0)

    # Numbers only
    assert lang_detector.detect_language("1234567890 987654321") == ("en", 0.0)

    # Only URLs
    assert lang_detector.detect_language("https://example.com/some/path") == ("en", 0.0)


def test_language_aggregation(lang_detector):
    texts = [
        "Hello world",
        "How are you today?",
        "Bonjour le monde",
        "Another English post",
        "Hola amigos, cómo están todos?",
    ]
    counts = lang_detector.aggregate_languages(texts)
    assert counts.get("en", 0) >= 2
    assert "fr" in counts
    assert "es" in counts


# ---------------------------------------------------------------------------
# 2. Geo Profiler Tests
# ---------------------------------------------------------------------------

def test_geo_profiler_exact_cities(geo_profiler):
    res_sf = geo_profiler.profile_location_string("San Francisco")
    assert res_sf.city == "San Francisco"
    assert res_sf.state_province == "California"
    assert res_sf.country == "United States"
    assert res_sf.country_code == "US"
    assert res_sf.coordinates is not None
    assert res_sf.confidence >= 0.85
    assert res_sf.source == "pattern_matched"

    res_ny = geo_profiler.profile_location_string("NYC")
    assert res_ny.city == "New York"
    assert res_ny.country == "United States"


def test_geo_profiler_delimited_patterns(geo_profiler):
    # US State Abbreviation
    res_austin = geo_profiler.profile_location_string("Austin, TX")
    assert res_austin.city == "Austin"
    assert res_austin.state_province == "Texas"
    assert res_austin.country == "United States"
    assert res_austin.country_code == "US"
    assert res_austin.confidence >= 0.9

    # US State Full Name
    res_miami = geo_profiler.profile_location_string("Miami, Florida")
    assert res_miami.city == "Miami"
    assert res_miami.state_province == "Florida"
    assert res_miami.country == "United States"

    # Canadian Province
    res_to = geo_profiler.profile_location_string("Toronto, ON")
    assert res_to.city == "Toronto"
    assert res_to.state_province == "Ontario"
    assert res_to.country == "Canada"
    assert res_to.country_code == "CA"

    # International Country
    res_ldn = geo_profiler.profile_location_string("London, UK")
    assert res_ldn.city == "London"
    assert res_ldn.country == "United Kingdom"
    assert res_ldn.country_code == "GB"

    # Colliding 2-letter codes: Canada vs California (CA)
    res_toronto_ca = geo_profiler.profile_location_string("Toronto, CA")
    assert res_toronto_ca.city == "Toronto"
    assert res_toronto_ca.state_province == "Ontario"
    assert res_toronto_ca.country == "Canada"
    assert res_toronto_ca.country_code == "CA"

    # Colliding 2-letter codes: India vs Indiana (IN)
    res_delhi_in = geo_profiler.profile_location_string("New Delhi, IN")
    assert res_delhi_in.city == "Delhi"
    assert res_delhi_in.country == "India"
    assert res_delhi_in.country_code == "IN"

    # Colliding 2-letter codes: Germany vs Delaware (DE)
    res_berlin_de = geo_profiler.profile_location_string("Berlin, DE")
    assert res_berlin_de.city == "Berlin"
    assert res_berlin_de.country == "Germany"
    assert res_berlin_de.country_code == "DE"

    # US State overriding foreign city name (Paris, TX -> Texas, US, not France)
    res_paris_tx = geo_profiler.profile_location_string("Paris, TX")
    assert res_paris_tx.city == "Paris"
    assert res_paris_tx.state_province == "Texas"
    assert res_paris_tx.country == "United States"
    assert res_paris_tx.country_code == "US"
    assert res_paris_tx.coordinates is None  # Does NOT borrow France coordinates

    # Cross-state identically-named city coordinate isolation
    res_kent_de = geo_profiler.profile_location_string("Kent, DE")
    assert res_kent_de.city == "Kent"
    assert res_kent_de.state_province == "Delaware"
    assert res_kent_de.coordinates is None  # Does NOT borrow Kent, WA coordinates

    res_kent_wa = geo_profiler.profile_location_string("Kent, WA")
    assert res_kent_wa.city == "Kent"
    assert res_kent_wa.state_province == "Washington"
    assert res_kent_wa.coordinates is not None
    assert round(res_kent_wa.coordinates.latitude, 2) == 47.38

    res_portland_me = geo_profiler.profile_location_string("Portland, ME")
    assert res_portland_me.city == "Portland"
    assert res_portland_me.state_province == "Maine"
    assert res_portland_me.coordinates is None  # Does NOT borrow Portland, OR coordinates

    res_portland_or = geo_profiler.profile_location_string("Portland, OR")
    assert res_portland_or.city == "Portland"
    assert res_portland_or.state_province == "Oregon"
    assert res_portland_or.coordinates is not None
    assert round(res_portland_or.coordinates.latitude, 2) == 45.52


def test_geo_profiler_gps_coordinates(geo_profiler):
    # Formatted GPS string (Cheng-Caverlee-Lee format)
    res_gps = geo_profiler.profile_location_string("UT: 37.7749,-122.4194")
    assert res_gps.confidence == 1.0
    assert res_gps.source == "explicit_gps"
    assert res_gps.city == "San Francisco"
    assert res_gps.country == "United States"

    # Raw arbitrary coordinates without nearest city
    raw_coords = GeoCoordinates(latitude=5.0, longitude=5.0)
    res_raw = geo_profiler.profile_location_string(raw="5.0, 5.0", explicit_coords=raw_coords)
    assert res_raw.confidence == 1.0
    assert res_raw.source == "explicit_gps"
    assert res_raw.coordinates.latitude == 5.0


def test_geo_profiler_text_extraction(geo_profiler):
    text = "Traveling from San Francisco to London tomorrow morning."
    locs = geo_profiler.extract_locations_from_text(text)
    cities = [loc.city for loc in locs]
    assert "San Francisco" in cities
    assert "London" in cities


def test_geo_profiler_user_profiling(geo_profiler):
    user = CanonicalUser(
        id="usr-99",
        platform=PlatformType.TWITTER,
        location_raw="Chicago, IL",
    )
    profile = geo_profiler.profile_user(user)
    assert profile.city == "Chicago"
    assert profile.state_province == "Illinois"
    assert profile.country == "United States"


# ---------------------------------------------------------------------------
# 3. Persona Profiler Tests
# ---------------------------------------------------------------------------

def test_persona_profiler_archetypes(persona_profiler):
    # Tech persona
    tech_bio = "Software architect & open-source Python engineer. Building distributed systems."
    res_tech = persona_profiler.profile_persona(user_id="u1", bio=tech_bio)
    assert res_tech.primary_persona == "Tech & Software Engineering"
    assert res_tech.persona_confidence > 0.35
    assert any("Software" in i or "Data" in i or "Open Source" in i for i in res_tech.top_interests)

    # Finance persona
    fin_bio = "Crypto trader, BTC/ETH hodler, angel investor. DeFi research and macroeconomics."
    res_fin = persona_profiler.profile_persona(user_id="u2", bio=fin_bio)
    assert res_fin.primary_persona == "Finance, Web3 & Crypto"
    assert res_fin.persona_confidence > 0.35

    # Media persona
    media_bio = "Journalist and podcast host. Writing stories, taking photos, producing documentaries."
    res_media = persona_profiler.profile_persona(user_id="u3", bio=media_bio)
    assert res_media.primary_persona == "Media, Design & Creative Arts"
    assert res_media.persona_confidence > 0.35


def test_persona_age_bracket_heuristics(persona_profiler):
    assert persona_profiler.infer_age_bracket("Undergrad student at Stanford class of '25") == "18-24"
    assert persona_profiler.infer_age_bracket("Young professional living in Austin, early career") == "25-34"
    assert persona_profiler.infer_age_bracket("VP of Marketing with 15+ years experience, proud father of 3") == "35-49"
    assert persona_profiler.infer_age_bracket("Retired university professor, proud grandpa") == "50+"
    assert persona_profiler.infer_age_bracket("Just a guy who likes coffee") == "unknown"


def test_persona_batch_profiling(persona_profiler):
    users = [
        CanonicalUser(id="1", platform=PlatformType.TWITTER, bio="AI machine learning engineer"),
        CanonicalUser(id="2", platform=PlatformType.TWITTER, bio="Personal trainer and fitness coach"),
        CanonicalUser(id="3", platform=PlatformType.TWITTER, bio=None),
    ]
    profiles = persona_profiler.profile_users_batch(users)
    assert len(profiles) == 3
    assert profiles[0].primary_persona == "Tech & Software Engineering"
    assert profiles[1].primary_persona == "Sports, Athletics & Fitness"
    assert profiles[2].primary_persona == "General & Everyday Lifestyle"


def test_persona_qdrant_sync(persona_profiler):
    qdrant = VectorStoreManager(":memory:")
    persona_profiler.vector_store = qdrant
    persona_profiler.sync_taxonomy_to_qdrant("persona_taxonomies")

    assert qdrant.count("persona_taxonomies") == len(persona_profiler.taxonomy_keys)


# ---------------------------------------------------------------------------
# 4. Behavioral Profiler Tests
# ---------------------------------------------------------------------------

def test_engagement_tier_classification():
    assert classify_engagement_tier(0) == "casual"
    assert classify_engagement_tier(5) == "casual"
    assert classify_engagement_tier(6) == "active"
    assert classify_engagement_tier(25) == "active"
    assert classify_engagement_tier(26) == "power_user"
    assert classify_engagement_tier(100) == "power_user"
    assert classify_engagement_tier(101) == "broadcaster"


def test_behavioral_profiler_from_timestamps(behavioral_profiler):
    timestamps = [
        datetime(2026, 3, 1, 14, 10, tzinfo=timezone.utc),
        datetime(2026, 3, 1, 14, 25, tzinfo=timezone.utc),
        datetime(2026, 3, 2, 14, 5, tzinfo=timezone.utc),
        datetime(2026, 3, 2, 9, 30, tzinfo=timezone.utc),
    ]
    profile = behavioral_profiler.profile_from_timestamps("user_test", timestamps)
    assert profile.user_id == "user_test"
    assert profile.total_posts == 4
    assert profile.peak_hour == 14
    assert profile.active_hours_distribution[14] == 3
    assert profile.active_hours_distribution[9] == 1
    assert profile.engagement_tier == "casual"


def test_behavioral_profiler_from_duckdb(behavioral_profiler):
    db = DuckDBManager(":memory:")
    posts = [
        CanonicalPost(id=f"p{i}", platform=PlatformType.TWITTER, author_id="u_duck", text=f"post {i}", timestamp=datetime(2026, 5, 1, 10 + (i % 2), 0, tzinfo=timezone.utc))
        for i in range(12)
    ]
    db.insert_posts(posts)

    profile = behavioral_profiler.profile_from_duckdb("u_duck", db)
    assert profile.total_posts == 12
    assert profile.engagement_tier == "active"
    assert 10 in profile.active_hours_distribution
    assert 11 in profile.active_hours_distribution

    batch_res = behavioral_profiler.profile_users_batch_from_duckdb(db, ["u_duck", "nonexistent"])
    assert "u_duck" in batch_res
    assert batch_res["u_duck"].total_posts == 12
    assert batch_res["nonexistent"].total_posts == 0


# ---------------------------------------------------------------------------
# 5. Demographics Engine & Audience Breakdown Tests
# ---------------------------------------------------------------------------

def test_demographics_engine_single_user(demographics_engine):
    user = CanonicalUser(
        id="dev-42",
        platform=PlatformType.TWITTER,
        screen_name="gopher_dev",
        bio="Staff backend engineer building microservices in Go. Proud dad.",
        location_raw="Seattle, WA",
    )
    posts = [
        CanonicalPost(id="dp1", platform=PlatformType.TWITTER, author_id="dev-42", text="Concurrency in Go is amazing", timestamp=datetime(2026, 6, 1, 15, 0, tzinfo=timezone.utc)),
        CanonicalPost(id="dp2", platform=PlatformType.TWITTER, author_id="dev-42", text="Benchmarking channels vs mutexes", timestamp=datetime(2026, 6, 1, 15, 30, tzinfo=timezone.utc)),
    ]

    profile = demographics_engine.profile_user(user, sample_posts=posts)
    assert profile.user_id == "dev-42"
    assert profile.geo.city == "Seattle"
    assert profile.geo.state_province == "Washington"
    assert profile.geo.country == "United States"
    assert profile.persona.primary_persona == "Tech & Software Engineering"
    assert profile.persona.age_bracket == "35-49"
    assert profile.behavioral is not None
    assert profile.behavioral.peak_hour == 15
    assert profile.behavioral.total_posts == 2


def test_demographics_engine_aggregation(demographics_engine):
    p1 = UserProfile(
        user_id="1",
        geo=GeoLocationProfile(city="San Francisco", country="United States"),
        persona=PersonaProfile(user_id="1", primary_persona="Tech & Software Engineering", language="en", age_bracket="25-34"),
        behavioral=BehavioralProfile(user_id="1", total_posts=15, engagement_tier="active"),
    )
    p2 = UserProfile(
        user_id="2",
        geo=GeoLocationProfile(city="San Francisco", country="United States"),
        persona=PersonaProfile(user_id="2", primary_persona="Finance, Web3 & Crypto", language="en", age_bracket="25-34"),
        behavioral=BehavioralProfile(user_id="2", total_posts=4, engagement_tier="casual"),
    )
    p3 = UserProfile(
        user_id="3",
        geo=GeoLocationProfile(city="London", country="United Kingdom"),
        persona=PersonaProfile(user_id="3", primary_persona="Media, Design & Creative Arts", language="en", age_bracket="35-49"),
        behavioral=BehavioralProfile(user_id="3", total_posts=40, engagement_tier="power_user"),
    )

    agg = demographics_engine.aggregate_demographics([p1, p2, p3])
    assert agg.total_users_profiled == 3
    assert agg.country_distribution["United States"] == 2
    assert agg.country_distribution["United Kingdom"] == 1
    assert agg.top_cities["San Francisco"] == 2
    assert agg.top_cities["London"] == 1
    assert agg.persona_distribution["Tech & Software Engineering"] == 1
    assert agg.age_bracket_distribution["25-34"] == 2
    assert agg.engagement_tier_distribution["active"] == 1
    assert agg.engagement_tier_distribution["casual"] == 1
    assert agg.engagement_tier_distribution["power_user"] == 1


def test_demographics_engine_duckdb_breakdown(demographics_engine):
    db = DuckDBManager(":memory:")
    # Insert users
    u1 = CanonicalUser(id="101", platform=PlatformType.TWITTER, screen_name="alice", bio="Crypto investor and trader", location_raw="Austin, TX")
    u2 = CanonicalUser(id="102", platform=PlatformType.TWITTER, screen_name="bob", bio="Graphic designer and artist", location_raw="London, UK")
    db.insert_users([u1, u2])

    # Insert posts
    posts = [
        CanonicalPost(id="p101_1", platform=PlatformType.TWITTER, author_id="101", text="Bull market is here!", timestamp=datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)),
        CanonicalPost(id="p102_1", platform=PlatformType.TWITTER, author_id="102", text="New logo concept", timestamp=datetime(2026, 7, 1, 16, 0, tzinfo=timezone.utc)),
    ]
    db.insert_posts(posts)

    agg = demographics_engine.get_audience_breakdown(db, limit=10)
    assert agg.total_users_profiled == 2
    assert "United States" in agg.country_distribution
    assert "United Kingdom" in agg.country_distribution
    assert "Finance, Web3 & Crypto" in agg.persona_distribution
    assert "Media, Design & Creative Arts" in agg.persona_distribution
