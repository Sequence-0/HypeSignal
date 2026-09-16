"""Unit and integration tests for Step 3: Multi-Stage Age Ensemble & Influencer Audience Profiler."""

import pytest

from hypesignal.demographics.age_classifier import (
    AGE_BRACKETS,
    MultiStageAgeClassifier,
)
from hypesignal.demographics.audience_profiler import (
    InfluencerAudienceProfile,
    InfluencerAudienceProfiler,
)
from hypesignal.models.canonical import CanonicalGraphEdge, CanonicalPost, CanonicalUser
from hypesignal.models.enums import PlatformType, RelationType
from hypesignal.storage.duckdb_manager import DuckDBManager


@pytest.fixture(scope="module")
def age_classifier():
    """Shared MultiStageAgeClassifier instance on CPU."""
    return MultiStageAgeClassifier(device="cpu", enable_nli=False, reference_year=2026)


@pytest.fixture
def mem_db():
    """In-memory DuckDB manager fixture."""
    db = DuckDBManager(":memory:")
    yield db
    db.close()


def test_stage1_explicit_regex_signals(age_classifier):
    """Test explicit regex extraction for ages, birth years, and life stage phrases."""
    # 1. Direct age declarations
    res_21 = age_classifier.classify_stage1_regex("Hey everyone, I am 21 and learning Python!")
    assert res_21 is not None
    assert res_21[0] == "18-24"
    assert res_21[1]["18-24"] == 0.90

    res_turning = age_classifier.classify_stage1_regex("Excited to be turning 30 next week! 🎉")
    assert res_turning is not None
    assert res_turning[0] == "25-34"

    res_yo = age_classifier.classify_stage1_regex("25yo software engineer based in NYC.")
    assert res_yo is not None
    assert res_yo[0] == "25-34"

    res_45 = age_classifier.classify_stage1_regex("Just another day at age 45.")
    assert res_45 is not None
    assert res_45[0] == "35-49"

    res_teen = age_classifier.classify_stage1_regex("I'm 16 and a high school sophomore.")
    assert res_teen is not None
    assert res_teen[0] == "<18"

    # 2. Birth year extraction (reference year 2026)
    # 1995 -> 31 -> 25-34
    res_birth = age_classifier.classify_stage1_regex("Born in 1995, building the open web.")
    assert res_birth is not None
    assert res_birth[0] == "25-34"

    # '08 -> 18 -> 18-24
    res_bday = age_classifier.classify_stage1_regex("bday: '08 gamer and coder")
    assert res_bday is not None
    assert res_bday[0] == "18-24"

    # 3. Life stage lexical cues
    res_college = age_classifier.classify_stage1_regex("University student, college freshman studying biology.")
    assert res_college is not None
    assert res_college[0] == "18-24"

    res_retired = age_classifier.classify_stage1_regex("Retired in 2020, proud grandpa of three beautiful grandkids.")
    assert res_retired is not None
    assert res_retired[0] == "65+"

    res_senior = age_classifier.classify_stage1_regex("Senior executive with 25+ years exp in corporate leadership.")
    assert res_senior is not None
    assert res_senior[0] == "50-64"

    # 4. Text with no explicit signals
    res_none = age_classifier.classify_stage1_regex("Just an AI enthusiast interested in open models.")
    assert res_none is None


def test_stage2_bio_vector_anchors(age_classifier):
    """Test semantic anchor matching via sentence transformer bio embeddings."""
    # Youth text
    youth_dist = age_classifier.classify_stage2_bio_vector("High school teen studying secondary school exams and anime.")
    assert youth_dist["<18"] > 0.20
    assert sum(youth_dist.values()) == pytest.approx(1.0, abs=0.01)

    # College text
    college_dist = age_classifier.classify_stage2_bio_vector("University undergraduate dorm life, studying for campus exams.")
    assert college_dist["18-24"] > 0.25
    assert sum(college_dist.values()) == pytest.approx(1.0, abs=0.01)

    # Young professional text
    young_prof_dist = age_classifier.classify_stage2_bio_vector("Corporate software engineer living in an apartment, career starter.")
    assert young_prof_dist["25-34"] > 0.25
    assert sum(young_prof_dist.values()) == pytest.approx(1.0, abs=0.01)

    # Retiree text
    retiree_dist = age_classifier.classify_stage2_bio_vector("Elderly retiree enjoying gardening, leisure cruises, and Florida sunshine.")
    assert retiree_dist["65+"] > 0.25
    assert sum(retiree_dist.values()) == pytest.approx(1.0, abs=0.01)


def test_weighted_ensemble_classification(age_classifier):
    """Test full 4-stage weighted ensemble age prediction."""
    # Profile with explicit cue: regex dominates
    pred_explicit = age_classifier.classify_age(
        bio="Software engineer at a startup. I am 22.",
        sample_posts=["Working on new open source protocols."],
    )
    assert pred_explicit.bracket == "18-24"
    assert pred_explicit.confidence >= 0.80
    assert pred_explicit.primary_source == "regex"
    assert "stage1_regex" in pred_explicit.stage_scores
    assert sum(pred_explicit.probabilities.values()) == pytest.approx(1.0, abs=0.01)

    # Profile with implicit bio semantics: bio vector dominates
    pred_implicit = age_classifier.classify_age(
        bio="Grandparent spending my golden years with family and grandchildren in Florida.",
        sample_posts=["Lovely morning in the garden."],
    )
    assert pred_implicit.bracket == "65+"
    assert pred_implicit.confidence > 0.30
    assert sum(pred_implicit.probabilities.values()) == pytest.approx(1.0, abs=0.01)

    # Profile with purely semantic text and zero regex cues: exercises bio_vector ensemble path (M1)
    pred_vector_only = age_classifier.classify_age(
        bio="Software engineer working at a tech startup, living in an apartment with my dog.",
        sample_posts=["Building distributed microservices and deploying Docker containers."],
    )
    assert pred_vector_only.primary_source == "bio_vector"
    assert "stage1_regex" not in pred_vector_only.stage_scores
    assert "stage2_bio_vector" in pred_vector_only.stage_scores
    assert pred_vector_only.bracket in ("25-34", "35-49")
    assert sum(pred_vector_only.probabilities.values()) == pytest.approx(1.0, abs=0.01)

    # Batch prediction on CanonicalUsers
    users = [
        CanonicalUser(id="u1", bio="I'm 19 and a college student."),
        CanonicalUser(id="u2", bio="Retired executive enjoying leisure cruises."),
    ]
    batch_preds = age_classifier.classify_users_batch(users)
    assert len(batch_preds) == 2
    assert batch_preds[0].bracket == "18-24"
    assert batch_preds[1].bracket == "65+"


def test_duckdb_get_user_followers_case_insensitive(mem_db):
    """Test that DuckDB get_user_followers works with relation_type regardless of case (C2)."""
    edges = [
        CanonicalGraphEdge(source_id="fan_a", target_id="creator_1", relation_type=RelationType.FOLLOWS),
        CanonicalGraphEdge(source_id="fan_b", target_id="creator_1", relation_type=RelationType.FOLLOWS),
    ]
    mem_db.insert_edges(edges)
    followers = mem_db.get_user_followers("creator_1")
    assert sorted(followers) == ["fan_a", "fan_b"]
    following = mem_db.get_user_following("fan_a")
    assert following == ["creator_1"]


def test_min_group_size_privacy_filter():
    """Test InfluencerAudienceProfiler.apply_min_group_size_filter."""
    raw_counts = {
        "25-34": 25,
        "18-24": 15,
        "35-49": 4,   # < 5: should be suppressed
        "<18": 2,     # < 5: should be suppressed
        "65+": 1,     # < 5: should be suppressed
    }

    filtered = InfluencerAudienceProfiler.apply_min_group_size_filter(raw_counts, min_size=5)

    assert "25-34" in filtered
    assert filtered["25-34"] == 25
    assert "18-24" in filtered
    assert filtered["18-24"] == 15

    # Small groups should NOT appear individually
    assert "35-49" not in filtered
    assert "<18" not in filtered
    assert "65+" not in filtered

    # Merged into 'Other / Low Count' with sum 4 + 2 + 1 = 7
    assert filtered["Other / Low Count"] == 7

    # When all counts exceed min_size, no Other bucket is created
    clean_counts = {"US": 10, "UK": 8}
    clean_filtered = InfluencerAudienceProfiler.apply_min_group_size_filter(clean_counts, min_size=5)
    assert "Other / Low Count" not in clean_filtered
    assert clean_filtered == {"US": 10, "UK": 8}


def test_influencer_audience_profiler_integration(age_classifier, mem_db):
    """Test end-to-end influencer audience profiling from DuckDB graph edges and user demographics."""
    # 1. Create target influencer and followers in DuckDB
    influencer_id = "influencer_prime"
    users = [
        CanonicalUser(id=influencer_id, screen_name="influencer_prime", bio="Tech thought leader and AI speaker"),
    ]

    # Create 6 young professionals (count >= 5 -> retained)
    for i in range(1, 7):
        users.append(
            CanonicalUser(
                id=f"follower_pro_{i}",
                screen_name=f"dev_{i}",
                bio="Young professional software engineer, 28 years old living in San Francisco, CA.",
                location_raw="San Francisco, CA, USA",
            )
        )

    # Create 3 college students (count 3 < 5 -> grouped into Other)
    for j in range(1, 4):
        users.append(
            CanonicalUser(
                id=f"follower_student_{j}",
                screen_name=f"student_{j}",
                bio="College freshman studying at university campus.",
                location_raw="Austin, TX, USA",
            )
        )

    mem_db.insert_users(users)

    # 2. Insert follower graph edges: follower -> FOLLOWS -> influencer
    edges = []
    for u in users[1:]:
        edges.append(
            CanonicalGraphEdge(
                source_id=u.id,
                target_id=influencer_id,
                relation_type=RelationType.FOLLOWS,
                weight=1.0,
            )
        )
    mem_db.insert_edges(edges)

    # 3. Profile influencer audience with min_group_size = 5
    profiler = InfluencerAudienceProfiler(
        db=mem_db,
        age_classifier=age_classifier,
        min_group_size=5,
    )

    profile = profiler.profile_influencer_audience(influencer_id, include_profiles=True)

    assert profile.influencer_id == influencer_id
    assert profile.total_followers_analyzed == 9
    assert profile.min_group_size_applied == 5

    # Verify age distribution with min_group_size filtering:
    # 6 young professionals -> "25-34" (count 6 >= 5, retained)
    # 3 students -> "18-24" (count 3 < 5, suppressed into 'Other / Low Count')
    assert "25-34" in profile.age_distribution
    assert profile.age_distribution["25-34"] == 6
    assert "18-24" not in profile.age_distribution
    assert profile.age_distribution["Other / Low Count"] == 3

    # Verify attached individual profiles
    assert profile.follower_profiles is not None
    assert len(profile.follower_profiles) == 9

    # 4. Test influencer with 0 followers
    empty_profile = profiler.profile_influencer_audience("unknown_influencer")
    assert empty_profile.total_followers_analyzed == 0
    assert len(empty_profile.age_distribution) == 0
