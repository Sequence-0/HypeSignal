"""Multi-stage age bracket classification ensemble (Pillar C).

Combines explicit regex cues, Qdrant/bio embedding vector anchors (MiniLM),
zero-shot NLI post scoring, and weighted ensembling across 6 standard age cohorts:
<18, 18-24, 25-34, 35-49, 50-64, 65+.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from hypesignal.models.canonical import CanonicalUser

logger = logging.getLogger(__name__)

# Standard 6-cohort age classification taxonomy
AGE_BRACKETS: List[str] = ["<18", "18-24", "25-34", "35-49", "50-64", "65+"]

# Semantic anchor descriptions for bio vector centroid comparison
AGE_ANCHORS: Dict[str, str] = {
    "<18": (
        "high school student teen teenager gaming anime discord secondary school "
        "adolescent youth homework TikTok dances kpop minor study exams"
    ),
    "18-24": (
        "university college student undergraduate dorm campus life frat sorority "
        "studying degree exam internship gen z early 20s bachelor club party"
    ),
    "25-34": (
        "young professional corporate software engineer career starter tech startup "
        "apartment millennial married wedding newlywed travel twenties thirties"
    ),
    "35-49": (
        "senior director parent soccer mom dad kids family homeowner mortgage "
        "mid career work-life balance parenting teenager child suburban executive"
    ),
    "50-64": (
        "executive empty nester college tuition veteran 25 years experience mature "
        "leadership consulting business owner pre-retirement senior director vice president"
    ),
    "65+": (
        "retired retiree grandchildren gardening grandpa grandma cruise pension "
        "senior citizen Florida golf leisure elder golden years emeritus"
    ),
}


class AgePrediction(BaseModel):
    """Ensemble prediction result for user age cohort."""
    bracket: str
    confidence: float
    stage_scores: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    probabilities: Dict[str, float] = Field(default_factory=dict)
    primary_source: str = "ensemble"  # 'regex', 'bio_vector', 'nli', 'ensemble'


class MultiStageAgeClassifier:
    """4-Stage demographic age bracket classification ensemble."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
        device: Optional[str] = None,
        enable_nli: bool = False,
        sentence_transformer: Optional[SentenceTransformer] = None,
        reference_year: Optional[int] = None,
    ) -> None:
        """Initialize classifier with sentence transformer embedder and anchor vectors.
        
        Args:
            model_name: SentenceTransformer model name for bio embeddings.
            nli_model_name: Cross-encoder or zero-shot model for post hypothesis testing.
            device: Device target ('cpu', 'cuda', etc.).
            enable_nli: Whether to evaluate Stage 3 zero-shot NLI (bypassed by default for offline speed).
            sentence_transformer: Optional pre-loaded SentenceTransformer instance.
            reference_year: Current reference year for birthdate calculation (default current year).
        """
        self.device = device or "cpu"
        self.enable_nli = enable_nli
        self.nli_model_name = nli_model_name
        self.reference_year = reference_year or datetime.now(timezone.utc).year

        # Load or reuse SentenceTransformer
        if sentence_transformer is not None:
            self.model = sentence_transformer
        else:
            logger.info("Loading SentenceTransformer model %s on %s...", model_name, self.device)
            self.model = SentenceTransformer(model_name, device=self.device)

        # Precompute and normalize anchor vectors for all 6 age cohorts
        self.anchor_brackets = AGE_BRACKETS
        anchor_texts = [AGE_ANCHORS[b] for b in self.anchor_brackets]
        self.anchor_embeddings = self.model.encode(
            anchor_texts,
            batch_size=len(anchor_texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        self._nli_pipeline: Optional[Any] = None

        # Regex patterns for Stage 1
        self._init_regex_patterns()

    def _init_regex_patterns(self) -> None:
        """Initialize compiled regular expressions for explicit age cue detection."""
        # 1. Numeric age declarations (e.g., "I am 21", "age 34", "25yo", "turning 30")
        self.re_age_i_am = re.compile(r"\b(?:i(?:'m| am)|age[:\s]?)\s*(\d{1,2})\b", re.IGNORECASE)
        self.re_age_yo = re.compile(r"\b(\d{1,2})\s*(?:yo|y/o|years?\s*old|yr\s*old)\b", re.IGNORECASE)
        self.re_turning = re.compile(r"\bturning\s*(\d{1,2})\b", re.IGNORECASE)

        # 2. Birth year declarations (e.g., "born in 1998", "born '95", "bday: 1990", "bday: 05/12/1990")
        self.re_born_year = re.compile(
            r"\b(?:born\s*(?:in\s*)?|bday:?\s*(?:is\s*)?(?:\d{1,2}[/-]\d{1,2}[/-])?)('?\d{2,4})\b",
            re.IGNORECASE,
        )

        # 3. Life stage lexical patterns mapped directly to brackets
        self.stage_cues: List[Tuple[re.Pattern, str]] = [
            (
                re.compile(
                    r"\b(high\s*school\s*(?:student|sophomore|freshman|junior|senior)|teen(?:ager)?|secondary\s*school|minor|underage)\b",
                    re.IGNORECASE,
                ),
                "<18",
            ),
            (
                re.compile(
                    r"\b(college\s*(?:freshman|sophomore|junior|senior|student)|undergrad(?:uate)?|university\s*student|campus\s*life|dorm\s*life|dormitory|sorority|fraternity|gen\s*z|uni\s*student)\b",
                    re.IGNORECASE,
                ),
                "18-24",
            ),
            (
                re.compile(
                    r"\b(young\s*professional|early\s*career|career\s*starter|recent\s*grad(?:uate)?|twenty-something|20-something|millennial)\b",
                    re.IGNORECASE,
                ),
                "25-34",
            ),
            (
                re.compile(
                    r"\b(father\s*of|mother\s*of|parent\s*of|raising\s*(?:kids|teens)|homeowner|mid-career|15\+\s*years\s*(?:of\s*)?exp)\b",
                    re.IGNORECASE,
                ),
                "35-49",
            ),
            (
                re.compile(
                    r"\b(empty\s*nester|college\s*tuition|25\+\s*years\s*(?:of\s*)?exp|30\+\s*years\s*(?:of\s*)?exp|nearing\s*retirement|senior\s*executive|veteran\s*leader)\b",
                    re.IGNORECASE,
                ),
                "50-64",
            ),
            (
                re.compile(
                    r"\b(retired\s*(?:in\s*\d{4})?|retiree|grandkid(?:s)?|grandchildren|grandma|grandpa|grandfather|grandmother|great-grandma|great-grandpa|senior\s*citizen|pensioner|golden\s*years)\b",
                    re.IGNORECASE,
                ),
                "65+",
            ),
        ]

    def _map_age_to_bracket(self, age: int) -> Optional[str]:
        """Map a numeric age value to an age bracket."""
        if age < 10 or age > 105:
            return None
        if age < 18:
            return "<18"
        if age <= 24:
            return "18-24"
        if age <= 34:
            return "25-34"
        if age <= 49:
            return "35-49"
        if age <= 64:
            return "50-64"
        return "65+"

    def _parse_birth_year(self, year_str: str) -> Optional[int]:
        """Parse 2-digit or 4-digit birth year string to numeric age."""
        clean_str = year_str.lstrip("'").strip()
        if not clean_str.isdigit():
            return None
        val = int(clean_str)
        if len(clean_str) == 2:
            cutoff = self.reference_year % 100
            year = (2000 + val) if val <= cutoff else (1900 + val)
        elif len(clean_str) == 4:
            year = val
        else:
            return None

        age = self.reference_year - year
        if 10 <= age <= 105:
            return age
        return None

    def classify_stage1_regex(self, text: str) -> Optional[Tuple[str, Dict[str, float]]]:
        """Stage 1: Explicit regex signals from bio or post text.
        
        Returns:
            Tuple of (matched_bracket, distribution_dict) if matched, else None.
        """
        if not text:
            return None

        matched_bracket: Optional[str] = None

        # 1. Check numeric age matches
        for pat in (self.re_age_i_am, self.re_age_yo, self.re_turning):
            m = pat.search(text)
            if m:
                age_val = int(m.group(1))
                bracket = self._map_age_to_bracket(age_val)
                if bracket:
                    matched_bracket = bracket
                    break

        # 2. Check birth year
        if not matched_bracket:
            m = self.re_born_year.search(text)
            if m:
                age_val = self._parse_birth_year(m.group(1))
                if age_val:
                    matched_bracket = self._map_age_to_bracket(age_val)

        # 3. Check life stage patterns
        if not matched_bracket:
            for pat, bracket in self.stage_cues:
                if pat.search(text):
                    matched_bracket = bracket
                    break

        if matched_bracket:
            # Produce high-confidence distribution (0.90 for matched bracket, distributed residual)
            residual = (1.0 - 0.90) / (len(AGE_BRACKETS) - 1)
            dist = {b: residual for b in AGE_BRACKETS}
            dist[matched_bracket] = 0.90
            return matched_bracket, dist

        return None

    def classify_stage2_bio_vector(self, text: str, temperature: float = 0.15) -> Dict[str, float]:
        """Stage 2: Qdrant / MiniLM bio vector anchor cosine comparison.
        
        Args:
            text: Bio and/or aggregated post text.
            temperature: Softmax temperature parameter for probability calibration.
            
        Returns:
            Normalized probability distribution over the 6 age cohorts.
        """
        if not text or len(text.strip()) < 3:
            # Uniform fallback for empty text
            prob = 1.0 / len(AGE_BRACKETS)
            return {b: prob for b in AGE_BRACKETS}

        text_emb = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        sims = np.dot(self.anchor_embeddings, text_emb)

        # Softmax with temperature scaling
        scaled_sims = sims / max(1e-3, temperature)
        exp_sims = np.exp(scaled_sims - np.max(scaled_sims))
        probs = exp_sims / np.sum(exp_sims)

        return {b: float(p) for b, p in zip(self.anchor_brackets, probs)}

    def classify_stage3_nli(self, posts: List[str]) -> Optional[Dict[str, float]]:
        """Stage 3: Zero-shot NLI post scorer.
        
        Bypassed if enable_nli is False or posts are empty.
        """
        if not self.enable_nli or not posts:
            return None

        try:
            from transformers import pipeline

            if self._nli_pipeline is None:
                logger.info("Initializing zero-shot NLI pipeline with %s...", self.nli_model_name)
                self._nli_pipeline = pipeline(
                    "zero-shot-classification",
                    model=self.nli_model_name,
                    device=0 if self.device.startswith("cuda") else -1,
                )

            combined_posts = " ".join([p.strip() for p in posts if p.strip()][:5])
            if not combined_posts:
                return None

            candidate_labels = [
                "under 18 years old teenager",
                "college university student 18 to 24 years old",
                "young professional adult 25 to 34 years old",
                "middle aged parent 35 to 49 years old",
                "mature adult 50 to 64 years old",
                "senior retired citizen 65 years or older",
            ]
            label_to_bracket = dict(zip(candidate_labels, AGE_BRACKETS))

            res = self._nli_pipeline(
                combined_posts,
                candidate_labels,
                hypothesis_template="This person is {}.",
                multi_label=False,
            )

            scores = dict(zip(res["labels"], res["scores"]))
            return {label_to_bracket[lbl]: float(scores[lbl]) for lbl in candidate_labels}
        except Exception as e:
            logger.debug("Zero-shot NLI evaluation bypassed or failed: %s", e)
            return None

    def classify_age(
        self,
        bio: Optional[str] = None,
        sample_posts: Optional[List[str]] = None,
    ) -> AgePrediction:
        """Stage 4: Full weighted ensemble demographic age prediction.
        
        Args:
            bio: User profile biography text.
            sample_posts: Optional sample post texts authored by the user.
            
        Returns:
            AgePrediction containing bracket, confidence, stage scores, and probabilities.
        """
        bio_text = (bio or "").strip()
        posts = [p.strip() for p in (sample_posts or []) if p and p.strip()]
        full_text = " ".join(([bio_text] if bio_text else []) + posts).strip()

        stage_scores: Dict[str, Dict[str, float]] = {}
        s1_dist: Optional[Dict[str, float]] = None

        # 1. Stage 1: Explicit Regex Signals (evaluated on bio, posts, or combined text)
        regex_result = self.classify_stage1_regex(full_text)
        if regex_result:
            s1_bracket, s1_dist = regex_result
            stage_scores["stage1_regex"] = s1_dist
            stage_scores["stage2_bio_vector"] = self.classify_stage2_bio_vector(full_text)
            return AgePrediction(
                bracket=s1_bracket,
                confidence=0.90,
                stage_scores=stage_scores,
                probabilities=s1_dist,
                primary_source="regex",
            )

        # 2. Stage 2: Qdrant Bio Vector Anchors
        s2_dist = self.classify_stage2_bio_vector(full_text)
        stage_scores["stage2_bio_vector"] = s2_dist

        # 3. Stage 3: Zero-shot NLI Scorer
        s3_dist = self.classify_stage3_nli(posts)
        if s3_dist:
            stage_scores["stage3_nli"] = s3_dist

        # 4. Stage 4: Weighted Ensembling
        if s3_dist is not None:
            w1 = 0.0
            w2 = 0.60
            w3 = 0.40
            primary_src = "ensemble"
        else:
            w1 = 0.0
            w2 = 1.0
            w3 = 0.0
            primary_src = "bio_vector"

        final_probs: Dict[str, float] = {}
        total_w = w1 + w2 + w3

        for b in AGE_BRACKETS:
            p1 = s1_dist.get(b, 0.0) if s1_dist else 0.0
            p2 = s2_dist.get(b, 0.0)
            p3 = s3_dist.get(b, 0.0) if s3_dist else 0.0
            final_probs[b] = round((w1 * p1 + w2 * p2 + w3 * p3) / total_w, 4)

        # Normalize and correct rounding drift on dominant key
        prob_sum = sum(final_probs.values())
        if prob_sum > 0.0:
            for b in final_probs:
                final_probs[b] = round(final_probs[b] / prob_sum, 4)
            drift = round(1.0 - sum(final_probs.values()), 4)
            if drift != 0.0:
                dominant_b = max(final_probs, key=lambda k: final_probs[k])
                final_probs[dominant_b] = round(final_probs[dominant_b] + drift, 4)

        dominant_bracket = max(final_probs, key=lambda k: final_probs[k])
        confidence = float(final_probs[dominant_bracket])

        return AgePrediction(
            bracket=dominant_bracket,
            confidence=confidence,
            stage_scores=stage_scores,
            probabilities=final_probs,
            primary_source=primary_src,
        )

    def classify_users_batch(
        self,
        users: List[CanonicalUser],
        user_posts_map: Optional[Dict[str, List[str]]] = None,
    ) -> List[AgePrediction]:
        """Classify a batch of users using the multi-stage ensemble."""
        if not users:
            return []

        posts_map = user_posts_map or {}
        results: List[AgePrediction] = []
        for u in users:
            posts = posts_map.get(u.id, [])
            results.append(self.classify_age(bio=u.bio, sample_posts=posts))
        return results
