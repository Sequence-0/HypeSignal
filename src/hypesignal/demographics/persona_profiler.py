"""Bio-persona and professional interest profiler.

Uses SentenceTransformer embeddings (all-MiniLM-L6-v2) for zero-shot persona
classification against a canonical taxonomy, heuristic age bracket inference,
and language detection. Optionally syncs embeddings to Qdrant vector store.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from hypesignal.demographics.language_detector import LanguageDetector
from hypesignal.demographics.schemas import PersonaProfile
from hypesignal.models.canonical import CanonicalUser
from hypesignal.storage.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

# Standardized Persona Archetypes & Associated Topic Interests
PERSONA_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "Tech & Software Engineering": {
        "description": "Software engineer, computer programmer, web developer, AI machine learning researcher, data scientist, cybersecurity, cloud DevOps, open source coder, python rust linux backend frontend",
        "interests": ["Software Engineering", "Artificial Intelligence", "Data Science", "Open Source", "Cloud Infrastructure"],
    },
    "Finance, Web3 & Crypto": {
        "description": "Financial analyst, trader, investor, venture capital, stock market, cryptocurrency, bitcoin, ethereum, defi, blockchain, private equity, fintech, hedge fund quantitative economics",
        "interests": ["Stock Trading", "Cryptocurrency & Web3", "Venture Capital", "DeFi", "Macroeconomics"],
    },
    "Media, Design & Creative Arts": {
        "description": "Journalist, author, writer, digital artist, graphic designer, content creator, video editor, podcaster, photographer, UI UX designer, music producer, filmmaker storyteller creative",
        "interests": ["Digital Design", "Journalism", "Content Creation", "Photography", "Creative Arts"],
    },
    "Science, Research & Academia": {
        "description": "University professor, academic researcher, PhD candidate, scientist, biology, chemistry, physics, scholar, peer reviewed publications, mathematics, laboratory scientific discoveries",
        "interests": ["Scientific Research", "Higher Education", "Physics & Mathematics", "Biotechnology", "Academic Publishing"],
    },
    "Healthcare & Medicine": {
        "description": "Medical doctor, physician, registered nurse, healthcare worker, medical researcher, surgeon, therapist, mental health psychologist, clinical medicine, public health, pharmacy wellness",
        "interests": ["Clinical Medicine", "Mental Health", "Public Health", "Medical Research", "Wellness"],
    },
    "Business, Marketing & Leadership": {
        "description": "Startup founder, CEO, entrepreneur, marketing director, product manager, sales executive, business development, strategy, consulting, corporate leadership executive B2B growth",
        "interests": ["Entrepreneurship", "Product Management", "Digital Marketing", "Business Strategy", "Executive Leadership"],
    },
    "Sports, Athletics & Fitness": {
        "description": "Professional athlete, fitness coach, personal trainer, bodybuilding, marathon runner, sports fan, soccer, basketball, football, baseball, gym workout, marathon, nutrition exercise",
        "interests": ["Fitness & Bodybuilding", "Competitive Athletics", "Sports Commentary", "Nutrition", "Outdoor Adventure"],
    },
    "Student & Youth": {
        "description": "University undergraduate student, high school student, college learner, studying computer science or business, campus life, studying for exams, intern student club",
        "interests": ["Campus Life", "Academic Studies", "Internships", "Student Projects", "Youth Culture"],
    },
    "General & Everyday Lifestyle": {
        "description": "Casual social media user, everyday thoughts, family, friends, pets, dogs, cats, food, travel, coffee, memes, living life, personal account pop culture enthusiast",
        "interests": ["Everyday Lifestyle", "Travel & Adventure", "Food & Dining", "Pets & Animals", "Pop Culture"],
    },
}

# Age bracket heuristic patterns
AGE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"\b(retired|grandparent|grandpa|grandma|emeritus|boomer|30\+\s*years|veteran|senior\s*citizen)\b",
            re.IGNORECASE,
        ),
        "50+",
    ),
    (
        re.compile(
            r"\b(student|undergrad|college|uni|freshman|sophomore|junior|senior|intern|high\s*school|gen\s*z|born\s*in\s*200\d)\b",
            re.IGNORECASE,
        ),
        "18-24",
    ),
    (
        re.compile(
            r"\b(young\s*professional|early\s*career|millennial|20-something|recent\s*grad|associate)\b",
            re.IGNORECASE,
        ),
        "25-34",
    ),
    (
        re.compile(
            r"\b(senior\s*(?:engineer|director|manager)|vp|vice\s*president|director|parent|father|mother|dad|mom|10\+\s*years|15\+\s*years|decade\s*of\s*experience)\b",
            re.IGNORECASE,
        ),
        "35-49",
    ),
]


class PersonaProfiler:
    """Zero-shot persona interest classifier and demographic profiler."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store: Optional[VectorStoreManager] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize SentenceTransformer and pre-encode taxonomy centroids.
        
        Args:
            model_name: HuggingFace sentence-transformers model checkpoint.
            vector_store: Optional VectorStoreManager for Qdrant storage.
            device: 'cuda', 'cpu', or None for automatic GPU detection.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info("Initializing PersonaProfiler with %s on %s", model_name, self.device)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.vector_store = vector_store
        self.lang_detector = LanguageDetector()

        # Precompute taxonomy centroids
        self.taxonomy_keys: List[str] = list(PERSONA_TAXONOMY.keys())
        taxonomy_texts = [PERSONA_TAXONOMY[k]["description"] for k in self.taxonomy_keys]
        self.centroid_embeddings: np.ndarray = self.model.encode(
            taxonomy_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def infer_age_bracket(self, text: Optional[str]) -> str:
        """Infer approximate age bracket from text clues and keywords."""
        if not text:
            return "unknown"
        for pattern, bracket in AGE_PATTERNS:
            if pattern.search(text):
                return bracket
        return "unknown"

    def profile_persona(
        self,
        user_id: str,
        bio: Optional[str] = None,
        sample_posts: Optional[List[str]] = None,
    ) -> PersonaProfile:
        """Infer persona profile from user bio and recent posts.
        
        Args:
            user_id: Unique user identifier.
            bio: Self-reported profile bio.
            sample_posts: Optional list of sample post texts authored by user.
            
        Returns:
            Structured PersonaProfile.
        """
        combined_parts: List[str] = []
        if bio and bio.strip():
            combined_parts.append(bio.strip())
        if sample_posts:
            # Include up to 5 non-empty sample posts
            valid_posts = [p.strip() for p in sample_posts if p and p.strip()][:5]
            if valid_posts:
                combined_parts.extend(valid_posts)

        full_text = " ".join(combined_parts).strip()

        # Detect language
        detected_lang, _ = self.lang_detector.detect_language(full_text if full_text else None)

        # Infer age bracket
        age_bracket = self.infer_age_bracket(full_text)

        # Fallback for empty or very short bios
        if not full_text or len(full_text) < 4:
            return PersonaProfile(
                user_id=user_id,
                primary_persona="General & Everyday Lifestyle",
                persona_confidence=0.0,
                top_interests=PERSONA_TAXONOMY["General & Everyday Lifestyle"]["interests"],
                age_bracket=age_bracket,
                language=detected_lang,
            )

        # Encode input text
        text_emb = self.model.encode(full_text, normalize_embeddings=True, show_progress_bar=False)

        # Calculate cosine similarity against precomputed taxonomy centroids
        similarities = np.dot(self.centroid_embeddings, text_emb)
        best_idx = int(np.argmax(similarities))
        primary_persona = self.taxonomy_keys[best_idx]
        raw_confidence = float(similarities[best_idx])
        confidence = float(np.clip(raw_confidence, 0.0, 1.0))

        # Extract top interests: combine top archetype interests + second best if close
        interests = list(PERSONA_TAXONOMY[primary_persona]["interests"])
        sorted_indices = np.argsort(similarities)[::-1]
        if len(sorted_indices) > 1:
            second_idx = int(sorted_indices[1])
            second_persona = self.taxonomy_keys[second_idx]
            if similarities[second_idx] > 0.35:
                # Add top interest from second persona
                second_interests = PERSONA_TAXONOMY[second_persona]["interests"]
                for item in second_interests[:2]:
                    if item not in interests:
                        interests.append(item)

        return PersonaProfile(
            user_id=user_id,
            primary_persona=primary_persona,
            persona_confidence=round(confidence, 3),
            top_interests=interests[:5],
            age_bracket=age_bracket,
            language=detected_lang,
        )

    def profile_users_batch(
        self,
        users: List[CanonicalUser],
        user_posts_map: Optional[Dict[str, List[str]]] = None,
    ) -> List[PersonaProfile]:
        """High-throughput batch persona profiling for multiple users.
        
        Args:
            users: List of CanonicalUser objects.
            user_posts_map: Optional mapping of user_id to lists of sample post texts.
            
        Returns:
            List of PersonaProfile instances corresponding to each user.
        """
        if not users:
            return []

        user_posts_map = user_posts_map or {}
        combined_texts: List[str] = []
        indices_with_text: List[int] = []

        for idx, user in enumerate(users):
            parts: List[str] = []
            if user.bio and user.bio.strip():
                parts.append(user.bio.strip())
            posts = user_posts_map.get(user.id, [])
            if posts:
                parts.extend([p.strip() for p in posts if p.strip()][:5])

            text = " ".join(parts).strip()
            combined_texts.append(text)
            if text and len(text) >= 4:
                indices_with_text.append(idx)

        # Batch encode all valid texts
        results: List[Optional[PersonaProfile]] = [None] * len(users)

        if indices_with_text:
            valid_texts = [combined_texts[i] for i in indices_with_text]
            embeddings = self.model.encode(
                valid_texts,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            for i, emb in zip(indices_with_text, embeddings):
                user = users[i]
                text = combined_texts[i]
                detected_lang, _ = self.lang_detector.detect_language(text)
                age_bracket = self.infer_age_bracket(text)

                sims = np.dot(self.centroid_embeddings, emb)
                best_idx = int(np.argmax(sims))
                primary_persona = self.taxonomy_keys[best_idx]
                confidence = float(np.clip(sims[best_idx], 0.0, 1.0))

                interests = list(PERSONA_TAXONOMY[primary_persona]["interests"])
                results[i] = PersonaProfile(
                    user_id=user.id,
                    primary_persona=primary_persona,
                    persona_confidence=round(confidence, 3),
                    top_interests=interests[:5],
                    age_bracket=age_bracket,
                    language=detected_lang,
                )

        # Handle empty/fallback users
        for i, res in enumerate(results):
            if res is None:
                user = users[i]
                text = combined_texts[i]
                lang, _ = self.lang_detector.detect_language(text if text else None)
                results[i] = PersonaProfile(
                    user_id=user.id,
                    primary_persona="General & Everyday Lifestyle",
                    persona_confidence=0.0,
                    top_interests=PERSONA_TAXONOMY["General & Everyday Lifestyle"]["interests"],
                    age_bracket="unknown",
                    language=lang,
                )

        return [r for r in results if r is not None]

    def sync_taxonomy_to_qdrant(self, collection_name: str = "persona_taxonomies") -> None:
        """Upsert taxonomy centroids into Qdrant vector store if configured."""
        if self.vector_store is None:
            logger.info("No vector store configured; skipping Qdrant taxonomy sync.")
            return

        self.vector_store.create_collection(collection_name=collection_name, vector_size=384)
        ids = list(range(1, len(self.taxonomy_keys) + 1))
        vectors = [self.centroid_embeddings[idx].tolist() for idx in range(len(self.taxonomy_keys))]
        payloads = [
            {
                "persona": key,
                "description": PERSONA_TAXONOMY[key]["description"],
                "interests": PERSONA_TAXONOMY[key]["interests"],
            }
            for key in self.taxonomy_keys
        ]

        self.vector_store.upsert_vectors(
            collection_name=collection_name,
            ids=ids,
            vectors=vectors,
            payloads=payloads,
        )
        logger.info("Successfully synced %d persona taxonomy centroids to Qdrant '%s'", len(ids), collection_name)
