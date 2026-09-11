"""Language detection utility for demographic profiling.

Wraps langdetect with robust error handling, URL/symbol stripping,
deterministic seeding, and distribution aggregation.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

logger = logging.getLogger(__name__)

# Enforce deterministic results across runs
DetectorFactory.seed = 0

# Strip noise before passing to langdetect
RE_CLEANUP = re.compile(r"https?://\S+|@\w+|#\w+|[^\w\s]", re.UNICODE)


class LanguageDetector:
    """Detects spoken and written languages for users and posts."""

    def __init__(self, default_lang: str = "en") -> None:
        self.default_lang = default_lang

    def _clean_text(self, text: str) -> str:
        """Strip URLs, mentions, and standalone punctuation."""
        cleaned = RE_CLEANUP.sub(" ", text)
        return re.sub(r"\s+", " ", cleaned).strip()

    def detect_language(self, text: Optional[str]) -> Tuple[str, float]:
        """Detect primary language code and confidence score.
        
        Args:
            text: Input text string.
            
        Returns:
            Tuple of (iso_639_code, confidence_between_0_and_1).
        """
        if not text:
            return self.default_lang, 0.0

        cleaned = self._clean_text(text)
        # If cleaned text has no alphabetic characters or is too short
        if not any(c.isalpha() for c in cleaned) or len(cleaned) < 3:
            return self.default_lang, 0.0

        try:
            results = detect_langs(cleaned)
            if results:
                best = results[0]
                return best.lang, float(best.prob)
            return self.default_lang, 0.0
        except LangDetectException:
            return self.default_lang, 0.0
        except Exception as e:
            logger.debug("Unexpected error during language detection: %s", e)
            return self.default_lang, 0.0

    def detect_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Detect languages for a batch of strings."""
        return [self.detect_language(t) for t in texts]

    def aggregate_languages(self, texts: List[str]) -> Dict[str, int]:
        """Aggregate language distribution count across a list of texts."""
        counts: Dict[str, int] = {}
        for text in texts:
            lang, _ = self.detect_language(text)
            counts[lang] = counts.get(lang, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
