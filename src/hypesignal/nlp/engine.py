"""Multi-dimensional sentiment, emotion, irony, and stance inference engine.

Uses pretrained zero-shot Cardiff NLP and DistilRoBERTa transformers with
GPU acceleration, float16 precision, and sarcasm-inversion logic.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hypesignal.models.enums import EmotionType, SentimentPolarity, StanceType
from hypesignal.nlp.schemas import (
    EmotionPrediction,
    IronyPrediction,
    MultiDimensionalResult,
    SentimentPrediction,
    StancePrediction,
)

logger = logging.getLogger(__name__)

# Official TweetEval Cardiff NLP preprocessing rules
RE_USER = re.compile(r"@\S+")
RE_HTTP = re.compile(r"https?://\S+")


def preprocess_tweet(text: str) -> str:
    """Normalize user mentions and URLs according to Cardiff NLP standard."""
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text).strip()


# Explicit lexical & arousal calibration markers for nuanced emotion isolation
ANXIETY_STRONG_MARKERS: Set[str] = {
    "panic", "panicked", "panicking", "terrified", "terrifying", "dread", "dreading",
    "anxiety", "anxious", "anxiously", "paralyzed", "horrified", "freaking out",
}
ANXIETY_MODERATE_MARKERS: Set[str] = {
    "worried", "worry", "worrying", "nervous", "stressed", "stressful", "stress",
    "uneasy", "apprehensive", "fearful", "troubled", "concerned", "overwhelmed", "hesitant", "distressed",
}

EXCITEMENT_STRONG_MARKERS: Set[str] = {
    "hyped", "hype", "ecstatic", "thrilled", "thrilling", "pumped", "electrifying",
    "unbelievable", "epic", "insane", "fire", "🚀", "🔥", "breathtaking", "phenomenal",
}
EXCITEMENT_MODERATE_MARKERS: Set[str] = {
    "excited", "exciting", "excitement", "can't wait", "cant wait", "amazing", "great",
    "fantastic", "awesome", "eager", "enthusiastic", "stoked", "delighted", "energized",
}

OPTIMISM_STRONG_MARKERS: Set[str] = {
    "hope", "hopeful", "optimistic", "optimism", "promising", "bright future",
}
OPTIMISM_MODERATE_MARKERS: Set[str] = {
    "looking forward", "progress", "confident", "positive outlook",
}


def _matches_marker(text_lower: str, marker: str) -> bool:
    """Check if marker matches text using word boundaries for text tokens, or exact match for emoji."""
    if any(ord(char) > 127 for char in marker):
        return marker in text_lower
    pattern = r"\b" + re.escape(marker) + r"\b"
    return bool(re.search(pattern, text_lower))


def compute_lexical_score(
    text: str,
    strong_markers: Set[str],
    moderate_markers: Set[str],
    cap: float = 2.0,
) -> float:
    """Compute normalized matched keyword intensity count for lexical calibration using word boundaries."""
    text_lower = text.lower()
    score = 0.0
    for marker in strong_markers:
        if _matches_marker(text_lower, marker):
            score += 1.0
    for marker in moderate_markers:
        if _matches_marker(text_lower, marker):
            score += 0.5
    if "!" in text:
        score += 0.25
    if any(word.isupper() and len(word) > 2 for word in text.split()):
        score += 0.25
    return min(1.0, score / cap)


class MultiDimensionalSentimentEngine:
    """Unified multi-dimensional NLP inference engine."""

    SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    IRONY_MODEL_NAME = "cardiffnlp/twitter-roberta-base-irony"
    EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
    DEFAULT_STANCE_MODEL = "cardiffnlp/twitter-roberta-base-stance-climate"

    SUPPORTED_STANCE_TARGETS: Dict[str, str] = {
        "climate": "cardiffnlp/twitter-roberta-base-stance-climate",
        "abortion": "cardiffnlp/twitter-roberta-base-stance-abortion",
        "atheism": "cardiffnlp/twitter-roberta-base-stance-atheism",
        "feminist": "cardiffnlp/twitter-roberta-base-stance-feminist",
        "hillary": "cardiffnlp/twitter-roberta-base-stance-hillary",
    }

    def __init__(
        self,
        device: Optional[str] = None,
        fp16: bool = True,
    ) -> None:
        """Initialize engine and select inference device (CUDA / CPU)."""
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fp16 = fp16 and (self.device.type == "cuda")
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        logger.info("Initialized MultiDimensionalSentimentEngine on device: %s (fp16=%s)", self.device, self.fp16)

    def _get_or_load_pipeline(self, model_name: str) -> Tuple[Any, Any]:
        """Lazy load model and tokenizer on the selected device."""
        if model_name not in self._models:
            logger.info("Loading model and tokenizer: %s", model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            if self.fp16:
                model = model.half()
            model.to(self.device)
            model.eval()

            self._models[model_name] = model
            self._tokenizers[model_name] = tokenizer

        return self._models[model_name], self._tokenizers[model_name]

    def _predict_logits(
        self,
        model_name: str,
        texts: List[str],
        batch_size: int = 32,
    ) -> Tuple[List[Dict[str, float]], List[int]]:
        """Batch inference returning softmax probabilities and argmax label indices."""
        model, tokenizer = self._get_or_load_pipeline(model_name)
        id2label = model.config.id2label

        all_probs: List[Dict[str, float]] = []
        all_pred_indices: List[int] = []

        preprocessed = [preprocess_tweet(t) for t in texts]

        for i in range(0, len(preprocessed), batch_size):
            batch = preprocessed[i : i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**encoded)
                probs = F.softmax(outputs.logits, dim=-1).cpu().float().numpy()

            for row in probs:
                prob_dict = {id2label[idx]: float(val) for idx, val in enumerate(row)}
                best_idx = int(row.argmax())
                all_probs.append(prob_dict)
                all_pred_indices.append(best_idx)

        return all_probs, all_pred_indices

    def analyze_sentiment_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[SentimentPrediction]:
        """Classify sentiment polarity (positive, neutral, negative)."""
        if not texts:
            return []

        probs, pred_indices = self._predict_logits(
            self.SENTIMENT_MODEL_NAME, texts, batch_size=batch_size
        )
        model, _ = self._get_or_load_pipeline(self.SENTIMENT_MODEL_NAME)
        id2label = model.config.id2label

        results = []
        for prob_dict, idx in zip(probs, pred_indices):
            raw_label = id2label[idx].lower()
            try:
                polarity = SentimentPolarity(raw_label)
            except ValueError:
                polarity = SentimentPolarity.NEUTRAL

            results.append(
                SentimentPrediction(
                    polarity=polarity,
                    score=prob_dict[raw_label],
                    probabilities=prob_dict,
                )
            )
        return results

    def analyze_irony_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[IronyPrediction]:
        """Detect sarcasm / irony."""
        if not texts:
            return []

        probs, pred_indices = self._predict_logits(
            self.IRONY_MODEL_NAME, texts, batch_size=batch_size
        )

        results = []
        for prob_dict, idx in zip(probs, pred_indices):
            irony_score = prob_dict.get("irony", prob_dict.get("1", 0.0))
            is_ironic = (idx == 1) or (irony_score > 0.5)

            results.append(
                IronyPrediction(
                    is_ironic=is_ironic,
                    irony_score=irony_score,
                    probabilities=prob_dict,
                )
            )
        return results

    def analyze_nuanced_emotions_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        alpha_anx: float = 0.6,
        alpha_exc: float = 0.6,
    ) -> List[EmotionPrediction]:
        """Classify nuanced emotions with lexical-arousal calibration for anxiety and excitement."""
        if not texts:
            return []

        probs, _ = self._predict_logits(
            self.EMOTION_MODEL_NAME, texts, batch_size=batch_size
        )

        results: List[EmotionPrediction] = []
        for text, raw_probs in zip(texts, probs):
            # Base emotion probabilities from model
            p_joy = float(raw_probs.get("joy", 0.0))
            p_fear = float(raw_probs.get("fear", 0.0))
            p_anger = float(raw_probs.get("anger", 0.0))
            p_sadness = float(raw_probs.get("sadness", 0.0))
            p_surprise = float(raw_probs.get("surprise", 0.0))
            p_disgust = float(raw_probs.get("disgust", 0.0))
            p_neutral = float(raw_probs.get("neutral", 0.0))

            # Lexical calibration scores
            s_anx = compute_lexical_score(text, ANXIETY_STRONG_MARKERS, ANXIETY_MODERATE_MARKERS)
            s_exc = compute_lexical_score(text, EXCITEMENT_STRONG_MARKERS, EXCITEMENT_MODERATE_MARKERS)
            s_opt = compute_lexical_score(text, OPTIMISM_STRONG_MARKERS, OPTIMISM_MODERATE_MARKERS)

            # Calibrate anxiety from fear only if anxiety lexical markers are present
            if s_anx > 0.0:
                p_anxiety = min(p_fear, p_fear * alpha_anx + (1.0 - alpha_anx) * s_anx)
                p_fear_remaining = max(0.0, p_fear - p_anxiety)
            else:
                p_anxiety = 0.0
                p_fear_remaining = p_fear

            # Calibrate excitement from joy only if excitement lexical markers are present
            if s_exc > 0.0:
                p_excitement = min(p_joy, p_joy * alpha_exc + (1.0 - alpha_exc) * s_exc)
                p_joy_remaining = max(0.0, p_joy - p_excitement)
            else:
                p_excitement = 0.0
                p_joy_remaining = p_joy

            # Optimism heuristic (decoupled from excitement; derived from joy and optimism markers)
            if s_opt > 0.0:
                p_optimism = min(p_joy_remaining, round(p_joy_remaining * 0.5 * s_opt, 4))
                p_joy_remaining = max(0.0, p_joy_remaining - p_optimism)
            else:
                p_optimism = 0.0

            calibrated_dict = {
                "joy": p_joy_remaining,
                "optimism": p_optimism,
                "anger": p_anger,
                "sadness": p_sadness,
                "fear": p_fear_remaining,
                "anxiety": p_anxiety,
                "excitement": p_excitement,
                "surprise": p_surprise,
                "disgust": p_disgust,
                "neutral": p_neutral,
            }

            # Normalize to sum 1.0 and fix rounding drift on dominant key
            total = sum(calibrated_dict.values())
            if total > 0.0:
                for k in calibrated_dict:
                    calibrated_dict[k] = round(calibrated_dict[k] / total, 4)
                drift = round(1.0 - sum(calibrated_dict.values()), 4)
                if drift != 0.0:
                    dominant_k = max(calibrated_dict, key=lambda k: calibrated_dict[k])
                    calibrated_dict[dominant_k] = round(calibrated_dict[dominant_k] + drift, 4)

            # Determine dominant emotion
            primary_label, max_score = max(calibrated_dict.items(), key=lambda item: item[1])
            emotion_enum = EmotionType(primary_label)

            results.append(
                EmotionPrediction(
                    primary_emotion=emotion_enum,
                    score=max_score,
                    probabilities=calibrated_dict,
                )
            )
        return results

    def analyze_emotion_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[EmotionPrediction]:
        """Classify nuanced emotions (calls calibrated analyze_nuanced_emotions_batch)."""
        return self.analyze_nuanced_emotions_batch(texts, batch_size=batch_size)

    def analyze_stance_batch(
        self,
        texts: List[str],
        target: str = "climate",
        batch_size: int = 32,
    ) -> List[StancePrediction]:
        """Classify stance towards a target (favor, against, none)."""
        if not texts:
            return []

        target_norm = target.lower().strip()
        if target_norm in self.SUPPORTED_STANCE_TARGETS:
            model_name = self.SUPPORTED_STANCE_TARGETS[target_norm]
        else:
            logger.warning(
                "Stance target '%s' not in supported targets %s. Falling back to default: %s",
                target,
                list(self.SUPPORTED_STANCE_TARGETS.keys()),
                self.DEFAULT_STANCE_MODEL,
            )
            model_name = self.DEFAULT_STANCE_MODEL

        probs, pred_indices = self._predict_logits(model_name, texts, batch_size=batch_size)
        model, _ = self._get_or_load_pipeline(model_name)
        id2label = model.config.id2label

        stance_map = {
            "favor": StanceType.SUPPORTIVE,
            "against": StanceType.AGAINST,
            "none": StanceType.NONE,
        }

        results = []
        for prob_dict, idx in zip(probs, pred_indices):
            raw_label = id2label[idx].lower()
            stance = stance_map.get(raw_label, StanceType.NONE)
            score = prob_dict.get(raw_label, 0.0)

            results.append(
                StancePrediction(
                    stance=stance,
                    score=score,
                    target=target,
                    probabilities=prob_dict,
                )
            )
        return results

    def analyze_multidimensional(
        self,
        texts: List[str],
        target: Optional[str] = None,
        irony_threshold: float = 0.85,
        batch_size: int = 32,
    ) -> List[MultiDimensionalResult]:
        """Perform comprehensive multi-dimensional inference with sarcasm-inversion logic."""
        if not texts:
            return []

        sentiments = self.analyze_sentiment_batch(texts, batch_size=batch_size)
        ironies = self.analyze_irony_batch(texts, batch_size=batch_size)
        emotions = self.analyze_emotion_batch(texts, batch_size=batch_size)

        stances: Optional[List[StancePrediction]] = None
        if target:
            stances = self.analyze_stance_towards_target(texts, target=target, batch_size=batch_size)

        results: List[MultiDimensionalResult] = []

        for i in range(len(texts)):
            text = texts[i]
            sent = sentiments[i]
            iro = ironies[i]
            emo = emotions[i]
            st = stances[i] if stances else None

            # Sarcasm Inversion Logic:
            # When surface sentiment is positive, but irony/sarcasm is detected with high confidence,
            # the effective sentiment inverts to NEGATIVE.
            is_inverted = False
            effective_polarity = sent.polarity
            explanation = None

            if sent.polarity == SentimentPolarity.POSITIVE and iro.irony_score >= irony_threshold:
                effective_polarity = SentimentPolarity.NEGATIVE
                is_inverted = True
                adjusted_score = -float(sent.score)
                explanation = (
                    f"Surface sentiment was positive ({sent.score:.2f}), but strong irony/sarcasm "
                    f"detected ({iro.irony_score:.2f} >= {irony_threshold}). Inverted to NEGATIVE."
                )
            elif effective_polarity == SentimentPolarity.POSITIVE:
                adjusted_score = float(sent.score)
            elif effective_polarity == SentimentPolarity.NEGATIVE:
                adjusted_score = -float(sent.score)
            else:
                adjusted_score = 0.0

            results.append(
                MultiDimensionalResult(
                    text=text,
                    sentiment=sent,
                    irony=iro,
                    emotion=emo,
                    stance=st,
                    effective_polarity=effective_polarity,
                    adjusted_sentiment_score=adjusted_score,
                    is_sarcasm_inverted=is_inverted,
                    explanation=explanation,
                )
            )

        return results

    def analyze_single(
        self,
        text: str,
        target: Optional[str] = None,
        irony_threshold: float = 0.85,
    ) -> MultiDimensionalResult:
        """Analyze a single text string."""
        return self.analyze_multidimensional([text], target=target, irony_threshold=irony_threshold)[0]

    def analyze_stance_towards_target(
        self,
        texts: List[str],
        target: str = "general",
        batch_size: int = 32,
    ) -> List[StancePrediction]:
        """Classify stance towards an explicit target (supportive, against, neutral)."""
        if not texts:
            return []

        target_norm = target.lower().strip() if target else "general"
        if target_norm in self.SUPPORTED_STANCE_TARGETS:
            return self.analyze_stance_batch(texts, target=target_norm, batch_size=batch_size)

        # Domain-agnostic stance inference via sentiment polarity alignment towards target
        sentiments = self.analyze_sentiment_batch(texts, batch_size=batch_size)
        results: List[StancePrediction] = []
        for s in sentiments:
            if s.polarity == SentimentPolarity.POSITIVE:
                stance = StanceType.SUPPORTIVE
            elif s.polarity == SentimentPolarity.NEGATIVE:
                stance = StanceType.AGAINST
            else:
                stance = StanceType.NEUTRAL

            results.append(
                StancePrediction(
                    stance=stance,
                    score=s.score,
                    target=target,
                    probabilities={
                        "supportive": s.probabilities.get("positive", 0.0),
                        "against": s.probabilities.get("negative", 0.0),
                        "neutral": s.probabilities.get("neutral", 0.0),
                    },
                )
            )
        return results

    def analyze_and_flatten(
        self,
        texts: List[str],
        post_ids: List[str],
        target: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """Run batch inference and return flattened dictionaries matching post_analytics schema."""
        if not texts or not post_ids:
            return []

        results = self.analyze_multidimensional(texts, target=target, batch_size=batch_size)
        flattened: List[Dict[str, Any]] = []

        for p_id, res in zip(post_ids, results):
            emo_probs = res.emotion.probabilities
            stance_val = res.stance.stance.value if res.stance else "neutral"
            stance_score = res.stance.score if res.stance else 0.0

            row = {
                "post_id": str(p_id),
                "effective_polarity": res.effective_polarity.value,
                "sentiment_score": float(res.adjusted_sentiment_score),
                "is_sarcastic": bool(res.is_sarcasm_inverted or res.irony.is_ironic),
                "irony_score": float(res.irony.irony_score),
                "primary_emotion": res.emotion.primary_emotion.value,
                "emotion_score": float(res.emotion.score),
                "joy": float(emo_probs.get("joy", 0.0)),
                "optimism": float(emo_probs.get("optimism", 0.0)),
                "anger": float(emo_probs.get("anger", 0.0)),
                "sadness": float(emo_probs.get("sadness", 0.0)),
                "fear": float(emo_probs.get("fear", 0.0)),
                "anxiety": float(emo_probs.get("anxiety", 0.0)),
                "excitement": float(emo_probs.get("excitement", 0.0)),
                "surprise": float(emo_probs.get("surprise", 0.0)),
                "disgust": float(emo_probs.get("disgust", 0.0)),
                "neutral": float(emo_probs.get("neutral", 0.0)),
                "stance": stance_val,
                "stance_score": float(stance_score),
            }
            flattened.append(row)
        return flattened

