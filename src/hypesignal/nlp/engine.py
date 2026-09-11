"""Multi-dimensional sentiment, emotion, irony, and stance inference engine.

Uses pretrained zero-shot Cardiff NLP and DistilRoBERTa transformers with
GPU acceleration, float16 precision, and sarcasm-inversion logic.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

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

    def analyze_emotion_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[EmotionPrediction]:
        """Classify nuanced emotions (joy, anger, sadness, fear/anxiety, surprise, disgust)."""
        if not texts:
            return []

        probs, pred_indices = self._predict_logits(
            self.EMOTION_MODEL_NAME, texts, batch_size=batch_size
        )
        model, _ = self._get_or_load_pipeline(self.EMOTION_MODEL_NAME)
        id2label = model.config.id2label

        # Map DistilRoBERTa emotion classes to EmotionType enum
        label_map = {
            "joy": EmotionType.JOY,
            "fear": EmotionType.ANXIETY,
            "anger": EmotionType.ANGER,
            "sadness": EmotionType.SADNESS,
            "surprise": EmotionType.SURPRISE,
            "disgust": EmotionType.DISGUST,
            "neutral": EmotionType.NEUTRAL,
        }

        results = []
        for prob_dict, idx in zip(probs, pred_indices):
            raw_emotion = id2label[idx].lower()
            mapped_emotion = label_map.get(raw_emotion, EmotionType.NEUTRAL)
            score = prob_dict.get(raw_emotion, 0.0)

            results.append(
                EmotionPrediction(
                    primary_emotion=mapped_emotion,
                    score=score,
                    probabilities=prob_dict,
                )
            )
        return results

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
            stances = self.analyze_stance_batch(texts, target=target, batch_size=batch_size)

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
