"""Ground-truth benchmark evaluator against TweetEval datasets.

Calculates accuracy, precision, recall, and macro-F1 scores to validate
zero-shot transformer predictions against the Cardiff NLP benchmark.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from hypesignal.models.enums import EmotionType, SentimentPolarity, StanceType
from hypesignal.nlp.engine import MultiDimensionalSentimentEngine
from hypesignal.nlp.evaluation.tweeteval_loader import TweetEvalLoader

logger = logging.getLogger(__name__)


def compute_classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Compute overall accuracy, per-class precision/recall, and macro-F1 score."""
    if not y_true or not y_pred:
        return {"accuracy": 0.0, "macro_f1": 0.0, "total": 0}

    classes = sorted(list(set(y_true) | set(y_pred)))
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total = len(y_true)
    accuracy = correct / total if total > 0 else 0.0

    per_class = {}
    f1_scores = []

    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[c] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for t in y_true if t == c),
        }
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "total_samples": total,
        "per_class": per_class,
    }


class ModelBenchmarkEvaluator:
    """Evaluates MultiDimensionalSentimentEngine models against TweetEval splits."""

    def __init__(
        self,
        engine: MultiDimensionalSentimentEngine,
        loader: Optional[TweetEvalLoader] = None,
    ) -> None:
        """Initialize evaluator."""
        self.engine = engine
        self.loader = loader or TweetEvalLoader()

    def evaluate_sentiment(
        self,
        split: str = "test",
        limit: Optional[int] = None,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Evaluate sentiment polarity classifier against TweetEval sentiment test set."""
        samples = self.loader.load_split("sentiment", split=split, limit=limit)
        texts = [s.text for s in samples]
        y_true = [s.label_name.lower() for s in samples]

        predictions = self.engine.analyze_sentiment_batch(texts, batch_size=batch_size)
        y_pred = [p.polarity.value for p in predictions]

        metrics = compute_classification_metrics(y_true, y_pred)
        metrics["task"] = "sentiment"
        metrics["split"] = split
        return metrics

    def evaluate_irony(
        self,
        split: str = "test",
        limit: Optional[int] = None,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Evaluate irony/sarcasm detector against TweetEval irony test set."""
        samples = self.loader.load_split("irony", split=split, limit=limit)
        texts = [s.text for s in samples]
        y_true = [s.label_name.lower() for s in samples]

        predictions = self.engine.analyze_irony_batch(texts, batch_size=batch_size)
        y_pred = ["irony" if p.is_ironic else "non_irony" for p in predictions]

        metrics = compute_classification_metrics(y_true, y_pred)
        metrics["task"] = "irony"
        metrics["split"] = split
        return metrics

    def evaluate_stance(
        self,
        target: str = "climate",
        split: str = "test",
        limit: Optional[int] = None,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Evaluate stance detector against TweetEval stance test set."""
        task_name = f"stance/{target.lower()}"
        samples = self.loader.load_split(task_name, split=split, limit=limit)
        texts = [s.text for s in samples]
        # TweetEval labels: 'favor', 'against', 'none'
        y_true = [s.label_name.lower() for s in samples]

        predictions = self.engine.analyze_stance_batch(texts, target=target, batch_size=batch_size)
        # Map StanceType back to string
        pred_map = {
            StanceType.SUPPORTIVE: "favor",
            StanceType.AGAINST: "against",
            StanceType.NONE: "none",
        }
        y_pred = [pred_map.get(p.stance, "none") for p in predictions]

        metrics = compute_classification_metrics(y_true, y_pred)
        metrics["task"] = task_name
        metrics["split"] = split
        return metrics
