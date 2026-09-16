"""Conversation and comment thread sentiment dynamics analyzer (Component B).

Computes conversation emotional arcs, polarity drift from root posts,
controversy dispersion index, hostility velocity, and supportive vs against ratios.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from hypesignal.models.canonical import CanonicalPost
from hypesignal.models.enums import EmotionType, PlatformType, SentimentPolarity, StanceType
from hypesignal.nlp.engine import MultiDimensionalSentimentEngine
from hypesignal.timeline.thread_manager import ConversationThreadManager

logger = logging.getLogger(__name__)


class ThreadSentimentAnalysis(BaseModel):
    """Aggregated sentiment, nuanced emotion, and controversy dynamics of a conversation thread."""
    root_post_id: str
    root_polarity: str
    root_sentiment_score: float
    root_primary_emotion: str
    comment_mean_sentiment: float
    polarity_drift: float  # comment_mean_sentiment - root_sentiment_score
    controversy_index: float  # standard deviation of comment sentiment scores
    hostility_velocity: float  # (anger + anxiety count) / elapsed_minutes
    supportive_ratio: float  # fraction of comments supportive of root claim
    against_ratio: float  # fraction of comments opposing root claim
    neutral_ratio: float  # fraction of neutral stance comments
    total_comments_analyzed: int
    dominant_thread_emotion: str
    emotional_trajectory: List[Dict[str, Any]] = Field(default_factory=list)


class ThreadSentimentAnalyzer:
    """Analyzes chronological sentiment and emotion trajectories across conversation comment threads."""

    def __init__(self, engine: MultiDimensionalSentimentEngine) -> None:
        """Initialize with multi-dimensional NLP engine."""
        self.engine = engine

    def analyze_thread(
        self,
        root_post: CanonicalPost,
        replies: List[CanonicalPost],
        target_topic: Optional[str] = None,
        batch_size: int = 32,
    ) -> ThreadSentimentAnalysis:
        """Analyze emotional trajectory and controversy across a root post and its replies.
        
        Args:
            root_post: The initiating root post.
            replies: Chronologically ordered list of reply posts.
            target_topic: Optional claim or topic for stance evaluation.
            batch_size: Inference batch size.
            
        Returns:
            ThreadSentimentAnalysis containing drift, controversy, and emotional arc.
        """
        target = target_topic or (root_post.text[:60] if root_post.text else "claim")

        # 1. Analyze root post
        root_res = self.engine.analyze_single(root_post.text, target=target)
        root_polarity = root_res.effective_polarity.value
        root_score = float(root_res.adjusted_sentiment_score)
        root_emotion = root_res.emotion.primary_emotion.value

        if not replies:
            return ThreadSentimentAnalysis(
                root_post_id=root_post.id,
                root_polarity=root_polarity,
                root_sentiment_score=round(root_score, 4),
                root_primary_emotion=root_emotion,
                comment_mean_sentiment=0.0,
                polarity_drift=0.0,
                controversy_index=0.0,
                hostility_velocity=0.0,
                supportive_ratio=0.0,
                against_ratio=0.0,
                neutral_ratio=1.0,
                total_comments_analyzed=0,
                dominant_thread_emotion=root_emotion,
                emotional_trajectory=[
                    {
                        "post_id": root_post.id,
                        "timestamp": root_post.timestamp.isoformat(),
                        "is_root": True,
                        "sentiment_score": round(root_score, 4),
                        "polarity": root_polarity,
                        "primary_emotion": root_emotion,
                    }
                ],
            )

        # 2. Analyze all replies in batches
        reply_texts = [r.text for r in replies]
        reply_results = self.engine.analyze_multidimensional(
            reply_texts, target=target, batch_size=batch_size
        )

        comment_scores = [float(r.adjusted_sentiment_score) for r in reply_results]
        comment_polarities = [r.effective_polarity.value for r in reply_results]
        comment_emotions = [r.emotion.primary_emotion.value for r in reply_results]

        # 3. Compute polarity drift
        mean_comment_score = float(sum(comment_scores) / len(comment_scores))
        polarity_drift = mean_comment_score - root_score

        # 4. Compute controversy index (standard deviation of sentiment scores)
        if len(comment_scores) > 1:
            variance = sum((s - mean_comment_score) ** 2 for s in comment_scores) / (len(comment_scores) - 1)
            controversy_index = math.sqrt(variance)
        else:
            controversy_index = 0.0

        # 5. Compute hostility velocity: (anger + anxiety count) / elapsed_minutes
        hostility_count = sum(
            1 for emo in comment_emotions if emo in ("anger", "anxiety", "disgust")
        )
        t_start = replies[0].timestamp
        t_end = replies[-1].timestamp
        elapsed_minutes = max(1.0, (t_end - t_start).total_seconds() / 60.0)
        hostility_velocity = hostility_count / elapsed_minutes

        # 6. Stance distribution (supportive, against, neutral)
        reply_stances = [r.stance for r in reply_results if r.stance is not None]
        stance_counts = Counter(s.stance.value for s in reply_stances)
        total_replies = len(replies)
        supportive_ratio = stance_counts.get("supportive", 0) / total_replies
        against_ratio = stance_counts.get("against", 0) / total_replies
        neutral_ratio = (stance_counts.get("neutral", 0) + stance_counts.get("none", 0)) / total_replies

        # 7. Dominant thread emotion
        dominant_emotion = Counter(comment_emotions).most_common(1)[0][0]

        # 8. Trajectory points
        trajectory: List[Dict[str, Any]] = [
            {
                "post_id": root_post.id,
                "timestamp": root_post.timestamp.isoformat(),
                "is_root": True,
                "sentiment_score": round(root_score, 4),
                "polarity": root_polarity,
                "primary_emotion": root_emotion,
            }
        ]
        for p, res in zip(replies, reply_results):
            trajectory.append(
                {
                    "post_id": p.id,
                    "timestamp": p.timestamp.isoformat(),
                    "is_root": False,
                    "sentiment_score": round(float(res.adjusted_sentiment_score), 4),
                    "polarity": res.effective_polarity.value,
                    "primary_emotion": res.emotion.primary_emotion.value,
                }
            )

        return ThreadSentimentAnalysis(
            root_post_id=root_post.id,
            root_polarity=root_polarity,
            root_sentiment_score=round(root_score, 4),
            root_primary_emotion=root_emotion,
            comment_mean_sentiment=round(mean_comment_score, 4),
            polarity_drift=round(polarity_drift, 4),
            controversy_index=round(controversy_index, 4),
            hostility_velocity=round(hostility_velocity, 4),
            supportive_ratio=round(supportive_ratio, 4),
            against_ratio=round(against_ratio, 4),
            neutral_ratio=round(neutral_ratio, 4),
            total_comments_analyzed=total_replies,
            dominant_thread_emotion=dominant_emotion,
            emotional_trajectory=trajectory,
        )

    def analyze_thread_from_db(
        self,
        root_post_id: str,
        thread_manager: ConversationThreadManager,
        target_topic: Optional[str] = None,
        batch_size: int = 32,
    ) -> Optional[ThreadSentimentAnalysis]:
        """Query DuckDB for thread posts and compute sentiment dynamics.
        
        Args:
            root_post_id: Root post ID.
            thread_manager: ConversationThreadManager instance.
            target_topic: Optional target topic for stance evaluation.
            batch_size: Inference batch size.
            
        Returns:
            ThreadSentimentAnalysis or None if root post not found.
        """
        thread_df = thread_manager.db.con.execute(
            "SELECT * FROM posts WHERE id = ?", [root_post_id]
        ).pl()
        if thread_df.is_empty():
            return None

        # Build root post
        r = thread_df.to_dicts()[0]
        ts = r["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        plat_val = r.get("platform")
        plat = PlatformType(plat_val) if plat_val else PlatformType.DATASET

        root_post = CanonicalPost(
            id=str(r["id"]),
            author_id=str(r["author_id"]),
            author_screen_name=r.get("author_screen_name"),
            text=r.get("text", ""),
            timestamp=ts,
            parent_id=str(r["parent_id"]) if r.get("parent_id") else None,
            platform=plat,
        )

        replies = thread_manager.get_chronological_reply_stream(root_post_id)
        return self.analyze_thread(
            root_post=root_post,
            replies=replies,
            target_topic=target_topic,
            batch_size=batch_size,
        )
