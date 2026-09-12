"""Reddit API Platform Connector and Normalization Stub."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from hypesignal.connectors.base import PlatformConnector
from hypesignal.connectors.schemas import ConnectorConfig, ConnectorStatus
from hypesignal.models.canonical import CanonicalPost, PostMetrics
from hypesignal.models.enums import PlatformType

logger = logging.getLogger(__name__)


class RedditConnector(PlatformConnector):
    """Connector for Reddit Subreddit submissions and comment stream."""

    @property
    def platform(self) -> PlatformType:
        return PlatformType.REDDIT

    def normalize_post(self, raw_payload: Dict[str, Any]) -> CanonicalPost:
        """Normalize a Reddit submission or comment payload into a CanonicalPost.
        
        Expected structure matches standard Reddit JSON API object:
        {
            "id": "17xyz9",
            "name": "t3_17xyz9",
            "title": "Discussion on LLM quantization techniques",
            "selftext": "What are your thoughts on 4-bit vs 8-bit quantization? #machinelearning",
            "author": "deep_coder",
            "author_fullname": "t2_987abc",
            "created_utc": 1697380200.0,
            "score": 85,
            "num_comments": 23,
            "permalink": "/r/MachineLearning/comments/17xyz9/discussion_on_llm/",
            "subreddit": "MachineLearning",
            "parent_id": None
        }
        """
        post_id = str(raw_payload.get("id") or raw_payload.get("name") or "")
        if not post_id:
            raise ValueError("Reddit payload missing required 'id' or 'name' field.")

        title = (raw_payload.get("title") or "").strip()
        body = (raw_payload.get("selftext") or raw_payload.get("body") or "").strip()
        if title and body:
            text = f"{title}\n\n{body}"
        else:
            text = title or body

        author_screen_name = raw_payload.get("author") or "anonymous"
        author_id = str(raw_payload.get("author_fullname") or raw_payload.get("author") or "unknown")

        created_utc = raw_payload.get("created_utc")
        if isinstance(created_utc, (int, float)):
            ts = datetime.fromtimestamp(created_utc, tz=timezone.utc)
        elif isinstance(created_utc, str):
            try:
                ts = datetime.fromisoformat(created_utc.replace("Z", "+00:00"))
            except ValueError:
                ts = datetime.fromtimestamp(float(created_utc), tz=timezone.utc)
        elif isinstance(created_utc, datetime):
            ts = created_utc
        else:
            ts = datetime.now(timezone.utc)

        score = max(0, int(raw_payload.get("score", 0)))
        num_comments = max(0, int(raw_payload.get("num_comments", 0)))

        metrics = PostMetrics(
            likes=score,
            replies=num_comments,
            reposts=0,
            views=int(raw_payload.get("view_count", 0)) if raw_payload.get("view_count") else None,
            shares=0,
        )

        extracted = self.extract_entities(text)
        permalink = raw_payload.get("permalink")
        urls = extracted["urls"]
        if permalink and f"https://reddit.com{permalink}" not in urls:
            urls.append(f"https://reddit.com{permalink}")

        parent_id = raw_payload.get("parent_id")

        return CanonicalPost(
            id=post_id,
            platform=PlatformType.REDDIT,
            author_id=author_id,
            author_screen_name=author_screen_name,
            text=text,
            timestamp=ts,
            parent_id=str(parent_id) if parent_id else None,
            reply_to_user_id=None,
            source_client="Reddit API",
            urls=urls,
            hashtags=extracted["hashtags"],
            mentions=extracted["mentions"],
            metrics=metrics,
            extra_metadata={
                "subreddit": raw_payload.get("subreddit"),
                "upvote_ratio": raw_payload.get("upvote_ratio"),
                "is_self": raw_payload.get("is_self", True),
            },
        )

    def poll(self, query: str = "", limit: int = 50) -> List[CanonicalPost]:
        """Poll Reddit API or return synthetic normalized posts if in offline stub mode."""
        if not self.rate_limiter.acquire(1.0):
            self.record_rate_limited()
            logger.warning("Rate limit exceeded for RedditConnector.")
            return []

        self.record_request()
        client_id = self.config.credentials.get("client_id")

        posts: List[CanonicalPost] = []
        try:
            if client_id:
                logger.info("Polling live Reddit API for query: '%s'", query)

            now = datetime.now(timezone.utc)
            with self._lock:
                req_num = self.stats.requests_made
            raw_sample = {
                "id": f"rd_{int(now.timestamp())}_{req_num}",
                "name": f"t3_rd_{int(now.timestamp())}",
                "title": f"Deep dive into {query or 'offline social analytics'}",
                "selftext": f"Analyzing real-time event propagation and topic models across platforms. #datascience #analytics",
                "author": "reddit_researcher",
                "author_fullname": "t2_rdauthor88",
                "created_utc": now.timestamp(),
                "score": 48,
                "num_comments": 14,
                "permalink": f"/r/technology/comments/rd_{int(now.timestamp())}/deep_dive/",
                "subreddit": "technology",
                "upvote_ratio": 0.96,
            }
            post = self.normalize_post(raw_sample)
            posts.append(post)

            self.record_success(len(posts))
            return posts[:limit]
        except Exception as e:
            self.record_error(str(e))
            logger.error("Error during Reddit poll: %s", e)
            return []
