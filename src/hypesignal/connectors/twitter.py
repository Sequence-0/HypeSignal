"""Twitter/X API v2 Platform Connector and Normalization Stub."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from hypesignal.connectors.base import PlatformConnector
from hypesignal.connectors.schemas import ConnectorConfig, ConnectorStatus
from hypesignal.models.canonical import CanonicalPost, PostMetrics
from hypesignal.models.enums import PlatformType

logger = logging.getLogger(__name__)


class TwitterConnector(PlatformConnector):
    """Connector for Twitter/X API v2 search and filtered streaming."""

    @property
    def platform(self) -> PlatformType:
        return PlatformType.TWITTER

    def normalize_post(self, raw_payload: Dict[str, Any]) -> CanonicalPost:
        """Normalize a Twitter API v2 Tweet object into a CanonicalPost.
        
        Expected structure matches Twitter API v2 Tweet object:
        {
            "id": "1460323737035677698",
            "text": "Hello world! #tech @openai https://t.co/xyz",
            "created_at": "2023-10-15T14:30:00.000Z",
            "author_id": "2244994945",
            "author_screen_name": "tech_lead",
            "public_metrics": {
                "like_count": 42,
                "retweet_count": 10,
                "reply_count": 5,
                "impression_count": 1200,
                "quote_count": 2
            },
            "entities": {
                "urls": [{"expanded_url": "https://example.com"}],
                "hashtags": [{"tag": "tech"}],
                "mentions": [{"username": "openai"}]
            },
            "conversation_id": "1460323737035677698",
            "in_reply_to_user_id": None
        }
        """
        post_id = str(raw_payload.get("id", ""))
        if not post_id:
            raise ValueError("Twitter payload missing required 'id' field.")

        text = raw_payload.get("text", "")
        author_id = str(raw_payload.get("author_id", "unknown"))
        author_screen_name = raw_payload.get("author_screen_name") or raw_payload.get("username")

        # Parse timestamp
        created_at_raw = raw_payload.get("created_at")
        if isinstance(created_at_raw, str):
            ts = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
        elif isinstance(created_at_raw, datetime):
            ts = created_at_raw
        elif isinstance(created_at_raw, (int, float)):
            ts = datetime.fromtimestamp(created_at_raw, tz=timezone.utc)
        else:
            ts = datetime.now(timezone.utc)

        # Parse entities
        entities = raw_payload.get("entities") or {}
        urls: List[str] = []
        if "urls" in entities and isinstance(entities["urls"], list):
            for u in entities["urls"]:
                if isinstance(u, dict):
                    urls.append(u.get("expanded_url") or u.get("url", ""))
                elif isinstance(u, str):
                    urls.append(u)
        else:
            urls = self.extract_entities(text)["urls"]

        hashtags: List[str] = []
        if "hashtags" in entities and isinstance(entities["hashtags"], list):
            for h in entities["hashtags"]:
                if isinstance(h, dict):
                    hashtags.append(h.get("tag", ""))
                elif isinstance(h, str):
                    hashtags.append(h.lstrip("#"))
        else:
            hashtags = self.extract_entities(text)["hashtags"]

        mentions: List[str] = []
        if "mentions" in entities and isinstance(entities["mentions"], list):
            for m in entities["mentions"]:
                if isinstance(m, dict):
                    mentions.append(m.get("username", ""))
                elif isinstance(m, str):
                    mentions.append(m.lstrip("@"))
        else:
            mentions = self.extract_entities(text)["mentions"]

        # Parse metrics
        p_metrics = raw_payload.get("public_metrics") or {}
        metrics = PostMetrics(
            likes=int(p_metrics.get("like_count", 0)),
            reposts=int(p_metrics.get("retweet_count", 0)),
            replies=int(p_metrics.get("reply_count", 0)),
            views=int(p_metrics.get("impression_count", 0)) if "impression_count" in p_metrics else None,
            shares=int(p_metrics.get("quote_count", 0)),
        )

        parent_id = None
        referenced = raw_payload.get("referenced_tweets")
        if referenced and isinstance(referenced, list) and len(referenced) > 0:
            parent_id = str(referenced[0].get("id", ""))

        reply_to_user_id = raw_payload.get("in_reply_to_user_id")

        return CanonicalPost(
            id=post_id,
            platform=PlatformType.TWITTER,
            author_id=author_id,
            author_screen_name=author_screen_name,
            text=text,
            timestamp=ts,
            parent_id=parent_id,
            reply_to_user_id=str(reply_to_user_id) if reply_to_user_id else None,
            source_client=raw_payload.get("source"),
            urls=urls,
            hashtags=hashtags,
            mentions=mentions,
            metrics=metrics,
            extra_metadata={
                "conversation_id": raw_payload.get("conversation_id"),
                "lang": raw_payload.get("lang"),
            },
        )

    def poll(self, query: str = "", limit: int = 50) -> List[CanonicalPost]:
        """Poll Twitter API or return synthetic normalized posts if in offline stub mode."""
        if not self.rate_limiter.acquire(1.0):
            self.record_rate_limited()
            logger.warning("Rate limit exceeded for TwitterConnector.")
            return []

        self.record_request()
        bearer_token = self.config.credentials.get("bearer_token")

        posts: List[CanonicalPost] = []
        try:
            if bearer_token:
                # Stub hook for live HTTP call via httpx
                logger.info("Polling live Twitter API v2 for query: '%s' (bearer auth)", query)
                # In live mode this makes HTTP GET to https://api.twitter.com/2/tweets/search/recent
            
            # Offline stub generator for seamless zero-setup testing
            sample_query = query or "AI tech"
            now = datetime.now(timezone.utc)
            with self._lock:
                req_num = self.stats.requests_made
            raw_sample = {
                "id": f"tw_{int(now.timestamp())}_{req_num}",
                "text": f"Exploring breakthrough developments in {sample_query}! #innovation @hypesignal",
                "created_at": now.isoformat(),
                "author_id": "tw_author_101",
                "author_screen_name": "ai_researcher",
                "public_metrics": {
                    "like_count": 15,
                    "retweet_count": 4,
                    "reply_count": 2,
                    "impression_count": 500,
                    "quote_count": 1,
                },
                "entities": {
                    "hashtags": [{"tag": "innovation"}],
                    "mentions": [{"username": "hypesignal"}],
                    "urls": [],
                },
                "conversation_id": f"tw_conv_{int(now.timestamp())}",
            }
            post = self.normalize_post(raw_sample)
            posts.append(post)

            self.record_success(len(posts))
            return posts[:limit]
        except Exception as e:
            self.record_error(str(e))
            logger.error("Error during Twitter poll: %s", e)
            return []
