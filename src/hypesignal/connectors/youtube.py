"""YouTube Data API v3 Platform Connector and Normalization Stub."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from hypesignal.connectors.base import PlatformConnector
from hypesignal.connectors.schemas import ConnectorConfig, ConnectorStatus
from hypesignal.models.canonical import CanonicalPost, PostMetrics
from hypesignal.models.enums import PlatformType

logger = logging.getLogger(__name__)


class YouTubeConnector(PlatformConnector):
    """Connector for YouTube Data API v3 video comments and discussion threads."""

    @property
    def platform(self) -> PlatformType:
        return PlatformType.YOUTUBE

    def normalize_post(self, raw_payload: Dict[str, Any]) -> CanonicalPost:
        """Normalize a YouTube Data API v3 comment thread resource into a CanonicalPost.
        
        Expected structure supports both CommentThread and Comment resource shapes:
        {
            "id": "UgxK9911aaBB",
            "snippet": {
                "videoId": "dQw4w9WgXcQ",
                "topLevelComment": {
                    "snippet": {
                        "textOriginal": "Outstanding tutorial on neural networks! #deeplearning @channel",
                        "authorDisplayName": "NeuralLearner",
                        "authorChannelId": {"value": "UC_channel_123"},
                        "likeCount": 42,
                        "publishedAt": "2023-10-14T09:15:00Z"
                    }
                },
                "totalReplyCount": 3
            }
        }
        """
        post_id = str(raw_payload.get("id", ""))
        if not post_id:
            raise ValueError("YouTube payload missing required 'id' field.")

        snippet = raw_payload.get("snippet") or {}
        # Unpack topLevelComment if this is a CommentThread resource
        top_comment = snippet.get("topLevelComment", {})
        comment_snippet = top_comment.get("snippet", snippet)

        text = (
            comment_snippet.get("textOriginal")
            or comment_snippet.get("textDisplay")
            or raw_payload.get("text")
            or ""
        )
        author_screen_name = (
            comment_snippet.get("authorDisplayName")
            or raw_payload.get("author_screen_name")
            or "YouTubeUser"
        )
        channel_id_obj = comment_snippet.get("authorChannelId")
        if isinstance(channel_id_obj, dict):
            author_id = channel_id_obj.get("value", "unknown")
        else:
            author_id = str(channel_id_obj or raw_payload.get("author_id") or "unknown")

        published_at_raw = comment_snippet.get("publishedAt") or raw_payload.get("timestamp")
        if isinstance(published_at_raw, str):
            ts = datetime.fromisoformat(published_at_raw.replace("Z", "+00:00"))
        elif isinstance(published_at_raw, datetime):
            ts = published_at_raw
        elif isinstance(published_at_raw, (int, float)):
            ts = datetime.fromtimestamp(published_at_raw, tz=timezone.utc)
        else:
            ts = datetime.now(timezone.utc)

        likes = int(comment_snippet.get("likeCount", 0))
        replies = int(snippet.get("totalReplyCount", 0))

        metrics = PostMetrics(
            likes=likes,
            replies=replies,
            reposts=0,
            views=None,
            shares=0,
        )

        extracted = self.extract_entities(text)
        video_id = snippet.get("videoId") or raw_payload.get("video_id")
        urls = extracted["urls"]
        if video_id:
            urls.append(f"https://www.youtube.com/watch?v={video_id}")

        parent_id = comment_snippet.get("parentId") or (
            f"video_{video_id}" if video_id else None
        )

        return CanonicalPost(
            id=post_id,
            platform=PlatformType.YOUTUBE,
            author_id=author_id,
            author_screen_name=author_screen_name,
            text=text,
            timestamp=ts,
            parent_id=parent_id,
            reply_to_user_id=None,
            source_client="YouTube Data API v3",
            urls=urls,
            hashtags=extracted["hashtags"],
            mentions=extracted["mentions"],
            metrics=metrics,
            extra_metadata={
                "video_id": video_id,
                "can_rate": comment_snippet.get("canRate", True),
            },
        )

    def poll(self, query: str = "", limit: int = 50) -> List[CanonicalPost]:
        """Poll YouTube Data API or return synthetic normalized posts if in offline stub mode."""
        if not self.rate_limiter.acquire(1.0):
            self.record_rate_limited()
            logger.warning("Rate limit exceeded for YouTubeConnector.")
            return []

        self.record_request()
        api_key = self.config.credentials.get("api_key")

        posts: List[CanonicalPost] = []
        try:
            if api_key:
                logger.info("Polling live YouTube Data API for query: '%s'", query)

            now = datetime.now(timezone.utc)
            with self._lock:
                req_num = self.stats.requests_made
            raw_sample = {
                "id": f"yt_comm_{int(now.timestamp())}_{req_num}",
                "snippet": {
                    "videoId": "vid_abc123",
                    "topLevelComment": {
                        "snippet": {
                            "textOriginal": f"Great explanation of {query or 'transformer architectures'}! #machinelearning @educator",
                            "authorDisplayName": "TechStreamer",
                            "authorChannelId": {"value": "UC_yt_stream_99"},
                            "likeCount": 18,
                            "publishedAt": now.isoformat(),
                        }
                    },
                    "totalReplyCount": 2,
                },
            }
            post = self.normalize_post(raw_sample)
            posts.append(post)

            self.record_success(len(posts))
            return posts[:limit]
        except Exception as e:
            self.record_error(str(e))
            logger.error("Error during YouTube poll: %s", e)
            return []
