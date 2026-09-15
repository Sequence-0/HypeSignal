"""YouTube Data API v3 Platform Connector with Quota Guardrails and Normalization."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from hypesignal.connectors.base import PlatformConnector
from hypesignal.connectors.schemas import ConnectorConfig
from hypesignal.models.canonical import CanonicalPost, PostMetrics
from hypesignal.models.enums import PlatformType

logger = logging.getLogger(__name__)

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"

# Costs in quota units per YouTube Data API v3 documentation:
# - commentThreads.list: 1 unit
# - comments.list: 1 unit
# - search.list: 100 units
QUOTA_COST_COMMENT_THREADS = 1
QUOTA_COST_SEARCH = 100
DEFAULT_DAILY_QUOTA_LIMIT = 5000  # Conservative safety buffer below 10,000 free tier quota


def _sanitize_key(key: Optional[str]) -> str:
    """Mask API key for safe logging."""
    if not key:
        return "<none>"
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def _sanitize_url(url: str) -> str:
    """Mask key parameter in URL strings to prevent accidental leakage in logs."""
    return re.sub(r"([?&]key=)[^&]+", r"\1[REDACTED]", str(url))


class YouTubeConnector(PlatformConnector):
    """Connector for YouTube Data API v3 video comments, threads, and topics with quota guardrails."""

    def __init__(
        self,
        config: Optional[ConnectorConfig] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        """Initialize YouTube connector with optional custom HTTP client and daily quota tracking."""
        super().__init__(config=config)
        self._http_client = http_client
        self._owns_client = http_client is None

        # Daily quota tracking guardrail
        creds = self.config.credentials
        self.daily_quota_limit = int(
            creds.get("daily_quota_limit")
            or os.getenv("YOUTUBE_DAILY_QUOTA_LIMIT", DEFAULT_DAILY_QUOTA_LIMIT)
        )
        self._quota_used = 0
        self._quota_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @property
    def platform(self) -> PlatformType:
        return PlatformType.YOUTUBE

    def _get_client(self) -> httpx.Client:
        """Lazily initialize thread-safe httpx client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.Client(timeout=10.0)
            self._owns_client = True
        return self._http_client

    def disconnect(self) -> None:
        """Disconnect and release HTTP client."""
        super().disconnect()
        if self._owns_client and self._http_client and not self._http_client.is_closed:
            self._http_client.close()

    def _check_and_consume_quota(self, cost: int) -> bool:
        """Verify whether sufficient daily quota remains before making a request."""
        with self._lock:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if today != self._quota_date:
                self._quota_date = today
                self._quota_used = 0

            if self._quota_used + cost > self.daily_quota_limit:
                logger.warning(
                    "YouTube daily quota limit reached (%d/%d units). Live API request blocked.",
                    self._quota_used,
                    self.daily_quota_limit,
                )
                return False

            self._quota_used += cost
            return True

    def get_quota_stats(self) -> Dict[str, Any]:
        """Retrieve current daily quota consumption metrics."""
        with self._lock:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if today != self._quota_date:
                self._quota_date = today
                self._quota_used = 0
            return {
                "daily_quota_limit": self.daily_quota_limit,
                "quota_used": self._quota_used,
                "quota_remaining": max(0, self.daily_quota_limit - self._quota_used),
                "quota_date": self._quota_date,
            }

    def reset_quota(self) -> None:
        """Reset quota consumption counters (primarily for testing)."""
        with self._lock:
            self._quota_used = 0
            self._quota_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

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

    def _generate_mock_posts(self, query: str, count: int = 2) -> List[CanonicalPost]:
        """Deterministic synthetic post generator for offline fallback and tests."""
        now = datetime.now(timezone.utc)
        with self._lock:
            req_num = self.stats.requests_made

        clean_topic = query.strip().lstrip("#") or "AI technology"
        mock_posts = []

        root_id = f"yt_comm_{int(now.timestamp())}_{req_num}"
        raw_root = {
            "id": root_id,
            "snippet": {
                "videoId": "vid_abc123",
                "topLevelComment": {
                    "snippet": {
                        "textOriginal": f"Great breakdown on {clean_topic}! Really clear explanation #machinelearning @educator",
                        "authorDisplayName": "TechStreamer",
                        "authorChannelId": {"value": "UC_yt_stream_99"},
                        "likeCount": 18,
                        "publishedAt": now.isoformat(),
                    }
                },
                "totalReplyCount": 1,
            },
        }
        mock_posts.append(self.normalize_post(raw_root))

        if count > 1:
            reply_id = f"yt_reply_{int(now.timestamp())}_{req_num}"
            raw_reply = {
                "id": reply_id,
                "snippet": {
                    "videoId": "vid_abc123",
                    "textOriginal": f"Completely agree, especially the {clean_topic} architecture details!",
                    "authorDisplayName": "DevViewer",
                    "authorChannelId": {"value": "UC_viewer_42"},
                    "likeCount": 4,
                    "publishedAt": now.isoformat(),
                    "parentId": root_id,
                },
            }
            mock_posts.append(self.normalize_post(raw_reply))

        return mock_posts

    def _resolve_video_id_for_keyword(self, api_key: str, keyword: str) -> Optional[str]:
        """Resolve the top video ID for a bare keyword query using /search (cost: 100 units)."""
        if not self._check_and_consume_quota(QUOTA_COST_SEARCH):
            logger.warning(
                "Insufficient daily quota (%d units required) to resolve bare keyword via /search.",
                QUOTA_COST_SEARCH,
            )
            return None

        client = self._get_client()
        endpoint = f"{YOUTUBE_API_BASE}/search"
        params = {
            "part": "id",
            "type": "video",
            "maxResults": 1,
            "q": keyword or "technology AI",
            "key": api_key,
        }
        try:
            resp = client.get(endpoint, params=params)
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                if items:
                    vid = items[0].get("id", {}).get("videoId")
                    if vid:
                        logger.info("Resolved keyword '%s' to YouTube videoId: %s", keyword, vid)
                        return str(vid)
            elif resp.status_code == 403:
                err_data = resp.json().get("error", {})
                errors = err_data.get("errors", [])
                reasons = [e.get("reason", "") for e in errors]
                if "quotaExceeded" in reasons or "dailyLimitExceeded" in reasons:
                    with self._lock:
                        self._quota_used = self.daily_quota_limit
            logger.warning("YouTube /search query for '%s' returned HTTP %d", keyword, resp.status_code)
        except Exception as e:
            logger.warning("YouTube /search request error: %s", _sanitize_url(str(e)))
        return None

    def _poll_live(self, api_key: str, query: str, limit: int) -> List[CanonicalPost]:
        """Execute live HTTP query against YouTube Data API v3 with quota checks."""
        # 1. Quota check: 1 unit for commentThreads.list
        if not self._check_and_consume_quota(QUOTA_COST_COMMENT_THREADS):
            logger.info("Daily quota exhausted; falling back to offline mock generator.")
            return self._generate_mock_posts(query, count=2)

        client = self._get_client()
        endpoint = f"{YOUTUBE_API_BASE}/commentThreads"

        # Determine target parameter required by YouTube commentThreads.list:
        # According to YouTube Data API v3, one of videoId, allThreadsRelatedToChannelId, channelId, or id is REQUIRED.
        query_str = query.strip()
        params: Dict[str, Any] = {
            "part": "snippet,replies",
            "maxResults": min(max(1, limit), 100),
            "textFormat": "plainText",
            "key": api_key,
        }

        if query_str.startswith("video:"):
            params["videoId"] = query_str.split("video:", 1)[1].strip()
        elif len(query_str) == 11 and re.match(r"^[a-zA-Z0-9_-]{11}$", query_str):
            params["videoId"] = query_str
        elif query_str.startswith("channel:"):
            params["allThreadsRelatedToChannelId"] = query_str.split("channel:", 1)[1].strip()
        elif query_str.startswith("UC") and len(query_str) == 24:
            params["allThreadsRelatedToChannelId"] = query_str
        else:
            # Bare keyword query: commentThreads.list strictly requires a videoId or channelId target.
            creds = self.config.credentials
            default_vid = creds.get("default_video_id") or os.getenv("YOUTUBE_DEFAULT_VIDEO_ID")
            default_cid = creds.get("default_channel_id") or os.getenv("YOUTUBE_DEFAULT_CHANNEL_ID")

            if default_vid:
                params["videoId"] = default_vid
                if query_str:
                    params["searchTerms"] = query_str
            elif default_cid:
                params["allThreadsRelatedToChannelId"] = default_cid
                if query_str:
                    params["searchTerms"] = query_str
            else:
                # Attempt to dynamically resolve top video ID via /search (100 units) if enabled
                enable_search = creds.get("enable_search_fallback", True)
                resolved_vid = self._resolve_video_id_for_keyword(api_key, query_str) if enable_search else None
                if resolved_vid:
                    params["videoId"] = resolved_vid
                    if query_str:
                        params["searchTerms"] = query_str
                else:
                    # Safe fallback channel (Google Developers channel: UC_x5XG1OV2P6uZZ5FSM9Ttw)
                    params["allThreadsRelatedToChannelId"] = "UC_x5XG1OV2P6uZZ5FSM9Ttw"
                    if query_str:
                        params["searchTerms"] = query_str

        # YouTube Data API v3 constraint: order="relevance" is strictly only supported when querying a videoId.
        # When querying channel threads, order must be "time" or omitted.
        if "videoId" in params:
            params["order"] = "relevance"
        elif "allThreadsRelatedToChannelId" in params or "channelId" in params:
            params["order"] = "time"

        try:
            resp = client.get(endpoint, params=params)
        except Exception as conn_err:
            sanitized_err = _sanitize_url(str(conn_err))
            logger.warning("YouTube network error (%s); falling back to mock.", sanitized_err)
            return self._generate_mock_posts(query, count=2)

        if resp.status_code == 200:
            data = resp.json()
            items = data.get("items", [])
            posts: List[CanonicalPost] = []
            for item in items:
                try:
                    # Top level comment
                    post = self.normalize_post(item)
                    posts.append(post)

                    # Extract nested comment replies if present
                    parent_video_id = (
                        item.get("snippet", {}).get("videoId")
                        or params.get("videoId")
                    )
                    replies_obj = item.get("replies", {})
                    for reply_raw in replies_obj.get("comments", []):
                        try:
                            # Attach parentId if missing
                            reply_snippet = reply_raw.setdefault("snippet", {})
                            if not reply_snippet.get("parentId"):
                                reply_snippet["parentId"] = post.id
                            if parent_video_id and "videoId" not in reply_snippet:
                                reply_snippet["videoId"] = parent_video_id
                            reply_post = self.normalize_post(reply_raw)
                            posts.append(reply_post)
                        except Exception as reply_parse_err:
                            logger.debug("Failed parsing reply: %s", reply_parse_err)
                except Exception as parse_err:
                    logger.debug("Failed parsing YouTube CommentThread: %s", parse_err)

            if posts:
                return posts[:limit]
            else:
                logger.info("YouTube commentThreads returned 0 items for query '%s'; generating fallback.", query)
                return self._generate_mock_posts(query, count=2)

        elif resp.status_code == 403:
            # Inspect error reason
            err_data = resp.json().get("error", {})
            errors = err_data.get("errors", [])
            reasons = [e.get("reason", "") for e in errors]
            if "quotaExceeded" in reasons or "dailyLimitExceeded" in reasons:
                with self._lock:
                    self._quota_used = self.daily_quota_limit
                logger.warning("YouTube API daily quota exceeded (403). Switched to mock mode for today.")
            else:
                logger.warning("YouTube API 403 Forbidden: %s", reasons)
            return self._generate_mock_posts(query, count=2)

        elif resp.status_code in (400, 401):
            logger.warning("YouTube API authentication or query error (%d). Key: %s", resp.status_code, _sanitize_key(api_key))
            return self._generate_mock_posts(query, count=2)

        else:
            logger.warning("YouTube API returned unexpected HTTP %d: %s", resp.status_code, _sanitize_url(resp.text[:200]))
            return self._generate_mock_posts(query, count=2)

    def poll(self, query: str = "", limit: int = 50) -> List[CanonicalPost]:
        """Poll YouTube Data API or return synthetic normalized posts if in offline stub mode."""
        if not self.rate_limiter.acquire(1.0):
            self.record_rate_limited()
            logger.warning("Rate limit exceeded for YouTubeConnector.")
            return []

        self.record_request()
        api_key = self.config.credentials.get("api_key") or os.getenv("YOUTUBE_API_KEY")
        use_mock = self.config.credentials.get("mock", False)

        posts: List[CanonicalPost] = []
        try:
            if api_key and not use_mock:
                logger.info("Polling live YouTube Data API for query: '%s' (Key: %s)", query, _sanitize_key(api_key))
                posts = self._poll_live(api_key=api_key, query=query, limit=limit)
            else:
                posts = self._generate_mock_posts(query=query, count=min(limit, 2))

            if posts:
                self.record_success(len(posts))
            return posts[:limit]
        except Exception as e:
            self.record_error(str(e))
            logger.error("Error during YouTube poll: %s", _sanitize_url(str(e)))
            return self._generate_mock_posts(query=query, count=min(limit, 2))
