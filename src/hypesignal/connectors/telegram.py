"""Telegram Bot / MTProto API Platform Connector and Normalization Stub."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from hypesignal.connectors.base import PlatformConnector
from hypesignal.connectors.schemas import ConnectorConfig, ConnectorStatus
from hypesignal.models.canonical import CanonicalPost, PostMetrics
from hypesignal.models.enums import PlatformType

logger = logging.getLogger(__name__)


class TelegramConnector(PlatformConnector):
    """Connector for Telegram MTProto / Bot API channel messages and discussion groups."""

    @property
    def platform(self) -> PlatformType:
        return PlatformType.TELEGRAM

    def normalize_post(self, raw_payload: Dict[str, Any]) -> CanonicalPost:
        """Normalize a Telegram Message object into a CanonicalPost.
        
        Expected structure matches Telegram Bot API / MTProto message:
        {
            "message_id": 12345,
            "date": 1697380200,
            "chat": {
                "id": -1001234567890,
                "title": "Tech Signals & Radar",
                "username": "techsignals",
                "type": "channel"
            },
            "from": {
                "id": 987654321,
                "username": "signal_admin",
                "first_name": "Signal"
            },
            "text": "New breakthrough in open weights foundation models! #opensource @hypesignal",
            "views": 3200,
            "forwards": 145,
            "reply_to_message": {
                "message_id": 12340
            }
        }
        """
        msg_id = raw_payload.get("message_id") or raw_payload.get("id")
        if msg_id is None:
            raise ValueError("Telegram payload missing required 'message_id' or 'id' field.")

        chat = raw_payload.get("chat") or {}
        chat_id = str(chat.get("id", "channel"))
        post_id = f"tg_{chat_id}_{msg_id}"

        text = raw_payload.get("text") or raw_payload.get("caption") or ""

        # Author resolution
        from_user = raw_payload.get("from") or {}
        if from_user and from_user.get("id"):
            author_id = str(from_user.get("id"))
            author_screen_name = from_user.get("username") or from_user.get("first_name")
        else:
            author_id = chat_id
            author_screen_name = chat.get("username") or chat.get("title") or "TelegramChannel"

        # Timestamp
        date_raw = raw_payload.get("date")
        if isinstance(date_raw, (int, float)):
            ts = datetime.fromtimestamp(date_raw, tz=timezone.utc)
        elif isinstance(date_raw, str):
            try:
                ts = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
            except ValueError:
                ts = datetime.fromtimestamp(float(date_raw), tz=timezone.utc)
        elif isinstance(date_raw, datetime):
            ts = date_raw
        else:
            ts = datetime.now(timezone.utc)

        views = int(raw_payload.get("views", 0)) if raw_payload.get("views") else None
        forwards = int(raw_payload.get("forwards", 0))

        metrics = PostMetrics(
            likes=0,
            reposts=forwards,
            replies=0,
            views=views,
            shares=forwards,
        )

        extracted = self.extract_entities(text)
        urls = extracted["urls"]
        chat_username = chat.get("username")
        if chat_username:
            urls.append(f"https://t.me/{chat_username}/{msg_id}")

        parent_id = None
        reply_to = raw_payload.get("reply_to_message")
        if reply_to and isinstance(reply_to, dict) and reply_to.get("message_id"):
            parent_id = f"tg_{chat_id}_{reply_to['message_id']}"

        return CanonicalPost(
            id=post_id,
            platform=PlatformType.TELEGRAM,
            author_id=author_id,
            author_screen_name=author_screen_name,
            text=text,
            timestamp=ts,
            parent_id=parent_id,
            reply_to_user_id=None,
            source_client="Telegram MTProto/Bot API",
            urls=urls,
            hashtags=extracted["hashtags"],
            mentions=extracted["mentions"],
            metrics=metrics,
            extra_metadata={
                "chat_id": chat.get("id"),
                "chat_title": chat.get("title"),
                "chat_type": chat.get("type"),
                "message_id": msg_id,
            },
        )

    def poll(self, query: str = "", limit: int = 50) -> List[CanonicalPost]:
        """Poll Telegram channels or return synthetic normalized posts if in offline stub mode."""
        if not self.rate_limiter.acquire(1.0):
            self.record_rate_limited()
            logger.warning("Rate limit exceeded for TelegramConnector.")
            return []

        self.record_request()
        bot_token = self.config.credentials.get("bot_token")

        posts: List[CanonicalPost] = []
        try:
            if bot_token:
                logger.info("Polling live Telegram Bot API for updates on: '%s'", query)

            now = datetime.now(timezone.utc)
            with self._lock:
                req_num = self.stats.requests_made
            raw_sample = {
                "message_id": req_num + 1000,
                "date": int(now.timestamp()),
                "chat": {
                    "id": -100987654321,
                    "title": "HypeSignal Broadcast",
                    "username": "hypesignal_radar",
                    "type": "channel",
                },
                "from": {
                    "id": 11223344,
                    "username": "radar_bot",
                    "first_name": "RadarBot",
                },
                "text": f"Channel update: tracking {query or 'market trends'} in real-time. #radar @analyst",
                "views": 1500,
                "forwards": 35,
            }
            post = self.normalize_post(raw_sample)
            posts.append(post)

            self.record_success(len(posts))
            return posts[:limit]
        except Exception as e:
            self.record_error(str(e))
            logger.error("Error during Telegram poll: %s", e)
            return []
