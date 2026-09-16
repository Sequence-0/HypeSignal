"""Telegram Bot / MTProto API Platform Connector with Test DC Configuration and Normalization."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from telethon import TelegramClient
from telethon.sessions import MemorySession, SQLiteSession, StringSession

from hypesignal.connectors.base import PlatformConnector
from hypesignal.connectors.schemas import ConnectorConfig
from hypesignal.models.canonical import CanonicalPost, PostMetrics
from hypesignal.models.enums import PlatformType

logger = logging.getLogger(__name__)


def _sanitize_key(key: Optional[str]) -> str:
    """Mask sensitive tokens or hashes for safe logging."""
    if not key:
        return "<none>"
    key_str = str(key).strip()
    if len(key_str) <= 8:
        return "***"
    return f"{key_str[:4]}...{key_str[-4:]}"


class _TelegramLoopThread:
    """Dedicated background event loop thread for MTProto network operations."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="TelegramLoopThread",
        )
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    def run(self, coro: Any, timeout: float = 25.0) -> Any:
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    def stop(self) -> None:
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2.0)


class TelegramConnector(PlatformConnector):
    """Connector for Telegram MTProto / Bot API channel messages and discussion groups."""

    def __init__(
        self,
        config: Optional[ConnectorConfig] = None,
        client: Optional[Any] = None,
    ) -> None:
        """Initialize Telegram connector with MTProto credentials and test environment options."""
        super().__init__(config=config)
        self._client = client
        self._owns_client = client is None
        self._loop_thread: Optional[_TelegramLoopThread] = None

        creds = self.config.credentials
        raw_api_id = creds["api_id"] if "api_id" in creds else os.getenv("TELEGRAM_API_ID")
        self.api_id: Optional[int] = None
        if raw_api_id is not None and str(raw_api_id).strip():
            try:
                self.api_id = int(raw_api_id)
            except (ValueError, TypeError):
                self.api_id = None

        self.api_hash: Optional[str] = (
            creds["api_hash"] if "api_hash" in creds else os.getenv("TELEGRAM_API_HASH")
        )
        self.app_title: str = (
            creds["app_title"] if "app_title" in creds else os.getenv("TELEGRAM_APP_TITLE", "HypeSignal")
        )
        self.short_name: str = (
            creds["short_name"] if "short_name" in creds else os.getenv("TELEGRAM_SHORT_NAME", "hypesignal")
        )
        self.bot_token: Optional[str] = (
            creds["bot_token"] if "bot_token" in creds else os.getenv("TELEGRAM_BOT_TOKEN")
        )
        self.session_string: Optional[str] = (
            creds["session_string"] if "session_string" in creds else os.getenv("TELEGRAM_SESSION_STRING")
        )
        self.session_name: Optional[str] = (
            creds["session_name"] if "session_name" in creds else os.getenv("TELEGRAM_SESSION_NAME")
        )

        test_mode_val = creds["test_mode"] if "test_mode" in creds else os.getenv("TELEGRAM_TEST_MODE", "false")
        if isinstance(test_mode_val, bool):
            self.test_mode = test_mode_val
        else:
            self.test_mode = str(test_mode_val).lower() in ("1", "true", "yes")

        raw_dc_id = creds["test_dc_id"] if "test_dc_id" in creds else os.getenv("TELEGRAM_TEST_DC_ID", "2")
        try:
            self.test_dc_id = int(raw_dc_id)
        except (ValueError, TypeError):
            self.test_dc_id = 2

        self.test_dc_ip = str(
            creds["test_dc_ip"] if "test_dc_ip" in creds else os.getenv("TELEGRAM_TEST_DC_IP", "149.154.167.40")
        )
        raw_dc_port = creds["test_dc_port"] if "test_dc_port" in creds else os.getenv("TELEGRAM_TEST_DC_PORT", "443")
        try:
            self.test_dc_port = int(raw_dc_port)
        except (ValueError, TypeError):
            self.test_dc_port = 443

        self.public_keys = (
            creds["public_keys"]
            if "public_keys" in creds
            else (os.getenv("TELEGRAM_PUBLIC_KEYS") or os.getenv("TELEGRAM_TEST_PUBLIC_KEY"))
        )

        channels_cfg = creds["channels"] if "channels" in creds else os.getenv("TELEGRAM_CHANNELS", "")
        if isinstance(channels_cfg, list):
            self.default_channels = [str(c).strip() for c in channels_cfg if str(c).strip()]
        elif isinstance(channels_cfg, str) and channels_cfg.strip():
            self.default_channels = [c.strip() for c in channels_cfg.split(",") if c.strip()]
        else:
            self.default_channels = []

    @property
    def platform(self) -> PlatformType:
        return PlatformType.TELEGRAM

    def _run_coro(self, coro: Any, timeout: float = 25.0) -> Any:
        """Run coroutine in dedicated event loop thread, or return direct if not a coroutine."""
        if not asyncio.iscoroutine(coro):
            return coro
        if self._loop_thread is None:
            self._loop_thread = _TelegramLoopThread()
        return self._loop_thread.run(coro, timeout=timeout)

    def _register_public_keys(self) -> None:
        """Register custom MTProto RSA public keys with Telethon if configured."""
        if not self.public_keys:
            return

        raw_inputs: List[str] = (
            [self.public_keys]
            if isinstance(self.public_keys, str)
            else list(self.public_keys)
        )

        for raw_entry in raw_inputs:
            entry = str(raw_entry).strip()
            if not entry:
                continue

            if os.path.isfile(entry):
                try:
                    with open(entry, "r", encoding="utf-8") as f:
                        content = f.read()
                    self._add_pem_keys_from_text(content)
                except Exception as e:
                    logger.warning("Failed to load Telegram public key file '%s': %s", entry, e)
            elif "-----BEGIN" in entry:
                self._add_pem_keys_from_text(entry)
            else:
                logger.warning("Unrecognized Telegram public key format: %s", _sanitize_key(entry))

    def _add_pem_keys_from_text(self, text: str) -> None:
        """Parse one or more PKCS#1 PEM keys from text and register with Telethon."""
        current_lines: List[str] = []
        for line in text.splitlines():
            line_str = line.strip()
            if "-----BEGIN" in line_str and current_lines:
                pem_block = "\n".join(current_lines).strip()
                self._add_single_pem_key(pem_block)
                current_lines = [line_str]
            elif line_str:
                current_lines.append(line_str)

        if current_lines:
            pem_block = "\n".join(current_lines).strip()
            self._add_single_pem_key(pem_block)

    def _add_single_pem_key(self, pem_block: str) -> None:
        """Add a single PKCS#1 PEM block to Telethon RSA server keys."""
        from telethon.crypto import rsa

        try:
            key_bytes = pem_block.encode("utf-8")
            rsa.add_key(key_bytes, old=False)
            logger.info("Successfully registered custom MTProto RSA public key.")
        except Exception as e:
            logger.warning("Failed to parse MTProto RSA public key: %s", e)

    def _get_client(self) -> Optional[TelegramClient]:
        """Lazily initialize TelegramClient on background event loop."""
        if self._client is not None:
            return self._client

        if not self.api_id or not self.api_hash:
            return None

        # Build session
        if self.session_string:
            session = StringSession(self.session_string)
        elif self.session_name:
            session = SQLiteSession(self.session_name)
        else:
            session = MemorySession()

        # Configure test DC if test_mode is enabled
        if self.test_mode:
            if hasattr(session, "set_dc"):
                session.set_dc(self.test_dc_id, self.test_dc_ip, self.test_dc_port)
            self._register_public_keys()

        # Ensure loop thread is running
        if self._loop_thread is None:
            self._loop_thread = _TelegramLoopThread()

        client = TelegramClient(
            session=session,
            api_id=int(self.api_id),
            api_hash=str(self.api_hash),
            device_model=self.app_title,
            app_version="1.0.0",
            system_version="Linux",
            loop=self._loop_thread.loop,
        )
        self._client = client
        self._owns_client = True
        return self._client

    def _do_connect(self) -> None:
        """Subclass hook to initialize and connect MTProto TelegramClient."""
        if not self.api_id or not self.api_hash:
            logger.info("Telegram credentials not configured; operating in offline/mock mode.")
            return

        client = self._get_client()
        if client:
            async def _connect_task():
                if not client.is_connected():
                    await client.connect()
                if self.bot_token and not await client.is_user_authorized():
                    try:
                        await client.sign_in(bot_token=self.bot_token)
                    except Exception as e:
                        logger.warning("Failed to sign in with TELEGRAM_BOT_TOKEN: %s", e)

            self._run_coro(_connect_task())
            logger.info(
                "Telegram MTProto client connected (DC: %s, test_mode=%s).",
                f"{self.test_dc_id} ({self.test_dc_ip}:{self.test_dc_port})" if self.test_mode else "production",
                self.test_mode,
            )

    def _do_disconnect(self) -> None:
        """Subclass hook to disconnect client and release background event loop thread."""
        if self._owns_client and self._client:
            try:
                if self._client.is_connected():
                    disconnect_res = self._client.disconnect()
                    if inspect.isawaitable(disconnect_res):
                        self._run_coro(disconnect_res)
            except Exception as e:
                logger.warning("Error disconnecting Telegram client: %s", e)
            self._client = None
        if self._loop_thread:
            self._loop_thread.stop()
            self._loop_thread = None

    def __del__(self) -> None:
        """Ensure clean teardown on garbage collection."""
        try:
            self.disconnect()
        except Exception:
            pass

    def normalize_post(self, raw_payload: Union[Dict[str, Any], Any]) -> CanonicalPost:
        """Normalize a Telegram Message object or dict payload into a CanonicalPost."""
        is_dict = isinstance(raw_payload, dict)

        # 1. Message ID extraction
        if is_dict:
            msg_id = raw_payload.get("message_id") or raw_payload.get("id")
        else:
            msg_id = getattr(raw_payload, "id", None)

        if msg_id is None:
            raise ValueError("Telegram payload missing required 'message_id' or 'id' field.")

        # 2. Chat / Channel resolution
        chat_title = None
        chat_username = None
        chat_type = "channel"
        if is_dict:
            chat = raw_payload.get("chat") or {}
            chat_id = str(chat.get("id", "channel"))
            chat_title = chat.get("title")
            chat_username = chat.get("username")
            chat_type = chat.get("type", "channel")
        else:
            chat_obj = None
            try:
                chat_obj = getattr(raw_payload, "chat", None)
            except Exception:
                chat_obj = None

            raw_chat_id = None
            try:
                raw_chat_id = getattr(raw_payload, "chat_id", None)
            except Exception:
                raw_chat_id = None

            if raw_chat_id is None and chat_obj:
                raw_chat_id = getattr(chat_obj, "id", None)
            chat_id = str(raw_chat_id) if raw_chat_id is not None else "channel"

            if chat_obj:
                chat_title = getattr(chat_obj, "title", None)
                chat_username = getattr(chat_obj, "username", None)
            if getattr(raw_payload, "is_channel", False):
                chat_type = "channel"
            elif getattr(raw_payload, "is_group", False):
                chat_type = "group"
            elif getattr(raw_payload, "is_private", False):
                chat_type = "private"

        post_id = f"tg_{chat_id}_{msg_id}"

        # 3. Content / Text extraction
        if is_dict:
            text = raw_payload.get("text") or raw_payload.get("caption") or ""
        else:
            text = (
                getattr(raw_payload, "raw_text", None)
                or getattr(raw_payload, "message", None)
                or getattr(raw_payload, "text", "")
                or ""
            )

        # 4. Author resolution
        if is_dict:
            from_user = raw_payload.get("from") or {}
            if from_user and from_user.get("id"):
                author_id = str(from_user.get("id"))
                author_screen_name = from_user.get("username") or from_user.get("first_name")
            else:
                author_id = chat_id
                author_screen_name = chat_username or chat_title or "TelegramChannel"
        else:
            sender = None
            try:
                sender = getattr(raw_payload, "sender", None)
            except Exception:
                sender = None

            sender_id = None
            try:
                sender_id = getattr(raw_payload, "sender_id", None)
            except Exception:
                sender_id = None

            if sender_id is None and sender:
                sender_id = getattr(sender, "id", None)
            author_id = str(sender_id) if sender_id else chat_id

            post_author = getattr(raw_payload, "post_author", None)
            sender_name = (
                getattr(sender, "username", None)
                or getattr(sender, "first_name", None)
                if sender
                else None
            )
            author_screen_name = post_author or sender_name or chat_username or chat_title or "TelegramChannel"

        # 5. Timestamp resolution
        if is_dict:
            date_raw = raw_payload.get("date")
        else:
            date_raw = getattr(raw_payload, "date", None)

        if isinstance(date_raw, datetime):
            ts = date_raw.astimezone(timezone.utc) if date_raw.tzinfo else date_raw.replace(tzinfo=timezone.utc)
        elif isinstance(date_raw, (int, float)):
            ts = datetime.fromtimestamp(date_raw, tz=timezone.utc)
        elif isinstance(date_raw, str):
            try:
                ts = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
            except ValueError:
                ts = datetime.fromtimestamp(float(date_raw), tz=timezone.utc)
        else:
            ts = datetime.now(timezone.utc)

        # 6. Metrics extraction
        if is_dict:
            views = int(raw_payload.get("views", 0)) if raw_payload.get("views") is not None else None
            forwards = int(raw_payload.get("forwards", 0) or 0)
            replies = int(raw_payload.get("replies", 0) or 0)
            likes = int(raw_payload.get("likes", 0) or 0)
        else:
            raw_views = getattr(raw_payload, "views", None)
            views = int(raw_views) if raw_views is not None else None
            forwards = int(getattr(raw_payload, "forwards", 0) or 0)

            replies_obj = getattr(raw_payload, "replies", None)
            replies = int(getattr(replies_obj, "replies", 0) or 0) if replies_obj else 0

            reactions_obj = getattr(raw_payload, "reactions", None)
            reaction_results = getattr(reactions_obj, "results", None) if reactions_obj else None
            likes = sum(getattr(r, "count", 0) for r in reaction_results) if reaction_results else 0

        metrics = PostMetrics(
            likes=likes,
            reposts=forwards,
            replies=replies,
            views=views,
            shares=forwards,
        )

        # 7. URLs, Hashtags, Mentions
        extracted = self.extract_entities(text)
        urls = list(extracted["urls"])
        if chat_username:
            urls.append(f"https://t.me/{chat_username}/{msg_id}")

        # 8. Parent ID (reply threading)
        parent_id = None
        if is_dict:
            reply_to = raw_payload.get("reply_to_message")
            if reply_to and isinstance(reply_to, dict) and reply_to.get("message_id"):
                parent_id = f"tg_{chat_id}_{reply_to['message_id']}"
        else:
            reply_to_msg_id = getattr(raw_payload, "reply_to_msg_id", None)
            if reply_to_msg_id:
                parent_id = f"tg_{chat_id}_{reply_to_msg_id}"

        return CanonicalPost(
            id=post_id,
            platform=PlatformType.TELEGRAM,
            author_id=author_id,
            author_screen_name=author_screen_name or "TelegramChannel",
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
                "chat_id": chat_id,
                "chat_title": chat_title,
                "chat_username": chat_username,
                "chat_type": chat_type,
                "message_id": msg_id,
                "test_mode": self.test_mode,
            },
        )

    def _generate_mock_posts(self, query: str = "", limit: int = 1) -> List[CanonicalPost]:
        """Generate synthetic normalized posts for testing or offline fallback mode."""
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
        return [post]

    async def _poll_live(self, query: str = "", limit: int = 50) -> List[CanonicalPost]:
        """Fetch messages via live MTProto TelegramClient."""
        client = self._get_client()
        if client is None:
            return []

        if not client.is_connected():
            await client.connect()

        if self.bot_token and not await client.is_user_authorized():
            try:
                await client.sign_in(bot_token=self.bot_token)
            except Exception as e:
                logger.warning("Failed to sign in with TELEGRAM_BOT_TOKEN: %s", e)

        if not await client.is_user_authorized():
            logger.warning(
                "TelegramClient connected to MTProto (%s) but not authorized. "
                "Set TELEGRAM_BOT_TOKEN or TELEGRAM_SESSION_STRING in .env to fetch live channel posts.",
                f"DC {self.test_dc_id} ({self.test_dc_ip}:{self.test_dc_port})" if self.test_mode else "production",
            )
            return []

        clean_q = (query or "").strip()
        targets: List[str] = []
        search_kw: Optional[str] = None

        if clean_q.startswith("@") or clean_q.startswith("https://t.me/") or clean_q.startswith("t.me/"):
            target = clean_q.replace("https://t.me/", "@").replace("t.me/", "@")
            targets = [target]
        elif self.default_channels:
            targets = list(self.default_channels)
            if clean_q:
                search_kw = clean_q
        elif clean_q and " " not in clean_q and len(clean_q) > 2:
            targets = [f"@{clean_q}"]
        else:
            targets = ["@telegram"]
            if clean_q:
                search_kw = clean_q

        raw_messages: List[Any] = []
        for target in targets:
            try:
                entity_target = target.lstrip("@") if target.startswith("@") else target
                async for msg in client.iter_messages(entity_target, limit=limit, search=search_kw):
                    if getattr(msg, "id", None) is not None:
                        raw_messages.append(msg)
                        if len(raw_messages) >= limit:
                            break
            except Exception as err:
                logger.warning("Error fetching Telegram messages from '%s': %s", target, err)

        return [self.normalize_post(m) for m in raw_messages]

    def poll(self, query: str = "", limit: int = 50) -> List[CanonicalPost]:
        """Poll Telegram channels or return synthetic normalized posts if in offline stub mode."""
        if not self.rate_limiter.acquire(1.0):
            self.record_rate_limited()
            logger.warning("Rate limit exceeded for TelegramConnector.")
            return []

        self.record_request()

        # If live credentials are not present, produce synthetic mock posts
        if not self.api_id or not self.api_hash:
            posts = self._generate_mock_posts(query=query, limit=limit)
            self.record_success(len(posts))
            return posts[:limit]

        # Live poll attempt
        try:
            posts = self._run_coro(self._poll_live(query=query, limit=limit), timeout=25.0)
            if posts:
                self.record_success(len(posts))
                return posts[:limit]

            # If live returned no posts (e.g. not authorized or no messages found), fallback to mock
            logger.info("Telegram live poll produced 0 posts. Using synthetic mock fallback.")
            posts = self._generate_mock_posts(query=query, limit=limit)
            self.record_success(len(posts))
            return posts[:limit]
        except Exception as e:
            err_msg = str(e)
            if self.api_hash:
                err_msg = err_msg.replace(str(self.api_hash), _sanitize_key(self.api_hash))
            if self.bot_token:
                err_msg = err_msg.replace(str(self.bot_token), _sanitize_key(self.bot_token))
            if self.session_string:
                err_msg = err_msg.replace(str(self.session_string), _sanitize_key(self.session_string))
            self.record_error(err_msg)
            logger.warning("Telegram live poll encountered error (%s); falling back to synthetic posts.", err_msg)
            posts = self._generate_mock_posts(query=query, limit=limit)
            return posts[:limit]
