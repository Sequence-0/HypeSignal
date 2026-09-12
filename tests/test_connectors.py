"""Tests for live platform ingestion stubs and rate limiting."""

import time
from datetime import datetime, timezone

import pytest

from hypesignal.connectors.base import TokenBucketRateLimiter
from hypesignal.connectors.reddit import RedditConnector
from hypesignal.connectors.schemas import (
    ConnectorConfig,
    ConnectorStatus,
)
from hypesignal.connectors.telegram import TelegramConnector
from hypesignal.connectors.twitter import TwitterConnector
from hypesignal.connectors.youtube import YouTubeConnector
from hypesignal.models.enums import PlatformType


def test_token_bucket_rate_limiter():
    """Test token consumption, capacity limit, and wait time calculation."""
    limiter = TokenBucketRateLimiter(rate=2.0, capacity=2.0)
    assert limiter.acquire(1.0) is True
    assert limiter.acquire(1.0) is True
    # Capacity exhausted
    assert limiter.acquire(1.0) is False

    # Wait time should be positive
    wait = limiter.wait_time(1.0)
    assert wait > 0.0

    # Reset
    limiter.reset()
    assert limiter.acquire(1.0) is True


def test_twitter_connector_normalization():
    """Test normalizing Twitter API v2 Tweet payload into CanonicalPost."""
    conn = TwitterConnector()
    payload = {
        "id": "123456789",
        "text": "Check out this AI paper https://arxiv.org/abs/123 #MachineLearning @hypesignal",
        "created_at": "2024-01-15T12:00:00.000Z",
        "author_id": "author_001",
        "author_screen_name": "ai_researcher",
        "public_metrics": {
            "like_count": 50,
            "retweet_count": 12,
            "reply_count": 8,
            "impression_count": 1500,
            "quote_count": 3,
        },
        "entities": {
            "urls": [{"expanded_url": "https://arxiv.org/abs/123"}],
            "hashtags": [{"tag": "MachineLearning"}],
            "mentions": [{"username": "hypesignal"}],
        },
        "conversation_id": "123456789",
    }

    post = conn.normalize_post(payload)
    assert post.id == "123456789"
    assert post.platform == PlatformType.TWITTER
    assert post.author_id == "author_001"
    assert post.author_screen_name == "ai_researcher"
    assert "https://arxiv.org/abs/123" in post.urls
    assert "MachineLearning" in post.hashtags
    assert "hypesignal" in post.mentions
    assert post.metrics.likes == 50
    assert post.metrics.reposts == 12
    assert post.metrics.views == 1500
    assert post.timestamp.tzinfo is not None

    # Test error on missing id
    with pytest.raises(ValueError):
        conn.normalize_post({"text": "no id"})


def test_reddit_connector_normalization():
    """Test normalizing Reddit submission payload into CanonicalPost."""
    conn = RedditConnector()
    payload = {
        "id": "post_reddit_1",
        "title": "Discussion on Fast Ingestion",
        "selftext": "How do you achieve 10k eps? #databases @mod",
        "author": "dev_user",
        "author_fullname": "t2_user99",
        "created_utc": 1700000000.0,
        "score": 120,
        "num_comments": 45,
        "permalink": "/r/programming/comments/post_reddit_1/discussion/",
        "subreddit": "programming",
        "upvote_ratio": 0.98,
    }

    post = conn.normalize_post(payload)
    assert post.id == "post_reddit_1"
    assert post.platform == PlatformType.REDDIT
    assert post.author_id == "t2_user99"
    assert post.author_screen_name == "dev_user"
    assert "Discussion on Fast Ingestion" in post.text
    assert post.metrics.likes == 120
    assert post.metrics.replies == 45
    assert "https://reddit.com/r/programming/comments/post_reddit_1/discussion/" in post.urls
    assert "databases" in post.hashtags

    # Test error on missing id
    with pytest.raises(ValueError):
        conn.normalize_post({"title": "no id"})


def test_youtube_connector_normalization():
    """Test normalizing YouTube Data API v3 CommentThread payload into CanonicalPost."""
    conn = YouTubeConnector()
    payload = {
        "id": "yt_comment_99",
        "snippet": {
            "videoId": "vid_xyz123",
            "topLevelComment": {
                "snippet": {
                    "textOriginal": "Amazing tutorial! #learnAI @creator",
                    "authorDisplayName": "LearnerOne",
                    "authorChannelId": {"value": "UC_channel_456"},
                    "likeCount": 35,
                    "publishedAt": "2024-02-01T10:30:00Z",
                }
            },
            "totalReplyCount": 7,
        },
    }

    post = conn.normalize_post(payload)
    assert post.id == "yt_comment_99"
    assert post.platform == PlatformType.YOUTUBE
    assert post.author_id == "UC_channel_456"
    assert post.author_screen_name == "LearnerOne"
    assert post.metrics.likes == 35
    assert post.metrics.replies == 7
    assert "https://www.youtube.com/watch?v=vid_xyz123" in post.urls
    assert "learnAI" in post.hashtags

    # Test error on missing id
    with pytest.raises(ValueError):
        conn.normalize_post({"snippet": {}})


def test_telegram_connector_normalization():
    """Test normalizing Telegram message payload into CanonicalPost."""
    conn = TelegramConnector()
    payload = {
        "message_id": 9876,
        "date": 1700005000,
        "chat": {
            "id": -100555666777,
            "title": "Tech Signals",
            "username": "tech_signals",
            "type": "channel",
        },
        "from": {
            "id": 112233,
            "username": "telegram_admin",
        },
        "text": "Market alert: GPU cluster online #infra @tech_signals",
        "views": 2500,
        "forwards": 80,
    }

    post = conn.normalize_post(payload)
    assert post.id == "tg_-100555666777_9876"
    assert post.platform == PlatformType.TELEGRAM
    assert post.author_id == "112233"
    assert post.author_screen_name == "telegram_admin"
    assert post.metrics.views == 2500
    assert post.metrics.reposts == 80
    assert "https://t.me/tech_signals/9876" in post.urls
    assert "infra" in post.hashtags

    # Test error on missing id
    with pytest.raises(ValueError):
        conn.normalize_post({"text": "no message_id"})


def test_connector_lifecycle():
    """Test connector connect, disconnect, and disabled configuration."""
    conn = TwitterConnector()
    assert conn.is_connected() is False
    assert conn.status == ConnectorStatus.DISCONNECTED

    # Connect
    assert conn.connect() is True
    assert conn.is_connected() is True
    assert conn.status == ConnectorStatus.CONNECTED

    # Disconnect
    assert conn.disconnect() is True
    assert conn.is_connected() is False
    assert conn.status == ConnectorStatus.DISCONNECTED

    # Disabled connector should not connect
    disabled_conn = RedditConnector(config=ConnectorConfig(platform=PlatformType.REDDIT, enabled=False))
    assert disabled_conn.connect() is False
    assert disabled_conn.status == ConnectorStatus.DISCONNECTED


def test_connector_polling_and_stats():
    """Test polling across all 4 connectors updates stats and returns canonical posts."""
    connectors = [
        TwitterConnector(),
        RedditConnector(),
        YouTubeConnector(),
        TelegramConnector(),
    ]

    for conn in connectors:
        assert conn.connect() is True
        posts = conn.poll(query="test query", limit=10)
        assert len(posts) > 0
        assert posts[0].platform == conn.platform

        info = conn.get_info()
        assert info.stats.requests_made >= 1
        assert info.stats.messages_ingested >= 1
        assert info.stats.last_active is not None


def test_connector_stats_thread_safety():
    """Test that concurrent polling safely increments requests and ingested counts without race conditions."""
    import concurrent.futures

    conn = TwitterConnector(config=ConnectorConfig(platform=PlatformType.TWITTER, max_requests_per_minute=1000))
    conn.connect()

    def do_poll():
        return conn.poll(query="concurrent test", limit=1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(do_poll) for _ in range(20)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    info = conn.get_info()
    assert info.stats.requests_made == 20
    assert info.stats.messages_ingested == 20
    assert info.stats.error_count == 0
