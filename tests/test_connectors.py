"""Tests for live platform ingestion stubs and rate limiting."""

import pytest

from hypesignal.connectors.base import TokenBucketRateLimiter
from hypesignal.connectors.reddit import RedditConnector
from hypesignal.connectors.schemas import (
    ConnectorConfig,
    ConnectorStatus,
)
from hypesignal.connectors.telegram import TelegramConnector
from hypesignal.connectors.twitter import TwitterConnector
from hypesignal.connectors.youtube import (
    YouTubeConnector,
    _sanitize_key,
    _sanitize_url,
)
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


def test_youtube_connector_live_polling_mock_transport():
    """Test live YouTube polling flow with httpx.MockTransport parsing comment threads and replies."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        assert "commentThreads" in str(request.url)
        assert request.url.params.get("key") == "AIzaSyFakeKey12345"
        payload = {
            "kind": "youtube#commentThreadListResponse",
            "items": [
                {
                    "id": "thread_abc123",
                    "snippet": {
                        "videoId": "vid_xyz",
                        "topLevelComment": {
                            "id": "thread_abc123",
                            "snippet": {
                                "textOriginal": "Awesome deep dive into transformer architecture! #ai #llm @engineer",
                                "authorDisplayName": "TechGuy",
                                "authorChannelId": {"value": "UC_channel_1"},
                                "likeCount": 15,
                                "publishedAt": "2024-03-01T12:00:00Z",
                            },
                        },
                        "totalReplyCount": 1,
                    },
                    "replies": {
                        "comments": [
                            {
                                "id": "comment_rep_1",
                                "snippet": {
                                    "videoId": "vid_xyz",
                                    "parentId": "thread_abc123",
                                    "textOriginal": "Totally agree with the attention mechanism point!",
                                    "authorDisplayName": "DevGal",
                                    "authorChannelId": {"value": "UC_channel_2"},
                                    "likeCount": 3,
                                    "publishedAt": "2024-03-01T12:05:00Z",
                                },
                            }
                        ]
                    },
                }
            ],
        }
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    config = ConnectorConfig(
        platform=PlatformType.YOUTUBE,
        credentials={"api_key": "AIzaSyFakeKey12345", "daily_quota_limit": 100},
    )
    conn = YouTubeConnector(config=config, http_client=client)
    assert conn.connect() is True

    # Poll with explicit video target: query="video:vid_xyz"
    posts = conn.poll(query="video:vid_xyz", limit=10)
    assert len(posts) == 2
    # Verify top-level comment
    top_post = posts[0]
    assert top_post.id == "thread_abc123"
    assert top_post.platform == PlatformType.YOUTUBE
    assert top_post.author_screen_name == "TechGuy"
    assert top_post.metrics.likes == 15
    assert top_post.metrics.replies == 1
    assert "transformer architecture" in top_post.text
    assert "ai" in top_post.hashtags

    # Verify reply comment inherited parent's videoId (Issue 3 fix)
    reply_post = posts[1]
    assert reply_post.id == "comment_rep_1"
    assert reply_post.parent_id == "thread_abc123"
    assert reply_post.author_screen_name == "DevGal"
    assert reply_post.metrics.likes == 3
    assert "https://www.youtube.com/watch?v=vid_xyz" in reply_post.urls
    assert reply_post.extra_metadata.get("video_id") == "vid_xyz"

    # Check quota consumption: 1 unit
    stats = conn.get_quota_stats()
    assert stats["quota_used"] == 1
    assert stats["quota_remaining"] == 99


def test_youtube_connector_bare_keyword_search_resolution():
    """Test bare keyword query resolves top video via /search and queries commentThreads."""
    import httpx

    search_called = False
    threads_called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal search_called, threads_called
        path = request.url.path
        if path.endswith("/search"):
            search_called = True
            assert request.url.params.get("q") == "deeplearning"
            return httpx.Response(
                200,
                json={"items": [{"id": {"videoId": "vid_found_99"}}]},
            )
        elif path.endswith("/commentThreads"):
            threads_called = True
            assert request.url.params.get("videoId") == "vid_found_99"
            payload = {
                "kind": "youtube#commentThreadListResponse",
                "items": [
                    {
                        "id": "thread_resolved_1",
                        "snippet": {
                            "videoId": "vid_found_99",
                            "topLevelComment": {
                                "id": "thread_resolved_1",
                                "snippet": {
                                    "textOriginal": "Great deep learning talk!",
                                    "authorDisplayName": "AIWatcher",
                                    "likeCount": 10,
                                },
                            },
                        },
                    }
                ],
            }
            return httpx.Response(200, json=payload)
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    config = ConnectorConfig(
        platform=PlatformType.YOUTUBE,
        credentials={"api_key": "AIzaSyFakeKey12345", "daily_quota_limit": 500},
    )
    conn = YouTubeConnector(config=config, http_client=client)
    conn.connect()

    posts = conn.poll(query="deeplearning", limit=5)
    assert search_called is True
    assert threads_called is True
    assert len(posts) == 1
    assert posts[0].id == "thread_resolved_1"
    assert conn.get_quota_stats()["quota_used"] == 101


def test_youtube_connector_channel_target_order_is_time():
    """Verify YouTube Data API v3 rule: channel target commentThreads MUST use order='time', not 'relevance'."""
    import httpx

    captured_order = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_order
        captured_order = request.url.params.get("order")
        return httpx.Response(200, json={"items": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    config = ConnectorConfig(
        platform=PlatformType.YOUTUBE,
        credentials={"api_key": "AIzaSyFakeKey12345", "daily_quota_limit": 500},
    )
    conn = YouTubeConnector(config=config, http_client=client)
    conn.connect()

    # Channel query: order must be "time"
    conn.poll(query="channel:UC_x5XG1OV2P6uZZ5FSM9Ttw", limit=5)
    assert captured_order == "time"
    assert captured_order != "relevance"

    # Video query: order must be "relevance"
    conn.poll(query="video:vid_xyz", limit=5)
    assert captured_order == "relevance"



def test_youtube_connector_quota_guardrail():
    """Test daily quota limit halts live requests and gracefully falls back to mock posts."""
    import httpx

    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, json={"items": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    # Daily quota limit = 2 units
    config = ConnectorConfig(
        platform=PlatformType.YOUTUBE,
        credentials={"api_key": "AIzaSyFakeKey12345", "daily_quota_limit": 2},
    )
    conn = YouTubeConnector(config=config, http_client=client)
    conn.connect()

    # 1st request (consumes 1 unit) -> live call
    assert len(conn.poll(query="q1", limit=2)) > 0
    assert request_count == 1
    assert conn.get_quota_stats()["quota_used"] == 1

    # 2nd request (consumes 1 unit) -> live call
    assert len(conn.poll(query="q2", limit=2)) > 0
    assert request_count == 2
    assert conn.get_quota_stats()["quota_used"] == 2
    assert conn.get_quota_stats()["quota_remaining"] == 0

    # 3rd request (needs 1 unit, but quota limit 2 is reached) -> blocked, mock fallback
    p3 = conn.poll(query="q3", limit=2)
    assert request_count == 2  # No extra HTTP request made!
    assert len(p3) > 0  # Fallback mock posts returned
    assert p3[0].author_screen_name == "TechStreamer"

    # Reset quota
    conn.reset_quota()
    assert conn.get_quota_stats()["quota_used"] == 0
    assert len(conn.poll(query="q4", limit=2)) > 0
    assert request_count == 3  # HTTP request allowed again


def test_youtube_connector_403_quota_exceeded():
    """Test that an HTTP 403 quotaExceeded response immediately depletes remaining quota and falls back."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        error_resp = {
            "error": {
                "code": 403,
                "message": "The request cannot be completed because you have exceeded your quota.",
                "errors": [{"reason": "quotaExceeded", "domain": "youtube.quota"}],
            }
        }
        return httpx.Response(403, json=error_resp)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    config = ConnectorConfig(
        platform=PlatformType.YOUTUBE,
        credentials={"api_key": "AIzaSyFakeKey12345", "daily_quota_limit": 5000},
    )
    conn = YouTubeConnector(config=config, http_client=client)
    conn.connect()

    posts = conn.poll(query="test", limit=2)
    # Verify fallback to mock posts without throwing exception
    assert len(posts) > 0
    # Quota should now be marked as depleted (daily_quota_limit)
    assert conn.get_quota_stats()["quota_remaining"] == 0
    assert conn.get_quota_stats()["quota_used"] == 5000


def test_youtube_connector_url_and_key_sanitization():
    """Test masking of sensitive API keys in URLs, error logs, and helper methods."""
    key = "AIzaSyB_1234567890abcdef"
    sanitized_key = _sanitize_key(key)
    assert sanitized_key == "AIza...cdef"
    assert key not in sanitized_key

    # None and short keys
    assert _sanitize_key(None) == "<none>"
    assert _sanitize_key("") == "<none>"
    assert _sanitize_key("short") == "***"

    # URL sanitization
    raw_url = "https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&key=AIzaSyB_1234567890abcdef&maxResults=50"
    cleaned_url = _sanitize_url(raw_url)
    assert "AIzaSyB" not in cleaned_url
    assert "key=[REDACTED]" in cleaned_url
    assert "part=snippet" in cleaned_url
    assert "maxResults=50" in cleaned_url

    # When key is first param
    first_param_url = "https://example.com/api?key=mysecretkey&other=1"
    assert _sanitize_url(first_param_url) == "https://example.com/api?key=[REDACTED]&other=1"

