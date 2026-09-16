"""Tests for HypeSignal configuration and .env auto-loader."""

import os
from pathlib import Path

from hypesignal.config import _parse_env_file, find_dotenv, load_env


def test_find_dotenv_discovers_file(tmp_path: Path, monkeypatch):
    """Test find_dotenv locates .env in directory tree."""
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=BAR\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    found = find_dotenv()
    assert found is not None
    assert found.samefile(env_file)


def test_parse_env_file_syntax(tmp_path: Path):
    """Test built-in .env parser handles exports, quotes, whitespace, and comments."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
        # Leading comment
        KEY_SIMPLE=hello
        export EXPORTED_KEY=world
        QUOTED_DOUBLE="spaced value with # hash"
        QUOTED_SINGLE='single quoted'
        EMPTY_VAL=
        # Inactive key
        # COMMENTED_OUT=123
        SPACED_KEY  =  trimmed_value  
        INLINE_COMMENT=true # this is a comment
        QUOTED_WITH_COMMENT="hello world" # trailing comment
        """,
        encoding="utf-8",
    )

    parsed = _parse_env_file(env_file)
    assert parsed["KEY_SIMPLE"] == "hello"
    assert parsed["EXPORTED_KEY"] == "world"
    assert parsed["QUOTED_DOUBLE"] == "spaced value with # hash"
    assert parsed["QUOTED_SINGLE"] == "single quoted"
    assert parsed["EMPTY_VAL"] == ""
    assert parsed["SPACED_KEY"] == "trimmed_value"
    assert parsed["INLINE_COMMENT"] == "true"
    assert parsed["QUOTED_WITH_COMMENT"] == "hello world"
    assert "COMMENTED_OUT" not in parsed


def test_load_env_populates_os_environ(tmp_path: Path):
    """Test load_env reads file and populates os.environ."""
    env_file = tmp_path / ".env.test"
    test_key = "HYPESIGNAL_TEST_VAR_123"
    env_file.write_text(f"{test_key}=active_value\n", encoding="utf-8")

    if test_key in os.environ:
        del os.environ[test_key]

    try:
        loaded = load_env(dotenv_path=env_file)
        assert test_key in loaded
        assert os.environ.get(test_key) == "active_value"
    finally:
        if test_key in os.environ:
            del os.environ[test_key]


def test_load_env_respects_existing_env_by_default(tmp_path: Path):
    """Test load_env does not overwrite existing ambient environment variables unless override=True."""
    env_file = tmp_path / ".env.test"
    test_key = "HYPESIGNAL_TEST_PRESERVE"
    os.environ[test_key] = "ambient_shell_value"
    env_file.write_text(f"{test_key}=dotenv_file_value\n", encoding="utf-8")

    try:
        # Default: override=False
        load_env(dotenv_path=env_file, override=False)
        assert os.environ[test_key] == "ambient_shell_value"

        # Explicit: override=True
        load_env(dotenv_path=env_file, override=True)
        assert os.environ[test_key] == "dotenv_file_value"
    finally:
        if test_key in os.environ:
            del os.environ[test_key]


def test_load_env_missing_file_returns_empty():
    """Test load_env gracefully handles missing file without throwing."""
    non_existent = Path("/path/that/does/not/exist/.env")
    result = load_env(dotenv_path=non_existent)
    assert result == {}


def test_env_example_template_exists_and_contains_keys():
    """Verify that .env.example exists in the repository and documents all core keys."""
    repo_root = Path(__file__).resolve().parent.parent
    env_example = repo_root / ".env.example"
    assert env_example.is_file(), ".env.example must exist in the repository root"

    content = env_example.read_text(encoding="utf-8")
    assert "YOUTUBE_API_KEY" in content
    assert "YOUTUBE_DAILY_QUOTA_LIMIT" in content
    assert "YOUTUBE_MAX_RPM" in content
    assert "HYPESIGNAL_DB_PATH" in content
    assert "YOUTUBE_DEFAULT_VIDEO_ID" in content
    assert "YOUTUBE_DEFAULT_CHANNEL_ID" in content
