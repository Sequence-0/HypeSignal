"""Central environment configuration and .env auto-loader for HypeSignal."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def find_dotenv(filename: str = ".env") -> Optional[Path]:
    """Locate a .env file by searching the current directory, project root, and parents."""
    # 1. Check current working directory
    cwd_path = Path.cwd() / filename
    if cwd_path.is_file():
        return cwd_path

    # 2. Check repository/project root relative to this module
    # src/hypesignal/config.py -> parent.parent.parent is project root
    pkg_root = Path(__file__).resolve().parent.parent.parent / filename
    if pkg_root.is_file():
        return pkg_root

    # 3. Ascend parent directories from cwd
    for parent in Path.cwd().parents:
        cand = parent / filename
        if cand.is_file():
            return cand

    return None


def _parse_env_file(filepath: Path) -> Dict[str, str]:
    """Lightweight zero-dependency .env parser supporting exports, quotes, and comments."""
    loaded: Dict[str, str] = {}
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        logger.debug("Failed reading .env file at %s: %s", filepath, e)
        return loaded

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[7:].strip()

        if "=" not in line:
            continue

        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()

        # Handle quoted values (preserving hashes inside quotes) with optional trailing comments
        if val.startswith('"'):
            end_quote = val.find('"', 1)
            if end_quote != -1:
                val = val[1:end_quote]
            else:
                val = val[1:]
        elif val.startswith("'"):
            end_quote = val.find("'", 1)
            if end_quote != -1:
                val = val[1:end_quote]
            else:
                val = val[1:]
        else:
            # Strip unquoted inline comments
            val = val.partition("#")[0].strip()

        if key:
            loaded[key] = val

    return loaded


def load_env(
    dotenv_path: Optional[str | Path] = None,
    override: bool = False,
) -> Dict[str, str]:
    """Automatically discover and load environment variables from a .env file.
    
    If python-dotenv is installed, delegates to it. Otherwise, uses built-in parser.
    Existing exported environment variables take precedence unless override=True.
    
    Returns:
        Dict of key-value pairs loaded into os.environ.
    """
    target_path = Path(dotenv_path) if dotenv_path else find_dotenv()
    if not target_path or not target_path.is_file():
        logger.debug("No .env file found; using ambient environment variables.")
        return {}

    loaded: Dict[str, str] = {}

    # Prefer python-dotenv if available
    try:
        from dotenv import dotenv_values, load_dotenv as py_load_dotenv

        py_load_dotenv(dotenv_path=target_path, override=override)
        loaded = {k: v for k, v in dotenv_values(dotenv_path=target_path).items() if v is not None}
        logger.info("Loaded %d environment variables from %s via python-dotenv.", len(loaded), target_path)
        return loaded
    except ImportError:
        pass

    # Built-in fallback parser
    parsed = _parse_env_file(target_path)
    for k, v in parsed.items():
        if override or k not in os.environ:
            os.environ[k] = v
            loaded[k] = v

    logger.info("Loaded %d environment variables from %s via built-in loader.", len(loaded), target_path)
    return loaded


# Auto-load on initial import
load_env()
