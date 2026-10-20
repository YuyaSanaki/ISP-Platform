"""Map the Docker project root ``/app`` onto the checkout when running without Docker.

Configs, defaults and saved runs refer to ``/app/...`` (the Compose bind mount of the
repository). On a host without ``/app`` (native macOS install), those paths are rewritten
to the repository root (``GENEFORMER_ROOT``, else the checkout containing this file).
Inside Docker ``/app`` exists and nothing changes.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

DOCKER_ROOT = "/app"
_APP_PATH = re.compile(r"(?<![\w./-])/app(?=/|$|[\s\"',\]}])")


def platform_root() -> Path:
    env = os.environ.get("GENEFORMER_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def needs_mapping() -> bool:
    return not Path(DOCKER_ROOT).is_dir() and platform_root() != Path(DOCKER_ROOT)


def localize_path(value: str | os.PathLike) -> str:
    s = os.fspath(value)
    if not needs_mapping():
        return s
    if s == DOCKER_ROOT or s.startswith(DOCKER_ROOT + "/"):
        return str(platform_root()) + s[len(DOCKER_ROOT):]
    return s


def localize(obj: Any) -> Any:
    """Config with every string that is an ``/app`` path rewritten (dicts and lists recursed)."""
    if not needs_mapping():
        return obj
    if isinstance(obj, dict):
        return {k: localize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [localize(v) for v in obj]
    if isinstance(obj, str):
        return localize_path(obj)
    return obj


def localize_text(text: str) -> str:
    """YAML or log text with ``/app`` path prefixes rewritten (comments included)."""
    if not needs_mapping():
        return text
    return _APP_PATH.sub(lambda _m: str(platform_root()), text)
