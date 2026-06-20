"""Package version metadata — computed lazily to avoid startup I/O."""

from __future__ import annotations

import importlib
import os
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

UNKNOWN_VERSION = "unknown"
VERSION_ENV_VARS = ("HEADROOM_VERSION", "HEADROOM_BUILD_VERSION")
RELEASE_VERSION_RE = re.compile(r"^v?\d+\.\d+\.\d+$")


def _clean_version(value: object) -> str | None:
    """Return a non-empty version string, if one is present."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def is_release_version(value: object) -> bool:
    """Return whether a value is a comparable release version."""
    cleaned = _clean_version(value)
    return bool(cleaned and RELEASE_VERSION_RE.fullmatch(cleaned))


def normalize_release_version(value: object) -> str | None:
    """Return a comparable release version without a display prefix."""
    cleaned = _clean_version(value)
    if cleaned is None or RELEASE_VERSION_RE.fullmatch(cleaned) is None:
        return None
    return cleaned[1:] if cleaned.startswith("v") else cleaned


def format_version_label(value: object) -> str:
    """Return a user-facing version label without prefixing source labels."""
    cleaned = _clean_version(value) or UNKNOWN_VERSION
    if is_release_version(cleaned) and not cleaned.startswith("v"):
        return f"v{cleaned}"
    return cleaned


def _env_version() -> str | None:
    """Return an explicit runtime/build version override."""
    for name in VERSION_ENV_VARS:
        value = _clean_version(os.environ.get(name))
        if value:
            return value
    return None


def _packaged_build_version() -> str | None:
    """Return Docker/image build metadata baked into the installed package."""
    try:
        build_info = importlib.import_module("headroom._build_info")
    except ModuleNotFoundError:
        return None
    return _clean_version(getattr(build_info, "BUILD_VERSION", None))

_CACHED_VERSION: str | None = None


def _source_root() -> Path | None:
    """Return the repository root when imported from a git checkout."""
    root = Path(__file__).resolve().parents[1]
    if (root / ".git").exists() and (root / "pyproject.toml").exists():
        return root
    return None


def _source_tree_version(root: Path) -> str | None:
    """Compute the version release automation would assign to this checkout."""
    try:
        from headroom.release_version import (
            compute_release_version,
            determine_bump_level,
            get_canonical_version,
            list_release_commits,
            list_release_tags,
        )

        tags = list_release_tags(root)
        previous_tag = compute_release_version(
            canonical_version=get_canonical_version(root),
            level="patch",
            tags=tags,
        ).previous_tag
        commits = list_release_commits(root, previous_tag)
        level = determine_bump_level(commits)
        return compute_release_version(
            canonical_version=get_canonical_version(root),
            level=level,
            tags=tags,
        ).version
    except Exception:
        return None


def get_version() -> str:
"""Return Headroom's runtime version, cached after first call."""
    global _CACHED_VERSION
    if _CACHED_VERSION is not None:
        return _CACHED_VERSION
    root = _source_root()
    if root is not None:
        source_version = _source_tree_version(root)
        if source_version:
            _CACHED_VERSION = source_version
            return _CACHED_VERSION

    build_version = _packaged_build_version()
    if build_version:
        return build_version

    build_version = _packaged_build_version()
    if build_version:
        return build_version

    try:
        _CACHED_VERSION = version("headroom-ai")
        return _CACHED_VERSION
    except PackageNotFoundError:
        _CACHED_VERSION = UNKNOWN_VERSION
        return _CACHED_VERSION


def __getattr__(name: str) -> Any:
    """Lazy attribute access for ``__version__``."""
    if name == "__version__":
        value = get_version()
        globals()["__version__"] = value
        return value
    if name == "__path__":
        raise AttributeError(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"__version__"})
