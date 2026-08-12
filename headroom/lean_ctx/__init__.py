"""Compatibility helpers for locating an optional lean-ctx executable."""

from __future__ import annotations

import shutil
from pathlib import Path


def get_lean_ctx_path() -> Path | None:
    """Return the lean-ctx executable found on PATH, if any."""
    path = shutil.which("lean-ctx")
    return Path(path) if path else None
