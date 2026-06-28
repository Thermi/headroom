"""Central memory budget — one env var to shrink all caches and caps.

Set ``HEADROOM_MEMORY_MODE`` to ``low`` or ``minimal`` to reduce the
per-component cache sizes, store capacities, and TTLs by 10× or 100×
respectively.  Individual settings can still be overridden with their
component-specific env vars (they take precedence).

Modes
-----
default
    Current production defaults — no change.
low
    10× reduction in cache entries, store capacities, and TTLs.
    Suitable for constrained deployments (e.g. 512 MB containers).
minimal
    100× reduction.  Only the compression cache survives at minimal
    size; batch context TTL drops to 60 s.  Use for memory-starved
    CI or dev without GPU.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_MODE = os.environ.get("HEADROOM_MEMORY_MODE", "default")
_DIVISOR = {"low": 10, "minimal": 100}.get(_MODE, 1)


def _scale(value: int) -> int:
    return max(1, value // _DIVISOR)


# Map of env var → unscaled default.  If the env var is already set,
# the user's explicit value wins (component-specific override).
# Otherwise the scaled budget default is injected.
_BUDGET_DEFAULTS: dict[str, int] = {
    # compression cache
    "HEADROOM_CACHE_SESSION_MAX_ENTRIES": 10000,
    "HEADROOM_COMPRESSION_CACHE_MAX_SESSIONS": 500,
    # CCR store
    "HEADROOM_CCR_MAX_ENTRIES": 1000,
    "HEADROOM_CCR_TTL_SECONDS": 1800,
    # session tracker
    "HEADROOM_SESSION_TRACKER_MAX_SESSIONS": 1000,
    # traffic learner
    "HEADROOM_TRAFFIC_LEARNER_MAX_PATTERNS": 5000,
    "HEADROOM_TRAFFIC_LEARNER_MAX_PERSISTED_IDS": 5000,
    # batch context store
    "HEADROOM_BATCH_STORE_MAX_CONTEXTS": 10000,
    "HEADROOM_BATCH_STORE_TTL": 86400,
    # request logger
    "HEADROOM_REQUEST_LOGGER_MAX_ENTRIES": 10000,
    "HEADROOM_REQUEST_LOGGER_MAX_BYTES": 100 * 1024 * 1024,
    # TOIN
    "HEADROOM_TOIN_MAX_PATTERNS": 5000,
    # semantic cache
    "HEADROOM_SEMANTIC_CACHE_MAX_ENTRIES": 1000,
    # context tracker
    "HEADROOM_CONTEXT_TRACKER_MAX_CONTEXTS": 100,
    # backend router
    "HEADROOM_BACKEND_ROUTER_MAX_OPEN": 16,
    # in-memory graph store
    "HEADROOM_GRAPH_MAX_ENTITIES": 50000,
    "HEADROOM_GRAPH_MAX_RELATIONSHIPS": 100000,
    # LRU memory cache
    "HEADROOM_LRU_CACHE_MAX_SIZE": 1000,
    # ONNX runtime
    "HEADROOM_ONNX_INTRA_OP_THREADS": 2,
    "HEADROOM_ONNX_INTER_OP_THREADS": 1,
}


def _apply_budget() -> None:
    """Inject scaled defaults into os.environ for any unset budget vars."""
    if _DIVISOR <= 1:
        return
    for env_var, unscaled_default in _BUDGET_DEFAULTS.items():
        if env_var not in os.environ:
            os.environ[env_var] = str(_scale(unscaled_default))
    logger.info(
        "Memory budget: mode=%s divisor=%d (%d vars set)",
        _MODE,
        _DIVISOR,
        len(_BUDGET_DEFAULTS),
    )


# Apply immediately at import time — all downstream code sees the
# scaled values via their usual os.environ.get() calls.
_apply_budget()


def get_memory_budget() -> MemoryBudget:
    """Return the current memory budget (cached singleton)."""
    return _get_budget()


_budget: MemoryBudget | None = None


def _get_budget() -> MemoryBudget:
    global _budget
    if _budget is None:
        _budget = MemoryBudget()
        if _DIVISOR > 1:
            logger.info("Memory budget: mode=%s divisor=%d", _MODE, _DIVISOR)
    return _budget
