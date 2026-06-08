"""Tests for compression cache eviction (islice-based, O(K) vs O(N+K))."""

from __future__ import annotations

import threading

import pytest

from headroom.proxy.helpers import MAX_COMPRESSION_CACHE_SESSIONS
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import HeadroomProxy


@pytest.fixture
def proxy() -> HeadroomProxy:
    """Minimal proxy instance for cache eviction tests."""
    config = ProxyConfig(
        optimize=False,
        mode="token",
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        code_aware_enabled=False,
    )
    return HeadroomProxy(config)


def test_eviction_removes_oldest_quarter(proxy: HeadroomProxy) -> None:
    """Verify islice-based eviction removes the expected number of oldest entries."""
    with proxy._compression_caches_lock:
        cache_cls = proxy._get_compression_cache.__globals__.get("CompressionCache")
        if cache_cls is None:
            from headroom.cache.compression_cache import CompressionCache as cache_cls

        for i in range(MAX_COMPRESSION_CACHE_SESSIONS):
            proxy._compression_caches[f"session_{i:04d}"] = cache_cls()

        assert len(proxy._compression_caches) == MAX_COMPRESSION_CACHE_SESSIONS

    # Trigger eviction by adding a new session
    new_cache = proxy._get_compression_cache("session_new")

    with proxy._compression_caches_lock:
        expected_remaining = (
            MAX_COMPRESSION_CACHE_SESSIONS
            - MAX_COMPRESSION_CACHE_SESSIONS // 4
            + 1  # +1 because we add the new session
        )
        assert len(proxy._compression_caches) == expected_remaining

        # The oldest MAX_COMPRESSION_CACHE_SESSIONS // 4 sessions should be gone
        for i in range(MAX_COMPRESSION_CACHE_SESSIONS // 4):
            assert f"session_{i:04d}" not in proxy._compression_caches

        # The newest sessions (after eviction point) should remain
        for i in range(MAX_COMPRESSION_CACHE_SESSIONS // 4, MAX_COMPRESSION_CACHE_SESSIONS):
            assert f"session_{i:04d}" in proxy._compression_caches

        # The new session should be present
        assert "session_new" in proxy._compression_caches

    assert new_cache is not None


def test_eviction_does_not_run_below_capacity(proxy: HeadroomProxy) -> None:
    """No eviction when cache is below max sessions."""
    with proxy._compression_caches_lock:
        cache_cls = proxy._get_compression_cache.__globals__.get("CompressionCache")
        if cache_cls is None:
            from headroom.cache.compression_cache import CompressionCache as cache_cls

        for i in range(MAX_COMPRESSION_CACHE_SESSIONS - 5):
            proxy._compression_caches[f"session_{i:04d}"] = cache_cls()

    proxy._get_compression_cache("session_new")

    with proxy._compression_caches_lock:
        assert len(proxy._compression_caches) == MAX_COMPRESSION_CACHE_SESSIONS - 4


def test_concurrent_eviction_same_session(proxy: HeadroomProxy) -> None:
    """Two threads requesting the same new session get the same cache."""
    with proxy._compression_caches_lock:
        cache_cls = proxy._get_compression_cache.__globals__.get("CompressionCache")
        if cache_cls is None:
            from headroom.cache.compression_cache import CompressionCache as cache_cls

        for i in range(MAX_COMPRESSION_CACHE_SESSIONS):
            proxy._compression_caches[f"session_{i:04d}"] = cache_cls()

    results: list[str] = []

    def get_cache(sid: str) -> None:
        c = proxy._get_compression_cache(sid)
        results.append(id(c))

    t1 = threading.Thread(target=get_cache, args=("session_new",))
    t2 = threading.Thread(target=get_cache, args=("session_new",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(results) == 2
    assert results[0] == results[1], "Same session must return identical cache instance"
