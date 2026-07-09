from __future__ import annotations

import time
from unittest.mock import patch

from headroom.proxy.models import RateLimitState
from headroom.proxy.rate_limiter import (
    MAX_RATE_LIMITER_BUCKETS,
    TokenBucketRateLimiter,
)


class TestTokenBucketRateLimiter:
    async def test_default_constructor(self) -> None:
        limiter = TokenBucketRateLimiter()
        assert limiter.requests_per_minute == 60
        assert limiter.tokens_per_minute == 100_000

    async def test_custom_constructor(self) -> None:
        limiter = TokenBucketRateLimiter(requests_per_minute=10, tokens_per_minute=500)
        assert limiter.requests_per_minute == 10
        assert limiter.tokens_per_minute == 500

    async def test_check_request_allows_within_limit(self) -> None:
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        allowed, wait = await limiter.check_request("key")
        assert allowed is True
        assert wait == 0.0

    async def test_check_request_exhausts_bucket(self) -> None:
        limiter = TokenBucketRateLimiter(requests_per_minute=2)
        assert (await limiter.check_request("key"))[0] is True
        assert (await limiter.check_request("key"))[0] is True
        allowed, wait = await limiter.check_request("key")
        assert allowed is False
        assert wait > 0.0

    async def test_check_request_different_keys_independent(self) -> None:
        limiter = TokenBucketRateLimiter(requests_per_minute=1)
        assert (await limiter.check_request("a"))[0] is True
        assert (await limiter.check_request("b"))[0] is True
        assert (await limiter.check_request("a"))[0] is False

    async def test_check_tokens_allows_within_limit(self) -> None:
        limiter = TokenBucketRateLimiter(tokens_per_minute=1000)
        allowed, wait = await limiter.check_tokens("key", 500)
        assert allowed is True
        assert wait == 0.0

    async def test_check_tokens_denies_over_limit(self) -> None:
        limiter = TokenBucketRateLimiter(tokens_per_minute=100)
        allowed, wait = await limiter.check_tokens("key", 200)
        assert allowed is False
        assert wait > 0.0

    async def test_check_tokens_partial_exhaustion(self) -> None:
        limiter = TokenBucketRateLimiter(tokens_per_minute=100)
        assert (await limiter.check_tokens("key", 60))[0] is True
        assert (await limiter.check_tokens("key", 60))[0] is False

    async def test_stats_structure(self) -> None:
        limiter = TokenBucketRateLimiter(requests_per_minute=30, tokens_per_minute=5000)
        await limiter.check_request("k1")
        await limiter.check_request("k2")
        stats = await limiter.stats()
        assert stats["requests_per_minute"] == 30
        assert stats["tokens_per_minute"] == 5000
        assert stats["active_keys"] == 2

    async def test_cleanup_stale_buckets(self) -> None:
        limiter = TokenBucketRateLimiter(requests_per_minute=10)
        await limiter.check_request("stale")
        assert "stale" in limiter._request_buckets
        limiter._request_buckets["stale"].last_update = time.time() - 700
        await limiter._cleanup_stale_buckets()
        assert "stale" not in limiter._request_buckets

    async def test_cleanup_removes_token_bucket_too(self) -> None:
        limiter = TokenBucketRateLimiter(requests_per_minute=10)
        await limiter.check_request("stale")
        limiter._token_buckets["stale"] = limiter._request_buckets["stale"].__class__(
            tokens=100, last_update=time.time() - 700
        )
        limiter._request_buckets["stale"].last_update = time.time() - 700
        await limiter._cleanup_stale_buckets()
        assert "stale" not in limiter._request_buckets
        assert "stale" not in limiter._token_buckets

    async def test_cleanup_skips_fresh_buckets(self) -> None:
        limiter = TokenBucketRateLimiter(requests_per_minute=10)
        await limiter.check_request("fresh")
        await limiter._cleanup_stale_buckets()
        assert "fresh" in limiter._request_buckets

    async def test_get_lock_creates_per_key_lock(self) -> None:
        limiter = TokenBucketRateLimiter()
        lock = limiter._get_lock("mykey")
        assert lock is limiter._get_lock("mykey")
        other = limiter._get_lock("other")
        assert lock is not other

    async def test_refill_adds_tokens_based_on_elapsed(self) -> None:
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        state = RateLimitState(tokens=0, last_update=time.time() - 30)
        available = limiter._refill(state, 60)
        assert available > 28
        assert available < 32
        assert state.last_update > time.time() - 1

    async def test_refill_caps_at_rate(self) -> None:
        limiter = TokenBucketRateLimiter(requests_per_minute=10)
        state = RateLimitState(tokens=0, last_update=time.time() - 3600)
        available = limiter._refill(state, 10)
        assert available == 10

    async def test_cleanup_triggered_when_bucket_count_exceeds_max(self) -> None:
        with patch.object(TokenBucketRateLimiter, "_cleanup_stale_buckets") as mock_cleanup:
            limiter = TokenBucketRateLimiter(requests_per_minute=10)
            for i in range(MAX_RATE_LIMITER_BUCKETS + 1):
                limiter._request_buckets[f"key{i}"] = RateLimitState(
                    tokens=10, last_update=time.time()
                )
            await limiter.check_request("trigger")
            mock_cleanup.assert_awaited_once()
