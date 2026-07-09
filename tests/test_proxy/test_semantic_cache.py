"""Tests for headroom.proxy.semantic_cache."""

from __future__ import annotations

from headroom.proxy.semantic_cache import SemanticCache


class TestSemanticCache:
    async def test_construct_with_defaults(self) -> None:
        cache = SemanticCache()
        assert cache.max_entries == 1000
        assert cache.ttl_seconds == 3600

    async def test_construct_with_custom_values(self) -> None:
        cache = SemanticCache(max_entries=50, ttl_seconds=600)
        assert cache.max_entries == 50
        assert cache.ttl_seconds == 600

    async def test_set_then_get_returns_entry(self) -> None:
        cache = SemanticCache(max_entries=10, ttl_seconds=3600)
        messages = [{"role": "user", "content": "hello"}]
        await cache.set(messages, "model-1", b"response body", {"x-header": "v"})
        result = await cache.get(messages, "model-1")
        assert result is not None
        assert result.response_body == b"response body"
        assert result.response_headers == {"x-header": "v"}

    async def test_get_returns_none_for_unknown_key(self) -> None:
        cache = SemanticCache(max_entries=10, ttl_seconds=3600)
        result = await cache.get([{"role": "user", "content": "no-such"}], "model-1")
        assert result is None

    async def test_get_returns_none_for_expired_entry(self) -> None:
        cache = SemanticCache(max_entries=10, ttl_seconds=1)
        messages = [{"role": "user", "content": "hello"}]
        await cache.set(messages, "model-1", b"response", {})

        result_before = await cache.get(messages, "model-1")
        assert result_before is not None

        import asyncio

        await asyncio.sleep(1.5)

        result_after = await cache.get(messages, "model-1")
        assert result_after is None

    async def test_get_increments_hit_count(self) -> None:
        cache = SemanticCache(max_entries=10, ttl_seconds=3600)
        messages = [{"role": "user", "content": "hello"}]
        await cache.set(messages, "model-1", b"response", {})
        await cache.get(messages, "model-1")
        await cache.get(messages, "model-1")
        stats = await cache.stats()
        assert stats["total_hits"] == 2

    async def test_get_moves_entry_to_end_for_lru(self) -> None:
        cache = SemanticCache(max_entries=2, ttl_seconds=3600)
        await cache.set([{"role": "user", "content": "a"}], "model-1", b"a", {})
        await cache.set([{"role": "user", "content": "b"}], "model-1", b"b", {})
        await cache.get([{"role": "user", "content": "a"}], "model-1")
        await cache.set([{"role": "user", "content": "c"}], "model-1", b"c", {})
        assert await cache.get([{"role": "user", "content": "a"}], "model-1") is not None
        assert await cache.get([{"role": "user", "content": "b"}], "model-1") is None
        assert await cache.get([{"role": "user", "content": "c"}], "model-1") is not None

    async def test_set_replaces_existing_entry(self) -> None:
        cache = SemanticCache(max_entries=10, ttl_seconds=3600)
        messages = [{"role": "user", "content": "hello"}]
        await cache.set(messages, "model-1", b"old-response", {})
        await cache.set(messages, "model-1", b"new-response", {})
        result = await cache.get(messages, "model-1")
        assert result is not None
        assert result.response_body == b"new-response"
        stats = await cache.stats()
        assert stats["entries"] == 1

    async def test_set_evicts_oldest_when_at_capacity(self) -> None:
        cache = SemanticCache(max_entries=3, ttl_seconds=3600)
        await cache.set([{"role": "user", "content": "a"}], "model-1", b"a", {})
        await cache.set([{"role": "user", "content": "b"}], "model-1", b"b", {})
        await cache.set([{"role": "user", "content": "c"}], "model-1", b"c", {})
        await cache.set([{"role": "user", "content": "d"}], "model-1", b"d", {})
        assert await cache.get([{"role": "user", "content": "a"}], "model-1") is None
        assert await cache.get([{"role": "user", "content": "b"}], "model-1") is not None
        assert await cache.get([{"role": "user", "content": "c"}], "model-1") is not None
        assert await cache.get([{"role": "user", "content": "d"}], "model-1") is not None

    async def test_set_computes_different_keys_for_different_messages(self) -> None:
        cache = SemanticCache(max_entries=10, ttl_seconds=3600)
        msg_a = [{"role": "user", "content": "hello"}]
        msg_b = [{"role": "user", "content": "world"}]
        await cache.set(msg_a, "model-1", b"resp-a", {})
        await cache.set(msg_b, "model-1", b"resp-b", {})
        result_a = await cache.get(msg_a, "model-1")
        result_b = await cache.get(msg_b, "model-1")
        assert result_a is not None
        assert result_b is not None
        assert result_a.response_body == b"resp-a"
        assert result_b.response_body == b"resp-b"

    async def test_set_computes_different_keys_for_different_models(self) -> None:
        cache = SemanticCache(max_entries=10, ttl_seconds=3600)
        messages = [{"role": "user", "content": "hello"}]
        await cache.set(messages, "model-a", b"resp-a", {})
        await cache.set(messages, "model-b", b"resp-b", {})
        result_a = await cache.get(messages, "model-a")
        result_b = await cache.get(messages, "model-b")
        assert result_a is not None
        assert result_b is not None
        assert result_a.response_body == b"resp-a"
        assert result_b.response_body == b"resp-b"

    async def test_strips_cache_control_from_key_fields(self) -> None:
        cache = SemanticCache(max_entries=10, ttl_seconds=3600)
        messages = [{"role": "user", "content": "hello"}]
        system_with = [{"type": "text", "text": "prompt", "cache_control": {"type": "ephemeral"}}]
        system_without = [{"type": "text", "text": "prompt"}]
        await cache.set(messages, "model-1", b"response", {}, system=system_without)
        result = await cache.get(messages, "model-1", system=system_with)
        assert result is not None
        assert result.response_body == b"response"

    async def test_stats_returns_correct_counts(self) -> None:
        cache = SemanticCache(max_entries=10, ttl_seconds=3600)
        messages = [{"role": "user", "content": "hello"}]
        await cache.set(messages, "model-1", b"resp", {})
        await cache.get(messages, "model-1")
        stats = await cache.stats()
        assert stats["entries"] == 1
        assert stats["max_entries"] == 10
        assert stats["total_hits"] == 1
        assert stats["ttl_seconds"] == 3600

    async def test_clear_empties_cache(self) -> None:
        cache = SemanticCache(max_entries=10, ttl_seconds=3600)
        messages = [{"role": "user", "content": "hello"}]
        await cache.set(messages, "model-1", b"response", {})
        await cache.clear()
        assert await cache.get(messages, "model-1") is None
        stats = await cache.stats()
        assert stats["entries"] == 0

    async def test_clear_on_empty_cache(self) -> None:
        cache = SemanticCache(max_entries=10, ttl_seconds=3600)
        await cache.clear()
        stats = await cache.stats()
        assert stats["entries"] == 0

    async def test_get_memory_stats(self) -> None:
        cache = SemanticCache(max_entries=10, ttl_seconds=3600)
        await cache.set(
            [{"role": "user", "content": "hello"}],
            "model-1",
            b"response data",
            {"x-hdr": "val"},
        )
        stats = cache.get_memory_stats()
        assert stats.name == "semantic_cache"
        assert stats.entry_count == 1
        assert stats.size_bytes > 0
        assert stats.hits == 0
