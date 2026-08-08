import hashlib

from headroom.cache.kompress_cache import KompressCache


def test_key_uses_full_sha256_and_input_byte_length():
    cache = KompressCache(max_entries=10, max_bytes=10_000, max_attempts=3)
    content = "cafe\N{LATIN SMALL LETTER E WITH ACUTE} " * 10

    cache.record_success(content, "kept", 20, 1)
    entry = cache.lookup(content)

    assert entry is not None
    assert entry.digest == hashlib.sha256(content.encode("utf-8")).digest()
    assert entry.input_bytes == len(content.encode("utf-8"))
    assert len(entry.digest) == 32


def test_identity_pair_keeps_forced_digest_collision_entries_separate(monkeypatch):
    digest = b"forced-collision".ljust(32, b"\0")
    monkeypatch.setattr(
        KompressCache,
        "_identity",
        staticmethod(lambda content: (digest, len(content.encode("utf-8")))),
    )
    cache = KompressCache(max_entries=10, max_bytes=10_000, max_attempts=3)
    first = "\N{LATIN SMALL LETTER E WITH ACUTE}"
    second = "abc"

    cache.record_success(first, "first result", 1, 1)
    cache.record_success(second, "second result", 1, 1)

    first_entry = cache.lookup(first)
    second_entry = cache.lookup(second)
    assert first_entry is not None
    assert second_entry is not None
    assert first_entry.compressed == "first result"
    assert second_entry.compressed == "second result"
    assert set(cache._entries) == {(digest, 2), (digest, 3)}


def test_different_content_does_not_hit():
    cache = KompressCache(max_entries=10, max_bytes=10_000, max_attempts=3)
    cache.record_success("same length", "kept", 2, 1)

    assert cache.lookup("different!") is None
    assert cache.stats()["misses"] == 1


def test_failure_attempts_exhaust_and_success_replaces_failure():
    cache = KompressCache(max_entries=10, max_bytes=10_000, max_attempts=2)

    first = cache.record_failure("large payload")
    second = cache.record_failure("large payload")

    assert first.attempts == 1
    assert first.exhausted is False
    assert second.attempts == 2
    assert second.exhausted is True

    cache.record_success("large payload", "compressed", 2, 1)
    entry = cache.lookup("large payload")

    assert entry is not None
    assert entry.compressed == "compressed"
    assert entry.attempts == 0
    assert entry.exhausted is False


def test_lfu_eviction_breaks_ties_by_oldest():
    cache = KompressCache(max_entries=2, max_bytes=10_000, max_attempts=2)
    cache.record_success("old", "x", 1, 1)
    cache.record_success("new", "y", 1, 1)
    assert cache.lookup("new") is not None

    cache.record_success("third", "z", 1, 1)

    assert cache.lookup("old") is None
    assert cache.lookup("new") is not None
    assert cache.lookup("third") is not None
    assert cache.stats()["evictions"] == 1


def test_byte_limit_evicts_repeatedly_until_satisfied():
    cache = KompressCache(max_entries=10, max_bytes=120, max_attempts=2)
    cache.record_success("one", "a" * 40, 1, 1)
    cache.record_success("two", "b" * 40, 1, 1)
    cache.record_success("three", "c" * 40, 1, 1)

    stats = cache.stats()
    assert stats["entries"] == 2
    assert stats["bytes"] <= 120
    assert stats["evictions"] == 1


def test_oversized_entries_are_not_cached_and_limits_apply_to_failures():
    cache = KompressCache(max_entries=10, max_bytes=20, max_attempts=2)

    cache.record_success("x" * 100, "y", 100, 1)
    assert cache.lookup("x" * 100) is None

    cache.record_failure("small")
    assert cache.stats()["entries"] == 1


def test_lookup_returns_detached_entry_and_updates_access_accounting():
    cache = KompressCache(max_entries=10, max_bytes=10_000, max_attempts=2)
    cache.record_success("content", "compressed", 2, 1)

    entry = cache.lookup("content")
    assert entry is not None
    entry.compressed = "changed"
    entry.access_count = 999

    fresh = cache.lookup("content")
    assert fresh is not None
    assert fresh.compressed == "compressed"
    assert fresh.access_count == 2
    assert cache.stats()["hits"] == 2


def test_stats_count_misses_failures_and_retries():
    cache = KompressCache(max_entries=10, max_bytes=10_000, max_attempts=3)

    assert cache.lookup("content") is None
    cache.record_failure("content")
    cache.record_failure("content")

    stats = cache.stats()
    assert stats["misses"] == 1
    assert stats["failures"] == 2
    assert stats["retries"] == 1


def test_invalid_environment_limits_use_positive_defaults(monkeypatch):
    monkeypatch.setenv("HEADROOM_KOMPRESS_CACHE_MAX_ENTRIES", "0")
    monkeypatch.setenv("HEADROOM_KOMPRESS_CACHE_MAX_BYTES", "not-an-int")
    monkeypatch.setenv("HEADROOM_KOMPRESS_CACHE_MAX_ATTEMPTS", "-1")

    cache = KompressCache.from_environment()

    assert cache.max_entries > 0
    assert cache.max_bytes > 0
    assert cache.max_attempts > 0
