"""Process-local bounded cache for Kompress results."""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass, replace

_DEFAULT_MAX_ENTRIES = 256
_DEFAULT_MAX_BYTES = 64 * 1024 * 1024
_DEFAULT_MAX_ATTEMPTS = 3
_MAX_LIMIT = 1_000_000_000
_FIXED_ENTRY_BYTES = 8
_DEFAULT_SINGLE_FLIGHT_WAIT_SECONDS = 5.0
_CacheKey = tuple[str, bytes, int]


@dataclass
class KompressCacheEntry:
    digest: bytes
    input_bytes: int
    attempts: int
    access_count: int
    created_sequence: int
    last_access_sequence: int
    compressed: str | None = None
    original_tokens: int | None = None
    compressed_tokens: int | None = None
    exhausted: bool = False


def _positive_setting(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
    except (TypeError, ValueError):
        return default
    return value if 0 < value <= _MAX_LIMIT else default


class KompressCache:
    """Thread-safe LFU cache for successful results and payload failures."""

    def __init__(self, max_entries: int, max_bytes: int, max_attempts: int) -> None:
        self.max_entries = max(1, max_entries)
        self.max_bytes = max(1, max_bytes)
        self.max_attempts = max(1, max_attempts)
        self._entries: dict[_CacheKey, KompressCacheEntry] = {}
        self._entry_bytes = 0
        self._sequence = 0
        self._hits = 0
        self._misses = 0
        self._failures = 0
        self._retries = 0
        self._evictions = 0
        self._lock = threading.RLock()
        # Only currently running inferences are tracked, so coordination stays
        # bounded even when callers submit an unbounded stream of unique payloads.
        self._inflight: dict[_CacheKey, threading.Event] = {}

    @classmethod
    def from_environment(cls) -> KompressCache:
        return cls(
            _positive_setting("HEADROOM_KOMPRESS_CACHE_MAX_ENTRIES", _DEFAULT_MAX_ENTRIES),
            _positive_setting("HEADROOM_KOMPRESS_CACHE_MAX_BYTES", _DEFAULT_MAX_BYTES),
            _positive_setting("HEADROOM_KOMPRESS_CACHE_MAX_ATTEMPTS", _DEFAULT_MAX_ATTEMPTS),
        )

    @staticmethod
    def _identity(content: str) -> tuple[bytes, int]:
        encoded = content.encode("utf-8")
        return hashlib.sha256(encoded).digest(), len(encoded)

    @classmethod
    def _key(cls, content: str, namespace: str) -> _CacheKey:
        digest, input_bytes = cls._identity(content)
        return namespace, digest, input_bytes

    @staticmethod
    def _estimate(entry: KompressCacheEntry) -> int:
        compressed_bytes = (
            len(entry.compressed.encode("utf-8")) if entry.compressed is not None else 0
        )
        return _FIXED_ENTRY_BYTES + entry.input_bytes + compressed_bytes

    @staticmethod
    def _detached(entry: KompressCacheEntry) -> KompressCacheEntry:
        return replace(entry)

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _evict_until_within_limits(self) -> None:
        while self._entries and (
            len(self._entries) > self.max_entries or self._entry_bytes > self.max_bytes
        ):
            victim = min(
                self._entries.values(),
                key=lambda entry: (entry.access_count, entry.created_sequence),
            )
            victim_key = next(key for key, entry in self._entries.items() if entry is victim)
            del self._entries[victim_key]
            self._entry_bytes -= self._estimate(victim)
            self._evictions += 1

    def lookup(self, content: str, namespace: str = "") -> KompressCacheEntry | None:
        key = self._key(content, namespace)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            entry.access_count += 1
            entry.last_access_sequence = self._next_sequence()
            self._hits += 1
            return self._detached(entry)

    def record_success(
        self,
        content: str,
        compressed: str,
        original_tokens: int,
        compressed_tokens: int,
        namespace: str = "",
    ) -> None:
        key = self._key(content, namespace)
        _, digest, input_bytes = key
        entry = KompressCacheEntry(
            digest=digest,
            input_bytes=input_bytes,
            attempts=0,
            access_count=0,
            created_sequence=0,
            last_access_sequence=0,
            compressed=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
        )
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._entry_bytes -= self._estimate(previous)
            if self._estimate(entry) > self.max_bytes:
                return
            sequence = self._next_sequence()
            entry.created_sequence = sequence
            entry.last_access_sequence = sequence
            self._entries[key] = entry
            self._entry_bytes += self._estimate(entry)
            self._evict_until_within_limits()

    def record_failure(self, content: str, namespace: str = "") -> KompressCacheEntry:
        key = self._key(content, namespace)
        _, digest, input_bytes = key
        with self._lock:
            previous = self._entries.get(key)
            if previous is not None and previous.compressed is not None:
                # A late failure from a duplicate inference must not erase a
                # successful result that another caller already published.
                return self._detached(previous)
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._entry_bytes -= self._estimate(previous)
            attempts = (previous.attempts + 1) if previous is not None else 1
            entry = KompressCacheEntry(
                digest=digest,
                input_bytes=input_bytes,
                attempts=attempts,
                access_count=0,
                created_sequence=self._next_sequence(),
                last_access_sequence=self._sequence,
                exhausted=attempts >= self.max_attempts,
            )
            self._failures += 1
            if previous is not None:
                self._retries += 1
            if self._estimate(entry) <= self.max_bytes:
                self._entries[key] = entry
                self._entry_bytes += self._estimate(entry)
                self._evict_until_within_limits()
            return self._detached(entry)

    def discard(self, content: str, namespace: str = "") -> None:
        """Remove the entry for exactly this content, if one exists."""
        key = self._key(content, namespace)
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._entry_bytes -= self._estimate(previous)

    def acquire_inference(
        self,
        content: str,
        namespace: str = "",
        *,
        timeout_seconds: float = _DEFAULT_SINGLE_FLIGHT_WAIT_SECONDS,
    ) -> bool | None:
        """Claim an uncached key or wait for its current inference to finish.

        The cache lock is released before waiting or inference. A waiter returns
        ``False`` and must perform a fresh lookup before deciding whether it
        needs to retry; ``None`` means its bounded wait expired and the caller
        must fail open without recording a cache failure.
        """
        key = self._key(content, namespace)
        with self._lock:
            event = self._inflight.get(key)
            if event is None:
                self._inflight[key] = threading.Event()
                return True
        if not event.wait(timeout=max(0.0, timeout_seconds)):
            return None
        return False

    def release_inference(self, content: str, namespace: str = "") -> None:
        """Release the single-flight claim for a key and wake its waiters."""
        key = self._key(content, namespace)
        with self._lock:
            event = self._inflight.pop(key, None)
        if event is not None:
            event.set()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self._entry_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "failures": self._failures,
                "retries": self._retries,
                "evictions": self._evictions,
            }


_kompress_cache: KompressCache | None = None
_singleton_lock = threading.Lock()


def get_kompress_cache() -> KompressCache:
    global _kompress_cache
    if _kompress_cache is None:
        with _singleton_lock:
            if _kompress_cache is None:
                _kompress_cache = KompressCache.from_environment()
    return _kompress_cache


def reset_kompress_cache() -> None:
    global _kompress_cache
    with _singleton_lock:
        _kompress_cache = None
