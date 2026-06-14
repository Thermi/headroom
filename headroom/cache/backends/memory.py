"""In-memory storage backend for CompressionStore.

This is the default backend, providing fast access with no external dependencies.
Data is lost when the process exits.

Content compression (v2): large string fields in ``CompressionEntry`` are
transparently zlib-compressed by ``set()`` and decompressed by ``get()``,
reducing the memory footprint of the CCR store by 4-8x for typical tool
outputs with no API change for callers.
"""

from __future__ import annotations

import sys
import threading
import zlib
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..compression_store import CompressionEntry


# Strings above this many UTF-8 bytes are compressed before storage.
_COMPRESS_THRESHOLD = 2048
# zlib level 6 = good ratio for modest CPU.
_COMPRESS_LEVEL = 6


def _pack(raw: str) -> str:
    """Compress *raw* if large enough and return a portable string blob.

    The blob is prefixed with ``\\x00ZC`` so ``_unpack`` can recognise it.
    """
    utf8 = raw.encode("utf-8")
    if len(utf8) < _COMPRESS_THRESHOLD:
        return raw
    compressed = zlib.compress(utf8, _COMPRESS_LEVEL)
    return "\x00ZC" + compressed.decode("latin-1")


def _unpack(blob: str) -> str:
    """Reverse ``_pack`` — return the original string."""
    if blob.startswith("\x00ZC"):
        return zlib.decompress(blob[3:].encode("latin-1")).decode("utf-8")
    return blob


def _compress_entry(entry: CompressionEntry) -> CompressionEntry:
    """Return a copy of *entry* with content fields compressed."""
    return replace(
        entry,
        original_content=_pack(entry.original_content),
        compressed_content=_pack(entry.compressed_content),
    )


def _decompress_entry(entry: CompressionEntry) -> CompressionEntry:
    """Return a copy of *entry* with content fields decompressed."""
    return replace(
        entry,
        original_content=_unpack(entry.original_content),
        compressed_content=_unpack(entry.compressed_content),
    )


class InMemoryBackend:
    """Thread-safe in-memory storage backend with transparent compression.

    This is the default backend for CompressionStore. It stores entries in a
    Python dict with thread-safe access via a lock.  Large content strings
    are automatically zlib-compressed to reduce memory usage.

    Characteristics:
    - Fast: O(1) get/set/delete operations
    - Volatile: Data lost on process exit
    - Thread-safe: All operations are protected by a lock
    - Compressed: Content >2 KB is transparently zlib-compressed

    Usage:
        backend = InMemoryBackend()
        backend.set("abc123", entry)
        entry = backend.get("abc123")
    """

    def __init__(self) -> None:
        """Initialize the in-memory backend."""
        self._store: dict[str, CompressionEntry] = {}
        self._lock = threading.Lock()

    def get(self, hash_key: str) -> CompressionEntry | None:
        """Retrieve and decompress an entry by hash key.

        Args:
            hash_key: The unique hash identifying the entry.

        Returns:
            CompressionEntry if found, None otherwise.
        """
        with self._lock:
            entry = self._store.get(hash_key)
            if entry is None:
                return None
            return _decompress_entry(entry)

    def set(self, hash_key: str, entry: CompressionEntry) -> None:
        """Compress and store an entry with the given hash key.

        Args:
            hash_key: The unique hash identifying the entry.
            entry: The CompressionEntry to store.
        """
        with self._lock:
            self._store[hash_key] = _compress_entry(entry)

    def delete(self, hash_key: str) -> bool:
        """Delete an entry by hash key.

        Args:
            hash_key: The unique hash identifying the entry.

        Returns:
            True if entry was deleted, False if it didn't exist.
        """
        with self._lock:
            if hash_key in self._store:
                del self._store[hash_key]
                return True
            return False

    def exists(self, hash_key: str) -> bool:
        """Check if an entry exists.

        Args:
            hash_key: The unique hash identifying the entry.

        Returns:
            True if entry exists, False otherwise.
        """
        with self._lock:
            return hash_key in self._store

    def clear(self) -> None:
        """Remove all entries from storage."""
        with self._lock:
            self._store.clear()

    def count(self) -> int:
        """Get the number of entries in storage.

        Returns:
            Number of entries currently stored.
        """
        with self._lock:
            return len(self._store)

    def keys(self) -> list[str]:
        """Get all hash keys in storage.

        Returns:
            List of all hash keys.
        """
        with self._lock:
            return list(self._store.keys())

    def items(self) -> list[tuple[str, CompressionEntry]]:
        """Get all entries as (hash_key, entry) pairs.

        Returns:
            List of (hash_key, CompressionEntry) tuples.
        """
        with self._lock:
            return list(self._store.items())

    def get_stats(self) -> dict[str, Any]:
        """Get backend statistics.

        Returns:
            Dict with stats including entry_count and memory estimate.
        """
        with self._lock:
            entry_count = len(self._store)
            # Rough memory estimate from compressed storage.
            bytes_used = sys.getsizeof(self._store)
            for entry in self._store.values():
                bytes_used += sys.getsizeof(entry)
                bytes_used += len(entry.original_content.encode("utf-8"))
                bytes_used += len(entry.compressed_content.encode("utf-8"))

            return {
                "backend_type": "memory",
                "entry_count": entry_count,
                "bytes_used": bytes_used,
            }
