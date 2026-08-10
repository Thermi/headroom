"""Request logger for the Headroom proxy.

Logs requests to an in-memory deque and optionally to a gzip-compressed
JSONL file. File writes are buffered and flushed periodically to reduce
IO overhead on the hot path.

Extracted from server.py for maintainability.
"""

from __future__ import annotations

import gzip
import logging
import os
import sys
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

from headroom.proxy import _json as json

if TYPE_CHECKING:
    from ..memory.tracker import ComponentStats

from headroom.proxy import request_log_redaction_policy
from headroom.proxy.models import RequestLog

IMAGE_BASE64_REDACT_THRESHOLD_BYTES = (
    request_log_redaction_policy.IMAGE_BASE64_REDACT_THRESHOLD_BYTES
)
IMAGE_BASE64_REPLACEMENT_TEMPLATE = request_log_redaction_policy.IMAGE_BASE64_REPLACEMENT_TEMPLATE
IMAGE_BEARING_FIELD_NAMES = request_log_redaction_policy.IMAGE_BEARING_FIELD_NAMES
_is_base64_image_payload = request_log_redaction_policy.is_base64_image_payload

logger = logging.getLogger(__name__)

# Constants for log redaction counter export (Prometheus). The
# Python proxy's ``/metrics`` exporter surfaces
# ``proxy_image_generation_call_log_redacted_total`` from this
# module-level counter. C3 remediation: the Rust proxy previously
# held a dead counter; that's been removed in favour of this
# Python-side counter, which is the natural owner.
_redactions_total: int = 0
_redactions_lock = Lock()


def redactions_total() -> int:
    """Return the running count of base64 redactions performed.

    Exposed for unit tests, the legacy Python ``/stats`` endpoint,
    and the Prometheus exporter
    (``proxy_image_generation_call_log_redacted_total``).
    """
    with _redactions_lock:
        return _redactions_total


def redact_image_base64(payload: Any) -> Any:
    """Public entry point for base64-image redaction.

    Walks ``payload`` (a dict, list, or string) and replaces any
    over-threshold base64 string with a size-only placeholder.
    Idempotent — applying twice yields the same structure.
    """
    global _redactions_total

    result = request_log_redaction_policy.redact_image_base64_value(payload)
    if result.redactions:
        with _redactions_lock:
            _redactions_total += result.redactions
    return result.value


class RequestLogger:
    """Log requests to gzip-compressed JSONL file with write buffering.

    Uses a deque with max 10,000 entries to prevent unbounded memory growth.
    Gracefully degrades to in-memory-only if the log file cannot be written
    (read-only filesystem, permissions error, etc.).

    File writes are buffered and flushed every ``FLUSH_INTERVAL_SECONDS``
    or when the buffer reaches ``MAX_BUFFER_SIZE`` bytes, whichever comes
    first. The buffer is held as gzip-compressed data to minimize memory
    pressure for the flush cycle.
    """

    MAX_LOG_ENTRIES = int(os.environ.get("HEADROOM_REQUEST_LOGGER_MAX_ENTRIES", "10000"))
    MAX_TOTAL_BYTES = int(
        os.environ.get("HEADROOM_REQUEST_LOGGER_MAX_BYTES", str(100 * 1024 * 1024))
    )
    MAX_BUFFER_SIZE = 512 * 1024  # 512 KB before forced flush
    FLUSH_INTERVAL_SECONDS = 5.0

    def __init__(self, log_file: str | None = None, log_full_messages: bool = False):
        self.log_file = Path(log_file) if log_file else None
        self.log_full_messages = log_full_messages
        self._logs: deque[RequestLog] = deque(maxlen=self.MAX_LOG_ENTRIES)
        self._logs_bytes: int = 0

        self._buffer: list[bytes] = []
        self._buffer_size = 0
        self._last_flush = time.monotonic()
        self._flush_lock = threading.Lock()
        self._flush_timer: threading.Timer | None = None

        if self.log_file:
            try:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.warning(
                    "Cannot create log directory %s: %s — logging to memory only",
                    self.log_file.parent,
                    e,
                )
                self.log_file = None

    def _flush_buffer(self) -> None:
        """Flush buffered log lines to the gzip-compressed JSONL file."""
        if not self.log_file or not self._buffer:
            return
        try:
            with gzip.open(self.log_file, "ab", compresslevel=6) as f:
                for chunk in self._buffer:
                    f.write(chunk)
        except OSError:
            pass
        self._buffer = []
        self._buffer_size = 0

    def _maybe_flush(self) -> None:
        """Flush if buffer exceeds size threshold or enough time has passed."""
        now = time.monotonic()
        if (
            self._buffer_size >= self.MAX_BUFFER_SIZE
            or now - self._last_flush >= self.FLUSH_INTERVAL_SECONDS
        ):
            with self._flush_lock:
                self._flush_buffer()
                self._last_flush = time.monotonic()

    def _queue_log_line(self, line: bytes) -> None:
        """Append a compressed log line to the write buffer."""
        self._buffer.append(line)
        self._buffer_size += len(line)
        self._maybe_flush()

    def flush(self) -> None:
        """Force-flush any buffered log lines. Called at shutdown."""
        with self._flush_lock:
            self._flush_buffer()

    def log(self, entry: RequestLog):
        """Log a request. Oldest entries are automatically removed when limit reached.

        Base64-encoded image payloads in ``request_messages`` /
        ``compressed_messages`` / ``response_content`` are redacted before
        write. File writes are buffered and gzip-compressed.
        """
        # Only redact if we're actually keeping them in memory or writing to disk
        if self.log_full_messages or self.log_file:
            if entry.request_messages is not None:
                entry.request_messages = redact_image_base64(entry.request_messages)
            if entry.compressed_messages is not None:
                entry.compressed_messages = redact_image_base64(entry.compressed_messages)
            if entry.response_content is not None:
                entry.response_content = redact_image_base64(entry.response_content)

        self._logs.append(entry)
        self._logs_bytes += sys.getsizeof(entry)

        # Evict oldest entries if total byte budget exceeded
        while self._logs_bytes > self.MAX_TOTAL_BYTES and len(self._logs) > 1:
            old = self._logs.popleft()
            self._logs_bytes -= sys.getsizeof(old)

        if self.log_file:
            log_dict = asdict(entry)
            if not self.log_full_messages:
                log_dict.pop("request_messages", None)
                log_dict.pop("compressed_messages", None)
                log_dict.pop("response_content", None)
            line = json.dumps(log_dict, separators=(",", ":")).encode("utf-8") + b"\n"
            self._queue_log_line(line)

    def get_recent(self, n: int = 100) -> list[dict]:
        """Get recent log entries (without request/compressed messages and response_content)."""
        # Convert deque to list for slicing (deque doesn't support slicing)
        entries = list(self._logs)[-n:]
        return [
            {
                k: v
                for k, v in asdict(e).items()
                if k not in ("request_messages", "compressed_messages", "response_content")
            }
            for e in entries
        ]

    def get_recent_with_messages(self, n: int = 20) -> list[dict]:
        """Get recent log entries including full request/response messages."""
        entries = list(self._logs)[-n:]
        return [asdict(e) for e in entries]

    def stats(self) -> dict:
        """Get logging statistics."""
        return {
            "total_logged": len(self._logs),
            "log_file": str(self.log_file) if self.log_file else None,
        }

    def get_memory_stats(self) -> ComponentStats:
        """Get memory statistics for the MemoryTracker.

        Returns:
            ComponentStats with current memory usage.
        """
        from ..memory.tracker import ComponentStats

        # Calculate size
        size_bytes = sys.getsizeof(self._logs)

        for log_entry in self._logs:
            size_bytes += sys.getsizeof(log_entry)
            # Add string fields
            if log_entry.request_id:
                size_bytes += len(log_entry.request_id)
            if log_entry.provider:
                size_bytes += len(log_entry.provider)
            if log_entry.model:
                size_bytes += len(log_entry.model)
            if log_entry.error:
                size_bytes += len(log_entry.error)
            # Messages and response can be large
            if log_entry.request_messages:
                size_bytes += sys.getsizeof(log_entry.request_messages)
            if log_entry.compressed_messages:
                size_bytes += sys.getsizeof(log_entry.compressed_messages)
            if log_entry.response_content:
                size_bytes += len(log_entry.response_content)

        return ComponentStats(
            name="request_logger",
            entry_count=len(self._logs),
            size_bytes=size_bytes,
            budget_bytes=None,
            hits=0,
            misses=0,
            evictions=0,
        )
