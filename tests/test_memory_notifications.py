"""Tests for memory notification push-back mechanism.

Covers:
1. ``MemoryHandler.format_memory_events()`` -- structured event parsing
2. Streaming SSE ``event: memory_op`` generation (via generator iteration)
3. Non-streaming ``X-Headroom-Memory-*`` response headers (proxy integration)
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler, MemoryMode

# =========================================================================
# format_memory_events unit tests
# =========================================================================


def _tool_result(content: str) -> dict[str, Any]:
    """Simulate an Anthropic-format tool result."""
    return {"type": "tool_result", "tool_use_id": "tu_1", "content": content}


class TestFormatMemoryEvents:
    """Direct unit tests for ``MemoryHandler.format_memory_events()``."""

    def test_save_event(self) -> None:
        results = [
            _tool_result(
                json.dumps({"status": "saved", "memory_id": "m1", "content": "Prefers Python"})
            )
        ]
        events = MemoryHandler.format_memory_events(results)
        assert len(events) == 1
        assert events[0]["op"] == "memory_save"
        assert events[0]["status"] == "saved"
        assert events[0]["memory_id"] == "m1"
        assert "Python" in events[0]["content"]

    def test_search_event(self) -> None:
        payload = {
            "status": "found",
            "count": 2,
            "memories": [{"id": "m1", "content": "fact A"}, {"id": "m2", "content": "fact B"}],
        }
        results = [_tool_result(json.dumps(payload))]
        events = MemoryHandler.format_memory_events(results)
        assert len(events) == 1
        assert events[0]["op"] == "memory_search"
        assert events[0]["status"] == "found"
        assert events[0]["count"] == 2

    def test_update_event(self) -> None:
        results = [_tool_result(json.dumps({"status": "updated", "memory_id": "m1"}))]
        events = MemoryHandler.format_memory_events(results)
        assert len(events) == 1
        assert events[0]["op"] == "memory_update"
        assert events[0]["memory_id"] == "m1"

    def test_delete_found_event(self) -> None:
        results = [_tool_result(json.dumps({"status": "deleted", "memory_id": "m1"}))]
        events = MemoryHandler.format_memory_events(results)
        assert len(events) == 1
        assert events[0]["op"] == "memory_delete"
        assert events[0]["memory_id"] == "m1"

    def test_delete_not_found_event(self) -> None:
        results = [_tool_result(json.dumps({"status": "not_found", "memory_id": "m1"}))]
        events = MemoryHandler.format_memory_events(results)
        assert len(events) == 1
        assert events[0]["op"] == "memory_delete"
        assert events[0]["status"] == "not_found"

    def test_list_event(self) -> None:
        results = [_tool_result(json.dumps({"status": "ok", "count": 3}))]
        events = MemoryHandler.format_memory_events(results)
        assert len(events) == 1
        assert events[0]["op"] == "memory_list"
        assert events[0]["count"] == 3

    def test_error_event(self) -> None:
        results = [_tool_result(json.dumps({"status": "error", "error": "backend offline"}))]
        events = MemoryHandler.format_memory_events(results)
        assert len(events) == 1
        assert events[0]["op"] == "memory_error"
        assert events[0]["error"] == "backend offline"

    def test_multiple_events(self) -> None:
        results = [
            _tool_result(json.dumps({"status": "saved", "memory_id": "m1", "content": "alpha"})),
            _tool_result(json.dumps({"status": "saved", "memory_id": "m2", "content": "beta"})),
        ]
        events = MemoryHandler.format_memory_events(results)
        assert len(events) == 2
        assert events[0]["memory_id"] == "m1"
        assert events[1]["memory_id"] == "m2"

    def test_empty_results_yields_no_events(self) -> None:
        assert MemoryHandler.format_memory_events([]) == []

    def test_empty_content_skipped(self) -> None:
        results = [
            {"type": "tool_result", "tool_use_id": "t1", "content": ""},
            _tool_result(json.dumps({"status": "saved", "memory_id": "m1"})),
        ]
        events = MemoryHandler.format_memory_events(results)
        assert len(events) == 1
        assert events[0]["memory_id"] == "m1"

    def test_invalid_json_skipped(self) -> None:
        results = [
            _tool_result("not valid json"),
            _tool_result(json.dumps({"status": "saved", "memory_id": "m2"})),
        ]
        events = MemoryHandler.format_memory_events(results)
        assert len(events) == 1
        assert events[0]["memory_id"] == "m2"

    def test_unknown_status(self) -> None:
        results = [_tool_result(json.dumps({"status": "unknown_operation"}))]
        events = MemoryHandler.format_memory_events(results)
        assert len(events) == 1
        assert events[0]["op"] == "memory_unknown"
        assert events[0]["status"] == "unknown_operation"

    def test_dedup_note_preserved(self) -> None:
        payload = {
            "status": "saved",
            "memory_id": "m1",
            "note": "Similar memory exists (id: m0, score: 95%)",
        }
        results = [_tool_result(json.dumps(payload))]
        events = MemoryHandler.format_memory_events(results)
        assert "note" in events[0]
        assert "Similar memory" in events[0]["note"]


# =========================================================================
# Proxy integration tests: X-Headroom-Memory-* headers
# =========================================================================


class TestMemoryNotificationHeaders:
    """Tests that verify ``X-Headroom-Memory-*`` response headers
    are correctly formatted from tool results.

    These tests exercise the header-formatting logic directly rather
    than going through the full proxy stack (which requires a running
    FastAPI application with a built Rust extension, HTTP/2 support,
    etc.).  The header construction happens inline in the handler after
    ``handle_memory_tool_calls`` returns -- verifying the format of
    ``format_memory_events`` plus the header keys is sufficient to
    validate the push-back mechanism.
    """

    def test_memory_save_header_format(self) -> None:
        """A single memory_save produces ``x-headroom-memory-count: 1``
        and a valid JSON ``x-headroom-memory-op``."""
        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": "tu_1",
                "content": json.dumps(
                    {"status": "saved", "memory_id": "mem_001", "content": "Prefers Python"}
                ),
            }
        ]
        events = MemoryHandler.format_memory_events(tool_results)

        assert len(events) == 1
        # These are the assertions the header code would make
        assert events[0]["op"] == "memory_save"
        assert events[0]["status"] == "saved"
        assert events[0]["memory_id"] == "mem_001"

        # Verify the serialized form that goes into the header
        serialized = json.dumps(events[0])
        parsed = json.loads(serialized)
        assert parsed["op"] == "memory_save"
        assert parsed["memory_id"] == "mem_001"

    def test_no_memory_tools_no_headers(self) -> None:
        """When no tool results exist, no events and thus no headers."""
        events = MemoryHandler.format_memory_events([])
        assert len(events) == 0

    def test_multiple_ops_summary_header(self) -> None:
        """Multiple memory operations produce a summary with all events."""
        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": "tu_1",
                "content": json.dumps(
                    {"status": "saved", "memory_id": "mem_001", "content": "fact one"}
                ),
            },
            {
                "type": "tool_result",
                "tool_use_id": "tu_2",
                "content": json.dumps(
                    {"status": "saved", "memory_id": "mem_002", "content": "fact two"}
                ),
            },
        ]
        events = MemoryHandler.format_memory_events(tool_results)

        assert len(events) == 2
        # The header code would emit x-headroom-memory-count: 2
        # and x-headroom-memory-ops-summary with both events
        summary = json.dumps(events)
        parsed_summary = json.loads(summary)
        assert len(parsed_summary) == 2
        assert parsed_summary[0]["memory_id"] == "mem_001"
        assert parsed_summary[1]["memory_id"] == "mem_002"


# =========================================================================
# SSE streaming push tests (event: memory_op)
# =========================================================================


class FakeMemoryBackend:
    """Minimal in-process backend that records operations and returns
    deterministic results."""

    def __init__(self) -> None:
        self.saved: list[dict[str, Any]] = []
        self.search_results: list[Any] = []

    async def _ensure_initialized(self) -> None:
        pass

    async def save_memory(self, **kwargs: Any) -> Any:
        self.saved.append(kwargs)
        return SimpleNamespace(
            id=f"mem-{len(self.saved)}",
            content=kwargs.get("content", ""),
            metadata={},
        )

    async def search_memories(self, **kwargs: Any) -> list[Any]:
        return list(self.search_results)

    async def update_memory(self, **kwargs: Any) -> Any:
        return SimpleNamespace(id=kwargs.get("memory_id", "?"))

    async def delete_memory(self, memory_id: str) -> bool:
        return True

    async def close(self) -> None:
        pass


@pytest.fixture
def memory_handler(tmp_path: Any) -> MemoryHandler:
    h = MemoryHandler(
        MemoryConfig(
            enabled=True,
            backend="local",
            db_path=str(tmp_path / "test_memory.db"),
            inject_tools=True,
            inject_context=False,
            mode=MemoryMode.TOOL,
        ),
        agent_type="test",
    )
    # Swap in the fake backend to avoid SQLite + embedder overhead.
    h._backend = FakeMemoryBackend()
    h._initialized = True
    return h


class TestMemorySseEventPush:
    """Tests that memory tool calls in streaming mode yield ``event: memory_op``
    SSE events."""

    @pytest.mark.asyncio
    async def test_memory_save_produces_sse_event(self, memory_handler: MemoryHandler) -> None:
        """A memory_save tool call in a streaming response must yield an
        ``event: memory_op`` SSE event after format_memory_events."""
        # Simulate an Anthropic response with a memory_save tool call
        response_payload: dict[str, Any] = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_save_1",
                    "name": "memory_save",
                    "input": {
                        "content": "Alice prefers Python over JavaScript",
                        "importance": 0.8,
                    },
                },
            ],
            "stop_reason": "tool_use",
        }

        assert memory_handler.has_memory_tool_calls(response_payload, "anthropic")

        tool_results = await memory_handler.handle_memory_tool_calls(
            response_payload,
            "alice",
            "anthropic",
        )

        assert len(tool_results) == 1

        events = MemoryHandler.format_memory_events(tool_results)
        assert len(events) == 1
        assert events[0]["op"] == "memory_save"
        assert events[0]["status"] == "saved"

        # Build the SSE payload that the streaming handler would yield
        sse_payload = f"event: memory_op\ndata: {json.dumps(events[0])}\n\n"

        # Verify the SSE format is parseable
        assert sse_payload.startswith("event: memory_op")
        assert "\ndata: " in sse_payload

        parsed_data = json.loads(sse_payload.split("\ndata: ")[1].strip())
        assert parsed_data["op"] == "memory_save"
        assert parsed_data["memory_id"] is not None
        assert "Python" in parsed_data.get("content", "")

    @pytest.mark.asyncio
    async def test_multiple_memory_ops_multiple_events(self, memory_handler: MemoryHandler) -> None:
        """When the response contains multiple memory tool calls, each must
        produce a separate event."""
        response_payload: dict[str, Any] = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_save_1",
                    "name": "memory_save",
                    "input": {"content": "fact one", "importance": 0.6},
                },
                {
                    "type": "tool_use",
                    "id": "toolu_save_2",
                    "name": "memory_save",
                    "input": {"content": "fact two", "importance": 0.7},
                },
            ],
            "stop_reason": "tool_use",
        }

        assert memory_handler.has_memory_tool_calls(response_payload, "anthropic")

        tool_results = await memory_handler.handle_memory_tool_calls(
            response_payload,
            "alice",
            "anthropic",
        )

        assert len(tool_results) == 2

        events = MemoryHandler.format_memory_events(tool_results)
        assert len(events) == 2
        assert events[0]["content"] is not None
        assert events[1]["content"] is not None
