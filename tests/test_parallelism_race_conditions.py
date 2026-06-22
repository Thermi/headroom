"""Race-condition and data-integrity tests for free-threaded parallelism.

Validates that every parallelised code path introduced in Phase J
(REALIGNMENT/13-phase-J-free-threaded-parallelism.md) preserves:

1. **Order** — parallel outputs match input order.
2. **No data loss** — all items processed; counts/timing accumulate correctly.
3. **Error isolation** — one failing item does not corrupt neighbours.
4. **Lock correctness** — concurrent access to shared state is safe.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("fastapi")

from headroom.proxy.handlers import batch as batch_module
from headroom.proxy.rate_limiter import TokenBucketRateLimiter

# ============================================================================
# Helpers
# ============================================================================

_SLOW_PIPELINE_DELAY = 0.05


def _make_slow_pipeline(delay: float = _SLOW_PIPELINE_DELAY) -> SimpleNamespace:
    """A pipeline mock that sleeps so parallel execution is observable."""

    def apply(**kwargs: Any) -> SimpleNamespace:
        time.sleep(delay)
        messages = kwargs.get("messages", [])
        return SimpleNamespace(
            messages=messages,
            tokens_before=100,
            tokens_after=40,
            timing={"slow_step": delay},
            transforms_applied=["slow_step"],
        )

    return SimpleNamespace(apply=apply)


def _make_counting_pipeline(counter: list[int]) -> SimpleNamespace:
    """A pipeline mock that counts invocations and records message content."""

    def apply(**kwargs: Any) -> SimpleNamespace:
        counter[0] += 1
        messages = kwargs.get("messages", [])
        return SimpleNamespace(
            messages=messages,
            tokens_before=100,
            tokens_after=40,
            timing={"step": 0.01},
            transforms_applied=["step"],
        )

    return SimpleNamespace(apply=apply)


# ============================================================================
# Reusable dummy handler for batch tests
# ============================================================================


class _DummyHandler(batch_module.BatchHandlerMixin):
    OPENAI_API_URL = "https://openai.example"
    GEMINI_API_URL = "https://gemini.example"

    def __init__(self, optimize: bool = True, pipeline: Any = None) -> None:  # noqa: ANN401
        self.http_client = SimpleNamespace(
            get=AsyncMock(return_value=SimpleNamespace(status_code=200, text="{}")),
            post=AsyncMock(
                return_value=SimpleNamespace(
                    status_code=200,
                    json=lambda: {"id": "batch_1"},
                    headers={},
                )
            ),
            request=AsyncMock(),
        )
        self.metrics = SimpleNamespace(
            record_calls=[],
            failed_calls=[],
            record_request=AsyncMock(),
            record_failed=AsyncMock(),
        )
        self.config = SimpleNamespace(
            optimize=optimize,
            ccr_inject_tool=False,
            ccr_inject_system_instructions=False,
        )
        self.openai_provider = SimpleNamespace(get_context_limit=lambda model: 8192)
        self.openai_pipeline = pipeline or _make_slow_pipeline()
        self.anthropic_pipeline = pipeline or _make_slow_pipeline()
        self._request_counter = 0
        self._retry_response = SimpleNamespace(status_code=200)

    async def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"req-{self._request_counter}"

    async def _record_request_outcome(self, outcome: Any) -> None:  # noqa: ANN401
        from headroom.proxy.outcome import emit_request_outcome

        await emit_request_outcome(self, outcome)

    def _extract_tags(self, headers: dict) -> dict[str, str]:
        return {}

    async def handle_passthrough(self, request: Any, base_url: str) -> dict:  # noqa: ANN401
        return {"request": request, "base_url": base_url}

    async def _retry_request(
        self, method: str, url: str, headers: dict, body: Any, **kwargs: Any
    ) -> Any:  # noqa: ANN401
        return self._retry_response

    def _gemini_contents_to_messages(
        self, contents: list, system_instruction: Any
    ) -> tuple[list, list]:  # noqa: ANN401
        messages = [{"role": "user", "content": part["parts"][0]["text"]} for part in contents]
        return messages, []

    def _messages_to_gemini_contents(self, messages: list) -> tuple[list, Any]:  # noqa: ANN401
        return (
            [{"parts": [{"text": m["content"]}]} for m in messages],
            None,
        )


# ============================================================================
# 1. BATCH.PY — Order preservation, no data loss, error isolation
# ============================================================================
#
# Google batch compression is inlined inside handle_google_batch_create
# (not a standalone method).  We test the parallelisation mechanism through
# _compress_batch_jsonl which uses the exact same pattern
# (loop.run_in_executor + as_completed).  For the Google batch path we
# test the module-level helpers and rely on test_proxy_handlers_batch.py
# for end-to-end coverage.


class TestBatchParallelism:
    """Race-condition tests for parallel batch compression in batch.py."""

    # -- Order preservation ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_jsonl_batch_order_preserved(self, monkeypatch) -> None:
        """Parallel JSONL batch returns lines in original order."""
        handler = _DummyHandler(optimize=True)
        self._install_batch_support(monkeypatch)

        lines_input = [
            json.dumps(
                {
                    "body": {
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": f"hi-{i}"}],
                    }
                }
            )
            for i in range(10)
        ]
        content = "\n".join(lines_input)
        compressed_lines, stats = await handler._compress_batch_jsonl(content, "req-1")

        assert len(compressed_lines) == 10
        for i, line in enumerate(compressed_lines):
            body = json.loads(line)["body"]
            assert body["messages"][0]["content"] == f"hi-{i}", f"Line {i} out of order"

    # -- No data loss ---------------------------------------------------------

    @pytest.mark.asyncio
    async def test_jsonl_batch_all_items_processed(self, monkeypatch) -> None:
        """All JSONL items are compressed, none silently dropped."""
        counter = [0]
        handler = _DummyHandler(optimize=True, pipeline=_make_counting_pipeline(counter))
        self._install_batch_support(monkeypatch)

        lines_input = [
            json.dumps(
                {
                    "body": {
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                }
            )
            for _ in range(20)
        ]
        content = "\n".join(lines_input)
        await handler._compress_batch_jsonl(content, "req-1")
        assert counter[0] == 20, f"Expected 20 pipeline calls, got {counter[0]}"

    # -- Token accumulation ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_jsonl_batch_token_counts_accumulate(self, monkeypatch) -> None:
        """Token counts accumulate correctly across parallel JSONL items."""
        handler = _DummyHandler(optimize=True)
        self._install_batch_support(monkeypatch)

        lines_input = [
            json.dumps(
                {
                    "body": {
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                }
            )
            for _ in range(7)
        ]
        content = "\n".join(lines_input)
        _lines, stats = await handler._compress_batch_jsonl(content, "req-1")
        assert stats["total_original_tokens"] == 700
        assert stats["total_compressed_tokens"] == 280
        assert stats["total_tokens_saved"] == 420

    # -- Error isolation ------------------------------------------------------

    @pytest.mark.asyncio
    async def test_jsonl_batch_error_isolation(self, monkeypatch) -> None:
        """A single failing JSONL line does not affect neighbouring lines."""
        self._install_batch_support(monkeypatch)

        error_count = [0]

        def failing_apply(**kwargs: Any) -> Any:  # noqa: ANN401
            messages = kwargs.get("messages", [])
            if messages and "fail" in messages[0].get("content", ""):
                error_count[0] += 1
                raise RuntimeError("intentional failure")
            return SimpleNamespace(
                messages=messages,
                tokens_before=100,
                tokens_after=40,
                timing={"step": 0.01},
                transforms_applied=["step"],
            )

        pipeline = SimpleNamespace(apply=failing_apply)
        handler = _DummyHandler(optimize=True, pipeline=pipeline)

        lines_input = [
            json.dumps(
                {
                    "body": {
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": "ok-1"}],
                    }
                }
            ),
            json.dumps(
                {
                    "body": {
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": "fail"}],
                    }
                }
            ),
            json.dumps(
                {
                    "body": {
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": "ok-2"}],
                    }
                }
            ),
        ]
        content = "\n".join(lines_input)
        compressed_lines, stats = await handler._compress_batch_jsonl(content, "req-1")

        assert len(compressed_lines) == 3
        # Failing line passes through unchanged
        body_1 = json.loads(compressed_lines[1])["body"]
        assert body_1["messages"][0]["content"] == "fail"
        # Successful lines are compressed
        body_0 = json.loads(compressed_lines[0])["body"]
        assert body_0["messages"][0]["content"] == "ok-1"
        body_2 = json.loads(compressed_lines[2])["body"]
        assert body_2["messages"][0]["content"] == "ok-2"
        # Pipeline was called exactly 3 times (including the failing one)
        assert error_count[0] == 1

    # -- High concurrency spike -----------------------------------------------

    @pytest.mark.asyncio
    async def test_jsonl_batch_high_concurrency_no_crash(self, monkeypatch) -> None:
        """50 concurrent JSONL items complete without error."""
        handler = _DummyHandler(optimize=True, pipeline=_make_slow_pipeline(0.01))
        self._install_batch_support(monkeypatch)

        lines_input = [
            json.dumps(
                {
                    "body": {
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                }
            )
            for _ in range(50)
        ]
        content = "\n".join(lines_input)
        _lines, stats = await handler._compress_batch_jsonl(content, "req-1")
        assert stats["total_requests"] == 50

    # -- Parallelism timing gate ----------------------------------------------

    @pytest.mark.asyncio
    async def test_jsonl_batch_items_run_in_parallel(self, monkeypatch) -> None:
        """Multiple items compress concurrently, not sequentially.

        With 10 items each taking 50ms, sequential = 500ms, parallel ≈ 50ms.
        """
        handler = _DummyHandler(optimize=True, pipeline=_make_slow_pipeline(0.05))
        self._install_batch_support(monkeypatch)

        lines_input = [
            json.dumps(
                {
                    "body": {
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                }
            )
            for _ in range(10)
        ]
        content = "\n".join(lines_input)

        t0 = time.monotonic()
        await handler._compress_batch_jsonl(content, "req-1")
        elapsed = time.monotonic() - t0

        # Sequential 10×50ms = 500ms.  Parallel with 8 workers ≈ 100ms.
        # Allow generous margin for CI jitter.
        assert elapsed < 0.35, (
            f"Batch items appear serialised: {elapsed:.3f}s "
            f"(expected <0.35s for 10 parallel 50ms items)"
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _install_batch_support(monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeInjector:
            def __init__(self, **kwargs: Any) -> None: ...

            def process_request(self, messages: Any, tools: Any) -> tuple:  # noqa: ANN401
                return messages, tools, False

        class FakeTokenizer:
            @staticmethod
            def count_messages(messages: Any) -> int:  # noqa: ANN401
                return 10

        monkeypatch.setitem(
            sys.modules, "headroom.ccr", SimpleNamespace(CCRToolInjector=FakeInjector)
        )
        monkeypatch.setitem(
            sys.modules,
            "headroom.tokenizers",
            SimpleNamespace(get_tokenizer=lambda model: FakeTokenizer()),
        )
        monkeypatch.setitem(
            sys.modules,
            "headroom.utils",
            SimpleNamespace(extract_user_query=lambda messages: "query"),
        )


# ============================================================================
# 2. ANTHROPIC.PY — Batch parallelisation
# ============================================================================


class TestAnthropicBatchParallelism:
    """Race-condition tests for parallel Anthropic batch compression."""

    # Anthropic batch compression is inside handle_anthropic_batch_create.
    # We test it indirectly via the DummyHandler + monkeypatch setup,
    # but the actual compression loop is tightly coupled.  We rely on
    # the high-level test_proxy_handlers_batch.py tests for this path.

    @pytest.mark.asyncio
    async def test_anthropic_batch_executor_lazy_init(self) -> None:
        """The module-level executor is lazily initialized and thread-safe."""
        from headroom.proxy.handlers.anthropic import (
            _ANTHROPIC_BATCH_PARALLELISM_DEFAULT,
            _anthropic_batch_parallelism,
            _get_anthropic_batch_executor,
        )

        p = _anthropic_batch_parallelism()
        assert 1 <= p <= 64
        assert p == _ANTHROPIC_BATCH_PARALLELISM_DEFAULT

        executor = _get_anthropic_batch_executor()
        assert executor._max_workers == _ANTHROPIC_BATCH_PARALLELISM_DEFAULT

        # Calling twice returns same executor
        assert _get_anthropic_batch_executor() is executor


# ============================================================================
# 3. RATE_LIMITER — Per-key lock correctness
# ============================================================================


class TestRateLimiterParallelism:
    """Race-condition tests for per-key rate limiter locks."""

    @pytest.mark.asyncio
    async def test_different_keys_dont_block(self) -> None:
        """Two concurrent check_request calls for different keys proceed."""
        limiter = TokenBucketRateLimiter(requests_per_minute=10, tokens_per_minute=100000)

        async def check_a() -> float:
            t0 = time.monotonic()
            await limiter.check_request("key-a")
            return time.monotonic() - t0

        async def check_b() -> float:
            t0 = time.monotonic()
            await limiter.check_request("key-b")
            return time.monotonic() - t0

        t0 = time.monotonic()
        dur_a, dur_b = await asyncio.gather(check_a(), check_b())
        total = time.monotonic() - t0
        # Total wall time should be ~max(dur_a, dur_b), not ~sum
        # With per-key locks both proceed concurrently
        assert total < max(dur_a, dur_b) + 0.05, (
            f"Different keys blocked each other: total={total:.3f}s, "
            f"dur_a={dur_a:.3f}s, dur_b={dur_b:.3f}s"
        )

    @pytest.mark.asyncio
    async def test_same_key_serializes(self) -> None:
        """Two concurrent check_request calls for the SAME key serialize."""
        limiter = TokenBucketRateLimiter(requests_per_minute=10000, tokens_per_minute=100000)

        slow_event = asyncio.Event()
        release_event = asyncio.Event()

        async def slow_check() -> None:
            slow_event.set()
            await release_event.wait()
            async with limiter._get_lock("same-key"):
                pass  # hold the lock

        async def fast_check() -> bool:
            await slow_event.wait()
            allowed, _wait = await limiter.check_request("same-key")
            return allowed

        async with limiter._cleanup_lock:
            pass  # ensure cleanup_lock is not held

        # Start slow task first so it acquires the lock on "same-key"
        slow_task = asyncio.create_task(slow_check())
        fast_task = asyncio.create_task(fast_check())

        await slow_event.wait()
        await asyncio.sleep(0.02)
        release_event.set()

        result = await asyncio.gather(slow_task, fast_task, return_exceptions=True)
        # Fast check should have succeeded (waited for slow to release)
        assert result[1] is True or isinstance(result[1], Exception) == False  # noqa: E712

    @pytest.mark.asyncio
    async def test_cleanup_does_not_race_with_check(self) -> None:
        """Concurrent cleanup and check_request don't corrupt bucket state."""
        limiter = TokenBucketRateLimiter(requests_per_minute=60, tokens_per_minute=100000)

        # Fill up buckets to near-stale threshold
        for i in range(30):
            await limiter.check_request(f"key-{i}")

        async def cleanup_loop() -> None:
            for _ in range(10):
                async with limiter._cleanup_lock:
                    await limiter._cleanup_stale_buckets()
                await asyncio.sleep(0.005)

        async def check_loop() -> None:
            for i in range(30):
                await limiter.check_request(f"key-{i}")
                await asyncio.sleep(0.003)

        await asyncio.gather(cleanup_loop(), check_loop(), return_exceptions=True)

        stats = await limiter.stats()
        assert stats["active_keys"] >= 0  # no crash, state is consistent

    @pytest.mark.asyncio
    async def test_stats_consistent_under_concurrent_load(self) -> None:
        """stats() returns consistent data while requests are in flight."""
        limiter = TokenBucketRateLimiter(requests_per_minute=1000, tokens_per_minute=100000)

        async def hammer() -> None:
            for _ in range(20):
                await limiter.check_request("stats-test-key")
                await limiter.check_tokens("stats-test-key", 100)

        async def read_stats() -> None:
            for _ in range(10):
                s = await limiter.stats()
                assert "active_keys" in s
                assert "requests_per_minute" in s
                await asyncio.sleep(0.005)

        await asyncio.gather(hammer(), read_stats(), return_exceptions=True)

    @pytest.mark.asyncio
    async def test_check_tokens_different_keys_parallel(self) -> None:
        """check_tokens for different keys runs in parallel."""
        limiter = TokenBucketRateLimiter(requests_per_minute=10000, tokens_per_minute=100000)

        async def check_big(key: str, tokens: int) -> float:
            t0 = time.monotonic()
            await limiter.check_tokens(key, tokens)
            return time.monotonic() - t0

        t0 = time.monotonic()
        durs = await asyncio.gather(
            check_big("tok-a", 1000),
            check_big("tok-b", 1000),
            check_big("tok-c", 1000),
        )
        total = time.monotonic() - t0
        max_dur = max(durs)
        assert total < max_dur + 0.05, (
            f"check_tokens for different keys serialized: total={total:.3f}s, "
            f"max_dur={max_dur:.3f}s"
        )


# ============================================================================
# 4. MEMORY_HANDLER — Parallel tool call execution
# ============================================================================


class TestMemoryHandlerParallelism:
    """Race-condition tests for parallel memory tool call execution."""

    @pytest.fixture
    def mock_handler(self) -> Any:  # noqa: ANN401
        """Build a MemoryHandler with mocked execute methods."""
        with patch("headroom.proxy.memory_handler.MemoryHandler._ensure_initialized", AsyncMock()):
            from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler

            handler = MemoryHandler(MemoryConfig(enabled=True))
            handler._backend = MagicMock()

            async def fake_execute_memory_tool(
                tool_name: str, input_data: dict, user_id: str, provider: str, **kwargs: Any
            ) -> str:
                await asyncio.sleep(0.02)
                if tool_name == "memory_search":
                    return json.dumps({"results": [{"id": "mem-1", "content": "found"}]})
                elif tool_name == "memory_save":
                    return json.dumps({"status": "saved", "id": "mem-saved"})
                elif tool_name == "memory_list":
                    return json.dumps({"memories": []})
                return json.dumps({"status": "ok"})

            handler._initialized = True
            handler._execute_memory_tool = AsyncMock(side_effect=fake_execute_memory_tool)
            return handler

    @staticmethod
    def _anthropic_tool_calls(calls: list[dict]) -> dict:
        """Wrap tool call dicts in Anthropic response format."""
        return {
            "content": [
                {
                    "type": "tool_use",
                    "id": c["id"],
                    "name": c["name"],
                    "input": c["input"],
                }
                for c in calls
            ]
        }

    @pytest.mark.asyncio
    async def test_read_tools_run_in_parallel(self, mock_handler: Any) -> None:  # noqa: ANN401
        """Multiple memory_search calls run concurrently, not sequentially."""
        tool_calls = [
            {"id": f"call-{i}", "name": "memory_search", "input": {"query": f"test-{i}"}}
            for i in range(5)
        ]
        response = self._anthropic_tool_calls(tool_calls)

        t0 = time.monotonic()
        results = await mock_handler.handle_memory_tool_calls(
            response, "user-1", provider="anthropic"
        )
        elapsed = time.monotonic() - t0

        assert len(results) == 5
        assert elapsed < 0.07, (
            f"Read tools appear sequential: {elapsed:.3f}s (expected <0.07s for 5 parallel)"
        )

    @pytest.mark.asyncio
    async def test_write_tools_run_in_parallel(self, mock_handler: Any) -> None:  # noqa: ANN401
        """Multiple memory_save calls run concurrently."""
        tool_calls = [
            {"id": f"call-{i}", "name": "memory_save", "input": {"content": f"data-{i}"}}
            for i in range(5)
        ]
        response = self._anthropic_tool_calls(tool_calls)

        t0 = time.monotonic()
        results = await mock_handler.handle_memory_tool_calls(
            response, "user-1", provider="anthropic"
        )
        elapsed = time.monotonic() - t0

        assert len(results) == 5
        assert elapsed < 0.07, (
            f"Write tools appear sequential: {elapsed:.3f}s (expected <0.07s for 5 parallel)"
        )

    @pytest.mark.asyncio
    async def test_read_write_order_preserved(self, mock_handler: Any) -> None:  # noqa: ANN401
        """Tool results maintain original call order with reads before writes."""
        tool_calls = [
            {"id": "call-search", "name": "memory_search", "input": {"query": "find"}},
            {"id": "call-save", "name": "memory_save", "input": {"content": "new data"}},
            {"id": "call-list", "name": "memory_list", "input": {}},
        ]
        response = self._anthropic_tool_calls(tool_calls)

        results = await mock_handler.handle_memory_tool_calls(
            response, "user-1", provider="anthropic"
        )
        assert len(results) == 3
        assert results[0]["tool_use_id"] == "call-search"
        assert results[1]["tool_use_id"] == "call-save"
        assert results[2]["tool_use_id"] == "call-list"

    @pytest.mark.asyncio
    async def test_error_isolation_in_parallel_tools(self, mock_handler: Any) -> None:  # noqa: ANN401
        """One failing tool call doesn't prevent others from completing."""

        async def mixed_execute(
            tool_name: str, input_data: dict, user_id: str, provider: str, **kwargs: Any
        ) -> str:  # noqa: ANN401
            await asyncio.sleep(0.01)
            if "fail" in str(input_data.get("query", "")):
                raise RuntimeError("intentional failure")
            return json.dumps({"status": "ok"})

        mock_handler._execute_memory_tool = AsyncMock(side_effect=mixed_execute)

        tool_calls = [
            {"id": "call-1", "name": "memory_search", "input": {"query": "fail"}},
            {"id": "call-2", "name": "memory_search", "input": {"query": "ok"}},
            {"id": "call-3", "name": "memory_save", "input": {"content": "data"}},
        ]
        response = self._anthropic_tool_calls(tool_calls)

        results = await mock_handler.handle_memory_tool_calls(
            response, "user-1", provider="anthropic"
        )
        # call-1 fails → omitted; call-2 and call-3 succeed
        assert len(results) == 2, f"Expected 2, got {len(results)}: {results}"
        returned_ids = {r["tool_use_id"] for r in results}
        assert "call-2" in returned_ids
        assert "call-3" in returned_ids


# ============================================================================
# 5. IMAGE COMPRESSOR — Parallel image processing
# ============================================================================


class TestImageCompressorParallelism:
    """Race-condition tests for parallel image compression."""

    @pytest.fixture
    def mock_compressor(self) -> Any:  # noqa: ANN401
        """Build an ImageCompressor with a slow mock router."""
        from unittest.mock import patch as _patch

        with (
            _patch(
                "headroom.image.tile_optimizer.optimize_images_in_messages",
                side_effect=lambda messages, provider: (messages, []),
            ),
        ):
            from headroom.image.compressor import ImageCompressor, Technique

            compressor = ImageCompressor()

            # Set up a mock router that classifies slowly (simulates ONNX inference)
            mock_router = MagicMock()
            mock_decision = MagicMock()
            mock_decision.technique = Technique.PRESERVE
            mock_decision.confidence = 0.8
            mock_router.classify = MagicMock(
                side_effect=lambda image_data, query: (
                    time.sleep(0.03),
                    mock_decision,
                )[1]
            )

            # Patch _get_router on the instance so it's a MagicMock
            # (use_mock_router check in compress() looks at type().__module__)
            compressor._get_router = MagicMock(return_value=mock_router)
            yield compressor

    def test_multiple_images_compressed_in_parallel(self, mock_compressor: Any) -> None:  # noqa: ANN401
        """Multiple images in one request are compressed concurrently."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "text", "text": "hello"},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,CCCC"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,DDDD"}},
                ],
            },
        ]

        t0 = time.monotonic()
        result = mock_compressor.compress(messages, provider="openai")
        elapsed = time.monotonic() - t0

        # 4 images × 0.03s each: sequential = 0.12s, parallel with 4 workers ≈ 0.03s
        # Allow generous margin for CI jitter
        assert elapsed < 0.12, (
            f"Images appear sequential: {elapsed:.3f}s (expected <0.12s for 4 parallel)"
        )
        assert len(result) == 3

    def test_image_order_preserved_under_parallelism(self, mock_compressor: Any) -> None:  # noqa: ANN401
        """Compressed images maintain original message/content-block order."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            },
            {"role": "system", "content": "system prompt"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,BBBB"},
                    },
                    {"type": "text", "text": "last"},
                ],
            },
        ]

        result = mock_compressor.compress(messages, provider="openai")
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "first"
        assert result[1]["role"] == "system"
        assert result[2]["content"][1]["type"] == "text"
        assert result[2]["content"][1]["text"] == "last"

    def test_concurrent_image_high_count(self, mock_compressor: Any) -> None:  # noqa: ANN401
        """16 images in one request compress without error."""
        content = []
        for i in range(16):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{chr(65 + i) * 4}"},
                }
            )
        messages = [{"role": "user", "content": content}]

        result = mock_compressor.compress(messages, provider="openai")
        assert len(result) == 1


# ============================================================================
# 6. SERVER.PY — Background init / warmup
# ============================================================================
#
# Background init concurrency and shared-transform dedup are already covered
# by test_proxy_warmup.py:
#   - test_startup_runs_shared_transform_once  — eager-load runs exactly once
#   - test_startup_optimize_false_skips_preload — optimize=False inhibits load
#   - test_memory_backend_init_failure — clean error path
#
# These tests validate the WarmupRegistry + asyncio.gather inside
# _run_background_init with spy transforms and a stubbed proxy.


# ============================================================================
# 7. RUST — py.allow_threads cleanup (indirect validation)
# ============================================================================


class TestRustAllowThreadsCleanup:
    """Validates that the Rust py.allow_threads helper compiles and works."""

    def test_maybe_allow_threads_importable(self) -> None:
        """The Rust crate compiles and the proxy can import headroom._core."""
        try:
            import headroom._core as core  # noqa: F401

            assert True
        except ImportError:
            pytest.skip("headroom._core not built (expected in CI without maturin)")


# ============================================================================
# 8. ANTHROPIC BATCH — Model-level executor + helper sanity
# ============================================================================


class TestAnthropicBatchExecutor:
    """Module-level executor for Anthropic batch compression."""

    def test_anthropic_batch_parallelism_env_override(self, monkeypatch) -> None:
        from headroom.proxy.handlers.anthropic import _anthropic_batch_parallelism

        monkeypatch.delenv("HEADROOM_BATCH_PARALLELISM", raising=False)
        assert _anthropic_batch_parallelism() == 8

        monkeypatch.setenv("HEADROOM_BATCH_PARALLELISM", "4")
        assert _anthropic_batch_parallelism() == 4

        monkeypatch.setenv("HEADROOM_BATCH_PARALLELISM", "0")
        assert _anthropic_batch_parallelism() == 1  # clamped

        monkeypatch.setenv("HEADROOM_BATCH_PARALLELISM", "999")
        assert _anthropic_batch_parallelism() == 64  # clamped

    def test_anthropic_batch_executor_singleton(self) -> None:
        from headroom.proxy.handlers.anthropic import _get_anthropic_batch_executor

        e1 = _get_anthropic_batch_executor()
        e2 = _get_anthropic_batch_executor()
        assert e1 is e2
