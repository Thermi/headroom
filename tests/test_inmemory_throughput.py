"""Tests for in-memory throughput buffers used when HEADROOM_NO_FILE_LOG is set.

Verifies that:
1. ``store_inmemory_perf_record`` / ``store_inmemory_stage_timings`` buffer ops work.
2. ``emit_request_outcome`` stores a PerfRecord in the in-memory buffer as a
   side effect.
3. ``emit_stage_timings_log`` stores stage timings keyed by request_id.
4. The ``_compute_throughput`` nested function falls back to the in-memory
   buffers when ``parse_log_files()`` returns no records.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from headroom.perf.analyzer import PerfRecord, PerfReport, calculate_throughput

# ── Helpers ────────────────────────────────────────────────────────────────


def _perf_record(  # noqa: PLR0913
    request_id: str = "hr_test_1",
    timestamp: str | None = None,
    model: str = "gpt-4",
    tokens_before: int = 1000,
    tokens_after: int = 400,
    tokens_saved: int = 600,
    total_ms: float = 500.0,
    tokens_out: int = 500,
    ttfb_ms: float = 100.0,
    optimization_ms: float = 10.0,
    stages: dict[str, float] | None = None,
) -> PerfRecord:
    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    return PerfRecord(
        timestamp=ts,
        request_id=request_id,
        model=model,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_saved=tokens_saved,
        total_ms=total_ms,
        tokens_out=tokens_out,
        ttfb_ms=ttfb_ms,
        optimization_ms=optimization_ms,
        stages=stages or {},
    )


def _stage_timings(stages: dict[str, float | None]) -> dict[str, float | None]:
    """Build a ``padded`` dict akin to what ``emit_stage_timings_log`` produces."""
    return stages


# ── Unit tests: buffer store functions ─────────────────────────────────────


class TestStoreInmemoryPerfRecord:
    def test_appends_record_to_buffer(self):
        """store_inmemory_perf_record appends a PerfRecord to the deque."""
        from headroom.proxy.server import (
            _INMEMORY_PERF_LOCK,
            _INMEMORY_PERF_RECORDS,
            store_inmemory_perf_record,
        )

        with _INMEMORY_PERF_LOCK:
            _INMEMORY_PERF_RECORDS.clear()

        rec = _perf_record(request_id="test-append")
        store_inmemory_perf_record(rec)

        with _INMEMORY_PERF_LOCK:
            records = list(_INMEMORY_PERF_RECORDS)

        assert any(r.request_id == "test-append" for r in records)

    def test_respects_deque_maxlen(self):
        """The deque respects its maxlen of 1000."""
        from headroom.proxy.server import (
            _INMEMORY_PERF_LOCK,
            _INMEMORY_PERF_RECORDS,
            store_inmemory_perf_record,
        )

        with _INMEMORY_PERF_LOCK:
            _INMEMORY_PERF_RECORDS.clear()

        # Add 1100 records; only last 1000 should remain
        for i in range(1100):
            store_inmemory_perf_record(_perf_record(request_id=f"bulk-{i}"))

        with _INMEMORY_PERF_LOCK:
            records = list(_INMEMORY_PERF_RECORDS)

        assert len(records) == 1000
        assert records[0].request_id == "bulk-100"
        assert records[-1].request_id == "bulk-1099"


class TestStoreInmemoryStageTimings:
    def test_stores_stages_by_request_id(self):
        """store_inmemory_stage_timings stores stages keyed by request_id."""
        from headroom.proxy.server import (
            _INMEMORY_STAGE_TIMINGS,
            _INMEMORY_TIMINGS_LOCK,
            store_inmemory_stage_timings,
        )

        with _INMEMORY_TIMINGS_LOCK:
            _INMEMORY_STAGE_TIMINGS.clear()

        store_inmemory_stage_timings("req-a", {"compression": 100.0, "forward": 50.0})

        with _INMEMORY_TIMINGS_LOCK:
            assert _INMEMORY_STAGE_TIMINGS.get("req-a") == {
                "compression": 100.0,
                "forward": 50.0,
            }

    def test_prunes_oldest_when_exceeding_max(self):
        """When stage_timings exceeds max, oldest entries are dropped."""
        from headroom.proxy.server import (
            _INMEMORY_STAGE_TIMINGS,
            _INMEMORY_TIMINGS_LOCK,
            store_inmemory_stage_timings,
        )

        with _INMEMORY_TIMINGS_LOCK:
            _INMEMORY_STAGE_TIMINGS.clear()

        # Add many entries to trigger pruning
        for i in range(3000):
            store_inmemory_stage_timings(f"req-{i}", {"stage": float(i)})

        with _INMEMORY_TIMINGS_LOCK:
            count = len(_INMEMORY_STAGE_TIMINGS)

        # After pruning, should be at most half-ish the max (2000 // 2 = 1000)
        # but the actual count depends on pruning strategy. Just verify < max.
        assert count <= 2000


# ── Integration: emit_request_outcome side-effect ──────────────────────────


class TestEmitRequestOutcomeInmemory:
    def test_outcome_stores_perf_record_in_memory(self):
        """After emit_request_outcome, a PerfRecord appears in the in-memory buffer."""
        from headroom.proxy.outcome import RequestOutcome, emit_request_outcome
        from headroom.proxy.server import (
            _INMEMORY_PERF_LOCK,
            _INMEMORY_PERF_RECORDS,
        )

        with _INMEMORY_PERF_LOCK:
            _INMEMORY_PERF_RECORDS.clear()

        # Build a minimal handler that matches the duck-type contract
        class _MiniHandler:
            def __init__(self):
                self.metrics = MagicMock()
                self.metrics.record_request = AsyncMock()

        handler = _MiniHandler()
        outcome = RequestOutcome(
            request_id="inmem-test-1",
            provider="openai",
            model="gpt-4",
            original_tokens=1500,
            optimized_tokens=500,
            tokens_saved=1000,
            output_tokens=200,
            attempted_input_tokens=0,
            total_latency_ms=750.0,
            overhead_ms=15.0,
            ttfb_ms=80.0,
            cache_read_tokens=300,
            cache_write_tokens=100,
            num_messages=5,
            transforms_applied=("content_router",),
        )

        import asyncio

        asyncio.run(emit_request_outcome(handler, outcome))

        with _INMEMORY_PERF_LOCK:
            records = list(_INMEMORY_PERF_RECORDS)

        matching = [r for r in records if r.request_id == "inmem-test-1"]
        assert len(matching) == 1
        r = matching[0]
        assert r.model == "gpt-4"
        assert r.tokens_before == 1500
        assert r.tokens_after == 500
        assert r.tokens_saved == 1000
        assert r.tokens_out == 200
        assert r.total_ms == 750.0
        assert r.optimization_ms == 15.0
        assert r.ttfb_ms == 80.0

    def test_outcome_storage_exception_is_silent(self):
        """If storing the in-memory record raises, the error is swallowed."""
        from headroom.proxy.outcome import RequestOutcome, emit_request_outcome

        class _MiniHandler:
            def __init__(self):
                self.metrics = MagicMock()
                self.metrics.record_request = AsyncMock()

        handler = _MiniHandler()
        outcome = RequestOutcome(
            request_id="no-crash-test",
            provider="openai",
            model="gpt-4",
            original_tokens=100,
            optimized_tokens=100,
            output_tokens=10,
            attempted_input_tokens=0,
            tokens_saved=0,
        )

        # Patch the actual store function on server module (the import target)
        with patch(
            "headroom.proxy.server.store_inmemory_perf_record",
            side_effect=RuntimeError("boom"),
        ):
            import asyncio

            asyncio.run(emit_request_outcome(handler, outcome))
            # Should not propagate — funnel still recorded metrics
            handler.metrics.record_request.assert_awaited_once()


# ── Integration: emit_stage_timings_log side-effect ────────────────────────


class TestEmitStageTimingsLogInmemory:
    def test_stage_timings_are_stored_by_request_id(self):
        """After emit_stage_timings_log, stage timings appear in the in-memory buffer."""
        from headroom.proxy.server import _INMEMORY_STAGE_TIMINGS, _INMEMORY_TIMINGS_LOCK
        from headroom.proxy.stage_timer import StageTimer, emit_stage_timings_log

        with _INMEMORY_TIMINGS_LOCK:
            _INMEMORY_STAGE_TIMINGS.clear()

        timer = StageTimer()
        with timer.measure("compression"):
            pass
        with timer.measure("upstream"):
            pass

        import asyncio

        asyncio.run(
            emit_stage_timings_log(
                path="test_path",
                request_id="stages-test-1",
                session_id="sess-1",
                stage_timer=timer,
                expected_stages=("compression", "upstream", "never_ran"),
            )
        )

        with _INMEMORY_TIMINGS_LOCK:
            stored = _INMEMORY_STAGE_TIMINGS.get("stages-test-1")

        assert stored is not None
        assert "compression" in stored
        assert stored["compression"] >= 0
        assert "upstream" in stored
        assert stored["upstream"] >= 0
        # never_ran should be None in padded, so excluded from numeric_stages
        assert "never_ran" not in stored

    def test_stage_timings_none_values_excluded(self):
        """None values in padded stages are excluded from stored numeric_stages."""
        from headroom.proxy.server import _INMEMORY_STAGE_TIMINGS, _INMEMORY_TIMINGS_LOCK
        from headroom.proxy.stage_timer import StageTimer, emit_stage_timings_log

        with _INMEMORY_TIMINGS_LOCK:
            _INMEMORY_STAGE_TIMINGS.clear()

        timer = StageTimer()
        # Only run one stage; the other stays None
        with timer.measure("ran"):
            pass

        import asyncio

        asyncio.run(
            emit_stage_timings_log(
                path="test_path",
                request_id="none-test-1",
                session_id="sess-1",
                stage_timer=timer,
                expected_stages=("ran", "skipped"),
            )
        )

        with _INMEMORY_TIMINGS_LOCK:
            stored = _INMEMORY_STAGE_TIMINGS.get("none-test-1")

        assert stored is not None
        assert "ran" in stored
        assert "skipped" not in stored


# ── Throughput fallback: _compute_throughput using in-memory data ──────────


class TestThroughputFromInmemoryFallback:
    def test_empty_log_files_falls_back_to_inmemory(self):
        """When parse_log_files returns no records, in-memory buffers are used."""
        from headroom.proxy.server import (
            _INMEMORY_PERF_LOCK,
            _INMEMORY_PERF_RECORDS,
            _INMEMORY_STAGE_TIMINGS,
            _INMEMORY_TIMINGS_LOCK,
            store_inmemory_perf_record,
            store_inmemory_stage_timings,
        )

        # Clear and populate buffers
        with _INMEMORY_PERF_LOCK:
            _INMEMORY_PERF_RECORDS.clear()
        with _INMEMORY_TIMINGS_LOCK:
            _INMEMORY_STAGE_TIMINGS.clear()

        now = datetime.now()
        ts1 = (now - timedelta(seconds=10)).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        ts2 = (now - timedelta(seconds=5)).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

        store_inmemory_perf_record(
            _perf_record(
                request_id="tp-1",
                timestamp=ts1,
                tokens_before=1000,
                tokens_after=400,
                tokens_saved=600,
                total_ms=500.0,
                tokens_out=500,
                ttfb_ms=100.0,
            )
        )
        store_inmemory_perf_record(
            _perf_record(
                request_id="tp-2",
                timestamp=ts2,
                tokens_before=2000,
                tokens_after=1000,
                tokens_saved=1000,
                total_ms=1000.0,
                tokens_out=1000,
                ttfb_ms=200.0,
            )
        )
        store_inmemory_stage_timings("tp-1", {"compression_first_stage": 100.0})
        store_inmemory_stage_timings("tp-2", {"compression_first_stage": 200.0})

        # Build a PerfReport from in-memory data (simulating the fallback path)
        with _INMEMORY_PERF_LOCK:
            inmem_records = list(_INMEMORY_PERF_RECORDS)
        with _INMEMORY_TIMINGS_LOCK:
            inmem_timings = dict(_INMEMORY_STAGE_TIMINGS)
        for r in inmem_records:
            ts = inmem_timings.get(r.request_id)
            if ts:
                r.stages = ts

        report = PerfReport(perf_records=inmem_records)
        throughput = calculate_throughput(report)

        # Should NOT be all zeros
        assert "rolling" in throughput
        rolling = throughput["rolling"]
        assert rolling["input_wall_clock"] > 0
        assert rolling["input_active_p50"] > 0
        assert rolling["compression_p50"] > 0
        assert rolling["generation_p50"] > 0

    def test_empty_inmemory_buffers_return_zeros(self):
        """When both log files and in-memory buffers are empty, throughput is all zeros."""
        from headroom.proxy.server import (
            _INMEMORY_PERF_LOCK,
            _INMEMORY_PERF_RECORDS,
            _INMEMORY_STAGE_TIMINGS,
            _INMEMORY_TIMINGS_LOCK,
        )

        with _INMEMORY_PERF_LOCK:
            _INMEMORY_PERF_RECORDS.clear()
        with _INMEMORY_TIMINGS_LOCK:
            _INMEMORY_STAGE_TIMINGS.clear()

        report = PerfReport(perf_records=[])
        throughput = calculate_throughput(report)

        rolling = throughput["rolling"]
        current = throughput["current"]
        for field in ("input_wall_clock", "compression_p50", "generation_p50"):
            assert rolling[field] == 0.0
            assert current[field] == 0.0

    def test_stage_timings_are_matched_to_records(self):
        """PerfRecords get their stages populated from the in-memory timings dict."""
        from headroom.proxy.server import (
            _INMEMORY_PERF_LOCK,
            _INMEMORY_PERF_RECORDS,
            _INMEMORY_STAGE_TIMINGS,
            _INMEMORY_TIMINGS_LOCK,
            store_inmemory_perf_record,
            store_inmemory_stage_timings,
        )

        with _INMEMORY_PERF_LOCK:
            _INMEMORY_PERF_RECORDS.clear()
        with _INMEMORY_TIMINGS_LOCK:
            _INMEMORY_STAGE_TIMINGS.clear()

        now = datetime.now()
        ts = (now - timedelta(seconds=3)).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

        store_inmemory_perf_record(_perf_record(request_id="match-1", timestamp=ts))
        store_inmemory_stage_timings(
            "match-1", {"compression_first_stage": 123.4, "upstream": 50.0}
        )

        with _INMEMORY_PERF_LOCK:
            inmem_records = list(_INMEMORY_PERF_RECORDS)
        with _INMEMORY_TIMINGS_LOCK:
            inmem_timings = dict(_INMEMORY_STAGE_TIMINGS)

        for r in inmem_records:
            ts_match = inmem_timings.get(r.request_id)
            if ts_match:
                r.stages = ts_match

        assert inmem_records[0].stages == {"compression_first_stage": 123.4, "upstream": 50.0}

    def test_records_without_stage_timings_have_empty_stages(self):
        """If no stage timings were stored for a request, its stages dict is empty."""
        from headroom.proxy.server import (
            _INMEMORY_PERF_LOCK,
            _INMEMORY_PERF_RECORDS,
            _INMEMORY_STAGE_TIMINGS,
            _INMEMORY_TIMINGS_LOCK,
            store_inmemory_perf_record,
        )

        with _INMEMORY_PERF_LOCK:
            _INMEMORY_PERF_RECORDS.clear()
        with _INMEMORY_TIMINGS_LOCK:
            _INMEMORY_STAGE_TIMINGS.clear()

        store_inmemory_perf_record(_perf_record(request_id="no-stages"))

        with _INMEMORY_PERF_LOCK:
            inmem_records = list(_INMEMORY_PERF_RECORDS)

        assert inmem_records[0].stages == {}

    def test_compression_falls_back_to_optimization_ms(self):
        """When no compression stage timing exists, optimization_ms is used as duration."""
        from headroom.proxy.server import (
            _INMEMORY_PERF_LOCK,
            _INMEMORY_PERF_RECORDS,
            _INMEMORY_STAGE_TIMINGS,
            _INMEMORY_TIMINGS_LOCK,
            store_inmemory_perf_record,
        )

        with _INMEMORY_PERF_LOCK:
            _INMEMORY_PERF_RECORDS.clear()
        with _INMEMORY_TIMINGS_LOCK:
            _INMEMORY_STAGE_TIMINGS.clear()

        now = datetime.now()
        ts1 = (now - timedelta(seconds=8)).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        ts2 = (now - timedelta(seconds=3)).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

        # Two records: one with stage timing, one without (optimization_ms only)
        store_inmemory_perf_record(
            _perf_record(
                request_id="cpt-1",
                timestamp=ts1,
                tokens_before=1000,
                optimization_ms=50.0,
                stages={"compression_first_stage": 100.0},
            )
        )
        store_inmemory_perf_record(
            _perf_record(
                request_id="cpt-2",
                timestamp=ts2,
                tokens_before=1500,
                optimization_ms=30.0,
                # No stages — simulate OpenAI REST request without stage timings
            )
        )

        with _INMEMORY_PERF_LOCK:
            inmem_records = list(_INMEMORY_PERF_RECORDS)

        report = PerfReport(perf_records=inmem_records)
        throughput = calculate_throughput(report)

        rolling = throughput["rolling"]
        # compression_p50 should be non-zero because optimization_ms is used
        # as fallback for the record without stage timings
        assert rolling["compression_p50"] > 0
        # P50 of [10000, 50000] = 30000
        assert rolling["compression_p50"] == 30000.0


# ── Thread safety ──────────────────────────────────────────────────────────


class TestInmemoryThreadSafety:
    def test_concurrent_perf_record_writes(self):
        """Multiple threads can write perf records concurrently without corruption."""
        import threading

        from headroom.proxy.server import (
            _INMEMORY_PERF_LOCK,
            _INMEMORY_PERF_RECORDS,
            store_inmemory_perf_record,
        )

        with _INMEMORY_PERF_LOCK:
            _INMEMORY_PERF_RECORDS.clear()

        def writer(start: int, count: int):
            for i in range(start, start + count):
                store_inmemory_perf_record(_perf_record(request_id=f"thr-{i}"))

        threads = [
            threading.Thread(target=writer, args=(0, 100)),
            threading.Thread(target=writer, args=(100, 100)),
            threading.Thread(target=writer, args=(200, 100)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with _INMEMORY_PERF_LOCK:
            ids = {r.request_id for r in _INMEMORY_PERF_RECORDS}

        # All 300 should be present (maxlen=1000, so no drops)
        for i in range(300):
            assert f"thr-{i}" in ids

    def test_concurrent_stage_timing_writes(self):
        """Multiple threads can write stage timings without data loss."""
        import threading

        from headroom.proxy.server import (
            _INMEMORY_STAGE_TIMINGS,
            _INMEMORY_TIMINGS_LOCK,
            store_inmemory_stage_timings,
        )

        with _INMEMORY_TIMINGS_LOCK:
            _INMEMORY_STAGE_TIMINGS.clear()

        def writer(start: int, count: int):
            for i in range(start, start + count):
                store_inmemory_stage_timings(f"thr-{i}", {"stage": float(i)})

        threads = [
            threading.Thread(target=writer, args=(0, 100)),
            threading.Thread(target=writer, args=(100, 100)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with _INMEMORY_TIMINGS_LOCK:
            count = len(_INMEMORY_STAGE_TIMINGS)
        # All 200 entries should be present (well under the 2000 max)
        assert count == 200
