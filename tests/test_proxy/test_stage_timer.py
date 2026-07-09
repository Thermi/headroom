from __future__ import annotations

import asyncio
import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from headroom.proxy.stage_timer import (
    StageMeasurement,
    StageTimer,
    emit_stage_timings_log,
)


class TestStageTimer:
    def test_empty_summary_on_creation(self) -> None:
        timer = StageTimer()
        assert timer.summary() == {}

    def test_measure_sync_records_duration(self) -> None:
        timer = StageTimer()
        with timer.measure("foo"):
            time.sleep(0.01)
        summary = timer.summary()
        assert "foo" in summary
        assert summary["foo"] > 0

    async def test_measure_async_records_duration(self) -> None:
        timer = StageTimer()
        async with timer.measure("bar"):
            await asyncio.sleep(0.01)
        summary = timer.summary()
        assert "bar" in summary
        assert summary["bar"] > 0

    def test_measure_records_on_body_raise(self) -> None:
        timer = StageTimer()
        with pytest.raises(RuntimeError):
            with timer.measure("baz"):
                time.sleep(0.005)
                raise RuntimeError("boom")
        summary = timer.summary()
        assert "baz" in summary
        assert summary["baz"] > 0

    def test_record_stores_value(self) -> None:
        timer = StageTimer()
        timer.record("x", 42.5)
        assert timer.summary() == {"x": 42.5}

    def test_record_replaces_existing_value(self) -> None:
        timer = StageTimer()
        timer.record("x", 10.0)
        timer.record("x", 99.9)
        assert timer.summary() == {"x": 99.9}

    def test_elapsed_ms_returns_positive(self) -> None:
        timer = StageTimer()
        time.sleep(0.005)
        assert timer.elapsed_ms() > 0

    def test_summary_returns_snapshot(self) -> None:
        timer = StageTimer()
        timer.record("a", 1.0)
        snap = timer.summary()
        timer.record("a", 2.0)
        assert snap == {"a": 1.0}
        assert timer.summary() == {"a": 2.0}

    def test_contains(self) -> None:
        timer = StageTimer()
        timer.record("present", 1.0)
        assert "present" in timer
        assert "absent" not in timer


class TestStageMeasurement:
    def test_finalize_with_none_start_does_not_record(self) -> None:
        timer = StageTimer()
        m = StageMeasurement(timer, "x")
        assert m._start is None
        m._finalize()
        assert "x" not in timer


class TestEmitStageTimingsLog:
    def test_pads_expected_stages(self, caplog) -> None:
        timer = StageTimer()
        timer.record("ran", 10.0)
        caplog.set_level(logging.INFO)

        with patch("headroom.proxy.server.store_inmemory_stage_timings"):
            asyncio.run(
                emit_stage_timings_log(
                    path="/test",
                    request_id="r1",
                    session_id="s1",
                    stage_timer=timer,
                    expected_stages=["ran", "not_ran"],
                )
            )

        assert "STAGE_TIMINGS" in caplog.text
        assert "ran" in caplog.text
        assert "not_ran" in caplog.text
        assert "r1" in caplog.text

    def test_includes_extra_stages(self, caplog) -> None:
        timer = StageTimer()
        timer.record("unexpected", 5.0)
        caplog.set_level(logging.INFO)

        with patch("headroom.proxy.server.store_inmemory_stage_timings"):
            asyncio.run(
                emit_stage_timings_log(
                    path="/test",
                    request_id="r2",
                    session_id="s2",
                    stage_timer=timer,
                    expected_stages=[],
                )
            )

        assert "unexpected" in caplog.text

    def test_emits_correct_json_structure(self, caplog) -> None:
        timer = StageTimer()
        timer.record("a", 1.0)
        caplog.set_level(logging.INFO)

        with patch("headroom.proxy.server.store_inmemory_stage_timings"):
            asyncio.run(
                emit_stage_timings_log(
                    path="/p",
                    request_id="r3",
                    session_id="s3",
                    stage_timer=timer,
                    expected_stages=["a"],
                )
            )

        assert "stage_timings" in caplog.text
        assert "/p" in caplog.text
        assert "r3" in caplog.text
        assert "s3" in caplog.text

    def test_json_serialization_failure_falls_back(self, caplog) -> None:
        timer = StageTimer()
        timer.record("bad", 42.0)
        caplog.set_level(logging.INFO)

        with patch("headroom.proxy.server.store_inmemory_stage_timings"):
            with patch(
                "headroom.proxy.stage_timer.json.dumps",
                side_effect=TypeError("no way"),
            ):
                asyncio.run(
                    emit_stage_timings_log(
                        path="/fail",
                        request_id="r4",
                        session_id="s4",
                        stage_timer=timer,
                        expected_stages=["bad"],
                    )
                )

        assert "STAGE_TIMINGS" in caplog.text

    @patch("headroom.proxy.server.store_inmemory_stage_timings")
    def test_metrics_record_called(self, mock_store) -> None:
        timer = StageTimer()
        timer.record("a", 1.0)
        metrics = MagicMock()
        metrics.record_stage_timings = AsyncMock()

        asyncio.run(
            emit_stage_timings_log(
                path="/p",
                request_id="r5",
                session_id="s5",
                stage_timer=timer,
                expected_stages=["a"],
                metrics=metrics,
            )
        )

        metrics.record_stage_timings.assert_awaited_once_with("/p", {"a": 1.0})

    @patch("headroom.proxy.server.store_inmemory_stage_timings")
    def test_metrics_raises_caught_at_debug(self, mock_store, caplog) -> None:
        timer = StageTimer()
        timer.record("a", 1.0)
        metrics = MagicMock()
        metrics.record_stage_timings = AsyncMock(side_effect=RuntimeError("metrics down"))
        caplog.set_level(logging.DEBUG)

        asyncio.run(
            emit_stage_timings_log(
                path="/p",
                request_id="r6",
                session_id="s6",
                stage_timer=timer,
                expected_stages=["a"],
                metrics=metrics,
            )
        )

        assert "record_stage_timings failed" in caplog.text
        assert "metrics down" in caplog.text
