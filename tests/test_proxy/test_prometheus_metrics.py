"""Tests for headroom.proxy.prometheus_metrics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from headroom.proxy.prometheus_metrics import (
    PrometheusMetrics,
    _append_metric,
    _escape_label_value,
    _format_labels,
)

# ── Helper functions ───────────────────────────────────────────────────


class TestEscapeLabelValue:
    def test_backslash(self) -> None:
        assert _escape_label_value(r"a\b") == r"a\\b"

    def test_newline(self) -> None:
        assert _escape_label_value("a\nb") == "a\\nb"

    def test_double_quote(self) -> None:
        assert _escape_label_value('a"b') == 'a\\"b'

    def test_all_three(self) -> None:
        assert _escape_label_value('a\n\\"b') == 'a\\n\\\\\\"b'

    def test_no_escaping_needed(self) -> None:
        assert _escape_label_value("hello") == "hello"

    def test_empty_string(self) -> None:
        assert _escape_label_value("") == ""


class TestFormatLabels:
    def test_none(self) -> None:
        assert _format_labels(None) == ""

    def test_empty_dict(self) -> None:
        assert _format_labels({}) == ""

    def test_single_label(self) -> None:
        result = _format_labels({"key": "val"})
        assert result == '{key="val"}'

    def test_multiple_labels_sorted(self) -> None:
        result = _format_labels({"b": "2", "a": "1"})
        assert result == '{a="1",b="2"}'

    def test_escapes_value(self) -> None:
        result = _format_labels({"k": 'a"b'})
        assert result == '{k="a\\"b"}'


class TestAppendMetric:
    def test_basic(self) -> None:
        lines: list[str] = []
        _append_metric(
            lines, name="foo_total", metric_type="counter", help_text="Foo counter", value=42
        )
        assert lines == [
            "# HELP foo_total Foo counter",
            "# TYPE foo_total counter",
            "foo_total 42",
            "",
        ]

    def test_with_labels(self) -> None:
        lines: list[str] = []
        _append_metric(
            lines,
            name="foo_total",
            metric_type="counter",
            help_text="Foo",
            value=1,
            labels={"env": "prod"},
        )
        assert lines == [
            "# HELP foo_total Foo",
            "# TYPE foo_total counter",
            'foo_total{env="prod"} 1',
            "",
        ]

    def test_float_value(self) -> None:
        lines: list[str] = []
        _append_metric(lines, name="bar", metric_type="gauge", help_text="Bar", value=3.14)
        assert "bar 3.14" in lines[2]


# ── PrometheusMetrics construction ──────────────────────────────────────


class TestPrometheusMetricsConstruction:
    def test_default_construction(self) -> None:
        m = PrometheusMetrics(stateless=True)
        assert m.requests_total == 0
        assert m.tokens_input_total == 0
        assert m.tokens_output_total == 0
        assert m.tokens_saved_total == 0
        assert m.latency_count == 0
        assert m.active_ws_sessions == 0
        assert m.active_relay_tasks == 0
        assert m._stateless is True
        assert m.savings_tracker is not None
        assert m.cost_tracker is None

    def test_with_savings_and_cost_tracker(self) -> None:
        savings_tracker = MagicMock()
        savings_tracker.snapshot.return_value = {"lifetime": {}}
        cost_tracker = MagicMock()
        m = PrometheusMetrics(
            savings_tracker=savings_tracker, cost_tracker=cost_tracker, stateless=True
        )
        assert m.savings_tracker is savings_tracker
        assert m.cost_tracker is cost_tracker

    def test_stateless_false_creates_default_tracker(self) -> None:
        m = PrometheusMetrics(stateless=False)
        assert m._stateless is False
        assert m.savings_tracker is not None


# ── Stack recording ────────────────────────────────────────────────────


class TestRecordStack:
    def test_none_stack_is_noop(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_stack(None)
        assert len(m.requests_by_stack) == 0

    def test_valid_stack(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_stack("adapter_ts_openai")
        assert m.requests_by_stack.get("adapter_ts_openai") == 1

    def test_increments_existing_stack(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_stack("my_stack")
        m.record_stack("my_stack")
        assert m.requests_by_stack["my_stack"] == 2

    def test_cardinality_cap(self) -> None:
        from headroom.telemetry.context import MAX_DISTINCT_STACKS

        m = PrometheusMetrics(stateless=True)
        for i in range(MAX_DISTINCT_STACKS):
            m.record_stack(f"stack_{i}")
        assert len(m.requests_by_stack) == MAX_DISTINCT_STACKS
        m.record_stack("stack_one_more")
        assert len(m.requests_by_stack) == MAX_DISTINCT_STACKS
        assert "stack_one_more" not in m.requests_by_stack


# ── Compression recording ──────────────────────────────────────────────


class TestRecordCompression:
    def test_increments_strategy_counter(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_compression("smart_crusher", original_tokens=100, compressed_tokens=60)
        assert m.compressions_by_strategy["smart_crusher"] == 1
        assert m.tokens_saved_by_strategy["smart_crusher"] == 40

    def test_multiple_strategies(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_compression("router", original_tokens=200, compressed_tokens=100)
        m.record_compression("smart_crusher", original_tokens=50, compressed_tokens=25)
        assert m.compressions_by_strategy["router"] == 1
        assert m.compressions_by_strategy["smart_crusher"] == 1
        assert m.tokens_saved_by_strategy["router"] == 100
        assert m.tokens_saved_by_strategy["smart_crusher"] == 25

    def test_negative_savings_clamped_to_zero(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_compression("test", original_tokens=10, compressed_tokens=20)
        assert m.tokens_saved_by_strategy["test"] == 0

    def test_zero_savings_noop(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_compression("test", original_tokens=10, compressed_tokens=10)
        assert m.tokens_saved_by_strategy.get("test", 0) == 0


# ── Router route counts ────────────────────────────────────────────────


class TestRecordRouterRouteCounts:
    def test_accumulates_counts(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_router_route_counts({"user_msg": 12, "recent_code": 4})
        assert m.router_route_counts["user_msg"] == 12
        assert m.router_route_counts["recent_code"] == 4

    def test_increments_existing(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_router_route_counts({"user_msg": 5})
        m.record_router_route_counts({"user_msg": 3})
        assert m.router_route_counts["user_msg"] == 8

    def test_ignores_zero_counts(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_router_route_counts({"user_msg": 0})
        assert "user_msg" not in m.router_route_counts


# ── Codex WS unit recording ────────────────────────────────────────────


class TestRecordCodexWsUnit:
    def test_records_unit(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_codex_ws_unit(
            strategy="noop",
            reason_category="small",
            elapsed_ms=5.0,
            text_bytes=100,
            tokens_before=50,
            tokens_after=40,
            tokens_saved=10,
            modified=False,
        )
        assert m.codex_ws_units_total == 1
        assert m.codex_ws_unit_elapsed_ms_sum == 5.0
        assert m.codex_ws_unit_bytes_sum == 100
        assert m.codex_ws_unit_tokens_before_sum == 50
        assert m.codex_ws_unit_tokens_after_sum == 40
        assert m.codex_ws_unit_tokens_saved_sum == 10

    def test_modified_unit(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_codex_ws_unit(
            strategy="test",
            reason_category="reason",
            elapsed_ms=1.0,
            text_bytes=10,
            tokens_before=10,
            tokens_after=5,
            tokens_saved=5,
            modified=True,
        )
        assert m.codex_ws_units_modified_total == 1

    def test_kompress_unit(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_codex_ws_unit(
            strategy="kompress",
            reason_category="reason",
            elapsed_ms=1.0,
            text_bytes=10,
            tokens_before=10,
            tokens_after=5,
            tokens_saved=5,
            modified=False,
        )
        assert m.codex_ws_units_to_kompress_total == 1
        assert m.codex_ws_units_kompress_attempted_total == 1

    def test_strategy_chain_kompress(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_codex_ws_unit(
            strategy="noop",
            reason_category="reason",
            elapsed_ms=1.0,
            text_bytes=10,
            tokens_before=10,
            tokens_after=5,
            tokens_saved=5,
            modified=False,
            strategy_chain=["kompress"],
        )
        assert m.codex_ws_units_kompress_attempted_total == 1
        assert m.codex_ws_units_to_kompress_total == 0

    def test_categories_and_content_type(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_codex_ws_unit(
            strategy="test",
            reason_category="small",
            elapsed_ms=1.0,
            text_bytes=10,
            tokens_before=10,
            tokens_after=5,
            tokens_saved=5,
            modified=False,
            content_type="text",
            text_shape="short",
        )
        assert m.codex_ws_units_by_category["small"] == 1
        assert m.codex_ws_units_by_content_type["text"] == 1
        assert m.codex_ws_units_by_text_shape["short"] == 1


# ── Codex WS frame recording ───────────────────────────────────────────


class TestRecordCodexWsFrame:
    def test_records_frame(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_codex_ws_frame(
            elapsed_ms=10.0,
            bytes_before=1000,
            bytes_after=600,
            attempted_tokens=200,
            tokens_saved=80,
            modified=True,
        )
        assert m.codex_ws_frames_attempted_total == 1
        assert m.codex_ws_frames_compressed_total == 1
        assert m.codex_ws_frames_failed_total == 0
        assert m.codex_ws_frame_elapsed_ms_sum == 10.0
        assert m.codex_ws_frame_bytes_before_sum == 1000
        assert m.codex_ws_frame_bytes_after_sum == 600
        assert m.codex_ws_frame_attempted_tokens_sum == 200
        assert m.codex_ws_frame_tokens_saved_sum == 80

    def test_failed_frame(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_codex_ws_frame(
            elapsed_ms=5.0,
            bytes_before=500,
            failed=True,
        )
        assert m.codex_ws_frames_attempted_total == 1
        assert m.codex_ws_frames_failed_total == 1
        assert m.codex_ws_frames_compressed_total == 0

    def test_kompress_in_final_strategies(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_codex_ws_frame(
            elapsed_ms=1.0,
            bytes_before=100,
            modified=True,
            final_strategies=["kompress"],
        )
        assert m.codex_ws_frames_to_kompress_total == 1
        assert m.codex_ws_frames_kompress_attempted_total == 1

    def test_kompress_in_strategy_chain(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_codex_ws_frame(
            elapsed_ms=1.0,
            bytes_before=100,
            strategy_chain=["kompress"],
        )
        assert m.codex_ws_frames_kompress_attempted_total == 1
        assert m.codex_ws_frames_to_kompress_total == 0


# ── Inbound request tracking ───────────────────────────────────────────


class TestInboundTracking:
    def test_record_request_increments(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_inbound_request(method="POST", path="/v1/chat/completions")
        assert m.inbound_requests_total == 1
        assert m.inbound_requests_active == 1
        assert m.inbound_requests_by_method["POST"] == 1
        assert m.inbound_requests_by_path["/v1/chat/completions"] == 1

    def test_record_response_decrements_active(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_inbound_request(method="GET", path="/health")
        m.record_inbound_response(status_code=200)
        assert m.inbound_requests_total == 1
        assert m.inbound_requests_completed == 1
        assert m.inbound_requests_active == 0
        assert m.inbound_responses_by_status["200"] == 1

    def test_record_aborted(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_inbound_request(method="POST", path="/test")
        m.record_inbound_aborted(reason="client_disconnect")
        assert m.inbound_requests_completed == 1
        assert m.inbound_requests_active == 0
        assert m.inbound_responses_by_status["aborted:client_disconnect"] == 1

    def test_active_never_below_zero_on_response(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_inbound_response(status_code=200)
        assert m.inbound_requests_active == 0

    def test_active_never_below_zero_on_aborted(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_inbound_aborted(reason="timeout")
        assert m.inbound_requests_active == 0

    def test_inbound_snapshot(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_inbound_request(method="POST", path="/v1/chat/completions")
        m.record_inbound_response(status_code=200)
        snap = m.inbound_snapshot()
        assert snap["total"] == 1
        assert snap["completed"] == 1
        assert snap["active"] == 0
        assert snap["by_method"] == {"POST": 1}
        assert snap["by_path"] == {"/v1/chat/completions": 1}
        assert snap["by_status"] == {"200": 1}


# ── WS session tracking ────────────────────────────────────────────────


class TestWsSessionTracking:
    def test_inc_dec_active_sessions(self) -> None:
        m = PrometheusMetrics(stateless=True)
        assert m.active_ws_sessions == 0
        m.inc_active_ws_sessions()
        assert m.active_ws_sessions == 1
        m.inc_active_ws_sessions()
        assert m.active_ws_sessions == 2
        m.dec_active_ws_sessions()
        assert m.active_ws_sessions == 1
        m.dec_active_ws_sessions()
        assert m.active_ws_sessions == 0

    def test_dec_active_sessions_never_below_zero(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.dec_active_ws_sessions()
        assert m.active_ws_sessions == 0
        m.dec_active_ws_sessions()
        assert m.active_ws_sessions == 0

    def test_inc_dec_relay_tasks(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.inc_active_relay_tasks()
        assert m.active_relay_tasks == 1
        m.inc_active_relay_tasks(3)
        assert m.active_relay_tasks == 4
        m.dec_active_relay_tasks(2)
        assert m.active_relay_tasks == 2

    def test_dec_relay_tasks_never_below_zero(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.dec_active_relay_tasks(5)
        assert m.active_relay_tasks == 0

    def test_record_ws_session_duration(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_ws_session_duration(duration_ms=1000.0, cause="client_disconnect")
        assert m.ws_session_duration_sum_ms["client_disconnect"] == 1000.0
        assert m.ws_session_duration_count["client_disconnect"] == 1
        assert m.ws_session_duration_max_ms["client_disconnect"] == 1000.0

    def test_ws_session_duration_tracks_max(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_ws_session_duration(duration_ms=500.0, cause="normal")
        m.record_ws_session_duration(duration_ms=1500.0, cause="normal")
        assert m.ws_session_duration_max_ms["normal"] == 1500.0

    def test_ws_session_duration_bogus_value_noop(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_ws_session_duration(duration_ms="not_a_number", cause="error")  # type: ignore[arg-type]
        assert len(m.ws_session_duration_sum_ms) == 0


# ── Export ──────────────────────────────────────────────────────────────


class TestExport:
    async def _export(self, m: PrometheusMetrics) -> str:
        return await m.export()

    def test_export_contains_help_and_type(self) -> None:
        m = PrometheusMetrics(stateless=True)
        text = asyncio_run(self._export(m))
        assert "# HELP headroom_requests_total" in text
        assert "# TYPE headroom_requests_total counter" in text
        assert "headroom_requests_total 0" in text

    def test_export_reflects_recorded_metrics(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_codex_ws_unit(
            strategy="test",
            reason_category="cat",
            elapsed_ms=5.0,
            text_bytes=100,
            tokens_before=50,
            tokens_after=40,
            tokens_saved=10,
            modified=True,
        )
        text = asyncio_run(self._export(m))
        assert "headroom_active_ws_sessions 0" in text
        assert "headroom_active_relay_tasks 0" in text

    def test_export_contains_inbound_metrics(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_inbound_request(method="POST", path="/test")
        m.record_inbound_response(status_code=200)
        text = asyncio_run(self._export(m))
        assert "headroom_inbound_requests_total 1" in text
        assert "headroom_inbound_requests_completed_total 1" in text
        assert "headroom_inbound_requests_active 0" in text

    def test_export_includes_active_ws_sessions(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.inc_active_ws_sessions()
        m.inc_active_ws_sessions()
        m.inc_active_relay_tasks(3)
        text = asyncio_run(self._export(m))
        assert "headroom_active_ws_sessions 2" in text
        assert "headroom_active_relay_tasks 3" in text

    def test_export_ws_session_duration_histogram(self) -> None:
        m = PrometheusMetrics(stateless=True)
        m.record_ws_session_duration(duration_ms=2000.0, cause="normal")
        text = asyncio_run(self._export(m))
        assert 'headroom_ws_session_duration_ms_sum{cause="normal"} 2000.0' in text
        assert 'headroom_ws_session_duration_ms_count{cause="normal"} 1' in text
        assert 'headroom_ws_session_duration_ms_max{cause="normal"} 2000.0' in text

    def test_export_is_valid_prometheus_format(self) -> None:
        m = PrometheusMetrics(stateless=True)
        text = asyncio_run(self._export(m))
        lines = text.split("\n")
        for line in lines:
            if not line or line.startswith("#"):
                continue
            # Every non-comment line must be a metric line: name[labels] value
            assert " " in line, f"Invalid metric line: {line}"

    def test_export_with_cache_miss_attribution(self) -> None:
        m = PrometheusMetrics(stateless=True)
        asyncio_run(m.record_cache_miss_attribution(provider="anthropic", reason="ttl_expiry"))
        asyncio_run(m.record_cache_miss_attribution(provider="anthropic", reason="prefix_change"))
        asyncio_run(m.record_cache_miss_attribution(provider="openai", reason="ttl_expiry"))
        text = asyncio_run(self._export(m))
        assert (
            'headroom_cache_miss_attribution_total{provider="anthropic",reason="ttl_expiry"} 1'
            in text
        )
        assert (
            'headroom_cache_miss_attribution_total{provider="anthropic",reason="prefix_change"} 1'
            in text
        )
        assert (
            'headroom_cache_miss_attribution_total{provider="openai",reason="ttl_expiry"} 1' in text
        )

    def test_export_with_cache_by_provider(self) -> None:
        metrics = PrometheusMetrics(stateless=True)
        # Feed a request with cache tokens to populate cache_by_provider
        metrics.savings_tracker = MagicMock()
        metrics.savings_tracker.snapshot.return_value = {
            "lifetime": {
                "requests": 1,
                "tokens_saved": 300,
                "total_input_tokens": 1000,
                "total_input_cost_usd": 0.02,
                "compression_savings_usd": 0.01,
            }
        }
        metrics.cost_tracker = MagicMock()
        metrics.cost_tracker.stats.return_value = {}

        with patch("headroom.proxy.prometheus_metrics.savings_ledger"):
            with patch("headroom.proxy.prometheus_metrics.get_otel_metrics") as mock_otel:

                async def _run() -> str:
                    await metrics.record_request(
                        provider="anthropic",
                        model="claude-3",
                        input_tokens=1000,
                        output_tokens=200,
                        tokens_saved=300,
                        latency_ms=500.0,
                        cache_read_tokens=100,
                        cache_write_tokens=50,
                        cache_write_5m_tokens=30,
                        cache_write_1h_tokens=20,
                        uncached_input_tokens=850,
                    )
                    return await metrics.export()

                mock_otel_instance = MagicMock()
                mock_otel_instance.record_proxy_request = MagicMock()
                mock_otel.return_value = mock_otel_instance

                text = asyncio_run(_run())

        assert 'headroom_cache_read_tokens_total{provider="anthropic"} 100' in text
        assert 'headroom_cache_write_tokens_total{provider="anthropic"} 50' in text
        assert 'headroom_provider_cache_requests_total{provider="anthropic"} 1' in text
        assert 'headroom_provider_cache_hit_requests_total{provider="anthropic"} 1' in text


# ── record_request ─────────────────────────────────────────────────────


class TestRecordRequest:
    @pytest.fixture
    def metrics(self) -> PrometheusMetrics:
        m = PrometheusMetrics(stateless=True)
        m.savings_tracker = MagicMock()
        m.savings_tracker.snapshot.return_value = {"lifetime": {}}
        m.cost_tracker = MagicMock()
        m.cost_tracker.stats.return_value = {}
        return m

    def test_records_basic_request(self, metrics: PrometheusMetrics) -> None:
        with patch("headroom.proxy.prometheus_metrics.get_otel_metrics") as mock_otel:
            mock_otel_instance = MagicMock()
            mock_otel_instance.record_proxy_request = MagicMock()
            mock_otel.return_value = mock_otel_instance

            asyncio_run(
                metrics.record_request(
                    provider="openai",
                    model="gpt-4",
                    input_tokens=100,
                    output_tokens=50,
                    tokens_saved=30,
                    latency_ms=200.0,
                )
            )

        assert metrics.requests_total == 1
        assert metrics.requests_by_provider["openai"] == 1
        assert metrics.requests_by_model["gpt-4"] == 1
        assert metrics.tokens_input_total == 100
        assert metrics.tokens_output_total == 50
        assert metrics.tokens_saved_total == 30
        assert metrics.latency_sum_ms == 200.0
        assert metrics.latency_count == 1

    def test_cached_request(self, metrics: PrometheusMetrics) -> None:
        with patch("headroom.proxy.prometheus_metrics.get_otel_metrics") as mock_otel:
            mock_otel_instance = MagicMock()
            mock_otel_instance.record_proxy_request = MagicMock()
            mock_otel.return_value = mock_otel_instance

            asyncio_run(
                metrics.record_request(
                    provider="anthropic",
                    model="claude-3",
                    input_tokens=100,
                    output_tokens=50,
                    tokens_saved=0,
                    latency_ms=100.0,
                    cached=True,
                )
            )

        assert metrics.requests_cached == 1

    def test_with_overhead_and_ttfb(self, metrics: PrometheusMetrics) -> None:
        with patch("headroom.proxy.prometheus_metrics.get_otel_metrics") as mock_otel:
            mock_otel_instance = MagicMock()
            mock_otel_instance.record_proxy_request = MagicMock()
            mock_otel.return_value = mock_otel_instance

            asyncio_run(
                metrics.record_request(
                    provider="openai",
                    model="gpt-4",
                    input_tokens=100,
                    output_tokens=50,
                    tokens_saved=10,
                    latency_ms=500.0,
                    overhead_ms=50.0,
                    ttfb_ms=300.0,
                )
            )

        assert metrics.overhead_sum_ms == 50.0
        assert metrics.overhead_count == 1
        assert metrics.ttfb_sum_ms == 300.0
        assert metrics.ttfb_count == 1

    def test_with_pipeline_timing(self, metrics: PrometheusMetrics) -> None:
        with patch("headroom.proxy.prometheus_metrics.get_otel_metrics") as mock_otel:
            mock_otel_instance = MagicMock()
            mock_otel_instance.record_proxy_request = MagicMock()
            mock_otel.return_value = mock_otel_instance

            asyncio_run(
                metrics.record_request(
                    provider="openai",
                    model="gpt-4",
                    input_tokens=100,
                    output_tokens=50,
                    tokens_saved=10,
                    latency_ms=500.0,
                    pipeline_timing={"router": 10.0, "compress": 40.0},
                )
            )

        assert metrics.transform_timing_sum["router"] == 10.0
        assert metrics.transform_timing_sum["compress"] == 40.0
        assert metrics.transform_timing_count["compress"] == 1
        assert metrics.transform_timing_max["compress"] == 40.0

    def test_with_waste_signals(self, metrics: PrometheusMetrics) -> None:
        with patch("headroom.proxy.prometheus_metrics.get_otel_metrics") as mock_otel:
            mock_otel_instance = MagicMock()
            mock_otel_instance.record_proxy_request = MagicMock()
            mock_otel.return_value = mock_otel_instance

            asyncio_run(
                metrics.record_request(
                    provider="openai",
                    model="gpt-4",
                    input_tokens=100,
                    output_tokens=50,
                    tokens_saved=10,
                    latency_ms=500.0,
                    waste_signals={"repeat": 50, "boilerplate": 30},
                )
            )

        assert metrics.waste_signals_total["repeat"] == 50
        assert metrics.waste_signals_total["boilerplate"] == 30

    def test_savings_history(self, metrics: PrometheusMetrics) -> None:
        with patch("headroom.proxy.prometheus_metrics.get_otel_metrics") as mock_otel:
            mock_otel_instance = MagicMock()
            mock_otel_instance.record_proxy_request = MagicMock()
            mock_otel.return_value = mock_otel_instance

            for i in range(3):
                asyncio_run(
                    metrics.record_request(
                        provider="openai",
                        model="gpt-4",
                        input_tokens=100,
                        output_tokens=50,
                        tokens_saved=10 * (i + 1),
                        latency_ms=100.0,
                    )
                )

        assert len(metrics.savings_history) == 3
        assert metrics.savings_history[-1][1] == 60  # 10+20+30

    def test_attempted_input_tokens(self, metrics: PrometheusMetrics) -> None:
        with patch("headroom.proxy.prometheus_metrics.get_otel_metrics") as mock_otel:
            mock_otel_instance = MagicMock()
            mock_otel_instance.record_proxy_request = MagicMock()
            mock_otel.return_value = mock_otel_instance

            asyncio_run(
                metrics.record_request(
                    provider="openai",
                    model="gpt-4",
                    input_tokens=100,
                    output_tokens=50,
                    tokens_saved=10,
                    latency_ms=100.0,
                    attempted_input_tokens=80,
                )
            )

        assert metrics.attempted_input_tokens_total == 80


# ── Async helpers ──────────────────────────────────────────────────────


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)
