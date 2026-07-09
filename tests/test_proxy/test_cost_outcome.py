from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from headroom.proxy.cost import (
    CostTracker,
    _aggregate_mcp_events,
    _summarize_transforms,
    build_prefix_cache_stats,
    build_session_summary,
    header_safe_transforms,
    merge_cost_stats,
)
from headroom.proxy.models import RequestLog
from headroom.proxy.outcome import (
    RequestOutcome,
    emit_request_outcome,
)
from headroom.proxy.warmup import WarmupRegistry, WarmupSlot

# ── WarmupSlot ────────────────────────────────────────────────────────


class TestWarmupSlot:
    def test_default_state(self) -> None:
        slot = WarmupSlot()
        assert slot.status == "null"
        assert slot.handle is None
        assert slot.error is None
        assert slot.info == {}

    def test_mark_loaded(self) -> None:
        slot = WarmupSlot()
        handle = object()
        slot.mark_loaded(handle=handle, language="python")
        assert slot.status == "loaded"
        assert slot.handle is handle
        assert slot.info["language"] == "python"

    def test_mark_loading(self) -> None:
        slot = WarmupSlot()
        slot.mark_loaded(handle=object())
        slot.mark_loading()
        assert slot.status == "loading"
        assert slot.handle is None

    def test_mark_null(self) -> None:
        slot = WarmupSlot()
        slot.mark_loaded(handle=object())
        slot.mark_null()
        assert slot.status == "null"
        assert slot.handle is None
        assert slot.error is None

    def test_mark_error(self) -> None:
        slot = WarmupSlot()
        slot.mark_error("something went wrong")
        assert slot.status == "error"
        assert slot.error == "something went wrong"
        assert slot.handle is None

    def test_to_dict_loaded(self) -> None:
        slot = WarmupSlot()
        slot.mark_loaded(handle=object(), lang="py")
        d = slot.to_dict()
        assert d["status"] == "loaded"
        assert d["info"]["lang"] == "py"
        assert "handle" not in d

    def test_to_dict_error(self) -> None:
        slot = WarmupSlot()
        slot.mark_error("fail")
        d = slot.to_dict()
        assert d["status"] == "error"
        assert d["error"] == "fail"

    def test_to_dict_null(self) -> None:
        slot = WarmupSlot()
        d = slot.to_dict()
        assert d == {"status": "null"}


# ── WarmupRegistry ────────────────────────────────────────────────────


class TestWarmupRegistry:
    def test_default_slots(self) -> None:
        r = WarmupRegistry()
        assert r.kompress.status == "null"
        assert r.magika.status == "null"
        assert r.code_aware.status == "null"
        assert r.tree_sitter.status == "null"
        assert r.smart_crusher.status == "null"
        assert r.memory_backend.status == "null"
        assert r.memory_embedder.status == "null"

    def test_to_dict_structure(self) -> None:
        r = WarmupRegistry()
        d = r.to_dict()
        assert set(d) == {
            "kompress",
            "magika",
            "code_aware",
            "tree_sitter",
            "smart_crusher",
            "memory_backend",
            "memory_embedder",
        }
        for v in d.values():
            assert "status" in v

    def test_merge_transform_status_promotes_to_loaded(self) -> None:
        r = WarmupRegistry()
        r.merge_transform_status({"kompress": "enabled"})
        assert r.kompress.status == "loaded"
        assert r.kompress.info.get("source_status") == "enabled"

    def test_merge_transform_status_ready(self) -> None:
        r = WarmupRegistry()
        r.merge_transform_status({"magika": "ready"})
        assert r.magika.status == "loaded"

    def test_merge_transform_status_loaded_prefix(self) -> None:
        r = WarmupRegistry()
        r.merge_transform_status({"tree_sitter": "loaded_with_extras"})
        assert r.tree_sitter.status == "loaded"

    def test_merge_transform_status_ignores_other_values(self) -> None:
        r = WarmupRegistry()
        r.merge_transform_status({"kompress": "disabled"})
        assert r.kompress.status == "null"
        assert r.kompress.info.get("source_status") == "disabled"

    def test_merge_transform_status_preserves_existing_loaded(self) -> None:
        r = WarmupRegistry()
        handle = object()
        r.kompress.mark_loaded(handle=handle)
        r.merge_transform_status({"kompress": "disabled"})
        assert r.kompress.status == "loaded"
        assert r.kompress.handle is handle

    def test_merge_transform_status_partial_update(self) -> None:
        r = WarmupRegistry()
        r.merge_transform_status({"code_aware": "ready", "smart_crusher": "absent"})
        assert r.code_aware.status == "loaded"
        assert r.smart_crusher.status == "null"

    def test_merge_transform_status_none_value_noop(self) -> None:
        r = WarmupRegistry()
        r.merge_transform_status({"kompress": None})
        assert r.kompress.status == "null"


# ── _summarize_transforms ─────────────────────────────────────────────


class TestSummarizeTransforms:
    def test_empty_list(self) -> None:
        assert _summarize_transforms([]) == "none"

    def test_single(self) -> None:
        assert _summarize_transforms(["router:excluded:tool"]) == "router:excluded:tool"

    def test_repeated(self) -> None:
        result = _summarize_transforms(["a", "a", "b"])
        assert "a*2" in result
        assert "b" in result

    def test_all_unique(self) -> None:
        result = _summarize_transforms(["a", "b", "c"])
        assert result == "a b c"


# ── header_safe_transforms ────────────────────────────────────────────


class TestHeaderSafeTransforms:
    def test_empty(self) -> None:
        assert header_safe_transforms([]) == []

    def test_passthrough_simple_tags(self) -> None:
        result = header_safe_transforms(["tag1", "tag2"])
        assert result == ["tag1", "tag2"]

    def test_collapses_smart_crush(self) -> None:
        result = header_safe_transforms(["smart_crush:3:read,write"])
        assert result == ["smart_crush:3"]

    def test_collapses_read_lifecycle(self) -> None:
        result = header_safe_transforms(["read_lifecycle:stale:foo.py,bar.py"])
        assert result == ["read_lifecycle:stale"]

    def test_preserves_other_tags(self) -> None:
        result = header_safe_transforms(
            [
                "router:excluded:tool",
                "smart_crush:2:a,b",
                "read_lifecycle:fresh:baz.py",
            ]
        )
        assert result == [
            "router:excluded:tool",
            "smart_crush:2",
            "read_lifecycle:fresh",
        ]


# ── merge_cost_stats ──────────────────────────────────────────────────


class TestMergeCostStats:
    def test_none_cost_stats(self) -> None:
        assert merge_cost_stats(None, {}) == {}

    def test_empty_cost_stats(self) -> None:
        result = merge_cost_stats({}, {})
        assert result["savings_usd"] == 0.0
        assert result["compression_savings_usd"] == 0.0
        assert result["cache_savings_usd"] == 0.0
        assert result["cli_tokens_avoided"] == 0

    def test_with_cache_savings(self) -> None:
        cache_stats = {"totals": {"net_savings_usd": 1.5}}
        cost_stats = {"savings_usd": 10.0}
        result = merge_cost_stats(cost_stats, cache_stats)
        assert result["savings_usd"] == 10.0
        assert result["cache_savings_usd"] == 1.5
        assert result["compression_savings_usd"] == 10.0

    def test_with_cli_tokens(self) -> None:
        result = merge_cost_stats({"savings_usd": 5.0}, {}, cli_tokens_avoided=100)
        assert result["cli_tokens_avoided"] == 100
        assert result["cli_filtering_tokens_avoided"] == 100
        assert result["cli_tokens_included_in_compression"] is True


# ── _aggregate_mcp_events ─────────────────────────────────────────────


class TestAggregateMcpEvents:
    def test_returns_zero_when_import_fails(self) -> None:
        result = _aggregate_mcp_events()
        assert result == {"compressions": 0, "tokens_removed": 0, "retrievals": 0}

    @patch("headroom.ccr.mcp_server._read_shared_events")
    def test_aggregates_events(self, mock_read) -> None:
        mock_read.return_value = [
            {"type": "compress", "input_tokens": 1000, "output_tokens": 600},
            {"type": "compress", "input_tokens": 500, "output_tokens": 200},
            {"type": "retrieve"},
            {"type": "compress", "input_tokens": 200, "output_tokens": 200},
        ]
        result = _aggregate_mcp_events()
        assert result["compressions"] == 3
        assert result["tokens_removed"] == (400 + 300 + 0)
        assert result["retrievals"] == 1


# ── RequestOutcome ────────────────────────────────────────────────────


class TestRequestOutcome:
    def test_cache_hit_from_cache_read(self) -> None:
        outcome = RequestOutcome(
            request_id="r1",
            provider="anthropic",
            model="claude-3",
            original_tokens=100,
            optimized_tokens=50,
            output_tokens=20,
            tokens_saved=50,
            attempted_input_tokens=80,
            cache_read_tokens=10,
        )
        assert outcome.cache_hit is True

    def test_cache_hit_from_response_cache(self) -> None:
        outcome = RequestOutcome(
            request_id="r1",
            provider="anthropic",
            model="claude-3",
            original_tokens=100,
            optimized_tokens=50,
            output_tokens=20,
            tokens_saved=50,
            attempted_input_tokens=80,
            from_response_cache=True,
        )
        assert outcome.cache_hit is True

    def test_cache_hit_false(self) -> None:
        outcome = RequestOutcome(
            request_id="r1",
            provider="anthropic",
            model="claude-3",
            original_tokens=100,
            optimized_tokens=50,
            output_tokens=20,
            tokens_saved=50,
            attempted_input_tokens=80,
        )
        assert outcome.cache_hit is False

    def test_cache_hit_pct(self) -> None:
        outcome = RequestOutcome(
            request_id="r1",
            provider="anthropic",
            model="claude-3",
            original_tokens=100,
            optimized_tokens=50,
            output_tokens=20,
            tokens_saved=50,
            attempted_input_tokens=80,
            cache_read_tokens=80,
            cache_write_tokens=20,
        )
        assert outcome.cache_hit_pct == 80

    def test_cache_hit_pct_zero_denom(self) -> None:
        outcome = RequestOutcome(
            request_id="r1",
            provider="anthropic",
            model="claude-3",
            original_tokens=100,
            optimized_tokens=50,
            output_tokens=20,
            tokens_saved=50,
            attempted_input_tokens=80,
        )
        assert outcome.cache_hit_pct == 0

    def test_savings_pct(self) -> None:
        outcome = RequestOutcome(
            request_id="r1",
            provider="anthropic",
            model="claude-3",
            original_tokens=200,
            optimized_tokens=100,
            output_tokens=20,
            tokens_saved=100,
            attempted_input_tokens=150,
        )
        assert outcome.savings_pct == 50.0

    def test_savings_pct_zero_original(self) -> None:
        outcome = RequestOutcome(
            request_id="r1",
            provider="anthropic",
            model="claude-3",
            original_tokens=0,
            optimized_tokens=0,
            output_tokens=0,
            tokens_saved=0,
            attempted_input_tokens=0,
        )
        assert outcome.savings_pct == 0.0

    def test_immutable(self) -> None:
        outcome = RequestOutcome(
            request_id="r1",
            provider="anthropic",
            model="claude-3",
            original_tokens=100,
            optimized_tokens=50,
            output_tokens=20,
            tokens_saved=50,
            attempted_input_tokens=80,
        )
        with pytest.raises(AttributeError):
            outcome.provider = "openai"  # type: ignore[misc]

    def test_value_equality(self) -> None:
        a = RequestOutcome(
            request_id="r1",
            provider="a",
            model="m",
            original_tokens=100,
            optimized_tokens=50,
            output_tokens=20,
            tokens_saved=50,
            attempted_input_tokens=80,
        )
        b = RequestOutcome(
            request_id="r1",
            provider="a",
            model="m",
            original_tokens=100,
            optimized_tokens=50,
            output_tokens=20,
            tokens_saved=50,
            attempted_input_tokens=80,
        )
        assert a == b


class TestRequestOutcomeFromStream:
    def test_basic_messages_body(self) -> None:
        outcome = RequestOutcome.from_stream(
            body={"messages": [{"role": "user", "content": "hi"}]},
            provider="anthropic",
            model="claude-3",
            request_id="r1",
            original_tokens=100,
            optimized_tokens=60,
            output_tokens=30,
            tokens_saved=40,
            transforms_applied=["crush"],
            total_latency_ms=500.0,
            overhead_ms=50.0,
            tags=None,
            client=None,
        )
        assert outcome.request_id == "r1"
        assert outcome.attempted_input_tokens == 100
        assert outcome.num_messages == 1
        assert outcome.transforms_applied == ("crush",)

    def test_gemini_contents_body(self) -> None:
        outcome = RequestOutcome.from_stream(
            body={"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
            provider="gemini",
            model="gemini-2.0",
            request_id="r2",
            original_tokens=100,
            optimized_tokens=60,
            output_tokens=30,
            tokens_saved=40,
            transforms_applied=[],
            total_latency_ms=200.0,
            overhead_ms=20.0,
            tags={"env": "test"},
            client="codex",
        )
        assert outcome.num_messages == 1
        assert outcome.client == "codex"
        assert outcome.tags == {"env": "test"}

    def test_log_full_messages(self) -> None:
        outcome = RequestOutcome.from_stream(
            body={"messages": [{"role": "user", "content": "hi"}]},
            provider="anthropic",
            model="claude-3",
            request_id="r3",
            original_tokens=100,
            optimized_tokens=60,
            output_tokens=30,
            tokens_saved=40,
            transforms_applied=[],
            total_latency_ms=0.0,
            overhead_ms=0.0,
            tags=None,
            client=None,
            log_full_messages=True,
        )
        assert outcome.request_messages == [{"role": "user", "content": "hi"}]

    def test_original_messages_provided(self) -> None:
        outcome = RequestOutcome.from_stream(
            body={"messages": [{"role": "assistant", "content": "compressed"}]},
            provider="anthropic",
            model="claude-3",
            request_id="r4",
            original_tokens=100,
            optimized_tokens=60,
            output_tokens=30,
            tokens_saved=40,
            transforms_applied=[],
            total_latency_ms=0.0,
            overhead_ms=0.0,
            tags=None,
            client=None,
            log_full_messages=True,
            original_messages=[{"role": "user", "content": "original long text"}],
        )
        assert outcome.request_messages == [{"role": "user", "content": "original long text"}]
        assert outcome.compressed_messages == [{"role": "assistant", "content": "compressed"}]

    def test_cache_params_passthrough(self) -> None:
        outcome = RequestOutcome.from_stream(
            body={"messages": [{"role": "user", "content": "hi"}]},
            provider="anthropic",
            model="claude-3",
            request_id="r5",
            original_tokens=200,
            optimized_tokens=100,
            output_tokens=50,
            tokens_saved=100,
            transforms_applied=[],
            total_latency_ms=1000.0,
            overhead_ms=100.0,
            tags=None,
            client=None,
            cache_read_tokens=80,
            cache_write_tokens=20,
            cache_write_5m_tokens=15,
            cache_write_1h_tokens=5,
            uncached_input_tokens=50,
            cache_inferred=True,
            ttfb_ms=300.0,
        )
        assert outcome.cache_read_tokens == 80
        assert outcome.cache_write_tokens == 20
        assert outcome.cache_write_5m_tokens == 15
        assert outcome.cache_write_1h_tokens == 5
        assert outcome.uncached_input_tokens == 50
        assert outcome.cache_inferred is True
        assert outcome.ttfb_ms == 300.0


class TestEmitRequestOutcome:
    def _make_handler(self, **kwargs: dict) -> object:
        from unittest.mock import AsyncMock

        handler = MagicMock()
        handler.metrics = AsyncMock()
        handler.cost_tracker = None
        handler.logger = None
        for k, v in kwargs.items():
            setattr(handler, k, v)
        return handler

    def test_all_downstream_called(self) -> None:
        handler = self._make_handler(
            cost_tracker=MagicMock(),
            logger=MagicMock(),
        )
        outcome = RequestOutcome(
            request_id="r1",
            provider="anthropic",
            model="claude-3",
            original_tokens=100,
            optimized_tokens=60,
            output_tokens=30,
            tokens_saved=40,
            attempted_input_tokens=100,
        )

        with patch("headroom.proxy.project_context.get_current_project", return_value=None):
            import asyncio

            asyncio.run(emit_request_outcome(handler, outcome))

        handler.metrics.record_request.assert_awaited_once()
        handler.cost_tracker.record_tokens.assert_called_once()
        handler.logger.log.assert_called_once()

    def test_emits_perf_log(self, caplog) -> None:
        handler = self._make_handler()
        outcome = RequestOutcome(
            request_id="r1",
            provider="anthropic",
            model="claude-3",
            original_tokens=100,
            optimized_tokens=60,
            output_tokens=30,
            tokens_saved=40,
            attempted_input_tokens=100,
        )
        caplog.set_level(0)

        with patch("headroom.proxy.project_context.get_current_project", return_value=None):
            import asyncio

            asyncio.run(emit_request_outcome(handler, outcome))

        assert "PERF" in caplog.text
        assert "r1" in caplog.text
        assert "tok_saved=40" in caplog.text

    def test_skips_cost_tracker_when_none(self) -> None:
        handler = self._make_handler(logger=MagicMock())
        outcome = RequestOutcome(
            request_id="r1",
            provider="anthropic",
            model="claude-3",
            original_tokens=100,
            optimized_tokens=60,
            output_tokens=30,
            tokens_saved=40,
            attempted_input_tokens=100,
        )

        with patch("headroom.proxy.project_context.get_current_project", return_value=None):
            import asyncio

            asyncio.run(emit_request_outcome(handler, outcome))

        handler.metrics.record_request.assert_awaited_once()
        handler.logger.log.assert_called_once()

    def test_skips_logger_when_none(self) -> None:
        handler = self._make_handler(cost_tracker=MagicMock())
        outcome = RequestOutcome(
            request_id="r1",
            provider="anthropic",
            model="claude-3",
            original_tokens=100,
            optimized_tokens=60,
            output_tokens=30,
            tokens_saved=40,
            attempted_input_tokens=100,
        )

        with patch("headroom.proxy.project_context.get_current_project", return_value=None):
            import asyncio

            asyncio.run(emit_request_outcome(handler, outcome))

        handler.metrics.record_request.assert_awaited_once()
        # logger.log should NOT be called
        assert handler.logger is None


# ── CostTracker ───────────────────────────────────────────────────────


class TestCostTracker:
    def test_default_constructor(self) -> None:
        ct = CostTracker()
        assert ct.budget_limit_usd is None
        assert ct.budget_period == "daily"
        assert ct._costs.maxlen == 100_000

    def test_custom_budget(self) -> None:
        ct = CostTracker(budget_limit_usd=50.0, budget_period="monthly")
        assert ct.budget_limit_usd == 50.0
        assert ct.budget_period == "monthly"

    def test_reset_runtime(self) -> None:
        ct = CostTracker()
        ct._tokens_saved_by_model["gpt-4"] = 1000
        ct._costs.append((datetime.now(), 0.05))
        ct.reset_runtime()
        assert ct._tokens_saved_by_model == {}
        assert len(ct._costs) == 0

    def test_get_period_cost_zero(self) -> None:
        ct = CostTracker()
        assert ct.get_period_cost() == 0.0

    def test_check_budget_no_limit(self) -> None:
        ct = CostTracker()
        allowed, remaining = ct.check_budget()
        assert allowed is True
        assert remaining == float("inf")

    def test_check_budget_within_limit(self) -> None:
        ct = CostTracker(budget_limit_usd=10.0)
        ct._costs.append((datetime.now(), 5.0))
        allowed, remaining = ct.check_budget()
        assert allowed is True
        assert remaining == 5.0

    def test_check_budget_over_limit(self) -> None:
        ct = CostTracker(budget_limit_usd=10.0)
        ct._costs.append((datetime.now(), 12.0))
        allowed, remaining = ct.check_budget()
        assert allowed is False
        assert remaining == 0.0

    def test_get_period_cost_hourly(self) -> None:
        ct = CostTracker(budget_period="hourly")
        ct._costs.append((datetime.now(), 1.0))
        ct._costs.append((datetime.now() - timedelta(hours=2), 2.0))
        assert ct.get_period_cost() == 1.0

    def test_get_period_cost_daily(self) -> None:
        ct = CostTracker(budget_period="daily")
        ct._costs.append((datetime.now(), 1.0))
        assert ct.get_period_cost() > 0

    def test_get_period_cost_monthly(self) -> None:
        ct = CostTracker(budget_period="monthly")
        ct._costs.append((datetime.now(), 1.0))
        assert ct.get_period_cost() > 0

    def test_record_tokens_updates_dicts(self) -> None:
        ct = CostTracker()
        ct.record_tokens(
            model="gpt-4",
            tokens_saved=500,
            tokens_sent=1000,
        )
        assert ct._tokens_saved_by_model["gpt-4"] == 500
        assert ct._tokens_sent_by_model["gpt-4"] == 1000
        assert ct._requests_by_model["gpt-4"] == 1

    def test_record_tokens_with_cache(self) -> None:
        ct = CostTracker()
        ct.record_tokens(
            model="gpt-4",
            tokens_saved=500,
            tokens_sent=1000,
            cache_read_tokens=50,
            cache_write_tokens=20,
            cache_write_5m_tokens=10,
            cache_write_1h_tokens=10,
            uncached_tokens=930,
            output_tokens=200,
        )
        assert ct._api_cache_read_by_model["gpt-4"] == 50
        assert ct._api_cache_write_by_model["gpt-4"] == 20
        assert ct._api_cache_write_5m_by_model["gpt-4"] == 10
        assert ct._api_cache_write_1h_by_model["gpt-4"] == 10
        assert ct._api_uncached_by_model["gpt-4"] == 930

    def test_stats_structure(self) -> None:
        ct = CostTracker()
        ct.record_tokens("gpt-4", tokens_saved=500, tokens_sent=1000)
        ct.record_tokens("claude-3", tokens_saved=200, tokens_sent=800)
        stats = ct.stats()
        assert "per_model" in stats
        assert "gpt-4" in stats["per_model"]
        assert "claude-3" in stats["per_model"]
        assert stats["total_tokens_saved"] == 700
        assert stats["cost_with_headroom_usd"] >= 0.0

    def test_prune_old_costs_skips_recent(self) -> None:
        ct = CostTracker()
        ct._costs.append((datetime.now(), 1.0))
        ct._prune_old_costs()
        assert len(ct._costs) == 1

    def test_prune_old_costs_removes_old(self) -> None:
        ct = CostTracker()
        ct._costs.append((datetime.now() - timedelta(hours=1000), 1.0))
        ct._costs.append((datetime.now(), 2.0))
        ct._last_prune_time = datetime.now() - timedelta(minutes=10)
        ct._prune_old_costs()
        assert len(ct._costs) == 1
        assert ct._costs[0][1] == 2.0


# ── build_prefix_cache_stats ──────────────────────────────────────────


class TestBuildPrefixCacheStats:
    def test_empty_metrics(self) -> None:
        metrics = MagicMock()
        metrics.cache_by_provider = {}
        metrics.cache_miss_attribution_by_provider = {}
        metrics.prefix_freeze_busts_avoided = 0
        metrics.prefix_freeze_tokens_preserved = 0
        metrics.prefix_freeze_compression_foregone = 0
        metrics.tokens_saved_total = 0
        metrics.cache_bust_tokens_lost = 0
        metrics.cache_bust_count = 0

        result = build_prefix_cache_stats(metrics, cost_tracker=None)
        assert result["by_provider"] == {}
        assert result["totals"]["requests"] == 0
        assert result["totals"]["hit_rate"] == 0

    def test_with_provider_data(self) -> None:
        metrics = MagicMock()
        metrics.cache_by_provider = {
            "anthropic": {
                "requests": 10,
                "hit_requests": 5,
                "cache_read_tokens": 1000,
                "cache_write_tokens": 200,
                "cache_write_5m_tokens": 150,
                "cache_write_1h_tokens": 50,
                "cache_write_5m_requests": 3,
                "cache_write_1h_requests": 2,
                "uncached_input_tokens": 8000,
                "bust_count": 1,
                "bust_write_tokens": 100,
            },
        }
        metrics.cache_miss_attribution_by_provider = {
            "anthropic": {"ttl_expiry": 3, "prefix_change": 1},
        }
        metrics.prefix_freeze_busts_avoided = 0
        metrics.prefix_freeze_tokens_preserved = 0
        metrics.prefix_freeze_compression_foregone = 0
        metrics.tokens_saved_total = 0
        metrics.cache_bust_tokens_lost = 0
        metrics.cache_bust_count = 0

        result = build_prefix_cache_stats(metrics, cost_tracker=None)
        assert "anthropic" in result["by_provider"]
        ap = result["by_provider"]["anthropic"]
        assert ap["requests"] == 10
        assert ap["hit_requests"] == 5
        assert ap["cache_read_tokens"] == 1000
        assert ap["hit_rate"] > 0
        assert result["miss_attribution"]["totals"]["ttl_expiry"] == 3
        assert result["miss_attribution"]["totals"]["prefix_change"] == 1


# ── build_session_summary ─────────────────────────────────────────────


class TestBuildSessionSummary:
    def test_basic_structure(self) -> None:
        proxy = MagicMock()
        proxy.config.mode = "token"
        proxy.cost_tracker = None
        proxy.logger = None

        metrics = MagicMock()
        metrics.requests_by_model = {}
        metrics.tokens_saved_total = 0

        result = build_session_summary(
            proxy,
            metrics,
            {"totals": {"net_savings_usd": 0.0}},
            cli_tokens_avoided=0,
            total_tokens_before=0,
        )
        assert result["mode"] == "token"
        assert result["api_requests"] == 0
        assert result["compression"]["requests_compressed"] == 0

    def test_with_request_logs(self) -> None:
        from datetime import datetime

        proxy = MagicMock()
        proxy.config.mode = "cache"
        proxy.cost_tracker = None

        log_entry = RequestLog(
            request_id="r1",
            timestamp=datetime.now().isoformat(),
            provider="anthropic",
            model="claude-3",
            input_tokens_original=1000,
            input_tokens_optimized=500,
            output_tokens=200,
            tokens_saved=500,
            savings_percent=50.0,
            optimization_latency_ms=50.0,
            total_latency_ms=1000.0,
            tags={},
            cache_hit=False,
            transforms_applied=[],
        )
        request_logger = MagicMock()
        request_logger._logs = [log_entry]
        proxy.logger = request_logger

        metrics = MagicMock()
        metrics.requests_by_model = {}
        metrics.tokens_saved_total = 500

        result = build_session_summary(
            proxy,
            metrics,
            {"totals": {"net_savings_usd": 0.0}},
            cli_tokens_avoided=0,
            total_tokens_before=1000,
        )
        assert result["compression"]["requests_compressed"] == 1
        assert result["compression"]["total_tokens_removed"] == 500
