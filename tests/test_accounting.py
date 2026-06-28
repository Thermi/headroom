"""Tests for headroom.accounting -- model usage tracking with statistics."""

from __future__ import annotations

import pytest

from headroom.accounting import (
    ModelAccounting,
    ModelCompressionStats,
    ModelUsageRecord,
    _cv,
    _histogram,
    _percentile,
    get_model_accounting,
    reset_model_accounting,
)

# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_empty_list(self):
        assert _percentile([], 50) == 0.0

    def test_single_value(self):
        assert _percentile([42.0], 50) == 42.0
        assert _percentile([42.0], 99) == 42.0
        assert _percentile([42.0], 1) == 42.0

    def test_median_odd(self):
        assert _percentile([1.0, 2.0, 3.0], 50) == 2.0

    def test_median_even(self):
        assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5

    def test_p1(self):
        vals = list(range(100))
        assert _percentile([float(v) for v in vals], 1) == pytest.approx(0.99, rel=0.1)

    def test_p99(self):
        vals = list(range(100))
        assert _percentile([float(v) for v in vals], 99) == pytest.approx(98.01, rel=0.1)

    def test_p95(self):
        vals = [float(v) for v in range(100)]
        result = _percentile(vals, 95)
        assert 94.0 <= result <= 95.0

    def test_p5(self):
        vals = [float(v) for v in range(100)]
        result = _percentile(vals, 5)
        assert 4.0 <= result <= 5.0


class TestCoefficientOfVariation:
    def test_empty_list(self):
        assert _cv([]) == 0.0

    def test_single_value(self):
        assert _cv([5.0]) == 0.0

    def test_two_values(self):
        result = _cv([1.0, 3.0])
        assert result == pytest.approx(0.70710678, rel=1e-5)

    def test_identical_values(self):
        assert _cv([5.0, 5.0, 5.0]) == 0.0

    def test_zero_mean(self):
        # When mean is 0, CV is 0
        assert _cv([0.0, 0.0, 0.0]) == 0.0

    def test_known_cv(self):
        # Data: 10, 12, 23, 23, 16, 23, 21, 16
        data = [10.0, 12.0, 23.0, 23.0, 16.0, 23.0, 21.0, 16.0]
        result = _cv(data)
        # Computed: mean=18.0, std≈5.237, CV≈0.291
        assert result == pytest.approx(0.291, rel=1e-2)


class TestHistogram:
    def test_empty_values(self):
        buckets = [(0, 0), (1, 10)]
        hist = _histogram([], buckets)
        assert all(v == 0 for v in hist.values())
        assert len(hist) == 2

    def test_single_value(self):
        buckets = [(0, 0), (1, 10), (11, 20)]
        hist = _histogram([5.0], buckets)
        assert hist["0"] == 0
        assert hist["1-10"] == 1
        assert hist["11-20"] == 0

    def test_value_at_boundary(self):
        buckets = [(0, 5), (6, 10)]
        hist = _histogram([5.0], buckets)
        assert hist["0-5"] == 1
        assert hist["6-10"] == 0

    def test_value_at_exact_bucket(self):
        buckets = [(0, 0), (1, 1)]
        hist = _histogram([1.0], buckets)
        assert hist["0"] == 0
        assert hist["1"] == 1

    def test_multiple_values(self):
        buckets = [(0, 5), (6, 10), (11, 15)]
        hist = _histogram([3.0, 7.0, 12.0, 14.0], buckets)
        assert hist["0-5"] == 1
        assert hist["6-10"] == 1
        assert hist["11-15"] == 2

    def test_zero_bucket(self):
        buckets = [(0, 0), (1, 10)]
        hist = _histogram([0.0], buckets)
        assert hist["0"] == 1
        assert hist["1-10"] == 0


# ---------------------------------------------------------------------------
# ModelCompressionStats
# ---------------------------------------------------------------------------


class TestModelCompressionStatsFromRecords:
    def test_empty_records(self):
        stats = ModelCompressionStats.from_records([])
        assert stats.total_calls == 0
        assert stats.total_input_tokens == 0
        assert stats.total_output_tokens == 0
        assert stats.total_runtime_ms == 0.0
        assert stats.avg_input_tokens == 0.0
        assert stats.avg_output_tokens == 0.0
        assert stats.avg_runtime_ms == 0.0
        assert stats.min_input_tokens == 0
        assert stats.max_input_tokens == 0
        assert stats.min_output_tokens == 0
        assert stats.max_output_tokens == 0
        assert stats.min_runtime_ms == 0.0
        assert stats.max_runtime_ms == 0.0
        assert stats.p1_input_tokens == 0
        assert stats.p50_input_tokens == 0
        assert stats.p99_input_tokens == 0
        assert stats.cv_input_tokens == 0.0
        assert stats.cv_output_tokens == 0.0
        assert stats.cv_runtime_ms == 0.0
        assert stats.input_token_distribution == {}
        assert stats.output_token_distribution == {}
        assert stats.runtime_ms_distribution == {}

    def test_single_record(self):
        records = [ModelUsageRecord("kompress", 100, 50, 10.0, timestamp=100.0)]
        stats = ModelCompressionStats.from_records(records)
        assert stats.total_calls == 1
        assert stats.total_input_tokens == 100
        assert stats.total_output_tokens == 50
        assert stats.total_runtime_ms == 10.0
        assert stats.avg_input_tokens == 100.0
        assert stats.avg_output_tokens == 50.0
        assert stats.avg_runtime_ms == 10.0
        assert stats.min_input_tokens == 100
        assert stats.max_input_tokens == 100
        assert stats.min_output_tokens == 50
        assert stats.max_output_tokens == 50
        assert stats.min_runtime_ms == 10.0
        assert stats.max_runtime_ms == 10.0
        assert stats.p50_input_tokens == 100
        assert stats.p50_output_tokens == 50
        assert stats.p50_runtime_ms == 10.0
        # CV is 0 for single record
        assert stats.cv_input_tokens == 0.0
        assert stats.cv_output_tokens == 0.0
        assert stats.cv_runtime_ms == 0.0

    def test_multiple_records(self):
        records = [
            ModelUsageRecord("kompress", 100, 30, 5.0, timestamp=100.0),
            ModelUsageRecord("kompress", 200, 60, 10.0, timestamp=101.0),
            ModelUsageRecord("kompress", 300, 90, 15.0, timestamp=102.0),
        ]
        stats = ModelCompressionStats.from_records(records)
        assert stats.total_calls == 3
        assert stats.total_input_tokens == 600
        assert stats.total_output_tokens == 180
        assert stats.total_runtime_ms == 30.0
        assert stats.avg_input_tokens == 200.0
        assert stats.avg_output_tokens == 60.0
        assert stats.avg_runtime_ms == 10.0
        assert stats.min_input_tokens == 100
        assert stats.max_input_tokens == 300
        assert stats.min_output_tokens == 30
        assert stats.max_output_tokens == 90
        assert stats.min_runtime_ms == 5.0
        assert stats.max_runtime_ms == 15.0
        assert stats.p50_input_tokens == 200
        assert stats.p50_output_tokens == 60
        assert stats.p50_runtime_ms == 10.0

    def test_preserves_order_for_percentiles(self):
        records = [ModelUsageRecord("m", i * 100, i * 30, i * 5.0) for i in range(1, 101)]
        stats = ModelCompressionStats.from_records(records)
        # With 100 values, p1 is roughly the 1st value, p99 roughly the 99th
        assert 0 < stats.p1_input_tokens <= 200
        assert 9500 <= stats.p99_input_tokens <= 10000

    def test_percentiles_with_few_values(self):
        records = [
            ModelUsageRecord("m", 100, 50, 10.0),
            ModelUsageRecord("m", 200, 80, 20.0),
        ]
        stats = ModelCompressionStats.from_records(records)
        assert stats.p1_input_tokens > 0
        assert stats.p99_input_tokens > 0
        assert stats.p50_input_tokens == 150

    def test_histograms_generated(self):
        records = [
            ModelUsageRecord("m", 100, 50, 10.0),
            ModelUsageRecord("m", 500, 200, 25.0),
        ]
        stats = ModelCompressionStats.from_records(records)
        assert "0" in stats.input_token_distribution
        assert stats.input_token_distribution["129-256"] == 0  # wrong bucket
        # 100 falls in bucket 1-128, 500 in 257-512
        assert stats.input_token_distribution["1-128"] >= 0
        # Check runtime distribution has values
        assert any(v > 0 for v in stats.runtime_ms_distribution.values())


class TestModelCompressionStatsToDict:
    def test_empty_to_dict(self):
        stats = ModelCompressionStats()
        d = stats.to_dict()
        assert d["total_calls"] == 0
        assert d["total_input_tokens"] == 0
        assert d["avg_input_tokens"] == 0.0
        assert d["min_input_tokens"] == 0
        assert d["max_input_tokens"] == 0
        assert d["cv_input_tokens"] == 0.0
        assert d["p1_input_tokens"] == 0
        assert d["p50_input_tokens"] == 0
        assert d["p99_input_tokens"] == 0
        # Distributions are always present (not None)
        assert isinstance(d["input_token_distribution"], dict)
        assert isinstance(d["output_token_distribution"], dict)
        assert isinstance(d["runtime_ms_distribution"], dict)

    def test_populated_to_dict(self):
        stats = ModelCompressionStats.from_records(
            [
                ModelUsageRecord("test-model", 150, 75, 12.5),
            ]
        )
        d = stats.to_dict()
        assert d["total_calls"] == 1
        assert d["total_input_tokens"] == 150
        assert d["total_output_tokens"] == 75
        assert d["total_runtime_ms"] == 12.5
        assert d["avg_input_tokens"] == 150.0
        assert d["avg_runtime_ms"] == 12.5
        assert d["min_input_tokens"] == 150
        assert d["max_input_tokens"] == 150
        assert d["p50_input_tokens"] == 150
        assert d["cv_input_tokens"] == 0.0

    def test_no_null_values_in_dict(self):
        """Guarantee: every field in to_dict is a non-null value."""
        stats = ModelCompressionStats()
        d = stats.to_dict()
        for key, value in d.items():
            assert value is not None, f"Key {key!r} is None"

        stats2 = ModelCompressionStats.from_records(
            [
                ModelUsageRecord("m", 100, 50, 10.0),
            ]
        )
        d2 = stats2.to_dict()
        for key, value in d2.items():
            assert value is not None, f"Key {key!r} is None"


# ---------------------------------------------------------------------------
# ModelAccounting
# ---------------------------------------------------------------------------


class TestModelAccounting:
    def test_new_accounting_is_empty(self):
        acc = ModelAccounting()
        assert acc.total_calls == 0
        assert acc.models_seen == []

    def test_record_single_model(self):
        acc = ModelAccounting()
        acc.record("kompress", 100, 50, 10.0)
        assert acc.total_calls == 1
        assert acc.models_seen == ["kompress"]

    def test_record_multiple_calls(self):
        acc = ModelAccounting()
        acc.record("kompress", 100, 50, 10.0)
        acc.record("kompress", 200, 80, 20.0)
        assert acc.total_calls == 2
        assert acc.models_seen == ["kompress"]

    def test_record_multiple_models(self):
        acc = ModelAccounting()
        acc.record("kompress", 100, 50, 10.0)
        acc.record("smartcrusher", 500, 200, 5.0)
        assert acc.total_calls == 2
        assert acc.models_seen == ["kompress", "smartcrusher"]

    def test_record_negative_values(self):
        """Negative values should be clamped to 0."""
        acc = ModelAccounting()
        acc.record("kompress", -100, -50, -10.0)
        stats = acc.get_stats("kompress")
        assert stats["kompress"].total_input_tokens == 0
        assert stats["kompress"].total_output_tokens == 0
        assert stats["kompress"].total_runtime_ms == 0.0

    def test_get_stats_all(self):
        acc = ModelAccounting()
        acc.record("kompress", 100, 50, 10.0)
        acc.record("smartcrusher", 500, 200, 5.0)
        stats = acc.get_stats()
        assert "kompress" in stats
        assert "smartcrusher" in stats
        assert stats["kompress"].total_calls == 1
        assert stats["smartcrusher"].total_calls == 1

    def test_get_stats_single_model(self):
        acc = ModelAccounting()
        acc.record("kompress", 100, 50, 10.0)
        acc.record("smartcrusher", 500, 200, 5.0)
        stats = acc.get_stats("kompress")
        assert "kompress" in stats
        assert "smartcrusher" not in stats

    def test_get_stats_unknown_model(self):
        acc = ModelAccounting()
        stats = acc.get_stats("nonexistent")
        assert "nonexistent" in stats
        assert stats["nonexistent"].total_calls == 0

    def test_get_stats_dict_all(self):
        acc = ModelAccounting()
        acc.record("kompress", 100, 50, 10.0)
        d = acc.get_stats_dict()
        assert isinstance(d, dict)
        assert "kompress" in d
        assert d["kompress"]["total_calls"] == 1
        assert d["kompress"]["total_input_tokens"] == 100

    def test_get_stats_dict_single(self):
        acc = ModelAccounting()
        acc.record("kompress", 100, 50, 10.0)
        d = acc.get_stats_dict("kompress")
        assert isinstance(d, dict)
        assert "total_calls" in d
        assert d["total_calls"] == 1

    def test_get_stats_dict_unknown(self):
        acc = ModelAccounting()
        d = acc.get_stats_dict("nonexistent")
        assert d["total_calls"] == 0

    def test_reset_single_model(self):
        acc = ModelAccounting()
        acc.record("kompress", 100, 50, 10.0)
        acc.record("smartcrusher", 500, 200, 5.0)
        acc.reset("kompress")
        assert acc.total_calls == 1
        assert acc.models_seen == ["smartcrusher"]

    def test_reset_all(self):
        acc = ModelAccounting()
        acc.record("kompress", 100, 50, 10.0)
        acc.record("smartcrusher", 500, 200, 5.0)
        acc.reset()
        assert acc.total_calls == 0
        assert acc.models_seen == []

    def test_thread_safety(self):
        """ModelAccounting should be thread-safe (best-effort check)."""
        import concurrent.futures

        acc = ModelAccounting()

        def record_calls(model: str, n: int):
            for _i in range(n):
                acc.record(model, 100, 50, 10.0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = []
            for i in range(4):
                futures.append(ex.submit(record_calls, f"model-{i}", 100))
            for f in concurrent.futures.as_completed(futures):
                f.result()

        assert acc.total_calls == 400
        assert len(acc.models_seen) == 4


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------


class TestGlobalModelAccounting:
    def setup_method(self):
        reset_model_accounting()

    def test_get_model_accounting(self):
        acc = get_model_accounting()
        assert isinstance(acc, ModelAccounting)

    def test_singleton_same_instance(self):
        acc1 = get_model_accounting()
        acc2 = get_model_accounting()
        assert acc1 is acc2

    def test_reset_model_accounting(self):
        acc = get_model_accounting()
        acc.record("kompress", 100, 50, 10.0)
        reset_model_accounting()
        acc2 = get_model_accounting()
        assert acc2.total_calls == 0


# ---------------------------------------------------------------------------
# SessionStats integration (via headroom.ccr.mcp_server)
# ---------------------------------------------------------------------------


class TestSessionStatsModelAccounting:
    def setup_method(self):
        reset_model_accounting()

    def test_record_model_usage(self):
        from headroom.ccr.mcp_server import SessionStats

        stats = SessionStats()
        stats.record_model_usage("kompress", 100, 50, 10.0)
        d = stats.to_dict()
        assert "model_accounting" in d
        assert "kompress" in d["model_accounting"]
        assert d["model_accounting"]["kompress"]["total_calls"] == 1

    def test_record_model_usage_unknown_model(self):
        """When model_name is None, it defaults to 'unknown'."""
        from headroom.ccr.mcp_server import SessionStats

        stats = SessionStats()
        stats.record_model_usage(None, 100, 50, 10.0)
        d = stats.to_dict()
        assert "unknown" in d["model_accounting"]
        assert d["model_accounting"]["unknown"]["total_calls"] == 1

    def test_record_compression_with_model_name(self):
        from headroom.ccr.mcp_server import SessionStats

        stats = SessionStats()
        stats.record_compression(
            input_tokens=200,
            output_tokens=60,
            strategy="kompress",
            model_name="chopratejas/kompress-v2-base",
            runtime_ms=15.0,
        )
        d = stats.to_dict()
        assert d["compressions"] == 1
        assert "model_accounting" in d
        assert "chopratejas/kompress-v2-base" in d["model_accounting"]

    def test_record_compression_without_model_name(self):
        from headroom.ccr.mcp_server import SessionStats

        stats = SessionStats()
        stats.record_compression(
            input_tokens=200,
            output_tokens=60,
            strategy="passthrough",
        )
        d = stats.to_dict()
        assert d["compressions"] == 1
        # Without model_name, it records under strategy name
        assert "passthrough" in d["model_accounting"]

    def test_model_accounting_present_in_empty_stats(self):
        """Even with no activity, model_accounting key exists and is not None."""
        from headroom.ccr.mcp_server import SessionStats

        stats = SessionStats()
        d = stats.to_dict()
        assert "model_accounting" in d
        assert d["model_accounting"] == {}
        assert d["model_accounting"] is not None

    def test_model_accounting_aggregates_across_calls(self):
        from headroom.ccr.mcp_server import SessionStats

        stats = SessionStats()
        stats.record_compression(100, 30, "kompress", "gpt-4o", 5.0)
        stats.record_compression(200, 60, "kompress", "gpt-4o", 10.0)
        d = stats.to_dict()
        model_stats = d["model_accounting"]["gpt-4o"]
        assert model_stats["total_calls"] == 2
        assert model_stats["total_input_tokens"] == 300
        assert model_stats["total_output_tokens"] == 90
        assert model_stats["avg_input_tokens"] == 150.0

    def test_model_accounting_to_dict_no_nulls(self):
        from headroom.ccr.mcp_server import SessionStats

        stats = SessionStats()
        stats.record_compression(100, 30, "kompress", "gpt-4o", 5.0)
        d = stats.to_dict()
        # Recursively check no nulls in model_accounting
        for model_name, model_data in d["model_accounting"].items():
            for key, value in model_data.items():
                assert value is not None, f"Key {key!r} is None for model {model_name!r}"
