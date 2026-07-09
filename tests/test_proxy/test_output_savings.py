from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from headroom.proxy.output_savings import (
    BaselineModel,
    SavingsEstimate,
    SavingsLedger,
    SavingsRecorder,
    _Accum,
    assign_arm,
    conversation_key_from_body,
    echo_ratio,
    get_recorder,
    input_bucket,
    model_family,
    parse_stratum_label,
    stratum_key,
    stratum_label,
)


class TestInputBucket:
    def test_xs_under_2000(self) -> None:
        assert input_bucket(0) == "xs"
        assert input_bucket(1999) == "xs"

    def test_s_2000_to_7999(self) -> None:
        assert input_bucket(2000) == "s"
        assert input_bucket(7999) == "s"

    def test_m_8000_to_31999(self) -> None:
        assert input_bucket(8000) == "m"
        assert input_bucket(31999) == "m"

    def test_l_32000_to_127999(self) -> None:
        assert input_bucket(32000) == "l"
        assert input_bucket(127999) == "l"

    def test_xl_128000_plus(self) -> None:
        assert input_bucket(128000) == "xl"
        assert input_bucket(999999) == "xl"


class TestModelFamily:
    def test_opus(self) -> None:
        assert model_family("claude-opus-4") == "opus"
        assert model_family("anthropic.claude-opus-v1") == "opus"

    def test_sonnet(self) -> None:
        assert model_family("claude-sonnet-4-20250514") == "sonnet"

    def test_haiku(self) -> None:
        assert model_family("claude-haiku-3-5") == "haiku"

    def test_fable(self) -> None:
        assert model_family("claude-fable") == "fable"

    def test_mythos(self) -> None:
        assert model_family("claude-mythos") == "mythos"

    def test_gpt(self) -> None:
        assert model_family("gpt-4o") == "gpt"
        assert model_family("chatgpt-4o-latest") == "gpt"

    def test_gemini(self) -> None:
        assert model_family("gemini-2.0-flash") == "gemini"

    def test_other(self) -> None:
        assert model_family("unknown-model-v1") == "other"
        assert model_family("") == "other"


class TestStratumKey:
    def test_builds_pipe_delimited_key(self) -> None:
        key = stratum_key(
            turn_kind="new_user_ask", input_tokens=10000, model="claude-sonnet-4", has_tools=True
        )
        assert key == "sonnet|new_user_ask|m|tools"

    def test_with_tools_flag_false(self) -> None:
        key = stratum_key(
            turn_kind="mechanical_continuation", input_tokens=500, model="gpt-4o", has_tools=False
        )
        assert key == "gpt|mechanical_continuation|xs|notools"

    def test_bucket_and_family_resolved(self) -> None:
        key = stratum_key(
            turn_kind="unknown", input_tokens=50000, model="claude-haiku-3", has_tools=False
        )
        assert key == "haiku|unknown|l|notools"


class TestConversationKeyFromBody:
    def test_uses_model_and_first_user_message(self) -> None:
        body = {
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "hello world"}],
        }
        key = conversation_key_from_body(body)
        assert isinstance(key, str)
        assert len(key) == 64  # sha256 hexdigest

    def test_deterministic(self) -> None:
        body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]}
        assert conversation_key_from_body(body) == conversation_key_from_body(body)

    def test_unwraps_response_create(self) -> None:
        body = {
            "type": "response.create",
            "response": {
                "model": "claude-sonnet-4",
                "messages": [{"role": "user", "content": "hi"}],
            },
        }
        key = conversation_key_from_body(body)
        assert len(key) == 64

    def test_different_input_yields_different_key(self) -> None:
        k1 = conversation_key_from_body(
            {"model": "a", "messages": [{"role": "user", "content": "x"}]}
        )
        k2 = conversation_key_from_body(
            {"model": "b", "messages": [{"role": "user", "content": "y"}]}
        )
        assert k1 != k2

    def test_content_blocks(self) -> None:
        body = {
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "analyze this"}]}],
        }
        key = conversation_key_from_body(body)
        assert len(key) == 64

    def test_empty_body_still_produces_hash(self) -> None:
        key = conversation_key_from_body(
            {"model": "test", "messages": [{"role": "user", "content": ""}]}
        )
        assert len(key) == 64


class TestAssignArm:
    def test_zero_holdout_all_treatment(self) -> None:
        assert assign_arm("conv_1", 0.0) == "treatment"
        assert assign_arm("conv_2", -0.1) == "treatment"

    def test_full_holdout_all_control(self) -> None:
        assert assign_arm("conv_1", 1.0) == "control"
        assert assign_arm("conv_2", 2.0) == "control"

    def test_half_holdout_distributes(self) -> None:
        keys = [f"conv_{i}" for i in range(100)]
        results = [assign_arm(k, 0.5) for k in keys]
        treatment_count = results.count("treatment")
        assert 30 < treatment_count < 70  # statistically plausible

    def test_deterministic_for_same_key(self) -> None:
        assert assign_arm("same_key", 0.3) == assign_arm("same_key", 0.3)

    def test_different_keys_can_differ(self) -> None:
        results = {assign_arm(f"key_{i}", 0.5) for i in range(100)}
        assert results == {"treatment", "control"}


class TestBaselineModel:
    def test_observe_adds_to_strata_and_glob(self) -> None:
        m = BaselineModel()
        m.observe("opus|ask|m|tools", 200)
        assert m.total_samples == 1
        assert m.strata["opus|ask|m|tools"].n == 1

    def test_lookup_returns_mean_var_n(self) -> None:
        m = BaselineModel()
        m.observe("opus|ask|m|tools", 200)
        m.observe("opus|ask|m|tools", 300)
        mean, var, n = m.lookup("opus|ask|m|tools")
        assert mean == 250.0
        assert n == 2

    def test_lookup_backoff_trims_fields(self) -> None:
        m = BaselineModel()
        m.observe("sonnet|ask|m|tools", 100)
        mean, var, n = m.lookup("sonnet|ask|m|notools")
        assert n == 1  # found by stripping "notools" -> "sonnet|ask|m" prefix
        assert mean == 100.0

    def test_lookup_falls_back_to_global(self) -> None:
        m = BaselineModel()
        m.observe("unknown|foo|xs|notools", 500)
        m.observe("other|bar|xl|tools", 300)
        mean, var, n = m.lookup("nonexistent|all|m|tools")
        assert n == 2
        assert mean == 400.0

    def test_lookup_empty_glob_returns_zero(self) -> None:
        m = BaselineModel()
        mean, var, n = m.lookup("any|key|xs|tools")
        assert mean == 0.0
        assert var == 0.0
        assert n == 0

    def test_merge_combines_accumulators(self) -> None:
        a = BaselineModel()
        a.observe("shared|key|m|tools", 100)
        b = BaselineModel()
        b.observe("shared|key|m|tools", 200)
        b.observe("other|key|s|notools", 50)
        a.merge(b)
        assert a.strata["shared|key|m|tools"].n == 2
        assert a.strata["shared|key|m|tools"].sum == 300.0
        assert a.strata["other|key|s|notools"].n == 1
        assert a.total_samples == 3

    def test_to_dict_from_dict_round_trip(self) -> None:
        m = BaselineModel()
        m.observe("a|b|xs|tools", 100)
        m.observe("c|d|m|notools", 200)
        restored = BaselineModel.from_dict(m.to_dict())
        assert restored.total_samples == 2
        assert restored.strata["a|b|xs|tools"].mean == 100.0
        assert restored.strata["c|d|m|notools"].mean == 200.0

    def test_total_samples_reflects_glob(self) -> None:
        m = BaselineModel()
        assert m.total_samples == 0
        m.observe("k1", 10)
        assert m.total_samples == 1
        m.observe("k2", 20)
        assert m.total_samples == 2


class TestSavingsLedger:
    def test_record_adds_to_treatment(self) -> None:
        ledger = SavingsLedger()
        ledger.record("treatment", "k1", 100)
        assert ledger.treatment["k1"].n == 1
        assert "k1" not in ledger.control

    def test_record_adds_to_control(self) -> None:
        ledger = SavingsLedger()
        ledger.record("control", "k1", 200)
        assert ledger.control["k1"].n == 1
        assert "k1" not in ledger.treatment

    def test_estimate_from_baseline_returns_estimate(self) -> None:
        ledger = SavingsLedger()
        ledger.baseline.observe("k1", 200)
        ledger.record("treatment", "k1", 150)
        est = ledger.estimate_from_baseline()
        assert isinstance(est, SavingsEstimate)
        assert est.kind == "estimated"
        assert est.n_requests == 1
        assert est.tokens_saved > 0

    def test_estimate_from_baseline_empty_treatment(self) -> None:
        ledger = SavingsLedger()
        est = ledger.estimate_from_baseline()
        assert est.kind == "estimated"
        assert est.n_requests == 0

    def test_estimate_from_holdout_returns_estimate(self) -> None:
        ledger = SavingsLedger()
        ledger.record("control", "k1", 200)
        ledger.record("control", "k1", 220)
        ledger.record("treatment", "k1", 150)
        ledger.record("treatment", "k1", 170)
        est = ledger.estimate_from_holdout()
        assert est is not None
        assert est.kind == "measured"
        assert est.n_requests == 2
        assert est.tokens_saved > 0

    def test_estimate_from_holdout_no_overlap_returns_none(self) -> None:
        ledger = SavingsLedger()
        ledger.record("control", "k1", 200)
        ledger.record("treatment", "k2", 150)
        assert ledger.estimate_from_holdout() is None

    def test_best_estimate_prefers_measured(self) -> None:
        ledger = SavingsLedger()
        ledger.baseline.observe("k1", 200)
        ledger.record("control", "k1", 220)
        ledger.record("treatment", "k1", 170)
        est = ledger.best_estimate()
        assert est.kind == "measured"

    def test_best_estimate_falls_back_to_estimated(self) -> None:
        ledger = SavingsLedger()
        ledger.baseline.observe("k1", 200)
        ledger.record("treatment", "k1", 150)
        est = ledger.best_estimate()
        assert est.kind == "estimated"

    def test_best_estimate_when_empty(self) -> None:
        ledger = SavingsLedger()
        est = ledger.best_estimate()
        assert est.kind == "estimated"
        assert est.n_requests == 0

    def test_to_dict_from_dict_round_trip(self) -> None:
        ledger = SavingsLedger()
        ledger.baseline.observe("k1", 200)
        ledger.record("treatment", "k1", 150)
        ledger.record("control", "k2", 100)
        restored = SavingsLedger.from_dict(ledger.to_dict())
        assert restored.baseline.total_samples == 1
        assert restored.treatment["k1"].n == 1
        assert restored.control["k2"].n == 1

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        p = tmp_path / "ledger.json"
        ledger = SavingsLedger()
        ledger.baseline.observe("k1", 200)
        ledger.record("treatment", "k1", 150)
        ledger.save(p)
        assert p.exists()
        loaded = SavingsLedger.load(p)
        assert loaded.baseline.total_samples == 1
        assert loaded.treatment["k1"].n == 1

    def test_load_missing_file_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "nonexistent.json"
        ledger = SavingsLedger.load(p)
        assert ledger.baseline.total_samples == 0
        assert ledger.treatment == {}
        assert ledger.control == {}

    def test_load_corrupted_file_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "corrupt.json"
        p.write_text("{not json}")
        ledger = SavingsLedger.load(p)
        assert ledger.baseline.total_samples == 0


class TestSavingsEstimate:
    def test_to_dict(self) -> None:
        est = SavingsEstimate(
            tokens_saved=100.0,
            baseline_tokens=500.0,
            pct=20.0,
            ci_low_pct=10.0,
            ci_high_pct=30.0,
            n_requests=5,
            kind="estimated",
        )
        d = est.to_dict()
        assert d["tokens_saved"] == 100.0
        assert d["kind"] == "estimated"
        assert d["n_requests"] == 5


class TestSavingsRecorder:
    def test_record_from_labels_matching(self, tmp_path: Path) -> None:
        p = tmp_path / "savings.json"
        recorder = SavingsRecorder(p, flush_every=100)
        labels = ["output_shaper:stratum:opus|new_user_ask|m|tools"]
        result = recorder.record_from_labels(labels, 150)
        assert result is True

    def test_record_from_labels_control_arm(self, tmp_path: Path) -> None:
        p = tmp_path / "savings.json"
        recorder = SavingsRecorder(p, flush_every=100)
        labels = ["output_shaper:control:sonnet|mechanical|s|tools"]
        result = recorder.record_from_labels(labels, 200)
        assert result is True

    def test_record_from_labels_no_match(self, tmp_path: Path) -> None:
        p = tmp_path / "savings.json"
        recorder = SavingsRecorder(p, flush_every=100)
        labels = ["some_other:label", "transforms_applied"]
        result = recorder.record_from_labels(labels, 150)
        assert result is False

    def test_record_from_labels_empty_labels(self, tmp_path: Path) -> None:
        p = tmp_path / "savings.json"
        recorder = SavingsRecorder(p, flush_every=100)
        result = recorder.record_from_labels([], 150)
        assert result is False

    def test_record_from_labels_none_labels(self, tmp_path: Path) -> None:
        p = tmp_path / "savings.json"
        recorder = SavingsRecorder(p, flush_every=100)
        result = recorder.record_from_labels(None, 150)
        assert result is False

    def test_flush_writes_to_disk(self, tmp_path: Path) -> None:
        p = tmp_path / "savings.json"
        recorder = SavingsRecorder(p, flush_every=100)
        labels = ["output_shaper:stratum:k1"]
        recorder.record_from_labels(labels, 150)
        with patch("headroom.paths.process_is_stateless", return_value=False):
            recorder.flush()
        assert p.exists()
        data = json.loads(p.read_text())
        assert "treatment" in data
        assert "k1" in data["treatment"]

    def test_flush_stateless_skips_write(self, tmp_path: Path) -> None:
        p = tmp_path / "savings.json"
        recorder = SavingsRecorder(p, flush_every=100)
        labels = ["output_shaper:stratum:k1"]
        recorder.record_from_labels(labels, 150)
        with patch("headroom.paths.process_is_stateless", return_value=True):
            recorder.flush()
        # no crash is sufficient; file may or may not exist but stateless skips write
        assert recorder._since_flush == 0

    def test_estimate_returns_ledger_estimate(self, tmp_path: Path) -> None:
        p = tmp_path / "savings.json"
        recorder = SavingsRecorder(p)
        recorder._ledger.baseline.observe("k1", 200)
        recorder._ledger.record("treatment", "k1", 150)
        est = recorder.estimate()
        assert isinstance(est, SavingsEstimate)
        assert est.n_requests == 1


class TestLabelHelpers:
    def test_stratum_label_treatment_prefix(self) -> None:
        label = stratum_label("treatment", "opus|ask|m|tools")
        assert label == "output_shaper:stratum:opus|ask|m|tools"

    def test_stratum_label_control_prefix(self) -> None:
        label = stratum_label("control", "sonnet|mechanical|s|notools")
        assert label == "output_shaper:control:sonnet|mechanical|s|notools"

    def test_parse_stratum_label_treatment(self) -> None:
        result = parse_stratum_label("output_shaper:stratum:opus|ask|m|tools")
        assert result == ("treatment", "opus|ask|m|tools")

    def test_parse_stratum_label_control(self) -> None:
        result = parse_stratum_label("output_shaper:control:sonnet|mechanical|s|notools")
        assert result == ("control", "sonnet|mechanical|s|notools")

    def test_parse_stratum_label_non_matching(self) -> None:
        assert parse_stratum_label("some_other:label") is None
        assert parse_stratum_label("") is None
        assert parse_stratum_label("output_shaper:stratum") is None  # prefix followed by nothing

    def test_round_trip(self) -> None:
        arm, key = "treatment", "opus|new_user_ask|m|tools"
        label = stratum_label(arm, key)
        parsed = parse_stratum_label(label)
        assert parsed == (arm, key)


@pytest.fixture(autouse=True)
def _reset_recorder_singleton() -> None:
    import headroom.proxy.output_savings as _os

    old = _os._RECORDER
    _os._RECORDER = None
    yield
    _os._RECORDER = old


class TestGetRecorder:
    def test_returns_savings_recorder(self, tmp_path: Path) -> None:
        with patch("headroom.paths.workspace_dir", return_value=tmp_path):
            recorder = get_recorder()
            assert isinstance(recorder, SavingsRecorder)

    def test_singleton(self, tmp_path: Path) -> None:
        with patch("headroom.paths.workspace_dir", return_value=tmp_path):
            r1 = get_recorder()
            r2 = get_recorder()
            assert r1 is r2


class TestEchoRatio:
    def test_output_shorter_than_n(self) -> None:
        assert echo_ratio("short", "some context here", n=8) == 0.0

    def test_no_overlap(self) -> None:
        output = "the quick brown fox jumps over the lazy dog"
        context = "completely unrelated text here for context"
        assert echo_ratio(output, context, n=3) == 0.0

    def test_complete_repeat(self) -> None:
        text = "a b c d e f g h i j"
        assert echo_ratio(text, text, n=3) == 1.0

    def test_fractional_overlap(self) -> None:
        output = "the cat sat on the mat with a hat"
        context = "the cat sat on the mat and then left"
        ratio = echo_ratio(output, context, n=3)
        assert 0.0 < ratio < 1.0

    def test_zero_ratio_on_empty_context(self) -> None:
        output = "a b c d e f g h i j"
        assert echo_ratio(output, "", n=3) == 0.0


class TestAccum:
    def test_add_updates_stats(self) -> None:
        a = _Accum()
        a.add(10.0)
        a.add(20.0)
        assert a.n == 2
        assert a.sum == 30.0
        assert a.mean == 15.0

    def test_var_single_sample(self) -> None:
        a = _Accum()
        a.add(10.0)
        assert a.var == 0.0

    def test_var_two_samples(self) -> None:
        a = _Accum()
        a.add(10.0)
        a.add(20.0)
        assert a.var == 50.0  # sample variance

    def test_merge(self) -> None:
        a = _Accum()
        a.add(10.0)
        a.add(20.0)
        b = _Accum()
        b.add(30.0)
        a.merge(b)
        assert a.n == 3
        assert a.sum == 60.0

    def test_to_dict_from_dict_round_trip(self) -> None:
        a = _Accum()
        a.add(10.0)
        a.add(20.0)
        restored = _Accum.from_dict(a.to_dict())
        assert restored.n == 2
        assert restored.sum == 30.0
        assert restored.sumsq == 500.0

    def test_empty_accum(self) -> None:
        a = _Accum()
        assert a.n == 0
        assert a.sum == 0.0
        assert a.mean == 0.0
        assert a.var == 0.0
