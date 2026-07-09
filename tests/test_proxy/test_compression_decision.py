from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from headroom.proxy.compression_decision import CompressionDecision


class TestCompressionDecisionDecide:
    def test_bypass_header(self) -> None:
        config = MagicMock(optimize=True)
        usage_reporter = MagicMock(should_compress=True)
        decision = CompressionDecision.decide(
            headers={"x-headroom-bypass": "true"},
            config=config,
            usage_reporter=usage_reporter,
            messages=[{"role": "user", "content": "hello"}],
        )
        assert decision.should_compress is False
        assert decision.passthrough_reason == "bypass_header"
        assert decision.bypass_header_set is True

    def test_compression_disabled(self) -> None:
        config = MagicMock(optimize=False)
        usage_reporter = MagicMock(should_compress=True)
        decision = CompressionDecision.decide(
            headers={},
            config=config,
            usage_reporter=usage_reporter,
            messages=[{"role": "user", "content": "hello"}],
        )
        assert decision.should_compress is False
        assert decision.passthrough_reason == "compression_disabled"
        assert decision.config_optimize_enabled is False

    def test_no_messages_none(self) -> None:
        config = MagicMock(optimize=True)
        usage_reporter = MagicMock(should_compress=True)
        decision = CompressionDecision.decide(
            headers={},
            config=config,
            usage_reporter=usage_reporter,
            messages=None,
        )
        assert decision.should_compress is False
        assert decision.passthrough_reason == "no_messages"
        assert decision.has_messages is False

    def test_no_messages_empty_list(self) -> None:
        config = MagicMock(optimize=True)
        usage_reporter = MagicMock(should_compress=True)
        decision = CompressionDecision.decide(
            headers={},
            config=config,
            usage_reporter=usage_reporter,
            messages=[],
        )
        assert decision.should_compress is False
        assert decision.passthrough_reason == "no_messages"
        assert decision.has_messages is False

    def test_license_denied(self) -> None:
        config = MagicMock(optimize=True)
        usage_reporter = MagicMock(should_compress=False)
        decision = CompressionDecision.decide(
            headers={},
            config=config,
            usage_reporter=usage_reporter,
            messages=[{"role": "user", "content": "hello"}],
        )
        assert decision.should_compress is False
        assert decision.passthrough_reason == "license_denied"
        assert decision.license_allows is False

    def test_usage_reporter_none_license_allows(self) -> None:
        config = MagicMock(optimize=True)
        decision = CompressionDecision.decide(
            headers={},
            config=config,
            usage_reporter=None,
            messages=[{"role": "user", "content": "hello"}],
        )
        assert decision.should_compress is True
        assert decision.passthrough_reason is None
        assert decision.license_allows is True

    def test_compresses_when_all_ok(self) -> None:
        config = MagicMock(optimize=True)
        usage_reporter = MagicMock(should_compress=True)
        decision = CompressionDecision.decide(
            headers={},
            config=config,
            usage_reporter=usage_reporter,
            messages=[{"role": "user", "content": "hello"}],
        )
        assert decision.should_compress is True
        assert decision.passthrough_reason is None
        assert decision.bypass_header_set is False
        assert decision.config_optimize_enabled is True
        assert decision.license_allows is True
        assert decision.has_messages is True

    def test_bypass_takes_highest_precedence(self) -> None:
        config = MagicMock(optimize=False)
        decision = CompressionDecision.decide(
            headers={"x-headroom-bypass": "true"},
            config=config,
            usage_reporter=None,
            messages=[],
        )
        assert decision.should_compress is False
        assert decision.passthrough_reason == "bypass_header"
        assert decision.config_optimize_enabled is False
        assert decision.has_messages is False

    def test_passthrough_mode_triggers_bypass(self) -> None:
        config = MagicMock(optimize=True)
        usage_reporter = MagicMock(should_compress=True)
        decision = CompressionDecision.decide(
            headers={"x-headroom-mode": "passthrough"},
            config=config,
            usage_reporter=usage_reporter,
            messages=[{"role": "user", "content": "hello"}],
        )
        assert decision.should_compress is False
        assert decision.passthrough_reason == "bypass_header"
        assert decision.bypass_header_set is True

    def test_config_without_optimize_attr_defaults_false(self) -> None:
        config = object()
        usage_reporter = MagicMock(should_compress=True)
        decision = CompressionDecision.decide(
            headers={},
            config=config,
            usage_reporter=usage_reporter,
            messages=[{"role": "user", "content": "hello"}],
        )
        assert decision.should_compress is False
        assert decision.passthrough_reason == "compression_disabled"
        assert decision.config_optimize_enabled is False


class TestCompressionDecisionApplyToTags:
    def test_stamps_passthrough_reason_when_not_compressing(self) -> None:
        decision = CompressionDecision(
            should_compress=False,
            passthrough_reason="no_messages",
            bypass_header_set=False,
            config_optimize_enabled=True,
            license_allows=True,
            has_messages=False,
        )
        tags: dict[str, str] = {}
        decision.apply_to_tags(tags)
        assert tags == {"passthrough_reason": "no_messages"}

    def test_noop_when_compressing(self) -> None:
        decision = CompressionDecision(
            should_compress=True,
            passthrough_reason=None,
            bypass_header_set=False,
            config_optimize_enabled=True,
            license_allows=True,
            has_messages=True,
        )
        tags: dict[str, str] = {}
        decision.apply_to_tags(tags)
        assert tags == {}

    def test_preserves_existing_tags(self) -> None:
        decision = CompressionDecision(
            should_compress=False,
            passthrough_reason="license_denied",
            bypass_header_set=False,
            config_optimize_enabled=True,
            license_allows=False,
            has_messages=True,
        )
        tags: dict[str, str] = {"client": "test"}
        decision.apply_to_tags(tags)
        assert tags == {"client": "test", "passthrough_reason": "license_denied"}


class TestCompressionDecisionConstruct:
    def test_frozen(self) -> None:
        decision = CompressionDecision(
            should_compress=True,
            passthrough_reason=None,
            bypass_header_set=False,
            config_optimize_enabled=True,
            license_allows=True,
            has_messages=True,
        )
        with pytest.raises(AttributeError):
            decision.should_compress = False  # type: ignore[misc]

    def test_value_equality(self) -> None:
        a = CompressionDecision(
            should_compress=True,
            passthrough_reason=None,
            bypass_header_set=False,
            config_optimize_enabled=True,
            license_allows=True,
            has_messages=True,
        )
        b = CompressionDecision(
            should_compress=True,
            passthrough_reason=None,
            bypass_header_set=False,
            config_optimize_enabled=True,
            license_allows=True,
            has_messages=True,
        )
        assert a == b

    def test_value_inequality(self) -> None:
        a = CompressionDecision(
            should_compress=True,
            passthrough_reason=None,
            bypass_header_set=False,
            config_optimize_enabled=True,
            license_allows=True,
            has_messages=True,
        )
        b = CompressionDecision(
            should_compress=False,
            passthrough_reason="no_messages",
            bypass_header_set=False,
            config_optimize_enabled=True,
            license_allows=True,
            has_messages=False,
        )
        assert a != b
