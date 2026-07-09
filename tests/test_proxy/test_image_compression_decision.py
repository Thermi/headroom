from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from headroom.proxy.image_compression_decision import ImageCompressionDecision


class TestImageCompressionDecisionDecide:
    def test_bypass_header(self) -> None:
        config = MagicMock(image_optimize=True)
        decision = ImageCompressionDecision.decide(
            headers={"x-headroom-bypass": "true"},
            config=config,
            messages=[{"role": "user", "content": [{"type": "image"}]}],
        )
        assert decision.should_compress is False
        assert decision.passthrough_reason == "bypass_header"
        assert decision.bypass_header_set is True

    def test_image_optimize_disabled(self) -> None:
        config = MagicMock(image_optimize=False)
        decision = ImageCompressionDecision.decide(
            headers={},
            config=config,
            messages=[{"role": "user", "content": [{"type": "image"}]}],
        )
        assert decision.should_compress is False
        assert decision.passthrough_reason == "image_optimize_disabled"
        assert decision.image_optimize_enabled is False

    def test_no_messages_none(self) -> None:
        config = MagicMock(image_optimize=True)
        decision = ImageCompressionDecision.decide(
            headers={},
            config=config,
            messages=None,
        )
        assert decision.should_compress is False
        assert decision.passthrough_reason == "no_messages"
        assert decision.has_messages is False

    def test_no_messages_empty_list(self) -> None:
        config = MagicMock(image_optimize=True)
        decision = ImageCompressionDecision.decide(
            headers={},
            config=config,
            messages=[],
        )
        assert decision.should_compress is False
        assert decision.passthrough_reason == "no_messages"
        assert decision.has_messages is False

    def test_compresses_when_all_ok(self) -> None:
        config = MagicMock(image_optimize=True)
        decision = ImageCompressionDecision.decide(
            headers={},
            config=config,
            messages=[{"role": "user", "content": [{"type": "image"}]}],
        )
        assert decision.should_compress is True
        assert decision.passthrough_reason is None
        assert decision.bypass_header_set is False
        assert decision.image_optimize_enabled is True
        assert decision.has_messages is True

    def test_bypass_takes_highest_precedence(self) -> None:
        config = MagicMock(image_optimize=False)
        decision = ImageCompressionDecision.decide(
            headers={"x-headroom-bypass": "true"},
            config=config,
            messages=[],
        )
        assert decision.should_compress is False
        assert decision.passthrough_reason == "bypass_header"
        assert decision.image_optimize_enabled is False
        assert decision.has_messages is False

    def test_passthrough_mode_triggers_bypass(self) -> None:
        config = MagicMock(image_optimize=True)
        decision = ImageCompressionDecision.decide(
            headers={"x-headroom-mode": "passthrough"},
            config=config,
            messages=[{"role": "user", "content": [{"type": "image"}]}],
        )
        assert decision.should_compress is False
        assert decision.passthrough_reason == "bypass_header"
        assert decision.bypass_header_set is True

    def test_config_without_image_optimize_attr_defaults_false(self) -> None:
        config = object()
        decision = ImageCompressionDecision.decide(
            headers={},
            config=config,
            messages=[{"role": "user", "content": [{"type": "image"}]}],
        )
        assert decision.should_compress is False
        assert decision.passthrough_reason == "image_optimize_disabled"
        assert decision.image_optimize_enabled is False


class TestImageCompressionDecisionApplyToTags:
    def test_stamps_skip_reason_when_not_compressing(self) -> None:
        decision = ImageCompressionDecision(
            should_compress=False,
            passthrough_reason="image_optimize_disabled",
            bypass_header_set=False,
            image_optimize_enabled=False,
            has_messages=True,
        )
        tags: dict[str, str] = {}
        decision.apply_to_tags(tags)
        assert tags == {"image_skip_reason": "image_optimize_disabled"}

    def test_noop_when_compressing(self) -> None:
        decision = ImageCompressionDecision(
            should_compress=True,
            passthrough_reason=None,
            bypass_header_set=False,
            image_optimize_enabled=True,
            has_messages=True,
        )
        tags: dict[str, str] = {}
        decision.apply_to_tags(tags)
        assert tags == {}

    def test_preserves_existing_tags(self) -> None:
        decision = ImageCompressionDecision(
            should_compress=False,
            passthrough_reason="no_messages",
            bypass_header_set=False,
            image_optimize_enabled=True,
            has_messages=False,
        )
        tags: dict[str, str] = {"client": "test"}
        decision.apply_to_tags(tags)
        assert tags == {"client": "test", "image_skip_reason": "no_messages"}


class TestImageCompressionDecisionConstruct:
    def test_frozen(self) -> None:
        decision = ImageCompressionDecision(
            should_compress=True,
            passthrough_reason=None,
            bypass_header_set=False,
            image_optimize_enabled=True,
            has_messages=True,
        )
        with pytest.raises(AttributeError):
            decision.should_compress = False  # type: ignore[misc]

    def test_value_equality(self) -> None:
        a = ImageCompressionDecision(
            should_compress=True,
            passthrough_reason=None,
            bypass_header_set=False,
            image_optimize_enabled=True,
            has_messages=True,
        )
        b = ImageCompressionDecision(
            should_compress=True,
            passthrough_reason=None,
            bypass_header_set=False,
            image_optimize_enabled=True,
            has_messages=True,
        )
        assert a == b

    def test_value_inequality(self) -> None:
        a = ImageCompressionDecision(
            should_compress=True,
            passthrough_reason=None,
            bypass_header_set=False,
            image_optimize_enabled=True,
            has_messages=True,
        )
        b = ImageCompressionDecision(
            should_compress=False,
            passthrough_reason="no_messages",
            bypass_header_set=False,
            image_optimize_enabled=True,
            has_messages=False,
        )
        assert a != b
