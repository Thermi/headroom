from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from headroom.proxy.memory_decision import MemoryDecision


class TestMemoryDecisionDecide:
    def test_bypass_header(self) -> None:
        headers = {"x-headroom-bypass": "true"}
        decision = MemoryDecision.decide(
            headers=headers,
            memory_handler=MagicMock(),
            memory_user_id="user123",
            mode_name="auto_tail",
        )
        assert decision.inject is False
        assert decision.skip_reason == "bypass_header"
        assert decision.bypass_header_set is True

    def test_no_handler(self) -> None:
        decision = MemoryDecision.decide(
            headers={},
            memory_handler=None,
            memory_user_id="user123",
            mode_name="auto_tail",
        )
        assert decision.inject is False
        assert decision.skip_reason == "no_handler"
        assert decision.memory_handler_present is False

    def test_no_user_id(self) -> None:
        decision = MemoryDecision.decide(
            headers={},
            memory_handler=MagicMock(),
            memory_user_id=None,
            mode_name="auto_tail",
        )
        assert decision.inject is False
        assert decision.skip_reason == "no_user_id"
        assert decision.memory_user_id_present is False

    def test_no_user_id_empty_string(self) -> None:
        decision = MemoryDecision.decide(
            headers={},
            memory_handler=MagicMock(),
            memory_user_id="",
            mode_name="auto_tail",
        )
        assert decision.inject is False
        assert decision.skip_reason == "no_user_id"
        assert decision.memory_user_id_present is False

    def test_mode_disabled(self) -> None:
        decision = MemoryDecision.decide(
            headers={},
            memory_handler=MagicMock(),
            memory_user_id="user123",
            mode_name="disabled",
        )
        assert decision.inject is False
        assert decision.skip_reason == "mode_disabled"
        assert decision.mode_name == "disabled"

    def test_mode_tool(self) -> None:
        decision = MemoryDecision.decide(
            headers={},
            memory_handler=MagicMock(),
            memory_user_id="user123",
            mode_name="tool",
        )
        assert decision.inject is False
        assert decision.skip_reason == "mode_tool"
        assert decision.mode_name == "tool"

    def test_injects_when_auto_tail(self) -> None:
        decision = MemoryDecision.decide(
            headers={},
            memory_handler=MagicMock(),
            memory_user_id="user123",
            mode_name="auto_tail",
        )
        assert decision.inject is True
        assert decision.skip_reason is None
        assert decision.bypass_header_set is False
        assert decision.memory_handler_present is True
        assert decision.memory_user_id_present is True
        assert decision.mode_name == "auto_tail"

    def test_injects_on_unknown_mode(self) -> None:
        decision = MemoryDecision.decide(
            headers={},
            memory_handler=MagicMock(),
            memory_user_id="user123",
            mode_name="unknown_mode",
        )
        assert decision.inject is True
        assert decision.skip_reason is None

    def test_bypass_takes_highest_precedence(self) -> None:
        decision = MemoryDecision.decide(
            headers={"x-headroom-bypass": "true"},
            memory_handler=None,
            memory_user_id=None,
            mode_name="disabled",
        )
        assert decision.inject is False
        assert decision.skip_reason == "bypass_header"
        assert decision.bypass_header_set is True
        assert decision.memory_handler_present is False
        assert decision.memory_user_id_present is False
        assert decision.mode_name == "disabled"

    def test_passthrough_mode_triggers_bypass(self) -> None:
        decision = MemoryDecision.decide(
            headers={"x-headroom-mode": "passthrough"},
            memory_handler=MagicMock(),
            memory_user_id="user123",
            mode_name="auto_tail",
        )
        assert decision.inject is False
        assert decision.skip_reason == "bypass_header"
        assert decision.bypass_header_set is True


class TestMemoryDecisionApplyToTags:
    def test_stamps_skip_reason_when_not_injecting(self) -> None:
        decision = MemoryDecision(
            inject=False,
            skip_reason="no_handler",
            bypass_header_set=False,
            memory_handler_present=False,
            memory_user_id_present=True,
            mode_name="auto_tail",
        )
        tags: dict[str, str] = {}
        decision.apply_to_tags(tags)
        assert tags == {"memory_skip_reason": "no_handler"}

    def test_noop_when_injecting(self) -> None:
        decision = MemoryDecision(
            inject=True,
            skip_reason=None,
            bypass_header_set=False,
            memory_handler_present=True,
            memory_user_id_present=True,
            mode_name="auto_tail",
        )
        tags: dict[str, str] = {}
        decision.apply_to_tags(tags)
        assert tags == {}

    def test_preserves_existing_tags(self) -> None:
        decision = MemoryDecision(
            inject=False,
            skip_reason="mode_disabled",
            bypass_header_set=False,
            memory_handler_present=True,
            memory_user_id_present=True,
            mode_name="disabled",
        )
        tags: dict[str, str] = {"client": "test"}
        decision.apply_to_tags(tags)
        assert tags == {"client": "test", "memory_skip_reason": "mode_disabled"}


class TestMemoryDecisionConstruct:
    def test_frozen(self) -> None:
        decision = MemoryDecision(
            inject=True,
            skip_reason=None,
            bypass_header_set=False,
            memory_handler_present=True,
            memory_user_id_present=True,
            mode_name="auto_tail",
        )
        with pytest.raises(AttributeError):
            decision.inject = False  # type: ignore[misc]

    def test_value_equality(self) -> None:
        a = MemoryDecision(
            inject=True,
            skip_reason=None,
            bypass_header_set=False,
            memory_handler_present=True,
            memory_user_id_present=True,
            mode_name="auto_tail",
        )
        b = MemoryDecision(
            inject=True,
            skip_reason=None,
            bypass_header_set=False,
            memory_handler_present=True,
            memory_user_id_present=True,
            mode_name="auto_tail",
        )
        assert a == b

    def test_value_inequality(self) -> None:
        a = MemoryDecision(
            inject=True,
            skip_reason=None,
            bypass_header_set=False,
            memory_handler_present=True,
            memory_user_id_present=True,
            mode_name="auto_tail",
        )
        b = MemoryDecision(
            inject=False,
            skip_reason="no_handler",
            bypass_header_set=False,
            memory_handler_present=False,
            memory_user_id_present=True,
            mode_name="auto_tail",
        )
        assert a != b
