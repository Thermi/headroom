from __future__ import annotations

from enum import Enum

import pytest

from headroom.proxy import runtime_env
from headroom.proxy.output_shaper import (
    _VERBOSITY_LEVELS,
    LEGACY_THINKING_FLOOR,
    OutputShaperSettings,
    ShapeResult,
    TurnKind,
    _replace_or_append_steering_block,
    apply_openai_responses_verbosity_steering,
    apply_verbosity_steering,
    classify_openai_responses_input,
    classify_turn,
    resolve_verbosity_level,
    route_effort,
    route_openai_reasoning_effort,
    route_openai_text_verbosity,
    shape_openai_responses_request,
    shape_request,
    steering_text,
)


@pytest.fixture(autouse=True)
def _clean_runtime() -> None:
    runtime_env.clear_overrides()
    yield


class TestTurnKind:
    def test_values(self) -> None:
        assert TurnKind.NEW_USER_ASK.value == "new_user_ask"
        assert TurnKind.MECHANICAL_CONTINUATION.value == "mechanical_continuation"
        assert TurnKind.ERROR_CONTINUATION.value == "error_continuation"
        assert TurnKind.UNKNOWN.value == "unknown"

    def test_is_enum(self) -> None:
        assert issubclass(TurnKind, Enum)


class TestOutputShaperSettingsFromEnv:
    def test_enabled_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "true")
        settings = OutputShaperSettings.from_env()
        assert settings.enabled is True

    def test_enabled_false_by_default(self) -> None:
        settings = OutputShaperSettings.from_env()
        assert settings.enabled is False

    def test_verbosity_level_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_VERBOSITY_LEVEL", "3")
        settings = OutputShaperSettings.from_env()
        assert settings.verbosity_level == 3

    def test_verbosity_level_default(self) -> None:
        settings = OutputShaperSettings.from_env()
        assert settings.verbosity_level == 2

    def test_verbosity_level_invalid_clamps_to_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_VERBOSITY_LEVEL", "99")
        settings = OutputShaperSettings.from_env()
        assert settings.verbosity_level == 4

    def test_verbosity_level_negative_clamps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_VERBOSITY_LEVEL", "-5")
        settings = OutputShaperSettings.from_env()
        assert settings.verbosity_level == 0

    def test_verbosity_level_non_int_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_VERBOSITY_LEVEL", "not-a-number")
        settings = OutputShaperSettings.from_env()
        assert settings.verbosity_level == 2

    def test_effort_router_enabled_by_default(self) -> None:
        settings = OutputShaperSettings.from_env()
        assert settings.effort_router_enabled is True

    def test_effort_router_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_EFFORT_ROUTER", "false")
        settings = OutputShaperSettings.from_env()
        assert settings.effort_router_enabled is False

    def test_mechanical_effort_default(self) -> None:
        settings = OutputShaperSettings.from_env()
        assert settings.mechanical_effort == "low"

    def test_mechanical_effort_invalid_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_MECHANICAL_EFFORT", "ultra")
        settings = OutputShaperSettings.from_env()
        assert settings.mechanical_effort == "low"


class TestResolveVerbosityLevel:
    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_VERBOSITY_LEVEL", "4")
        settings = OutputShaperSettings.from_env()
        level, source = resolve_verbosity_level(settings)
        assert level == 4
        assert source == "env"

    def test_default_when_no_env_or_files(self) -> None:
        settings = OutputShaperSettings(verbosity_level=2)
        level, source = resolve_verbosity_level(settings)
        assert level == 2
        assert source == "default"


class TestClassifyTurn:
    def test_empty_messages(self) -> None:
        assert classify_turn([]) == TurnKind.UNKNOWN

    def test_last_message_not_user(self) -> None:
        messages = [{"role": "assistant", "content": "hello"}]
        assert classify_turn(messages) == TurnKind.UNKNOWN

    def test_string_content_new_ask(self) -> None:
        messages = [{"role": "user", "content": "write a function"}]
        assert classify_turn(messages) == TurnKind.NEW_USER_ASK

    def test_string_content_empty(self) -> None:
        messages = [{"role": "user", "content": ""}]
        assert classify_turn(messages) == TurnKind.UNKNOWN

    def test_string_content_whitespace_only(self) -> None:
        messages = [{"role": "user", "content": "   "}]
        assert classify_turn(messages) == TurnKind.UNKNOWN

    def test_tool_result_mechanical(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "file contents here", "is_error": False},
                ],
            }
        ]
        assert classify_turn(messages) == TurnKind.MECHANICAL_CONTINUATION

    def test_tool_result_with_error(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "error!", "is_error": True},
                ],
            }
        ]
        assert classify_turn(messages) == TurnKind.ERROR_CONTINUATION

    def test_text_block(self) -> None:
        messages = [{"role": "user", "content": [{"type": "text", "text": "do something"}]}]
        assert classify_turn(messages) == TurnKind.NEW_USER_ASK

    def test_image_block(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "source": {"type": "base64", "data": "..."}}],
            }
        ]
        assert classify_turn(messages) == TurnKind.NEW_USER_ASK

    def test_document_block(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [{"type": "document", "source": {"type": "text", "data": "doc"}}],
            }
        ]
        assert classify_turn(messages) == TurnKind.NEW_USER_ASK

    def test_mixed_tool_result_and_text(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "result"},
                    {"type": "text", "text": "also a new ask"},
                ],
            }
        ]
        assert classify_turn(messages) == TurnKind.NEW_USER_ASK

    def test_unknown_block_types_ignored(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "unknown_type", "data": "something"},
                ],
            }
        ]
        assert classify_turn(messages) == TurnKind.UNKNOWN

    def test_non_dict_block_returns_unknown(self) -> None:
        messages = [{"role": "user", "content": ["just a string"]}]
        assert classify_turn(messages) == TurnKind.UNKNOWN

    def test_content_not_list_or_str(self) -> None:
        messages = [{"role": "user", "content": 42}]
        assert classify_turn(messages) == TurnKind.UNKNOWN

    def test_last_not_dict(self) -> None:
        messages = ["not a dict"]
        assert classify_turn(messages) == TurnKind.UNKNOWN


class TestClassifyOpenAiResponsesInput:
    def test_string_input_new_ask(self) -> None:
        assert classify_openai_responses_input("hello") == TurnKind.NEW_USER_ASK

    def test_string_input_empty(self) -> None:
        assert classify_openai_responses_input("") == TurnKind.UNKNOWN

    def test_empty_list(self) -> None:
        assert classify_openai_responses_input([]) == TurnKind.UNKNOWN

    def test_not_list_or_str(self) -> None:
        assert classify_openai_responses_input(42) == TurnKind.UNKNOWN

    def test_tool_output_items_mechanical(self) -> None:
        items = [
            {"type": "function_call_output", "output": "result"},
            {"type": "custom_tool_call_output", "output": "data"},
        ]
        assert classify_openai_responses_input(items) == TurnKind.MECHANICAL_CONTINUATION

    def test_user_items_new_ask(self) -> None:
        items = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
                "type": "message",
            },
        ]
        assert classify_openai_responses_input(items) == TurnKind.NEW_USER_ASK

    def test_mixed_tool_and_user(self) -> None:
        items = [
            {"type": "function_call_output", "output": "result"},
            {"role": "user", "content": "hello"},
        ]
        assert classify_openai_responses_input(items) == TurnKind.NEW_USER_ASK

    def test_input_text_type(self) -> None:
        items = [{"type": "input_text", "text": "hello"}]
        assert classify_openai_responses_input(items) == TurnKind.NEW_USER_ASK

    def test_input_image_type(self) -> None:
        items = [{"type": "input_image", "image": "..."}]
        assert classify_openai_responses_input(items) == TurnKind.NEW_USER_ASK

    def test_known_non_user_types_ignored(self) -> None:
        items = [{"type": "message", "role": "assistant", "content": "response"}]
        assert classify_openai_responses_input(items) == TurnKind.UNKNOWN

    def test_unknown_types_without_tool_output(self) -> None:
        items = [{"type": "some_random_type"}]
        assert classify_openai_responses_input(items) == TurnKind.UNKNOWN

    def test_user_with_input_file(self) -> None:
        items = [
            {
                "role": "user",
                "content": [{"type": "input_file", "filename": "test.py"}],
                "type": "message",
            }
        ]
        assert classify_openai_responses_input(items) == TurnKind.NEW_USER_ASK

    def test_user_with_input_image(self) -> None:
        items = [
            {
                "role": "user",
                "content": [{"type": "input_image", "image": "..."}],
                "type": "message",
            }
        ]
        assert classify_openai_responses_input(items) == TurnKind.NEW_USER_ASK

    def test_tool_output_with_unknown_still_unknown(self) -> None:
        items = [
            {"type": "function_call_output", "output": "result"},
            {"type": "unknown_type"},
        ]
        assert classify_openai_responses_input(items) == TurnKind.UNKNOWN


class TestSteeringText:
    def test_level_0_none(self) -> None:
        assert steering_text(0) is None

    def test_level_5_none(self) -> None:
        assert steering_text(5) is None

    def test_level_1_has_sentinel(self) -> None:
        text = steering_text(1)
        assert text is not None
        assert text.startswith("<headroom_output_shaping>")
        assert text.endswith("</headroom_output_shaping>")

    def test_level_2_content(self) -> None:
        text = steering_text(2)
        assert text is not None
        assert "Skip preamble" in text

    def test_level_3_content(self) -> None:
        text = steering_text(3)
        assert text is not None
        assert "Skip preamble" in text

    def test_level_4_content(self) -> None:
        text = steering_text(4)
        assert text is not None
        assert "Minimum tokens" in text


class TestApplyVerbositySteering:
    def test_no_system_creates_block(self) -> None:
        body: dict = {}
        assert apply_verbosity_steering(body, 2) is True
        assert len(body["system"]) == 1
        assert body["system"][0]["type"] == "text"
        assert "<headroom_output_shaping>" in body["system"][0]["text"]

    def test_string_system_converts_to_list(self) -> None:
        body: dict = {"system": "Be helpful."}
        assert apply_verbosity_steering(body, 1) is True
        assert isinstance(body["system"], list)
        assert len(body["system"]) == 2
        assert body["system"][0]["text"] == "Be helpful."
        assert "<headroom_output_shaping>" in body["system"][1]["text"]

    def test_list_with_existing_steering_replaces(self) -> None:
        body: dict = {
            "system": [
                {"type": "text", "text": "original"},
                {
                    "type": "text",
                    "text": "<headroom_output_shaping>\nbe concise\n</headroom_output_shaping>",
                },
            ]
        }
        assert apply_verbosity_steering(body, 1) is True
        assert len(body["system"]) == 2
        assert "<headroom_output_shaping>" in body["system"][1]["text"]

    def test_list_same_level_no_op(self) -> None:
        existing = steering_text(1)
        assert existing is not None
        body: dict = {
            "system": [
                {"type": "text", "text": existing},
            ]
        }
        assert apply_verbosity_steering(body, 1) is False

    def test_list_without_steering_appends(self) -> None:
        body: dict = {
            "system": [
                {"type": "text", "text": "original block"},
            ]
        }
        assert apply_verbosity_steering(body, 2) is True
        assert len(body["system"]) == 2
        assert body["system"][1]["type"] == "text"
        assert "<headroom_output_shaping>" in body["system"][1]["text"]

    def test_invalid_system_type_returns_false(self) -> None:
        body: dict = {"system": 42}
        assert apply_verbosity_steering(body, 2) is False

    def test_level_0_returns_false(self) -> None:
        body: dict = {}
        assert apply_verbosity_steering(body, 0) is False
        assert "system" not in body


class TestApplyOpenaiResponsesVerbositySteering:
    def test_no_instructions_creates(self) -> None:
        body: dict = {}
        assert apply_openai_responses_verbosity_steering(body, 2) is True
        assert "<headroom_output_shaping>" in body["instructions"]

    def test_existing_appends_steering(self) -> None:
        body: dict = {"instructions": "Be brief."}
        assert apply_openai_responses_verbosity_steering(body, 2) is True
        assert "<headroom_output_shaping>" in body["instructions"]
        assert body["instructions"].startswith("Be brief.")

    def test_existing_replaces_steering(self) -> None:
        body: dict = {
            "instructions": "Be nice. <headroom_output_shaping>\nold\n</headroom_output_shaping>"
        }
        assert apply_openai_responses_verbosity_steering(body, 2) is True
        assert "old" not in body["instructions"]

    def test_non_string_instructions_returns_false(self) -> None:
        body: dict = {"instructions": ["list", "of", "strings"]}
        assert apply_openai_responses_verbosity_steering(body, 2) is False

    def test_level_0_returns_false(self) -> None:
        body: dict = {}
        assert apply_openai_responses_verbosity_steering(body, 0) is False
        assert "instructions" not in body


class TestReplaceOrAppendSteeringBlock:
    def test_appends_when_no_sentinel(self) -> None:
        updated, changed = _replace_or_append_steering_block("original", "<block>new</block>")
        assert changed is True
        assert "<block>new</block>" in updated
        assert "original" in updated

    def test_replaces_existing_sentinel(self) -> None:
        existing = "prefix <headroom_output_shaping>\nold\n</headroom_output_shaping> suffix"
        updated, changed = _replace_or_append_steering_block(
            existing, "<headroom_output_shaping>\nnew\n</headroom_output_shaping>"
        )
        assert changed is True
        assert "old" not in updated
        assert "new" in updated

    def test_no_change_when_content_identical(self) -> None:
        text = "stuff\n\n<headroom_output_shaping>\ncontent\n</headroom_output_shaping>"
        updated, changed = _replace_or_append_steering_block(
            text, "<headroom_output_shaping>\ncontent\n</headroom_output_shaping>"
        )
        assert changed is False

    def test_empty_existing_string(self) -> None:
        updated, changed = _replace_or_append_steering_block("", "<block>content</block>")
        assert changed is True
        assert updated == "<block>content</block>"


class TestRouteEffort:
    def test_non_mechanical_returns_empty(self) -> None:
        body: dict = {"output_config": {"effort": "xhigh"}}
        labels = route_effort(body, TurnKind.NEW_USER_ASK, OutputShaperSettings())
        assert labels == []

    def test_mechanical_high_effort_lowered(self) -> None:
        body: dict = {"output_config": {"effort": "xhigh"}}
        labels = route_effort(
            body, TurnKind.MECHANICAL_CONTINUATION, OutputShaperSettings(mechanical_effort="low")
        )
        assert labels == ["output_shaper:effort:xhigh->low"]
        assert body["output_config"]["effort"] == "low"

    def test_mechanical_medium_effort_lowered(self) -> None:
        body: dict = {"output_config": {"effort": "high"}}
        labels = route_effort(
            body, TurnKind.MECHANICAL_CONTINUATION, OutputShaperSettings(mechanical_effort="low")
        )
        assert labels == ["output_shaper:effort:high->low"]

    def test_mechanical_already_low_no_change(self) -> None:
        body: dict = {"output_config": {"effort": "low"}}
        labels = route_effort(
            body, TurnKind.MECHANICAL_CONTINUATION, OutputShaperSettings(mechanical_effort="medium")
        )
        assert labels == []

    def test_legacy_thinking_budget_clamped(self) -> None:
        body: dict = {"thinking": {"type": "enabled", "budget_tokens": 10000}}
        labels = route_effort(body, TurnKind.MECHANICAL_CONTINUATION, OutputShaperSettings())
        assert labels == [f"output_shaper:thinking_budget:10000->{LEGACY_THINKING_FLOOR}"]
        assert body["thinking"]["budget_tokens"] == LEGACY_THINKING_FLOOR

    def test_legacy_thinking_budget_at_floor_not_clamped(self) -> None:
        body: dict = {"thinking": {"type": "enabled", "budget_tokens": LEGACY_THINKING_FLOOR}}
        labels = route_effort(body, TurnKind.MECHANICAL_CONTINUATION, OutputShaperSettings())
        assert labels == []

    def test_no_output_config_no_labels(self) -> None:
        body: dict = {}
        labels = route_effort(body, TurnKind.MECHANICAL_CONTINUATION, OutputShaperSettings())
        assert labels == []

    def test_output_config_not_dict(self) -> None:
        body: dict = {"output_config": "invalid"}
        labels = route_effort(body, TurnKind.MECHANICAL_CONTINUATION, OutputShaperSettings())
        assert labels == []

    def test_effort_not_in_rank(self) -> None:
        body: dict = {"output_config": {"effort": "unknown"}}
        labels = route_effort(body, TurnKind.MECHANICAL_CONTINUATION, OutputShaperSettings())
        assert labels == []


class TestRouteOpenaiReasoningEffort:
    def test_non_mechanical_returns_empty(self) -> None:
        body: dict = {"reasoning": {"effort": "high"}}
        labels = route_openai_reasoning_effort(body, TurnKind.NEW_USER_ASK, OutputShaperSettings())
        assert labels == []

    def test_high_effort_lowered(self) -> None:
        body: dict = {"reasoning": {"effort": "high"}}
        labels = route_openai_reasoning_effort(
            body, TurnKind.MECHANICAL_CONTINUATION, OutputShaperSettings()
        )
        assert labels == ["output_shaper:reasoning_effort:high->low"]
        assert body["reasoning"]["effort"] == "low"

    def test_already_low_no_change(self) -> None:
        body: dict = {"reasoning": {"effort": "low"}}
        labels = route_openai_reasoning_effort(
            body, TurnKind.MECHANICAL_CONTINUATION, OutputShaperSettings()
        )
        assert labels == []

    def test_reasoning_not_dict(self) -> None:
        body: dict = {"reasoning": "string"}
        labels = route_openai_reasoning_effort(
            body, TurnKind.MECHANICAL_CONTINUATION, OutputShaperSettings()
        )
        assert labels == []

    def test_reasoning_not_present(self) -> None:
        body: dict = {}
        labels = route_openai_reasoning_effort(
            body, TurnKind.MECHANICAL_CONTINUATION, OutputShaperSettings()
        )
        assert labels == []


class TestRouteOpenaiTextVerbosity:
    def test_no_text_config_gpt5_creates(self) -> None:
        body: dict = {"model": "gpt-5"}
        labels = route_openai_text_verbosity(body)
        assert labels == ["output_shaper:text_verbosity:unset->low"]
        assert body["text"] == {"verbosity": "low"}

    def test_no_text_config_non_gpt5_no_op(self) -> None:
        body: dict = {"model": "gpt-4o"}
        labels = route_openai_text_verbosity(body)
        assert labels == []

    def test_existing_low_verbosity_no_change(self) -> None:
        body: dict = {"model": "gpt-5", "text": {"verbosity": "low"}}
        labels = route_openai_text_verbosity(body)
        assert labels == []

    def test_existing_high_verbosity_lowered(self) -> None:
        body: dict = {"model": "gpt-5", "text": {"verbosity": "high"}}
        labels = route_openai_text_verbosity(body)
        assert labels == ["output_shaper:text_verbosity:high->low"]
        assert body["text"]["verbosity"] == "low"

    def test_text_config_not_dict(self) -> None:
        body: dict = {"model": "gpt-5", "text": "invalid"}
        labels = route_openai_text_verbosity(body)
        assert labels == []

    def test_verbosity_unset_without_gpt5_no_op(self) -> None:
        body: dict = {"model": "o3", "text": {}}
        labels = route_openai_text_verbosity(body)
        assert labels == []


class TestShapeRequest:
    def test_disabled_returns_noop(self) -> None:
        settings = OutputShaperSettings(enabled=False)
        body: dict = {"messages": [{"role": "user", "content": "hi"}]}
        result = shape_request(body, settings)
        assert result.changed is False
        assert result.labels == []

    def test_enabled_with_verbosity(self) -> None:
        settings = OutputShaperSettings(
            enabled=True, verbosity_level=2, effort_router_enabled=False
        )
        body: dict = {"messages": [{"role": "user", "content": "hi"}]}
        result = shape_request(body, settings)
        assert result.changed is True
        assert "output_shaper:verbosity:L2" in result.labels

    def test_with_effort_router_labels_included(self) -> None:
        settings = OutputShaperSettings(
            enabled=True, verbosity_level=1, effort_router_enabled=True, mechanical_effort="low"
        )
        body: dict = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "content": "ok", "is_error": False}],
                }
            ],
            "output_config": {"effort": "xhigh"},
        }
        result = shape_request(body, settings)
        assert result.changed is True
        assert any("effort" in lbl for lbl in result.labels)

    def test_level_override_takes_precedence(self) -> None:
        settings = OutputShaperSettings(enabled=True, verbosity_level=1)
        body: dict = {"messages": [{"role": "user", "content": "hi"}]}
        result = shape_request(body, settings, level_override=4)
        assert "output_shaper:verbosity:L4" in result.labels

    def test_level_0_skips_verbosity(self) -> None:
        settings = OutputShaperSettings(
            enabled=True, verbosity_level=0, effort_router_enabled=False
        )
        body: dict = {"messages": [{"role": "user", "content": "hi"}]}
        result = shape_request(body, settings)
        assert result.changed is False

    def test_default_settings_when_none(self) -> None:
        body: dict = {"messages": [{"role": "user", "content": "hi"}]}
        result = shape_request(body)
        assert result.changed is False  # disabled by default


class TestShapeOpenaiResponsesRequest:
    def test_disabled_returns_noop(self) -> None:
        settings = OutputShaperSettings(enabled=False)
        body: dict = {"input": "hello"}
        result = shape_openai_responses_request(body, settings)
        assert result.changed is False
        assert result.labels == []

    def test_enabled_with_verbosity(self) -> None:
        settings = OutputShaperSettings(
            enabled=True, verbosity_level=2, effort_router_enabled=False
        )
        body: dict = {"input": "hello"}
        result = shape_openai_responses_request(body, settings)
        assert result.changed is True
        assert "output_shaper:verbosity:L2" in result.labels

    def test_with_reasoning_effort(self) -> None:
        settings = OutputShaperSettings(
            enabled=True, verbosity_level=0, effort_router_enabled=True, mechanical_effort="low"
        )
        body: dict = {
            "input": [{"type": "function_call_output", "output": "result"}],
            "reasoning": {"effort": "xhigh"},
        }
        result = shape_openai_responses_request(body, settings)
        assert result.changed is True
        assert any("reasoning_effort" in lbl for lbl in result.labels)

    def test_with_text_verbosity(self) -> None:
        settings = OutputShaperSettings(
            enabled=True, verbosity_level=0, effort_router_enabled=False
        )
        body: dict = {"input": "hello", "model": "gpt-5"}
        result = shape_openai_responses_request(body, settings)
        assert result.changed is True
        assert any("text_verbosity" in lbl for lbl in result.labels)

    def test_level_override(self) -> None:
        settings = OutputShaperSettings(enabled=True, verbosity_level=1)
        body: dict = {"input": "hello"}
        result = shape_openai_responses_request(body, settings, level_override=3)
        assert "output_shaper:verbosity:L3" in result.labels

    def test_default_settings_when_none(self) -> None:
        body: dict = {"input": "hello"}
        result = shape_openai_responses_request(body)
        assert result.changed is False


class TestShapeResult:
    def test_default_constructor(self) -> None:
        result = ShapeResult()
        assert result.changed is False
        assert result.labels == []

    def test_labels_initialized_when_none(self) -> None:
        result = ShapeResult(changed=True)
        assert result.labels == []


class TestVerbosityLevels:
    def test_all_levels_present(self) -> None:
        for level in (1, 2, 3, 4):
            assert level in _VERBOSITY_LEVELS
        assert 0 not in _VERBOSITY_LEVELS

    def test_levels_are_strings(self) -> None:
        for v in _VERBOSITY_LEVELS.values():
            assert isinstance(v, str)
            assert len(v) > 0
