from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from headroom.proxy.helpers import (
    _BODY_TOO_LARGE_STATUS_ENV,
    _CODEX_WIRE_DEBUG_DIR_ENV,
    _CODEX_WIRE_DEBUG_ENV,
    _MEMORY_INJECTION_MODE_ENV,
    _NO_INLINE_TOOLS_HEADER,
    _NO_INLINE_TOOLS_QUERY_PARAM,
    _PYTHON_FORWARDER_MODE_ENV,
    _SSE_EVENT_MAX_BYTES_ENV,
    WS_COMPRESSION_FAIL_OPEN_ENV,
    WS_COMPRESSION_OVERSIZE_BYTES_DEFAULT,
    WS_COMPRESSION_OVERSIZE_BYTES_ENV,
    BodyMutationTracker,
    CompressionFailureAction,
    _headroom_bypass_enabled,
    _headroom_no_inline_tool_injection,
    append_text_to_latest_user_chat_message,
    append_text_to_latest_user_input_item,
    capture_codex_wire_debug,
    codex_wire_debug_enabled,
    decide_compression_failure_action,
    extract_tags,
    get_body_too_large_status,
    get_memory_injection_mode,
    get_python_forwarder_mode,
    get_sse_event_max_bytes,
    hash_query_for_log,
    is_anthropic_auth,
    jitter_delay_ms,
    log_memory_injection,
    log_outbound_request,
    parse_sse_events_from_byte_buffer,
    prepare_outbound_body_bytes,
    redact_for_wire_debug,
    retry_after_ms,
    safe_decode_for_logging,
    serialize_body_canonical,
)


@contextmanager
def _env(**overrides: str | None) -> Iterator[None]:
    saved: dict[str, str | None] = {}
    for key, value in overrides.items():
        saved[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, prior in saved.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


# ── Section 1: Codex Wire Debug ───────────────────────────────────────


class TestCodexWireDebugEnabled:
    def test_default_false(self) -> None:
        with _env(**{_CODEX_WIRE_DEBUG_ENV: None}):
            assert codex_wire_debug_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", "Yes", "ON"])
    def test_truthy_values(self, value: str) -> None:
        with _env(**{_CODEX_WIRE_DEBUG_ENV: value}):
            assert codex_wire_debug_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "maybe"])
    def test_falsy_values(self, value: str) -> None:
        with _env(**{_CODEX_WIRE_DEBUG_ENV: value}):
            assert codex_wire_debug_enabled() is False

    def test_whitespace_trimmed(self) -> None:
        with _env(**{_CODEX_WIRE_DEBUG_ENV: "  true  "}):
            assert codex_wire_debug_enabled() is True


class TestRedactForWireDebug:
    def test_redacts_exact_match_keys(self) -> None:
        data = {"x-api-key": "sk-secret", "model": "claude-3"}
        result = redact_for_wire_debug(data)
        assert result["x-api-key"] == "[REDACTED]"
        assert result["model"] == "claude-3"

    def test_redacts_normalized_underscore_keys(self) -> None:
        data = {"api-key": "secret", "set-cookie": "sessionid=abc"}
        result = redact_for_wire_debug(data)
        assert result["api-key"] == "[REDACTED]"
        assert result["set-cookie"] == "[REDACTED]"

    @pytest.mark.parametrize(
        "key",
        [
            "authorization",
            "cookie",
            "api-key",
            "x-api-key",
            "openai-api-key",
            "anthropic-api-key",
            "access_token",
            "refresh_token",
            "id_token",
            "bearer",
            "password",
            "secret",
            "token",
            "credential",
        ],
    )
    def test_all_secret_key_names(self, key: str) -> None:
        data = {key: "sensitive"}
        result = redact_for_wire_debug(data)
        assert result[key] == "[REDACTED]"

    def test_preserves_safe_keys(self) -> None:
        data = {"content": "hello", "role": "user", "type": "text"}
        result = redact_for_wire_debug(data)
        assert result == data

    def test_nested_dict_redaction(self) -> None:
        data = {"headers": {"x-api-key": "sk-xxx"}, "body": {"text": "hello"}}
        result = redact_for_wire_debug(data)
        assert result["headers"]["x-api-key"] == "[REDACTED]"
        assert result["body"]["text"] == "hello"

    def test_list_items_redacted(self) -> None:
        data = [{"token": "abc"}, {"name": "safe"}]
        result = redact_for_wire_debug(data)
        assert result[0]["token"] == "[REDACTED]"
        assert result[1]["name"] == "safe"

    def test_scalar_values_passthrough(self) -> None:
        assert redact_for_wire_debug("hello") == "hello"
        assert redact_for_wire_debug(42) == 42
        assert redact_for_wire_debug(None) is None

    def test_suffix_patterns_redacted(self) -> None:
        data = {
            "user_api_key": "sk-xxx",
            "db_secret": "password123",
            "admin_password": "hunter2",
            "app_access_token": "eyJ",
            "sess_refresh_token": "rft",
            "safe_param": "hello",
        }
        result = redact_for_wire_debug(data)
        assert result["user_api_key"] == "[REDACTED]"
        assert result["db_secret"] == "[REDACTED]"
        assert result["admin_password"] == "[REDACTED]"
        assert result["app_access_token"] == "[REDACTED]"
        assert result["sess_refresh_token"] == "[REDACTED]"
        assert result["safe_param"] == "hello"

    def test_empty_dict(self) -> None:
        assert redact_for_wire_debug({}) == {}

    def test_empty_list(self) -> None:
        assert redact_for_wire_debug([]) == []


class TestCaptureCodexWireDebug:
    def test_disabled_returns_none(self) -> None:
        with _env(**{_CODEX_WIRE_DEBUG_ENV: None}):
            result = capture_codex_wire_debug(
                "test_event",
                transport="http",
                direction="inbound",
            )
            assert result is None

    def test_enabled_writes_file(self, tmp_path: Path) -> None:
        debug_dir = tmp_path / "wire_debug"
        with _env(**{_CODEX_WIRE_DEBUG_ENV: "true", _CODEX_WIRE_DEBUG_DIR_ENV: str(debug_dir)}):
            result = capture_codex_wire_debug(
                "test_event",
                request_id="req-1",
                transport="http",
                direction="inbound",
                method="POST",
                url="https://api.anthropic.com/v1/messages",
                headers={"x-api-key": "sk-secret", "content-type": "application/json"},
                body={"prompt": "hello"},
                status_code=200,
                metadata={"source": "test"},
            )
        assert result is not None
        assert isinstance(result, Path)
        assert result.exists()
        assert result.parent == debug_dir
        assert "test_event" in result.name
        content = result.read_text("utf-8")
        assert "test_event" in content
        assert "req-1" in content
        assert "sk-secret" not in content
        assert "[REDACTED]" in content
        assert "hello" in content

    def test_creates_directory_if_not_exists(self, tmp_path: Path) -> None:
        debug_dir = tmp_path / "does_not_exist_yet" / "nested"
        with _env(**{_CODEX_WIRE_DEBUG_ENV: "true", _CODEX_WIRE_DEBUG_DIR_ENV: str(debug_dir)}):
            result = capture_codex_wire_debug(
                "mkdir_test",
                transport="ws",
                direction="outbound",
            )
        assert result is not None
        assert result.exists()

    def test_safe_event_name_sanitized(self, tmp_path: Path) -> None:
        debug_dir = tmp_path / "wire"
        with _env(**{_CODEX_WIRE_DEBUG_ENV: "true", _CODEX_WIRE_DEBUG_DIR_ENV: str(debug_dir)}):
            result = capture_codex_wire_debug(
                "bad/event:name",
                transport="http",
                direction="inbound",
            )
        assert result is not None
        assert "/" not in result.name
        assert ":" not in result.name

    def test_logger_called_on_success(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level("INFO")
        debug_dir = tmp_path / "wire_log"
        with _env(**{_CODEX_WIRE_DEBUG_ENV: "true", _CODEX_WIRE_DEBUG_DIR_ENV: str(debug_dir)}):
            capture_codex_wire_debug(
                "log_test",
                request_id="req-2",
                session_id="sess-1",
                transport="http",
                direction="outbound",
                status_code=200,
                metadata={"key1": "v1"},
            )
        assert "event=codex_wire_debug_capture" in caplog.text
        assert "event=codex_wire_debug_frame" in caplog.text
        assert "req-2" in caplog.text
        assert "sess-1" in caplog.text
        assert "meta_keys" in caplog.text


# ── Section 2: Memory Injection Mode ──────────────────────────────────


class TestGetMemoryInjectionMode:
    def test_default_is_live_zone_tail(self) -> None:
        with _env(**{_MEMORY_INJECTION_MODE_ENV: None}):
            assert get_memory_injection_mode() == "live_zone_tail"

    def test_explicit_live_zone_tail(self) -> None:
        with _env(**{_MEMORY_INJECTION_MODE_ENV: "live_zone_tail"}):
            assert get_memory_injection_mode() == "live_zone_tail"

    def test_disabled(self) -> None:
        with _env(**{_MEMORY_INJECTION_MODE_ENV: "disabled"}):
            assert get_memory_injection_mode() == "disabled"

    def test_whitespace_trimmed(self) -> None:
        with _env(**{_MEMORY_INJECTION_MODE_ENV: "  live_zone_tail  "}):
            assert get_memory_injection_mode() == "live_zone_tail"

    def test_case_insensitive(self) -> None:
        with _env(**{_MEMORY_INJECTION_MODE_ENV: "DISABLED"}):
            assert get_memory_injection_mode() == "disabled"

    def test_invalid_value_raises(self) -> None:
        with _env(**{_MEMORY_INJECTION_MODE_ENV: "system_prompt"}):
            with pytest.raises(ValueError, match="system_prompt"):
                get_memory_injection_mode()

    def test_invalid_value_empty_uses_default(self) -> None:
        with _env(**{_MEMORY_INJECTION_MODE_ENV: ""}):
            assert get_memory_injection_mode() == "live_zone_tail"


# ── Section 3: Hash Query for Log ─────────────────────────────────────


class TestHashQueryForLog:
    def test_stable_hash(self) -> None:
        q = "what is the meaning of life?"
        assert hash_query_for_log(q) == hash_query_for_log(q)

    def test_different_queries_different_hashes(self) -> None:
        assert hash_query_for_log("hello") != hash_query_for_log("world")

    def test_returns_16_hex_chars(self) -> None:
        h = hash_query_for_log("some query")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_empty_string(self) -> None:
        h = hash_query_for_log("")
        assert len(h) == 16

    def test_unicode(self) -> None:
        h = hash_query_for_log("héllo wörld 🌍")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_large_query(self) -> None:
        q = "x" * 10000
        h = hash_query_for_log(q)
        assert len(h) == 16


# ── Section 4: Python Forwarder Mode ──────────────────────────────────


class TestGetPythonForwarderMode:
    def test_default_is_byte_faithful(self) -> None:
        with _env(**{_PYTHON_FORWARDER_MODE_ENV: None}):
            assert get_python_forwarder_mode() == "byte_faithful"

    def test_explicit_byte_faithful(self) -> None:
        with _env(**{_PYTHON_FORWARDER_MODE_ENV: "byte_faithful"}):
            assert get_python_forwarder_mode() == "byte_faithful"

    def test_legacy_json_kwarg(self) -> None:
        with _env(**{_PYTHON_FORWARDER_MODE_ENV: "legacy_json_kwarg"}):
            assert get_python_forwarder_mode() == "legacy_json_kwarg"

    def test_invalid_value_raises(self) -> None:
        with _env(**{_PYTHON_FORWARDER_MODE_ENV: "something_else"}):
            with pytest.raises(ValueError, match="something_else"):
                get_python_forwarder_mode()

    def test_empty_uses_default(self) -> None:
        with _env(**{_PYTHON_FORWARDER_MODE_ENV: ""}):
            assert get_python_forwarder_mode() == "byte_faithful"


# ── Section 5: Extract Tags ───────────────────────────────────────────


class TestExtractTags:
    def test_empty_headers(self) -> None:
        assert extract_tags({}) == {}

    def test_single_tag(self) -> None:
        headers = {"x-headroom-tag": "test"}
        assert extract_tags(headers) == {"tag": "test"}

    def test_multiple_tags(self) -> None:
        headers = {"x-headroom-user-id": "abc", "x-headroom-stack": "full"}
        assert extract_tags(headers) == {"user-id": "abc", "stack": "full"}

    def test_case_insensitive(self) -> None:
        headers = {"X-HEADROOM-FOO": "bar", "X-Headroom-Baz": "qux"}
        assert extract_tags(headers) == {"foo": "bar", "baz": "qux"}

    def test_non_headroom_headers_ignored(self) -> None:
        headers = {
            "x-headroom-id": "42",
            "content-type": "application/json",
            "authorization": "Bearer sk-ant-xxx",
        }
        assert extract_tags(headers) == {"id": "42"}

    def test_prefix_stripped(self) -> None:
        headers = {"x-headroom-x-custom": "val"}
        result = extract_tags(headers)
        assert "x-headroom-x-custom" not in result
        assert result["x-custom"] == "val"

    def test_mixed_case_prefix(self) -> None:
        headers = {"X-HeadRoom-Key": "value"}
        assert extract_tags(headers) == {"key": "value"}


# ── Section 6: Bypass Detection ───────────────────────────────────────


class TestHeadroomBypassEnabled:
    def test_default_not_bypassed(self) -> None:
        headers = {"content-type": "application/json"}
        assert _headroom_bypass_enabled(headers) is False

    def test_bypass_header_true(self) -> None:
        headers = {"x-headroom-bypass": "true"}
        assert _headroom_bypass_enabled(headers) is True

    def test_mode_header_passthrough(self) -> None:
        headers = {"x-headroom-mode": "passthrough"}
        assert _headroom_bypass_enabled(headers) is True

    def test_bypass_false_not_enabled(self) -> None:
        headers = {"x-headroom-bypass": "false"}
        assert _headroom_bypass_enabled(headers) is False

    def test_mode_other_not_enabled(self) -> None:
        headers = {"x-headroom-mode": "compress"}
        assert _headroom_bypass_enabled(headers) is False

    def test_case_insensitive_value(self) -> None:
        headers = {"x-headroom-bypass": "True"}
        assert _headroom_bypass_enabled(headers) is True
        headers2 = {"x-headroom-mode": "PASSTHROUGH"}
        assert _headroom_bypass_enabled(headers2) is True

    def test_attribute_error_returns_false(self) -> None:
        assert _headroom_bypass_enabled(None) is False


# ── Section 7: No Inline Tool Injection ───────────────────────────────


class TestHeadroomNoInlineToolInjection:
    def test_default_false(self) -> None:
        assert _headroom_no_inline_tool_injection({}, {}) is False

    def test_header_true(self) -> None:
        headers = {_NO_INLINE_TOOLS_HEADER: "true"}
        assert _headroom_no_inline_tool_injection(headers, {}) is True

    def test_header_1(self) -> None:
        headers = {_NO_INLINE_TOOLS_HEADER: "1"}
        assert _headroom_no_inline_tool_injection(headers, {}) is True

    def test_header_yes(self) -> None:
        headers = {_NO_INLINE_TOOLS_HEADER: "yes"}
        assert _headroom_no_inline_tool_injection(headers, {}) is True

    def test_query_param_true(self) -> None:
        params = {_NO_INLINE_TOOLS_QUERY_PARAM: "true"}
        assert _headroom_no_inline_tool_injection({}, params) is True

    def test_query_param_1(self) -> None:
        params = {_NO_INLINE_TOOLS_QUERY_PARAM: "1"}
        assert _headroom_no_inline_tool_injection({}, params) is True

    def test_query_param_yes(self) -> None:
        params = {_NO_INLINE_TOOLS_QUERY_PARAM: "yes"}
        assert _headroom_no_inline_tool_injection({}, params) is True

    def test_query_param_false_not_enabled(self) -> None:
        params = {_NO_INLINE_TOOLS_QUERY_PARAM: "false"}
        assert _headroom_no_inline_tool_injection({}, params) is False

    def test_header_takes_precedence(self) -> None:
        headers = {_NO_INLINE_TOOLS_HEADER: "true"}
        params = {_NO_INLINE_TOOLS_QUERY_PARAM: "false"}
        assert _headroom_no_inline_tool_injection(headers, params) is True

    def test_attribute_error_on_header_handled(self) -> None:
        assert _headroom_no_inline_tool_injection(object(), {}) is False

    def test_none_query_params(self) -> None:
        headers = {_NO_INLINE_TOOLS_HEADER: "false"}
        assert _headroom_no_inline_tool_injection(headers, None) is False

    def test_attribute_error_on_query_params(self) -> None:
        headers = {}
        params = {"no_inline_tools": None}
        assert _headroom_no_inline_tool_injection(headers, params) is False


# ── Section 8: Body Serialization ─────────────────────────────────────


class TestSerializeBodyCanonical:
    def test_compact_separators(self) -> None:
        body = {"key": "value"}
        result = serialize_body_canonical(body)
        assert result == b'{"key":"value"}'

    def test_utf8_no_escape(self) -> None:
        body = {"text": "héllo"}
        result = serialize_body_canonical(body)
        assert b"h\\u00e9" not in result
        assert "héllo".encode() in result

    def test_empty_dict(self) -> None:
        assert serialize_body_canonical({}) == b"{}"

    def test_nested(self) -> None:
        body = {"a": {"b": [1, 2, 3]}}
        result = serialize_body_canonical(body)
        assert result == b'{"a":{"b":[1,2,3]}}'

    def test_preserves_order(self) -> None:
        body = {"z": 1, "a": 2, "m": 3}
        result = serialize_body_canonical(body)
        assert result.index(b'"z"') < result.index(b'"a"')


# ── Section 9: BodyMutationTracker ────────────────────────────────────


class TestBodyMutationTracker:
    def test_initial_state(self) -> None:
        t = BodyMutationTracker()
        assert t.mutated is False
        assert t.reasons == []

    def test_mark_mutated_sets_flag(self) -> None:
        t = BodyMutationTracker()
        t.mark_mutated("memory_injection")
        assert t.mutated is True

    def test_reasons_after_mark(self) -> None:
        t = BodyMutationTracker()
        t.mark_mutated("memory_injection")
        assert t.reasons == ["memory_injection"]

    def test_multiple_reasons(self) -> None:
        t = BodyMutationTracker()
        t.mark_mutated("injection_a")
        t.mark_mutated("injection_b")
        assert t.reasons == ["injection_a", "injection_b"]

    def test_dedup_reasons(self) -> None:
        t = BodyMutationTracker()
        t.mark_mutated("same")
        t.mark_mutated("same")
        assert t.reasons == ["same"]

    def test_dedup_preserves_order(self) -> None:
        t = BodyMutationTracker()
        t.mark_mutated("first")
        t.mark_mutated("second")
        t.mark_mutated("second")
        t.mark_mutated("first")
        assert t.reasons == ["first", "second"]

    def test_empty_reason_raises(self) -> None:
        t = BodyMutationTracker()
        with pytest.raises(ValueError, match="reason must be non-empty"):
            t.mark_mutated("")

    def test_reasons_returns_copy(self) -> None:
        t = BodyMutationTracker()
        t.mark_mutated("test")
        r = t.reasons
        r.append("injected")
        assert t.reasons == ["test"]

    def test_mark_not_mutated_stays_false(self) -> None:
        t = BodyMutationTracker()
        t.mark_mutated("compression")
        t.mark_mutated("compression")
        assert t.mutated is True


# ── Section 10: Prepare Outbound Body Bytes ───────────────────────────


class TestPrepareOutboundBodyBytes:
    def test_passthrough_unmutated(self) -> None:
        original = b'{"key":"value"}'
        body = {"key": "value"}
        result, source = prepare_outbound_body_bytes(
            body=body,
            original_body_bytes=original,
            body_mutated=False,
            forwarder_mode="byte_faithful",
        )
        assert result is original
        assert source == "passthrough"

    def test_canonical_mutated(self) -> None:
        body = {"key": "value"}
        result, source = prepare_outbound_body_bytes(
            body=body,
            original_body_bytes=b'{"key":"old"}',
            body_mutated=True,
            forwarder_mode="byte_faithful",
        )
        assert result == b'{"key":"value"}'
        assert source == "canonical"

    def test_canonical_no_original(self) -> None:
        body = {"key": "value"}
        result, source = prepare_outbound_body_bytes(
            body=body,
            original_body_bytes=None,
            body_mutated=False,
            forwarder_mode="byte_faithful",
        )
        assert result == b'{"key":"value"}'
        assert source == "canonical"

    def test_legacy_mode(self) -> None:
        body = {"key": "value"}
        result, source = prepare_outbound_body_bytes(
            body=body,
            original_body_bytes=b'{"key":"value"}',
            body_mutated=False,
            forwarder_mode="legacy_json_kwarg",
        )
        assert source == "legacy"
        assert result == b'{"key": "value"}'

    def test_legacy_ensure_ascii(self) -> None:
        body = {"text": "héllo"}
        result, source = prepare_outbound_body_bytes(
            body=body,
            original_body_bytes=None,
            body_mutated=False,
            forwarder_mode="legacy_json_kwarg",
        )
        assert source == "legacy"
        assert b"\\u00e9" in result

    def test_forwarder_mode_defaults(self) -> None:
        with _env(**{_PYTHON_FORWARDER_MODE_ENV: None}):
            body = {"key": "value"}
            result, source = prepare_outbound_body_bytes(
                body=body,
                original_body_bytes=None,
                body_mutated=False,
            )
            assert source == "canonical"


# ── Section 11: Log Functions ─────────────────────────────────────────


class TestLogOutboundRequest:
    def test_basic_log(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("INFO")
        log_outbound_request(
            forwarder="anthropic",
            method="POST",
            path="/v1/messages",
            body_bytes_count=42,
            body_mutated=False,
            mutation_reasons=[],
            request_id="req-1",
            source="passthrough",
        )
        assert "event=outbound_request" in caplog.text
        assert "req-1" in caplog.text
        assert "body_mutated=false" in caplog.text

    def test_with_mutation_reasons(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("INFO")
        log_outbound_request(
            forwarder="streaming",
            method="POST",
            path="/v1/chat/completions",
            body_bytes_count=128,
            body_mutated=True,
            mutation_reasons=["memory_injection", "compression"],
            request_id="req-2",
            source="canonical",
        )
        assert "event=outbound_request" in caplog.text
        assert "body_mutated=true" in caplog.text
        assert "memory_injection" in caplog.text
        assert "compression" in caplog.text


class TestLogMemoryInjection:
    def test_basic_log(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("INFO")
        log_memory_injection(
            request_id="req-1",
            session_id="sess-1",
            decision="injected",
            bytes_injected=256,
            query="user query",
        )
        assert "event=memory_injection" in caplog.text
        assert "req-1" in caplog.text
        assert "sess-1" in caplog.text
        assert "decision=injected" in caplog.text
        assert "bytes_injected=256" in caplog.text

    def test_without_query(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("INFO")
        log_memory_injection(
            request_id="req-2",
            session_id=None,
            decision="skipped",
            bytes_injected=0,
        )
        assert "event=memory_injection" in caplog.text
        assert "query_hash=" in caplog.text

    def test_query_hash_is_not_raw_query(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("INFO")
        log_memory_injection(
            request_id="req-3",
            session_id=None,
            decision="injected",
            bytes_injected=100,
            query="my secret query",
        )
        assert "my secret query" not in caplog.text
        assert "query_hash=" in caplog.text


# ── Section 12: Append Text to Messages ───────────────────────────────


class TestAppendTextToLatestUserChatMessage:
    def test_string_content_appended(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        result, count = append_text_to_latest_user_chat_message(messages, "world")
        assert result[0]["content"] == "hello\n\nworld"
        assert count == 5

    def test_list_content_first_text_block(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "url": "http://example.com/img.png"},
                ],
            }
        ]
        result, count = append_text_to_latest_user_chat_message(messages, "context")
        assert result[0]["content"][0]["text"] == "hello\n\ncontext"
        assert result[0]["content"][1]["type"] == "image_url"
        assert count == 7

    def test_input_text_type(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ]
        result, count = append_text_to_latest_user_chat_message(messages, "ctx")
        assert result[0]["content"][0]["text"] == "hello\n\nctx"
        assert count == 3

    def test_no_user_message_returns_zero(self) -> None:
        messages = [{"role": "assistant", "content": "hello"}]
        result, count = append_text_to_latest_user_chat_message(messages, "ctx")
        assert count == 0
        assert result == messages

    def test_empty_context_returns_zero(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        result, count = append_text_to_latest_user_chat_message(messages, "")
        assert count == 0
        assert result == messages

    def test_empty_messages_returns_zero(self) -> None:
        result, count = append_text_to_latest_user_chat_message([], "ctx")
        assert count == 0
        assert result == []

    def test_non_dict_items_skipped(self) -> None:
        messages = ["not a dict", {"role": "user", "content": "hello"}]
        result, count = append_text_to_latest_user_chat_message(messages, "ctx")
        assert result[1]["content"] == "hello\n\nctx"
        assert count == 3

    def test_roll_does_not_exist(self) -> None:
        messages = [{"role": "system", "content": "be helpful"}]
        result, count = append_text_to_latest_user_chat_message(messages, "ctx")
        assert count == 0
        assert result == messages

    def test_appends_to_latest_user_message(self) -> None:
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "second"},
        ]
        result, count = append_text_to_latest_user_chat_message(messages, "tail")
        assert result[2]["content"] == "second\n\ntail"
        assert count == 4

    def test_user_without_eligible_text_block_untouched(self) -> None:
        messages = [{"role": "user", "content": [{"type": "image_url", "url": "img.png"}]}]
        result, count = append_text_to_latest_user_chat_message(messages, "ctx")
        assert count == 0
        assert result == messages

    def test_does_not_mutate_original(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        original_id = id(messages[0]["content"])
        result, count = append_text_to_latest_user_chat_message(messages, "ctx")
        assert id(result[0]["content"]) != original_id


class TestAppendTextToLatestUserInputItem:
    def test_string_content_appended(self) -> None:
        inp = [{"role": "user", "content": "hello"}]
        result, count = append_text_to_latest_user_input_item(inp, "world")
        assert result[0]["content"] == "hello\n\nworld"
        assert count == 5

    def test_list_content_first_text_block(self) -> None:
        inp = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "hello"},
                    {"type": "file", "source": {"url": "file:///tmp/doc.pdf"}},
                ],
            }
        ]
        result, count = append_text_to_latest_user_input_item(inp, "context")
        assert result[0]["content"][0]["text"] == "hello\n\ncontext"
        assert result[0]["content"][1]["type"] == "file"
        assert count == 7

    def test_text_type_also_matches(self) -> None:
        inp = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        result, count = append_text_to_latest_user_input_item(inp, "ctx")
        assert result[0]["content"][0]["text"] == "hello\n\nctx"

    def test_no_user_item_returns_zero(self) -> None:
        inp = [{"role": "assistant", "content": "hello"}]
        result, count = append_text_to_latest_user_input_item(inp, "ctx")
        assert count == 0
        assert result == inp

    def test_empty_context_returns_zero(self) -> None:
        inp = [{"role": "user", "content": "hello"}]
        result, count = append_text_to_latest_user_input_item(inp, "")
        assert count == 0
        assert result == inp

    def test_empty_input_returns_zero(self) -> None:
        result, count = append_text_to_latest_user_input_item([], "ctx")
        assert count == 0
        assert result == []

    def test_non_dict_items_skipped(self) -> None:
        inp = ["not a dict", {"role": "user", "content": "hello"}]
        result, count = append_text_to_latest_user_input_item(inp, "ctx")
        assert result[1]["content"] == "hello\n\nctx"
        assert count == 3

    def test_appends_to_latest_user_item(self) -> None:
        inp = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "second"},
        ]
        result, count = append_text_to_latest_user_input_item(inp, "tail")
        assert result[2]["content"] == "second\n\ntail"
        assert count == 4

    def test_user_without_eligible_text_block_untouched(self) -> None:
        inp = [{"role": "user", "content": [{"type": "file", "url": "doc.pdf"}]}]
        result, count = append_text_to_latest_user_input_item(inp, "ctx")
        assert count == 0
        assert result == inp


# ── Section 13: SSE Helpers ───────────────────────────────────────────


class TestGetSseEventMaxBytes:
    def test_default(self) -> None:
        with _env(**{_SSE_EVENT_MAX_BYTES_ENV: None}):
            assert get_sse_event_max_bytes() == 1 * 1024 * 1024

    def test_custom(self) -> None:
        with _env(**{_SSE_EVENT_MAX_BYTES_ENV: "65536"}):
            assert get_sse_event_max_bytes() == 65536

    def test_negative_raises(self) -> None:
        with _env(**{_SSE_EVENT_MAX_BYTES_ENV: "-1"}):
            with pytest.raises(ValueError, match="must be positive"):
                get_sse_event_max_bytes()

    def test_zero_raises(self) -> None:
        with _env(**{_SSE_EVENT_MAX_BYTES_ENV: "0"}):
            with pytest.raises(ValueError, match="must be positive"):
                get_sse_event_max_bytes()

    def test_non_int_raises(self) -> None:
        with _env(**{_SSE_EVENT_MAX_BYTES_ENV: "not_an_int"}):
            with pytest.raises(ValueError, match="must be an integer"):
                get_sse_event_max_bytes()

    def test_empty_uses_default(self) -> None:
        with _env(**{_SSE_EVENT_MAX_BYTES_ENV: ""}):
            assert get_sse_event_max_bytes() == 1 * 1024 * 1024


class TestGetBodyTooLargeStatus:
    def test_default(self) -> None:
        with _env(**{_BODY_TOO_LARGE_STATUS_ENV: None}):
            assert get_body_too_large_status() == 413

    def test_custom_4xx(self) -> None:
        with _env(**{_BODY_TOO_LARGE_STATUS_ENV: "429"}):
            assert get_body_too_large_status() == 429

    def test_custom_5xx(self) -> None:
        with _env(**{_BODY_TOO_LARGE_STATUS_ENV: "507"}):
            assert get_body_too_large_status() == 507

    def test_non_4xx_5xx_raises(self) -> None:
        with _env(**{_BODY_TOO_LARGE_STATUS_ENV: "200"}):
            with pytest.raises(ValueError, match="must be a 4xx/5xx"):
                get_body_too_large_status()

    def test_non_int_raises(self) -> None:
        with _env(**{_BODY_TOO_LARGE_STATUS_ENV: "four thirteen"}):
            with pytest.raises(ValueError, match="must be an integer"):
                get_body_too_large_status()

    def test_empty_uses_default(self) -> None:
        with _env(**{_BODY_TOO_LARGE_STATUS_ENV: ""}):
            assert get_body_too_large_status() == 413


class TestSafeDecodeForLogging:
    def test_basic_decode(self) -> None:
        raw = b"hello world"
        assert safe_decode_for_logging(raw) == "hello world"

    def test_invalid_utf8_uses_replacement(self) -> None:
        raw = b"hello\xffworld"
        result = safe_decode_for_logging(raw)
        assert "hello" in result
        assert "world" in result
        assert "\ufffd" in result

    def test_max_bytes_truncates(self) -> None:
        raw = b"abcdefghij"
        result = safe_decode_for_logging(raw, max_bytes=5)
        assert len(result) <= 5

    def test_empty_bytes(self) -> None:
        assert safe_decode_for_logging(b"") == ""

    def test_unicode_passthrough(self) -> None:
        raw = "héllo 𝄞".encode()
        assert safe_decode_for_logging(raw) == "héllo 𝄞"

    def test_max_bytes_none_returns_all(self) -> None:
        raw = b"x" * 1000
        result = safe_decode_for_logging(raw, max_bytes=None)
        assert len(result) == 1000


class TestParseSseEventsFromByteBuffer:
    def test_basic_lf_event(self) -> None:
        buf = bytearray(b"event: completion\ndata: hello\n\n")
        events = parse_sse_events_from_byte_buffer(buf)
        assert len(events) == 1
        assert events[0] == ("completion", "hello")
        assert buf == bytearray()

    def test_crlf_event(self) -> None:
        buf = bytearray(b"event: message\r\ndata: hello\r\n\r\n")
        events = parse_sse_events_from_byte_buffer(buf)
        assert len(events) == 1
        assert events[0] == ("message", "hello")
        assert buf == bytearray()

    def test_multiple_events(self) -> None:
        buf = bytearray(b"event: a\ndata: 1\n\nevent: b\ndata: 2\n\n")
        events = parse_sse_events_from_byte_buffer(buf)
        assert len(events) == 2
        assert events[0] == ("a", "1")
        assert events[1] == ("b", "2")
        assert buf == bytearray()

    def test_partial_event_remains(self) -> None:
        buf = bytearray(b"event: a\ndata: 1\n\nevent: b\ndata: 2")
        events = parse_sse_events_from_byte_buffer(buf)
        assert len(events) == 1
        assert events[0] == ("a", "1")
        assert buf == bytearray(b"event: b\ndata: 2")

    def test_no_event_name_defaults_to_none(self) -> None:
        buf = bytearray(b"data: hello\n\n")
        events = parse_sse_events_from_byte_buffer(buf)
        assert len(events) == 1
        assert events[0][0] is None
        assert events[0][1] == "hello"

    def test_comment_line_ignored(self) -> None:
        buf = bytearray(b": this is a comment\ndata: hello\n\n")
        events = parse_sse_events_from_byte_buffer(buf)
        assert len(events) == 1
        assert events[0] == (None, "hello")

    def test_multiple_data_lines_joined(self) -> None:
        buf = bytearray(b"event: chunk\ndata: line1\ndata: line2\n\n")
        events = parse_sse_events_from_byte_buffer(buf)
        assert len(events) == 1
        assert events[0] == ("chunk", "line1\nline2")

    def test_no_data_lines_skipped(self) -> None:
        buf = bytearray(b"event: empty\n\n")
        events = parse_sse_events_from_byte_buffer(buf)
        assert len(events) == 0

    def test_empty_buffer(self) -> None:
        buf = bytearray()
        events = parse_sse_events_from_byte_buffer(buf)
        assert events == []

    def test_event_without_data_skipped(self) -> None:
        buf = bytearray(b"event: noop\n: comment only\n\n")
        events = parse_sse_events_from_byte_buffer(buf)
        assert len(events) == 0

    def test_mixed_lf_crlf(self) -> None:
        buf = bytearray(b"event: first\ndata: 1\n\nevent: second\r\ndata: 2\r\n\r\n")
        events = parse_sse_events_from_byte_buffer(buf)
        assert len(events) == 2
        assert events[0] == ("first", "1")
        assert events[1] == ("second", "2")
        assert buf == bytearray()

    def test_invalid_utf8_raises(self) -> None:
        buf = bytearray(b"data: hello\xffworld\n\n")
        with pytest.raises(UnicodeDecodeError):
            parse_sse_events_from_byte_buffer(buf)


# ── Section 14: Compression Failure Action ────────────────────────────


class TestCompressionFailureAction:
    def test_frozen_dataclass(self) -> None:
        a = CompressionFailureAction(refuse=True, reason="timeout", frame_bytes=128)
        with pytest.raises(AttributeError):
            a.refuse = False

    def test_fields(self) -> None:
        a = CompressionFailureAction(refuse=True, reason="timeout", frame_bytes=128)
        assert a.refuse is True
        assert a.reason == "timeout"
        assert a.frame_bytes == 128


class TestDecideCompressionFailureAction:
    def test_timeout_refuses(self) -> None:
        with _env(**{WS_COMPRESSION_FAIL_OPEN_ENV: None, WS_COMPRESSION_OVERSIZE_BYTES_ENV: None}):
            action = decide_compression_failure_action(asyncio.TimeoutError(), frame_bytes=128)
        assert action.refuse is True
        assert action.reason == "timeout"

    def test_codex_timeout_fails_open(self) -> None:
        with _env(**{WS_COMPRESSION_FAIL_OPEN_ENV: None, WS_COMPRESSION_OVERSIZE_BYTES_ENV: None}):
            action = decide_compression_failure_action(
                asyncio.TimeoutError(), frame_bytes=128, client="codex"
            )
        assert action.refuse is False
        assert action.reason == "client_override:codex"

    def test_non_codex_timeout_still_refuses(self) -> None:
        with _env(**{WS_COMPRESSION_FAIL_OPEN_ENV: None, WS_COMPRESSION_OVERSIZE_BYTES_ENV: None}):
            action = decide_compression_failure_action(
                asyncio.TimeoutError(), frame_bytes=128, client="claude-code"
            )
        assert action.refuse is True
        assert action.reason == "timeout"

    def test_oversize_refuses(self) -> None:
        big = WS_COMPRESSION_OVERSIZE_BYTES_DEFAULT + 1024
        with _env(**{WS_COMPRESSION_FAIL_OPEN_ENV: None, WS_COMPRESSION_OVERSIZE_BYTES_ENV: None}):
            action = decide_compression_failure_action(RuntimeError("crashed"), frame_bytes=big)
        assert action.refuse is True
        assert "oversize" in action.reason

    def test_small_transient_passthrough(self) -> None:
        with _env(**{WS_COMPRESSION_FAIL_OPEN_ENV: None, WS_COMPRESSION_OVERSIZE_BYTES_ENV: None}):
            action = decide_compression_failure_action(RuntimeError("glitch"), frame_bytes=4 * 1024)
        assert action.refuse is False
        assert action.reason == "small_frame_transient"

    def test_env_fail_open_overrides(self) -> None:
        big = WS_COMPRESSION_OVERSIZE_BYTES_DEFAULT + 1
        with _env(
            **{WS_COMPRESSION_FAIL_OPEN_ENV: "true", WS_COMPRESSION_OVERSIZE_BYTES_ENV: None}
        ):
            action = decide_compression_failure_action(asyncio.TimeoutError(), frame_bytes=big)
        assert action.refuse is False
        assert action.reason == "env_override:fail_open"

    def test_custom_threshold(self) -> None:
        with _env(
            **{WS_COMPRESSION_FAIL_OPEN_ENV: None, WS_COMPRESSION_OVERSIZE_BYTES_ENV: "1024"}
        ):
            action = decide_compression_failure_action(RuntimeError(), frame_bytes=2048)
        assert action.refuse is True
        assert "threshold=1024" in action.reason

        with _env(
            **{WS_COMPRESSION_FAIL_OPEN_ENV: None, WS_COMPRESSION_OVERSIZE_BYTES_ENV: "1024"}
        ):
            action_small = decide_compression_failure_action(RuntimeError(), frame_bytes=512)
        assert action_small.refuse is False

    def test_invalid_threshold_falls_back(self) -> None:
        big = WS_COMPRESSION_OVERSIZE_BYTES_DEFAULT + 1
        with _env(
            **{
                WS_COMPRESSION_FAIL_OPEN_ENV: None,
                WS_COMPRESSION_OVERSIZE_BYTES_ENV: "not-a-number",
            }
        ):
            action = decide_compression_failure_action(RuntimeError(), frame_bytes=big)
        assert action.refuse is True
        assert f"threshold={WS_COMPRESSION_OVERSIZE_BYTES_DEFAULT}" in action.reason

    def test_frame_bytes_in_result(self) -> None:
        with _env(**{WS_COMPRESSION_FAIL_OPEN_ENV: None, WS_COMPRESSION_OVERSIZE_BYTES_ENV: None}):
            action = decide_compression_failure_action(RuntimeError(), frame_bytes=999)
        assert action.frame_bytes == 999


# ── Section 15: Delay Helpers ─────────────────────────────────────────


class TestJitterDelayMs:
    def test_range_within_expected_bounds(self) -> None:
        for _ in range(100):
            d = jitter_delay_ms(base_ms=100, max_ms=1000, attempt=0)
            assert 50 <= d <= 150

    def test_capped_at_max(self) -> None:
        for _ in range(100):
            d = jitter_delay_ms(base_ms=1000, max_ms=200, attempt=3)
            assert 100 <= d <= 300

    def test_exponential_growth(self) -> None:
        results = set()
        for _ in range(10):
            d = jitter_delay_ms(base_ms=100, max_ms=10000, attempt=3)
            assert d >= 100 * (2**3) * 0.5
            assert d <= 10000 * 1.5
            results.add(int(d))
        assert len(results) > 1

    def test_zero_base(self) -> None:
        for _ in range(10):
            d = jitter_delay_ms(base_ms=0, max_ms=1000, attempt=5)
            assert 0 <= d <= 1500

    def test_large_max(self) -> None:
        d = jitter_delay_ms(base_ms=10, max_ms=1_000_000, attempt=10)
        assert d >= 10 * (2**10) * 0.5
        assert d <= 1_000_000 * 1.5


class TestRetryAfterMs:
    def test_numeric_seconds(self) -> None:
        response = MagicMock(spec_set=["headers"])
        response.headers = {"retry-after": "5"}
        result = retry_after_ms(response, max_ms=10000)
        assert result is not None
        assert 4900 <= result <= 5100

    def test_absent_returns_none(self) -> None:
        response = MagicMock(spec_set=["headers"])
        response.headers = {}
        assert retry_after_ms(response, max_ms=10000) is None

    def test_unparseable_returns_none(self) -> None:
        response = MagicMock(spec_set=["headers"])
        response.headers = {"retry-after": "not-a-date-or-number"}
        assert retry_after_ms(response, max_ms=10000) is None

    def test_capped_at_max_ms(self) -> None:
        response = MagicMock(spec_set=["headers"])
        response.headers = {"retry-after": "100"}
        result = retry_after_ms(response, max_ms=5000)
        assert result is not None
        assert result == 5000.0

    def test_negative_seconds_clamped_to_zero(self) -> None:
        response = MagicMock(spec_set=["headers"])
        response.headers = {"retry-after": "-5"}
        result = retry_after_ms(response, max_ms=10000)
        assert result is not None
        assert result == 0.0


# ── Section 16: Auth Helpers ──────────────────────────────────────────


class TestIsAnthropicAuth:
    def test_x_api_key(self) -> None:
        headers = {"x-api-key": "sk-ant-xxx"}
        assert is_anthropic_auth(headers) is True

    def test_anthropic_version(self) -> None:
        headers = {"anthropic-version": "2023-06-01"}
        assert is_anthropic_auth(headers) is True

    def test_bearer_sk_ant(self) -> None:
        headers = {"authorization": "Bearer sk-ant-v2-xxxxx"}
        assert is_anthropic_auth(headers) is True

    def test_no_auth(self) -> None:
        headers = {"content-type": "application/json"}
        assert is_anthropic_auth(headers) is False

    def test_empty_headers(self) -> None:
        assert is_anthropic_auth({}) is False

    def test_bearer_other_provider(self) -> None:
        headers = {"authorization": "Bearer sk-other-xxxxx"}
        assert is_anthropic_auth(headers) is False

    def test_bearer_without_prefix(self) -> None:
        headers = {"authorization": "sk-ant-xxx"}
        assert is_anthropic_auth(headers) is False

    def test_both_api_key_and_version(self) -> None:
        headers = {"x-api-key": "sk-ant-xxx", "anthropic-version": "2023-06-01"}
        assert is_anthropic_auth(headers) is True
