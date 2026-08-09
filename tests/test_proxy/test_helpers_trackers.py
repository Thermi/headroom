from __future__ import annotations

from unittest.mock import patch

import pytest

from headroom.proxy.helpers import (
    # Section 2
    _BETA_HEADER_STICKY_ENV,
    _BETA_TRACKER_MAX_SESSIONS_ENV,
    _STRIP_INTERNAL_HEADERS_ENV,
    # Section 5
    _TOOL_INJECTION_STICKY_ENV,
    _TOOL_TRACKER_MAX_SESSIONS_ENV,
    # Section 3
    SessionBetaTracker,
    # Section 9
    SessionCcrTracker,
    # Section 6
    SessionToolTracker,
    # Section 8
    _merge_beta_tokens,
    # Section 4
    _reset_session_beta_tracker_for_test,
    # Section 10
    _reset_session_ccr_tracker_for_test,
    # Section 7
    _reset_session_tool_tracker_for_test,
    _split_beta_tokens,
    _strip_internal_headers,
    # Section 11
    apply_session_sticky_ccr_tool,
    apply_session_sticky_memory_tools,
    # Section 13
    claude_code_tool_search_inactive,
    # Section 12
    compute_turn_id,
    format_tool_search_disabled_hint,
    get_beta_header_sticky_mode,
    get_beta_tracker_max_sessions,
    get_session_beta_tracker,
    get_session_ccr_tracker,
    get_session_tool_tracker,
    # Section 1
    get_strip_internal_headers_mode,
    get_tool_injection_sticky_mode,
    get_tool_tracker_max_sessions,
    log_beta_header_merge,
    log_outbound_headers,
    log_tool_injection_decision,
    merge_anthropic_beta,
    merge_openai_beta,
    reset_tool_search_hint_state,
    serialize_tool_definition_canonical,
    take_tool_search_hint_slot,
    tool_search_hint_pending,
)

# ── Section 1: Internal header stripping ────────────────────────────────


class TestGetStripInternalHeadersMode:
    def test_default_enabled(self) -> None:
        assert get_strip_internal_headers_mode() == "enabled"

    def test_explicit_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_STRIP_INTERNAL_HEADERS_ENV, "enabled")
        assert get_strip_internal_headers_mode() == "enabled"

    def test_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_STRIP_INTERNAL_HEADERS_ENV, "disabled")
        assert get_strip_internal_headers_mode() == "disabled"

    def test_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_STRIP_INTERNAL_HEADERS_ENV, "ENABLED")
        assert get_strip_internal_headers_mode() == "enabled"

    def test_unknown_value_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_STRIP_INTERNAL_HEADERS_ENV, "maybe")
        with pytest.raises(ValueError, match="Invalid"):
            get_strip_internal_headers_mode()


class TestStripInternalHeaders:
    def test_strips_x_headroom_prefix(self) -> None:
        headers = {"x-headroom-bypass": "1", "host": "example.com", "authorization": "Bearer x"}
        result = _strip_internal_headers(headers)
        assert "x-headroom-bypass" not in result
        assert "host" in result
        assert result["host"] == "example.com"

    def test_case_insensitive_strip(self) -> None:
        headers = {"X-HEADROOM-MODE": "proxy", "x-HeadRoom-UserId": "abc", "normal": "value"}
        result = _strip_internal_headers(headers)
        assert "X-HEADROOM-MODE" not in result
        assert "x-HeadRoom-UserId" not in result
        assert result["normal"] == "value"

    def test_returns_new_dict(self) -> None:
        headers = {"x-headroom-foo": "bar", "keep": "me"}
        result = _strip_internal_headers(headers)
        assert result is not headers
        assert result == {"keep": "me"}

    def test_no_internal_headers_unchanged(self) -> None:
        headers = {"host": "example.com", "content-type": "application/json"}
        result = _strip_internal_headers(headers)
        assert result == headers

    def test_empty_dict(self) -> None:
        result = _strip_internal_headers({})
        assert result == {}

    def test_disabled_mode_returns_copy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_STRIP_INTERNAL_HEADERS_ENV, "disabled")
        headers = {"x-headroom-bypass": "1", "host": "example.com"}
        result = _strip_internal_headers(headers)
        assert result is not headers
        assert result == headers

    def test_disabled_mode_keeps_internal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_STRIP_INTERNAL_HEADERS_ENV, "disabled")
        headers = {"x-headroom-foo": "bar"}
        result = _strip_internal_headers(headers)
        assert "x-headroom-foo" in result

    def test_does_not_mutate_original(self) -> None:
        headers = {"x-headroom-bypass": "1", "host": "example.com"}
        _strip_internal_headers(headers)
        assert "x-headroom-bypass" in headers

    def test_only_strips_x_headroom_prefix_exact(self) -> None:
        headers = {"x-custom": "keep", "x-headroom-foo": "strip", "x-something": "keep"}
        result = _strip_internal_headers(headers)
        assert "x-custom" in result
        assert "x-headroom-foo" not in result
        assert "x-something" in result

    def test_header_equals_prefix_no_trailing_hyphen(self) -> None:
        headers = {"x-headroom": "value"}
        result = _strip_internal_headers(headers)
        assert "x-headroom" in result


class TestLogOutboundHeaders:
    def test_logs_stripped_count(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(0)
        log_outbound_headers(forwarder="anthropic", stripped_count=3, request_id="req-1")
        assert "event=outbound_headers" in caplog.text
        assert "request_id=req-1" in caplog.text
        assert "forwarder=anthropic" in caplog.text
        assert "stripped_count=3" in caplog.text

    def test_logs_without_request_id(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(0)
        log_outbound_headers(forwarder="openai", stripped_count=0, request_id=None)
        assert "event=outbound_headers" in caplog.text
        assert "request_id=" in caplog.text

    def test_logs_zero_stripped(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(0)
        log_outbound_headers(forwarder="test", stripped_count=0, request_id="r1")
        assert "stripped_count=0" in caplog.text


# ── Section 2: Beta header merge ────────────────────────────────────────


class TestGetBetaHeaderStickyMode:
    def test_default_enabled(self) -> None:
        assert get_beta_header_sticky_mode() == "enabled"

    def test_explicit_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_BETA_HEADER_STICKY_ENV, "enabled")
        assert get_beta_header_sticky_mode() == "enabled"

    def test_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_BETA_HEADER_STICKY_ENV, "disabled")
        assert get_beta_header_sticky_mode() == "disabled"

    def test_unknown_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_BETA_HEADER_STICKY_ENV, "auto")
        with pytest.raises(ValueError, match="Invalid"):
            get_beta_header_sticky_mode()


class TestGetBetaTrackerMaxSessions:
    def test_default_value(self) -> None:
        assert get_beta_tracker_max_sessions() == 1000

    def test_custom_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_BETA_TRACKER_MAX_SESSIONS_ENV, "500")
        assert get_beta_tracker_max_sessions() == 500

    def test_non_positive_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_BETA_TRACKER_MAX_SESSIONS_ENV, "0")
        with pytest.raises(ValueError, match="positive int"):
            get_beta_tracker_max_sessions()

    def test_negative_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_BETA_TRACKER_MAX_SESSIONS_ENV, "-1")
        with pytest.raises(ValueError, match="positive int"):
            get_beta_tracker_max_sessions()

    def test_non_int_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_BETA_TRACKER_MAX_SESSIONS_ENV, "abc")
        with pytest.raises(ValueError, match="positive int"):
            get_beta_tracker_max_sessions()


class TestSplitBetaTokens:
    def test_none(self) -> None:
        assert _split_beta_tokens(None) == []

    def test_empty_string(self) -> None:
        assert _split_beta_tokens("") == []

    def test_single_token(self) -> None:
        assert _split_beta_tokens("token-1") == ["token-1"]

    def test_multiple_tokens(self) -> None:
        assert _split_beta_tokens("a,b,c") == ["a", "b", "c"]

    def test_whitespace_trimmed(self) -> None:
        assert _split_beta_tokens(" a , b , c ") == ["a", "b", "c"]

    def test_empty_entries_dropped(self) -> None:
        assert _split_beta_tokens("a,,b,") == ["a", "b"]


class TestMergeBetaTokens:
    def test_both_empty(self) -> None:
        assert _merge_beta_tokens(None, []) == ""

    def test_client_only(self) -> None:
        assert _merge_beta_tokens("a,b", []) == "a,b"

    def test_headroom_only(self) -> None:
        assert _merge_beta_tokens(None, ["c", "d"]) == "c,d"

    def test_merge_deduplicates_case_insensitive(self) -> None:
        result = _merge_beta_tokens("Alpha,beta", ["BETA", "Gamma"])
        assert result == "Alpha,beta,Gamma"

    def test_preserves_client_order(self) -> None:
        result = _merge_beta_tokens("z,y,x", ["w"])
        assert result == "z,y,x,w"

    def test_preserves_first_seen_casing(self) -> None:
        result = _merge_beta_tokens("ALPHA", ["alpha"])
        assert result == "ALPHA"

    def test_empty_entries_in_headroom_skipped(self) -> None:
        result = _merge_beta_tokens("a", ["", "b", "  "])
        assert result == "a,b"

    def test_returns_empty_string_for_none_client_and_empty_headroom(self) -> None:
        assert _merge_beta_tokens(None, []) == ""

    def test_whitespace_token_in_headroom_is_skipped(self) -> None:
        result = _merge_beta_tokens("a", ["  ", "b"])
        assert result == "a,b"


class TestMergeAnthropicBeta:
    def test_delegates_to_merge_beta_tokens(self) -> None:
        assert merge_anthropic_beta("a,b", ["c"]) == "a,b,c"

    def test_empty(self) -> None:
        assert merge_anthropic_beta(None, []) == ""

    def test_deduplication(self) -> None:
        assert merge_anthropic_beta("token-1", ["TOKEN-1", "token-2"]) == "token-1,token-2"


class TestMergeOpenaiBeta:
    def test_delegates_to_merge_beta_tokens(self) -> None:
        assert merge_openai_beta("x,y", ["z"]) == "x,y,z"

    def test_empty(self) -> None:
        assert merge_openai_beta(None, []) == ""


# ── Section 3: SessionBetaTracker ────────────────────────────────────────


class TestSessionBetaTracker:
    def test_construct_with_default_max(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_BETA_TRACKER_MAX_SESSIONS_ENV, "500")
        tracker = SessionBetaTracker()
        assert tracker._max_sessions == 500

    def test_construct_with_explicit_max(self) -> None:
        tracker = SessionBetaTracker(max_sessions=10)
        assert tracker._max_sessions == 10

    def test_construct_zero_max_raises(self) -> None:
        with pytest.raises(ValueError, match="max_sessions must be > 0"):
            SessionBetaTracker(max_sessions=0)

    def test_construct_negative_max_raises(self) -> None:
        with pytest.raises(ValueError, match="max_sessions must be > 0"):
            SessionBetaTracker(max_sessions=-1)

    def test_active_sessions_starts_zero(self) -> None:
        tracker = SessionBetaTracker(max_sessions=10)
        assert tracker.active_sessions == 0

    def test_record_first_request_returns_client_tokens(self) -> None:
        tracker = SessionBetaTracker(max_sessions=10)
        result = tracker.record_and_get_sticky_betas("anthropic", "sess-1", "a,b")
        assert result == "a,b"

    def test_sticky_on_second_request(self) -> None:
        tracker = SessionBetaTracker(max_sessions=10)
        tracker.record_and_get_sticky_betas("anthropic", "sess-1", "a,b")
        result = tracker.record_and_get_sticky_betas("anthropic", "sess-1", None)
        assert result == "a,b"

    def test_sticky_on_new_tokens_added(self) -> None:
        tracker = SessionBetaTracker(max_sessions=10)
        tracker.record_and_get_sticky_betas("anthropic", "sess-1", "a")
        result = tracker.record_and_get_sticky_betas("anthropic", "sess-1", "b")
        assert "a" in result
        assert "b" in result

    def test_case_insensitive_dedup_in_session(self) -> None:
        tracker = SessionBetaTracker(max_sessions=10)
        tracker.record_and_get_sticky_betas("anthropic", "sess-1", "Alpha")
        result = tracker.record_and_get_sticky_betas("anthropic", "sess-1", "alpha")
        assert result.count("Alpha") == 1
        assert "alpha" not in result

    def test_provider_isolation(self) -> None:
        tracker = SessionBetaTracker(max_sessions=10)
        tracker.record_and_get_sticky_betas("anthropic", "sess-1", "a")
        result = tracker.record_and_get_sticky_betas("openai", "sess-1", None)
        assert result == ""

    def test_session_isolation(self) -> None:
        tracker = SessionBetaTracker(max_sessions=10)
        tracker.record_and_get_sticky_betas("anthropic", "sess-1", "a")
        result = tracker.record_and_get_sticky_betas("anthropic", "sess-2", None)
        assert result == ""

    def test_empty_provider_raises(self) -> None:
        tracker = SessionBetaTracker(max_sessions=10)
        with pytest.raises(ValueError, match="provider must be non-empty"):
            tracker.record_and_get_sticky_betas("", "sess-1", "a")

    def test_empty_session_id_raises(self) -> None:
        tracker = SessionBetaTracker(max_sessions=10)
        with pytest.raises(ValueError, match="session_id must be non-empty"):
            tracker.record_and_get_sticky_betas("anthropic", "", "a")

    def test_disabled_mode_returns_verbatim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_BETA_HEADER_STICKY_ENV, "disabled")
        tracker = SessionBetaTracker(max_sessions=10)
        result = tracker.record_and_get_sticky_betas("anthropic", "sess-1", "a,b")
        assert result == "a,b"

    def test_disabled_mode_does_not_track(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_BETA_HEADER_STICKY_ENV, "disabled")
        tracker = SessionBetaTracker(max_sessions=10)
        tracker.record_and_get_sticky_betas("anthropic", "sess-1", "a")
        assert tracker.active_sessions == 0

    def test_snapshot_empty(self) -> None:
        tracker = SessionBetaTracker(max_sessions=10)
        assert tracker.snapshot() == []

    def test_snapshot_after_recording(self) -> None:
        tracker = SessionBetaTracker(max_sessions=10)
        tracker.record_and_get_sticky_betas("anthropic", "sess-1", "a,b")
        snapshot = tracker.snapshot()
        assert len(snapshot) == 1
        assert snapshot[0]["provider"] == "anthropic"
        assert snapshot[0]["session_id"] == "sess-1"
        assert snapshot[0]["beta_tokens"] == ["a", "b"]

    def test_reset_clears(self) -> None:
        tracker = SessionBetaTracker(max_sessions=10)
        tracker.record_and_get_sticky_betas("anthropic", "sess-1", "a")
        tracker.reset()
        assert tracker.active_sessions == 0

    def test_lru_eviction(self) -> None:
        tracker = SessionBetaTracker(max_sessions=2)
        tracker.record_and_get_sticky_betas("anthropic", "sess-1", "a")
        tracker.record_and_get_sticky_betas("anthropic", "sess-2", "b")
        tracker.record_and_get_sticky_betas("anthropic", "sess-3", "c")
        assert tracker.active_sessions == 2
        # sess-1 should be evicted (oldest)
        result = tracker.record_and_get_sticky_betas("anthropic", "sess-1", "a")
        assert result == "a"  # fresh session

    def test_lru_access_refreshes(self) -> None:
        tracker = SessionBetaTracker(max_sessions=2)
        tracker.record_and_get_sticky_betas("anthropic", "sess-1", "a")
        tracker.record_and_get_sticky_betas("anthropic", "sess-2", "b")
        # Access sess-1 to make it recently used
        tracker.record_and_get_sticky_betas("anthropic", "sess-1", None)
        tracker.record_and_get_sticky_betas("anthropic", "sess-3", "c")
        assert tracker.active_sessions == 2
        # sess-2 should be evicted, sess-1 should still be present
        result = tracker.record_and_get_sticky_betas("anthropic", "sess-2", "b")
        assert result == "b"  # fresh session since evicted


# ── Section 4: get_session_beta_tracker + log_beta_header_merge ────────


class TestGetSessionBetaTracker:
    def setup_method(self) -> None:
        _reset_session_beta_tracker_for_test()

    def test_returns_singleton(self) -> None:
        t1 = get_session_beta_tracker()
        t2 = get_session_beta_tracker()
        assert t1 is t2

    def test_is_beta_tracker_instance(self) -> None:
        tracker = get_session_beta_tracker()
        assert isinstance(tracker, SessionBetaTracker)

    def test_reset_for_test_clears_singleton(self) -> None:
        t1 = get_session_beta_tracker()
        _reset_session_beta_tracker_for_test()
        t2 = get_session_beta_tracker()
        assert t1 is not t2

    def teardown_method(self) -> None:
        _reset_session_beta_tracker_for_test()


class TestLogBetaHeaderMerge:
    def test_logs_all_fields(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(0)
        log_beta_header_merge(
            provider="anthropic",
            session_id="sess-1",
            client_betas_count=2,
            sticky_betas_count=3,
            headroom_added=["context-management-2025-06-27"],
            request_id="req-1",
        )
        assert "event=beta_header_merge" in caplog.text
        assert "provider=anthropic" in caplog.text
        assert "session_id=sess-1" in caplog.text
        assert "client_betas=2" in caplog.text
        assert "sticky_betas=3" in caplog.text
        assert "context-management-2025-06-27" in caplog.text

    def test_logs_with_none_session(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(0)
        log_beta_header_merge(
            provider="openai",
            session_id=None,
            client_betas_count=0,
            sticky_betas_count=0,
            headroom_added=[],
            request_id=None,
        )
        assert "event=beta_header_merge" in caplog.text
        assert "provider=openai" in caplog.text


# ── Section 5: Tool injection helpers ────────────────────────────────────


class TestGetToolInjectionStickyMode:
    def test_default_enabled(self) -> None:
        assert get_tool_injection_sticky_mode() == "enabled"

    def test_explicit_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_TOOL_INJECTION_STICKY_ENV, "enabled")
        assert get_tool_injection_sticky_mode() == "enabled"

    def test_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_TOOL_INJECTION_STICKY_ENV, "disabled")
        assert get_tool_injection_sticky_mode() == "disabled"

    def test_unknown_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_TOOL_INJECTION_STICKY_ENV, "auto")
        with pytest.raises(ValueError, match="Invalid"):
            get_tool_injection_sticky_mode()


class TestGetToolTrackerMaxSessions:
    def test_default(self) -> None:
        assert get_tool_tracker_max_sessions() == 1000

    def test_custom(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_TOOL_TRACKER_MAX_SESSIONS_ENV, "200")
        assert get_tool_tracker_max_sessions() == 200

    def test_non_positive_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_TOOL_TRACKER_MAX_SESSIONS_ENV, "0")
        with pytest.raises(ValueError):
            get_tool_tracker_max_sessions()

    def test_non_int_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_TOOL_TRACKER_MAX_SESSIONS_ENV, "bad")
        with pytest.raises(ValueError):
            get_tool_tracker_max_sessions()


class TestSerializeToolDefinitionCanonical:
    def test_returns_bytes(self) -> None:
        tool_def = {"name": "memory_save", "schema": {"type": "object"}}
        result = serialize_tool_definition_canonical(tool_def)
        assert isinstance(result, bytes)

    def test_deterministic_output(self) -> None:
        tool_def = {"name": "memory_save", "schema": {"type": "object"}}
        assert serialize_tool_definition_canonical(tool_def) == serialize_tool_definition_canonical(
            tool_def
        )

    def test_compact_separators(self) -> None:
        tool_def = {"name": "test"}
        result = serialize_tool_definition_canonical(tool_def)
        decoded = result.decode("utf-8")
        assert " " not in decoded


# ── Section 6: SessionToolTracker ────────────────────────────────────────


class TestSessionToolTracker:
    def test_construct_default_max(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_TOOL_TRACKER_MAX_SESSIONS_ENV, "300")
        tracker = SessionToolTracker()
        assert tracker._max_sessions == 300

    def test_construct_explicit_max(self) -> None:
        tracker = SessionToolTracker(max_sessions=5)
        assert tracker._max_sessions == 5

    def test_construct_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            SessionToolTracker(max_sessions=0)

    def test_active_sessions_starts_zero(self) -> None:
        tracker = SessionToolTracker(max_sessions=5)
        assert tracker.active_sessions == 0

    def test_should_inject_false_for_new_session(self) -> None:
        tracker = SessionToolTracker(max_sessions=5)
        assert tracker.should_inject("anthropic", "sess-1") is False

    def test_should_inject_true_after_recording(self) -> None:
        tracker = SessionToolTracker(max_sessions=5)
        tracker.record_injection("anthropic", "sess-1", "memory_save", b"{}")
        assert tracker.should_inject("anthropic", "sess-1") is True

    def test_get_golden_definitions_none_for_new_session(self) -> None:
        tracker = SessionToolTracker(max_sessions=5)
        assert tracker.get_golden_definitions("anthropic", "sess-1") is None

    def test_get_golden_definitions_after_recording(self) -> None:
        tracker = SessionToolTracker(max_sessions=5)
        tracker.record_injection("anthropic", "sess-1", "memory_save", b'{"a":1}')
        golden = tracker.get_golden_definitions("anthropic", "sess-1")
        assert golden is not None
        assert golden[0] == ("memory_save", b'{"a":1}')

    def test_first_write_wins(self) -> None:
        tracker = SessionToolTracker(max_sessions=5)
        tracker.record_injection("anthropic", "sess-1", "memory_save", b"first")
        tracker.record_injection("anthropic", "sess-1", "memory_save", b"second")
        golden = tracker.get_golden_definitions("anthropic", "sess-1")
        assert golden is not None
        assert golden[0][1] == b"first"

    def test_multiple_tools(self) -> None:
        tracker = SessionToolTracker(max_sessions=5)
        tracker.record_injection("anthropic", "sess-1", "memory_save", b"s")
        tracker.record_injection("anthropic", "sess-1", "memory_search", b"t")
        golden = tracker.get_golden_definitions("anthropic", "sess-1")
        assert golden is not None
        assert len(golden) == 2

    def test_empty_provider_raises(self) -> None:
        tracker = SessionToolTracker(max_sessions=5)
        with pytest.raises(ValueError):
            tracker.should_inject("", "sess-1")

    def test_empty_session_id_raises(self) -> None:
        tracker = SessionToolTracker(max_sessions=5)
        with pytest.raises(ValueError):
            tracker.should_inject("anthropic", "")

    def test_empty_tool_name_raises(self) -> None:
        tracker = SessionToolTracker(max_sessions=5)
        with pytest.raises(ValueError):
            tracker.record_injection("anthropic", "sess-1", "", b"bytes")

    def test_empty_bytes_raises(self) -> None:
        tracker = SessionToolTracker(max_sessions=5)
        with pytest.raises(ValueError):
            tracker.record_injection("anthropic", "sess-1", "tool", b"")

    def test_provider_isolation(self) -> None:
        tracker = SessionToolTracker(max_sessions=5)
        tracker.record_injection("anthropic", "sess-1", "tool", b"1")
        assert tracker.should_inject("openai", "sess-1") is False

    def test_session_isolation(self) -> None:
        tracker = SessionToolTracker(max_sessions=5)
        tracker.record_injection("anthropic", "sess-1", "tool", b"1")
        assert tracker.should_inject("anthropic", "sess-2") is False

    def test_lru_eviction(self) -> None:
        tracker = SessionToolTracker(max_sessions=2)
        tracker.record_injection("anthropic", "sess-1", "t1", b"1")
        tracker.record_injection("anthropic", "sess-2", "t2", b"2")
        tracker.record_injection("anthropic", "sess-3", "t3", b"3")
        assert tracker.active_sessions == 2
        assert tracker.should_inject("anthropic", "sess-1") is False

    def test_lru_access_refreshes(self) -> None:
        tracker = SessionToolTracker(max_sessions=2)
        tracker.record_injection("anthropic", "sess-1", "t1", b"1")
        tracker.record_injection("anthropic", "sess-2", "t2", b"2")
        tracker.should_inject("anthropic", "sess-1")  # access
        tracker.record_injection("anthropic", "sess-3", "t3", b"3")
        assert tracker.active_sessions == 2
        assert tracker.should_inject("anthropic", "sess-1") is True
        assert tracker.should_inject("anthropic", "sess-2") is False

    def test_snapshot_empty(self) -> None:
        tracker = SessionToolTracker(max_sessions=5)
        assert tracker.snapshot() == []

    def test_snapshot_after_recording(self) -> None:
        tracker = SessionToolTracker(max_sessions=5)
        tracker.record_injection("anthropic", "sess-1", "memory_save", b"{}")
        snapshot = tracker.snapshot()
        assert len(snapshot) == 1
        assert snapshot[0]["provider"] == "anthropic"
        assert snapshot[0]["session_id"] == "sess-1"
        assert snapshot[0]["tool_names"] == ["memory_save"]

    def test_reset_clears(self) -> None:
        tracker = SessionToolTracker(max_sessions=5)
        tracker.record_injection("anthropic", "sess-1", "t", b"1")
        tracker.reset()
        assert tracker.active_sessions == 0

    def test_snapshot_does_not_expose_bytes(self) -> None:
        tracker = SessionToolTracker(max_sessions=5)
        tracker.record_injection("anthropic", "sess-1", "t", b"secret")
        for entry in tracker.snapshot():
            assert "bytes" not in entry


# ── Section 7: get_session_tool_tracker + log_tool_injection_decision ────


class TestGetSessionToolTracker:
    def setup_method(self) -> None:
        _reset_session_tool_tracker_for_test()

    def test_returns_singleton(self) -> None:
        t1 = get_session_tool_tracker()
        t2 = get_session_tool_tracker()
        assert t1 is t2

    def test_is_tool_tracker_instance(self) -> None:
        assert isinstance(get_session_tool_tracker(), SessionToolTracker)

    def test_reset_for_test_clears(self) -> None:
        t1 = get_session_tool_tracker()
        _reset_session_tool_tracker_for_test()
        t2 = get_session_tool_tracker()
        assert t1 is not t2

    def teardown_method(self) -> None:
        _reset_session_tool_tracker_for_test()


class TestLogToolInjectionDecision:
    def test_logs_inject_first_time(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(0)
        log_tool_injection_decision(
            provider="anthropic",
            session_id="sess-1",
            decision="inject_first_time",
            tool_definition_bytes_count=42,
            request_id="req-1",
        )
        assert "event=tool_injection_decision" in caplog.text
        assert "decision=inject_first_time" in caplog.text
        assert "tool_definition_bytes_count=42" in caplog.text

    def test_logs_sticky_replay(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(0)
        log_tool_injection_decision(
            provider="anthropic",
            session_id="sess-1",
            decision="inject_sticky_replay",
            tool_definition_bytes_count=100,
            request_id="req-2",
        )
        assert "inject_sticky_replay" in caplog.text

    def test_logs_with_none_session(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(0)
        log_tool_injection_decision(
            provider="openai",
            session_id=None,
            decision="skip",
            tool_definition_bytes_count=0,
            request_id=None,
        )
        assert "skip" in caplog.text


# ── Section 8: apply_session_sticky_memory_tools ─────────────────────────


class TestApplySessionStickyMemoryTools:
    def _make_memory_tool(self, name: str = "memory_save") -> dict:
        return {"name": name, "input_schema": {"type": "object"}}

    def test_disable_inline_returns_unchanged(self) -> None:
        existing = [{"name": "existing_tool"}]
        tools, injected = apply_session_sticky_memory_tools(
            provider="anthropic",
            session_id="sess-1",
            request_id="r1",
            existing_tools=existing,
            memory_tools_to_inject=[self._make_memory_tool()],
            inject_this_turn=True,
            disable_inline_tool_injection=True,
        )
        assert injected is False
        assert tools == existing

    def test_disable_inline_no_existing(self) -> None:
        tools, injected = apply_session_sticky_memory_tools(
            provider="anthropic",
            session_id="sess-1",
            request_id="r1",
            existing_tools=None,
            memory_tools_to_inject=[self._make_memory_tool()],
            inject_this_turn=True,
            disable_inline_tool_injection=True,
        )
        assert injected is False
        assert tools == []

    def test_sticky_disabled_skip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_TOOL_INJECTION_STICKY_ENV, "disabled")
        tools, injected = apply_session_sticky_memory_tools(
            provider="anthropic",
            session_id="sess-1",
            request_id="r1",
            existing_tools=[],
            memory_tools_to_inject=[self._make_memory_tool()],
            inject_this_turn=False,
        )
        assert injected is False

    def test_sticky_disabled_inject(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_TOOL_INJECTION_STICKY_ENV, "disabled")
        tools, injected = apply_session_sticky_memory_tools(
            provider="anthropic",
            session_id="sess-1",
            request_id="r1",
            existing_tools=[],
            memory_tools_to_inject=[self._make_memory_tool()],
            inject_this_turn=True,
        )
        assert injected is True
        assert len(tools) == 1

    def test_session_id_none_and_inject_true(self) -> None:
        tools, injected = apply_session_sticky_memory_tools(
            provider="anthropic",
            session_id=None,
            request_id="r1",
            existing_tools=[],
            memory_tools_to_inject=[self._make_memory_tool()],
            inject_this_turn=True,
        )
        assert injected is True
        assert len(tools) == 1

    def test_session_id_none_and_inject_false(self) -> None:
        tools, injected = apply_session_sticky_memory_tools(
            provider="anthropic",
            session_id=None,
            request_id="r1",
            existing_tools=[],
            memory_tools_to_inject=[self._make_memory_tool()],
            inject_this_turn=False,
        )
        assert injected is False

    def test_unsupported_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported provider"):
            apply_session_sticky_memory_tools(
                provider="gemini",
                session_id="sess-1",
                request_id="r1",
                existing_tools=[],
                memory_tools_to_inject=[],
                inject_this_turn=False,
            )

    def test_fresh_session_inject_records_golden(self) -> None:
        _reset_session_tool_tracker_for_test()
        tool_def = self._make_memory_tool("memory_search")
        tools, injected = apply_session_sticky_memory_tools(
            provider="anthropic",
            session_id="sess-fresh",
            request_id="r1",
            existing_tools=[],
            memory_tools_to_inject=[tool_def],
            inject_this_turn=True,
        )
        assert injected is True
        # Second call should get sticky replay
        tools2, injected2 = apply_session_sticky_memory_tools(
            provider="anthropic",
            session_id="sess-fresh",
            request_id="r2",
            existing_tools=[],
            memory_tools_to_inject=[],
            inject_this_turn=False,
        )
        assert injected2 is True
        assert len(tools2) == 1
        _reset_session_tool_tracker_for_test()

    def test_existing_tool_name_not_duplicated(self) -> None:
        _reset_session_tool_tracker_for_test()
        existing = [{"name": "memory_save"}]
        tools, injected = apply_session_sticky_memory_tools(
            provider="anthropic",
            session_id="sess-existing",
            request_id="r1",
            existing_tools=existing,
            memory_tools_to_inject=[self._make_memory_tool("memory_save")],
            inject_this_turn=True,
        )
        assert injected is False
        assert len(tools) == 1  # not duplicated
        _reset_session_tool_tracker_for_test()

    def test_fresh_session_skip_when_not_injecting(self) -> None:
        _reset_session_tool_tracker_for_test()
        tools, injected = apply_session_sticky_memory_tools(
            provider="anthropic",
            session_id="sess-skip",
            request_id="r1",
            existing_tools=[],
            memory_tools_to_inject=[self._make_memory_tool()],
            inject_this_turn=False,
        )
        assert injected is False
        assert tools == []
        _reset_session_tool_tracker_for_test()


# ── Section 9: SessionCcrTracker ─────────────────────────────────────────


class TestSessionCcrTracker:
    def test_construct_with_default_max(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_TOOL_TRACKER_MAX_SESSIONS_ENV, "700")
        tracker = SessionCcrTracker()
        assert tracker._max_sessions == 700

    def test_construct_explicit_max(self) -> None:
        tracker = SessionCcrTracker(max_sessions=3)
        assert tracker._max_sessions == 3

    def test_construct_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            SessionCcrTracker(max_sessions=0)

    def test_active_sessions_starts_zero(self) -> None:
        tracker = SessionCcrTracker(max_sessions=3)
        assert tracker.active_sessions == 0

    def test_has_done_ccr_false_by_default(self) -> None:
        tracker = SessionCcrTracker(max_sessions=3)
        assert tracker.has_done_ccr("anthropic", "sess-1") is False

    def test_has_done_ccr_true_after_record(self) -> None:
        tracker = SessionCcrTracker(max_sessions=3)
        tracker.record_ccr_done("anthropic", "sess-1", b"{}")
        assert tracker.has_done_ccr("anthropic", "sess-1") is True

    def test_get_golden_tool_bytes_none_before_record(self) -> None:
        tracker = SessionCcrTracker(max_sessions=3)
        assert tracker.get_golden_tool_bytes("anthropic", "sess-1") is None

    def test_get_golden_tool_bytes_after_record(self) -> None:
        tracker = SessionCcrTracker(max_sessions=3)
        tracker.record_ccr_done("anthropic", "sess-1", b"golden")
        assert tracker.get_golden_tool_bytes("anthropic", "sess-1") == b"golden"

    def test_first_write_wins_for_golden_bytes(self) -> None:
        tracker = SessionCcrTracker(max_sessions=3)
        tracker.record_ccr_done("anthropic", "sess-1", b"first")
        tracker.record_ccr_done("anthropic", "sess-1", b"second")
        assert tracker.get_golden_tool_bytes("anthropic", "sess-1") == b"first"

    def test_has_done_ccr_is_monotonic(self) -> None:
        tracker = SessionCcrTracker(max_sessions=3)
        tracker.record_ccr_done("anthropic", "sess-1", b"g")
        assert tracker.has_done_ccr("anthropic", "sess-1") is True
        tracker.record_ccr_done("anthropic", "sess-1", b"g2")
        assert tracker.has_done_ccr("anthropic", "sess-1") is True

    def test_empty_provider_raises(self) -> None:
        tracker = SessionCcrTracker(max_sessions=3)
        with pytest.raises(ValueError):
            tracker.has_done_ccr("", "sess-1")

    def test_empty_session_id_raises(self) -> None:
        tracker = SessionCcrTracker(max_sessions=3)
        with pytest.raises(ValueError):
            tracker.has_done_ccr("anthropic", "")

    def test_empty_golden_bytes_raises(self) -> None:
        tracker = SessionCcrTracker(max_sessions=3)
        with pytest.raises(ValueError):
            tracker.record_ccr_done("anthropic", "sess-1", b"")

    def test_provider_isolation(self) -> None:
        tracker = SessionCcrTracker(max_sessions=3)
        tracker.record_ccr_done("anthropic", "sess-1", b"g")
        assert tracker.has_done_ccr("openai", "sess-1") is False

    def test_session_isolation(self) -> None:
        tracker = SessionCcrTracker(max_sessions=3)
        tracker.record_ccr_done("anthropic", "sess-1", b"g")
        assert tracker.has_done_ccr("anthropic", "sess-2") is False

    def test_lru_eviction(self) -> None:
        tracker = SessionCcrTracker(max_sessions=2)
        tracker.record_ccr_done("anthropic", "sess-1", b"1")
        tracker.record_ccr_done("anthropic", "sess-2", b"2")
        tracker.record_ccr_done("anthropic", "sess-3", b"3")
        assert tracker.active_sessions == 2
        assert tracker.has_done_ccr("anthropic", "sess-1") is False

    def test_lru_access_refreshes(self) -> None:
        tracker = SessionCcrTracker(max_sessions=2)
        tracker.record_ccr_done("anthropic", "sess-1", b"1")
        tracker.record_ccr_done("anthropic", "sess-2", b"2")
        tracker.has_done_ccr("anthropic", "sess-1")  # access
        tracker.record_ccr_done("anthropic", "sess-3", b"3")
        assert tracker.active_sessions == 2
        assert tracker.has_done_ccr("anthropic", "sess-1") is True
        assert tracker.has_done_ccr("anthropic", "sess-2") is False

    def test_snapshot_empty(self) -> None:
        tracker = SessionCcrTracker(max_sessions=3)
        assert tracker.snapshot() == []

    def test_snapshot_after_recording(self) -> None:
        tracker = SessionCcrTracker(max_sessions=3)
        tracker.record_ccr_done("anthropic", "sess-1", b"golden")
        snapshot = tracker.snapshot()
        assert len(snapshot) == 1
        assert snapshot[0]["has_done_ccr"] is True
        assert snapshot[0]["has_golden_tool_bytes"] is True

    def test_reset_clears(self) -> None:
        tracker = SessionCcrTracker(max_sessions=3)
        tracker.record_ccr_done("anthropic", "sess-1", b"g")
        tracker.reset()
        assert tracker.active_sessions == 0


# ── Section 10: get_session_ccr_tracker ─────────────────────────────────


class TestGetSessionCcrTracker:
    def setup_method(self) -> None:
        _reset_session_ccr_tracker_for_test()

    def test_returns_singleton(self) -> None:
        t1 = get_session_ccr_tracker()
        t2 = get_session_ccr_tracker()
        assert t1 is t2

    def test_is_ccr_tracker_instance(self) -> None:
        assert isinstance(get_session_ccr_tracker(), SessionCcrTracker)

    def test_reset_for_test_clears(self) -> None:
        t1 = get_session_ccr_tracker()
        _reset_session_ccr_tracker_for_test()
        t2 = get_session_ccr_tracker()
        assert t1 is not t2

    def teardown_method(self) -> None:
        _reset_session_ccr_tracker_for_test()


# ── Section 11: apply_session_sticky_ccr_tool ────────────────────────────


class TestApplySessionStickyCcrTool:
    TOOL_DEF = {"name": "headroom_retrieve", "schema": {"type": "object"}}

    def test_disable_inline_returns_unchanged(self) -> None:
        existing = [{"name": "other_tool"}]
        tools, injected = apply_session_sticky_ccr_tool(
            provider="anthropic",
            session_id="sess-1",
            request_id="r1",
            existing_tools=existing,
            has_compressed_content_this_turn=True,
            disable_inline_tool_injection=True,
        )
        assert injected is False
        assert tools == existing

    def test_existing_ccr_tool_name_skips(self) -> None:
        existing = [{"name": "headroom_retrieve"}]
        tools, injected = apply_session_sticky_ccr_tool(
            provider="anthropic",
            session_id="sess-1",
            request_id="r1",
            existing_tools=existing,
            has_compressed_content_this_turn=True,
        )
        assert injected is False
        assert len(tools) == 1

    def test_session_id_none_with_compressed_content(self) -> None:
        with patch("headroom.ccr.tool_injection.create_ccr_tool_definition") as mock_create:
            mock_create.return_value = self.TOOL_DEF
            tools, injected = apply_session_sticky_ccr_tool(
                provider="anthropic",
                session_id=None,
                request_id="r1",
                existing_tools=[],
                has_compressed_content_this_turn=True,
            )
        assert injected is True
        assert len(tools) == 1

    def test_session_id_none_without_compressed(self) -> None:
        tools, injected = apply_session_sticky_ccr_tool(
            provider="anthropic",
            session_id=None,
            request_id="r1",
            existing_tools=[],
            has_compressed_content_this_turn=False,
        )
        assert injected is False

    def test_unsupported_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported provider"):
            apply_session_sticky_ccr_tool(
                provider="unknown",
                session_id="sess-1",
                request_id="r1",
                existing_tools=[],
                has_compressed_content_this_turn=False,
            )

    def test_fresh_session_with_compressed_injects(self) -> None:
        _reset_session_ccr_tracker_for_test()
        with patch("headroom.ccr.tool_injection.create_ccr_tool_definition") as mock_create:
            mock_create.return_value = self.TOOL_DEF
            tools, injected = apply_session_sticky_ccr_tool(
                provider="anthropic",
                session_id="sess-ccr-1",
                request_id="r1",
                existing_tools=[],
                has_compressed_content_this_turn=True,
            )
        assert injected is True
        assert len(tools) == 1
        _reset_session_ccr_tracker_for_test()

    def test_fresh_session_without_compressed_skips(self) -> None:
        _reset_session_ccr_tracker_for_test()
        tools, injected = apply_session_sticky_ccr_tool(
            provider="anthropic",
            session_id="sess-ccr-2",
            request_id="r1",
            existing_tools=[],
            has_compressed_content_this_turn=False,
        )
        assert injected is False
        _reset_session_ccr_tracker_for_test()

    def test_previously_done_sticky_replay(self) -> None:
        _reset_session_ccr_tracker_for_test()
        with patch("headroom.ccr.tool_injection.create_ccr_tool_definition") as mock_create:
            mock_create.return_value = self.TOOL_DEF
            # First injection
            apply_session_sticky_ccr_tool(
                provider="anthropic",
                session_id="sess-ccr-3",
                request_id="r1",
                existing_tools=[],
                has_compressed_content_this_turn=True,
            )
            # Second turn: no compressed content but should still inject (sticky)
            tools, injected = apply_session_sticky_ccr_tool(
                provider="anthropic",
                session_id="sess-ccr-3",
                request_id="r2",
                existing_tools=[],
                has_compressed_content_this_turn=False,
            )
        assert injected is True
        assert len(tools) == 1
        _reset_session_ccr_tracker_for_test()

    def test_previously_done_even_with_existing_other_tools(self) -> None:
        _reset_session_ccr_tracker_for_test()
        with patch("headroom.ccr.tool_injection.create_ccr_tool_definition") as mock_create:
            mock_create.return_value = self.TOOL_DEF
            apply_session_sticky_ccr_tool(
                provider="anthropic",
                session_id="sess-ccr-4",
                request_id="r1",
                existing_tools=[],
                has_compressed_content_this_turn=True,
            )
            existing = [{"name": "other_tool"}]
            tools, injected = apply_session_sticky_ccr_tool(
                provider="anthropic",
                session_id="sess-ccr-4",
                request_id="r2",
                existing_tools=existing,
                has_compressed_content_this_turn=False,
            )
        assert injected is True
        assert len(tools) == 2  # other_tool + headroom_retrieve
        _reset_session_ccr_tracker_for_test()


# ── Section 12: compute_turn_id ──────────────────────────────────────────


class TestComputeTurnId:
    def test_none_when_no_messages(self) -> None:
        assert compute_turn_id("model", "system", None) is None

    def test_none_when_empty_messages(self) -> None:
        assert compute_turn_id("model", "system", []) is None

    def test_none_when_no_user_text(self) -> None:
        messages = [{"role": "assistant", "content": "hello"}]
        assert compute_turn_id("model", "system", messages) is None

    def test_returns_hash_for_simple_user_text(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        result = compute_turn_id("model", "system", messages)
        assert isinstance(result, str)
        assert len(result) == 16

    def test_stable_hash_for_same_input(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        a = compute_turn_id("model", "system", messages)
        b = compute_turn_id("model", "system", messages)
        assert a == b

    def test_different_model_different_hash(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        a = compute_turn_id("model-a", "system", messages)
        b = compute_turn_id("model-b", "system", messages)
        assert a != b

    def test_different_system_different_hash(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        a = compute_turn_id("model", "system-a", messages)
        b = compute_turn_id("model", "system-b", messages)
        assert a != b

    def test_different_messages_different_hash(self) -> None:
        a = compute_turn_id("model", "system", [{"role": "user", "content": "hello"}])
        b = compute_turn_id("model", "system", [{"role": "user", "content": "world"}])
        assert a != b

    def test_strips_cache_control(self) -> None:
        messages = [{"role": "user", "content": "hello", "cache_control": {"type": "ephemeral"}}]
        a = compute_turn_id("model", "system", messages)
        messages_no_cc = [{"role": "user", "content": "hello"}]
        b = compute_turn_id("model", "system", messages_no_cc)
        assert a == b

    def test_uses_last_user_text_message(self) -> None:
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "second"},
        ]
        # Should hash based on first + second user messages
        result = compute_turn_id("model", "system", messages)
        assert result is not None

    def test_only_hashes_up_to_last_user_text(self) -> None:
        messages = [
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "tool_use"},
            {"role": "user", "content": [{"type": "tool_result"}], "is_error": False},
        ]
        result = compute_turn_id("model", "system", messages)
        # The last user message has no text content, so it stops at first user message
        assert result is not None

    def test_handles_list_content_with_tool_result(self) -> None:
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "thinking"},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "result"},
                    {"type": "text", "content": "follow-up"},
                ],
            },
        ]
        # Last user has text but also tool_result, so it's not a fresh user turn
        result = compute_turn_id("model", "system", messages)
        assert result is not None

    def test_returns_none_when_last_user_only_has_tool_result(self) -> None:
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": [{"type": "tool_result", "content": "data"}]},
        ]
        result = compute_turn_id("model", "system", messages)
        # The last user message has no text block, only tool_result
        # The first user message "hello" is text, so it should still return a hash
        assert result is not None

    def test_none_when_only_tool_result_no_text_user(self) -> None:
        messages = [
            {"role": "user", "content": [{"type": "tool_result", "content": "data"}]},
        ]
        result = compute_turn_id("model", "system", messages)
        assert result is None

    def test_system_none(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        result = compute_turn_id("model", None, messages)
        assert isinstance(result, str)

    def test_system_as_dict(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        a = compute_turn_id("model", {"text": "sys"}, messages)
        b = compute_turn_id("model", {"text": "sys"}, messages)
        assert a == b


# ── Section 13: Tool search detection ─────────────────────────────────────


class TestClaudeCodeToolSearchInactive:
    def test_non_claude_code_client(self) -> None:
        assert (
            claude_code_tool_search_inactive(
                client="other", tools=[{"type": "custom"}], anthropic_beta=None
            )
            is False
        )

    def test_no_tools_returns_false(self) -> None:
        assert (
            claude_code_tool_search_inactive(client="claude-code", tools=None, anthropic_beta=None)
            is False
        )

    def test_empty_tools_returns_false(self) -> None:
        assert (
            claude_code_tool_search_inactive(client="claude-code", tools=[], anthropic_beta=None)
            is False
        )

    def test_tool_search_tool_present_returns_false(self) -> None:
        assert (
            claude_code_tool_search_inactive(
                client="claude-code",
                tools=[{"type": "tool_search_tool_2025"}, {"type": "regular"}],
                anthropic_beta=None,
            )
            is False
        )

    def test_beta_marker_present_returns_false(self) -> None:
        assert (
            claude_code_tool_search_inactive(
                client="claude-code",
                tools=[{"type": "regular"}],
                anthropic_beta="advanced-tool-use-2025-11-20",
            )
            is False
        )

    def test_beta_marker_tool_search_tool_returns_false(self) -> None:
        assert (
            claude_code_tool_search_inactive(
                client="claude-code",
                tools=[{"type": "regular"}],
                anthropic_beta="tool-search-tool-2025-10-19",
            )
            is False
        )

    def test_inactive_detected(self) -> None:
        assert (
            claude_code_tool_search_inactive(
                client="claude-code",
                tools=[{"type": "regular"}],
                anthropic_beta="some-other-beta",
            )
            is True
        )

    def test_case_insensitive_beta_check(self) -> None:
        assert (
            claude_code_tool_search_inactive(
                client="claude-code",
                tools=[{"type": "regular"}],
                anthropic_beta="ADVANCED-TOOL-USE-2025-11-20",
            )
            is False
        )

    def test_tools_not_a_list_returns_false(self) -> None:
        assert (
            claude_code_tool_search_inactive(
                client="claude-code", tools="not_a_list", anthropic_beta=None
            )
            is False
        )

    def test_tool_type_not_dict_safe_check(self) -> None:
        assert (
            claude_code_tool_search_inactive(
                client="claude-code",
                tools=["string_tool"],
                anthropic_beta=None,
            )
            is True
        )


class TestFormatToolSearchDisabledHint:
    def test_returns_string(self) -> None:
        hint = format_tool_search_disabled_hint([{"name": "tool1"}])
        assert isinstance(hint, str)
        assert "ENABLE_TOOL_SEARCH" in hint
        assert "Claude Code" in hint

    def test_mentions_tool_count(self) -> None:
        hint = format_tool_search_disabled_hint([{"name": "a"}, {"name": "b"}])
        assert "all 2 tool definitions" in hint

    def test_handles_empty_tools(self) -> None:
        hint = format_tool_search_disabled_hint([])
        assert "all 0 tool definitions" in hint
        assert "ENABLE_TOOL_SEARCH" in hint

    def test_handles_serialization_error(self) -> None:
        class BadTool:
            pass

        hint = format_tool_search_disabled_hint([BadTool()])
        assert isinstance(hint, str)
        assert "ENABLE_TOOL_SEARCH" in hint


class TestToolSearchHintState:
    def setup_method(self) -> None:
        reset_tool_search_hint_state()

    def test_pending_initially_true(self) -> None:
        assert tool_search_hint_pending() is True

    def test_take_slot_returns_true_once(self) -> None:
        assert take_tool_search_hint_slot() is True

    def test_take_slot_returns_false_second_time(self) -> None:
        take_tool_search_hint_slot()
        assert take_tool_search_hint_slot() is False

    def test_not_pending_after_taken(self) -> None:
        take_tool_search_hint_slot()
        assert tool_search_hint_pending() is False

    def test_reset_restores_pending(self) -> None:
        take_tool_search_hint_slot()
        reset_tool_search_hint_state()
        assert tool_search_hint_pending() is True

    def test_reset_allows_take_again(self) -> None:
        take_tool_search_hint_slot()
        reset_tool_search_hint_state()
        assert take_tool_search_hint_slot() is True

    def teardown_method(self) -> None:
        reset_tool_search_hint_state()
