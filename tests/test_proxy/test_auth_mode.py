"""Tests for headroom.proxy.auth_mode — auth-mode classifier."""

from __future__ import annotations

from headroom.proxy.auth_mode import (
    CLIENT_UA_MAP,
    CODEX_RESPONSES_PATH,
    SUBSCRIPTION_UA_PREFIXES,
    AuthMode,
    classify_auth_mode,
    classify_client,
    should_stamp_codex_client,
)


def _h(**headers: str) -> dict[str, str]:
    return dict(headers)


# ── AuthMode enum ─────────────────────────────────────────────────────


class TestAuthMode:
    def test_str_values(self) -> None:
        assert AuthMode.PAYG == "payg"
        assert AuthMode.OAUTH == "oauth"
        assert AuthMode.SUBSCRIPTION == "subscription"

    def test_is_str_subclass(self) -> None:
        assert isinstance(AuthMode.PAYG, str)


# ── classify_auth_mode ───────────────────────────────────────────────


class TestClassifyAuthMode:
    # 1. Subscription UA prefix
    def test_claude_code_ua(self) -> None:
        assert (
            classify_auth_mode(_h(**{"user-agent": "claude-code/1.2.3"})) == AuthMode.SUBSCRIPTION
        )

    def test_codex_cli_ua(self) -> None:
        assert classify_auth_mode(_h(**{"user-agent": "codex-cli/2.0"})) == AuthMode.SUBSCRIPTION

    def test_cursor_ua(self) -> None:
        assert classify_auth_mode(_h(**{"user-agent": "cursor/0.40"})) == AuthMode.SUBSCRIPTION

    def test_subscription_ua_wins_over_bearer(self) -> None:
        """UA prefix takes priority over bearer token shape."""
        assert (
            classify_auth_mode(
                _h(
                    **{
                        "user-agent": "claude-code/1.0",
                        "authorization": "Bearer sk-ant-oat01-xxx",
                    }
                )
            )
            == AuthMode.SUBSCRIPTION
        )

    # 2. Bearer sk-ant-oat* → OAUTH
    def test_bearer_sk_ant_oat(self) -> None:
        assert (
            classify_auth_mode(_h(**{"authorization": "Bearer sk-ant-oat01-abcdef"}))
            == AuthMode.OAUTH
        )

    # 3. Bearer sk-ant-api* or sk-* → PAYG
    def test_bearer_sk_ant_api(self) -> None:
        assert (
            classify_auth_mode(_h(**{"authorization": "Bearer sk-ant-api01-xxx"})) == AuthMode.PAYG
        )

    def test_bearer_sk_openai(self) -> None:
        assert (
            classify_auth_mode(_h(**{"authorization": "Bearer sk-someother-yyy"})) == AuthMode.PAYG
        )

    # 4. JWT (3+ dot segments) → OAUTH
    def test_bearer_jwt(self) -> None:
        jwt = "eyJhbGci.eyJzdWIi.eyJpc3Mi.s"
        assert classify_auth_mode(_h(**{"authorization": f"Bearer {jwt}"})) == AuthMode.OAUTH

    # 5. Non-Bearer Authorization → OAUTH
    def test_basic_auth(self) -> None:
        assert classify_auth_mode(_h(**{"authorization": "Basic dXNlcjpwYXNz"})) == AuthMode.OAUTH

    def test_aws_sigv4(self) -> None:
        assert (
            classify_auth_mode(_h(**{"authorization": "AWS4-HMAC-SHA256 Credential=..."}))
            == AuthMode.OAUTH
        )

    # 6. x-api-key → PAYG
    def test_x_api_key(self) -> None:
        assert classify_auth_mode(_h(**{"x-api-key": "sk-xxx"})) == AuthMode.PAYG

    # 7. x-goog-api-key → PAYG
    def test_x_goog_api_key(self) -> None:
        assert classify_auth_mode(_h(**{"x-goog-api-key": "AIza..."})) == AuthMode.PAYG

    # 8. Default → PAYG
    def test_empty_headers(self) -> None:
        assert classify_auth_mode({}) == AuthMode.PAYG

    def test_no_auth_no_key(self) -> None:
        assert classify_auth_mode(_h(**{"user-agent": "custom-app/1.0"})) == AuthMode.PAYG

    # Malformed headers
    def test_bytes_authorization(self) -> None:
        assert classify_auth_mode(_h(**{"authorization": b"Bearer sk-test"})) == AuthMode.PAYG

    def test_non_utf8_authorization(self) -> None:
        assert classify_auth_mode(_h(**{"authorization": b"\xff\xfe"})) == AuthMode.PAYG

    def test_bytes_user_agent(self) -> None:
        assert classify_auth_mode(_h(**{"user-agent": b"claude-code/1.0"})) == AuthMode.SUBSCRIPTION

    def test_non_utf8_user_agent(self) -> None:
        assert classify_auth_mode(_h(**{"user-agent": b"\xff\xfe"})) == AuthMode.PAYG

    def test_none_headers_object(self) -> None:
        assert classify_auth_mode(None) == AuthMode.PAYG

    def test_bearer_unknown_short_token(self) -> None:
        """Bearer with a short non- sk- non-JWT token falls through to PAYG."""
        assert classify_auth_mode(_h(**{"authorization": "Bearer abc"})) == AuthMode.PAYG

    def test_subscription_prefix_all(self) -> None:
        for prefix in SUBSCRIPTION_UA_PREFIXES:
            assert classify_auth_mode(_h(**{"user-agent": f"{prefix}1.0"})) == AuthMode.SUBSCRIPTION


# ── classify_client ───────────────────────────────────────────────────


class TestClassifyClient:
    def test_explicit_x_client(self) -> None:
        assert classify_client(_h(**{"x-client": "aider"})) == "aider"

    def test_explicit_x_client_case_insensitive(self) -> None:
        assert classify_client(_h(**{"x-client": "Aider"})) == "aider"

    def test_x_client_trimmed(self) -> None:
        assert classify_client(_h(**{"x-client": "  aider  "})) == "aider"

    def test_x_client_wins_over_ua(self) -> None:
        assert classify_client(_h(**{"x-client": "custom", "user-agent": "cursor/1.0"})) == "custom"

    def test_ua_match_claude_code(self) -> None:
        assert classify_client(_h(**{"user-agent": "claude-code/1.2.3"})) == "claude-code"

    def test_ua_match_codex(self) -> None:
        assert classify_client(_h(**{"user-agent": "codex-cli/2.0"})) == "codex"

    def test_ua_match_cursor(self) -> None:
        assert classify_client(_h(**{"user-agent": "cursor/0.40"})) == "cursor"

    def test_ua_match_aider(self) -> None:
        assert classify_client(_h(**{"user-agent": "aider/0.50"})) == "aider"

    def test_ua_match_substring(self) -> None:
        """Match works even if the needle is in the middle of the UA."""
        assert classify_client(_h(**{"user-agent": "corporate-wrapper cursor/1.0"})) == "cursor"

    def test_no_match_returns_none(self) -> None:
        assert classify_client(_h(**{"user-agent": "custom-app/1.0"})) is None

    def test_no_match_with_default(self) -> None:
        assert (
            classify_client(_h(**{"user-agent": "custom-app/1.0"}), default="unknown") == "unknown"
        )

    def test_empty_headers(self) -> None:
        assert classify_client({}) is None

    def test_all_client_map_entries(self) -> None:
        for prefix, name in CLIENT_UA_MAP:
            assert classify_client(_h(**{"user-agent": f"{prefix}1.0"})) == name


# ── should_stamp_codex_client ─────────────────────────────────────────


class TestShouldStampCodexClient:
    def test_responses_path_unidentified(self) -> None:
        assert should_stamp_codex_client(CODEX_RESPONSES_PATH, {}) is True

    def test_responses_path_with_subpath(self) -> None:
        assert should_stamp_codex_client(CODEX_RESPONSES_PATH + "/abc", {}) is True

    def test_other_path_unidentified(self) -> None:
        assert should_stamp_codex_client("/v1/messages", {}) is False

    def test_responses_path_identified_client(self) -> None:
        assert (
            should_stamp_codex_client(
                CODEX_RESPONSES_PATH,
                _h(**{"user-agent": "cursor/1.0"}),
            )
            is False
        )

    def test_responses_path_explicit_x_client(self) -> None:
        assert (
            should_stamp_codex_client(
                CODEX_RESPONSES_PATH,
                _h(**{"x-client": "aider"}),
            )
            is False
        )

    def test_empty_path(self) -> None:
        assert should_stamp_codex_client("", {}) is False
