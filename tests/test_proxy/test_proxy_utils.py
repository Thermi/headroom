from __future__ import annotations

import ipaddress
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from headroom.proxy.forwarded_headers import (
    TRUSTED_GATEWAY_CIDRS_ENV,
    _header_first,
    _normalize_ip,
    _parse_cidr_list,
    _peer_host,
    _read_header,
    load_trusted_gateway_cidrs,
    peer_is_trusted_gateway,
    resolve_client_ip,
    trusted_forwarded_headers,
)
from headroom.proxy.loopback_guard import (
    LOOPBACK_HOSTS,
    is_loopback_host,
    is_loopback_host_header,
)
from headroom.proxy.modes import (
    PROXY_MODE_CACHE,
    PROXY_MODE_TOKEN,
    is_cache_mode,
    is_token_mode,
    normalize_proxy_mode,
)
from headroom.proxy.runtime_env import (
    RUNTIME_ENV_KNOBS,
    clear_overrides,
    effective_runtime_env,
    explicit_env,
    getenv,
    set_overrides,
)
from headroom.proxy.verbosity_controller import (
    ControllerState,
    Signal,
    VerbosityController,
    load_state,
    save_state,
)

# ── loopback_guard ────────────────────────────────────────────────────


class TestLoopbackGuard:
    def test_loopback_hosts_frozenset(self) -> None:
        assert LOOPBACK_HOSTS == frozenset({"127.0.0.1", "::1", "localhost"})

    def test_is_loopback_host_none(self) -> None:
        assert is_loopback_host(None) is True

    def test_is_loopback_host_ipv4(self) -> None:
        assert is_loopback_host("127.0.0.1") is True

    def test_is_loopback_host_ipv6(self) -> None:
        assert is_loopback_host("::1") is True

    def test_is_loopback_host_localhost(self) -> None:
        assert is_loopback_host("localhost") is True

    def test_is_loopback_host_localhost_case_insensitive(self) -> None:
        assert is_loopback_host("LOCALHOST") is True

    def test_is_loopback_host_non_loopback(self) -> None:
        assert is_loopback_host("192.168.1.1") is False

    def test_is_loopback_host_invalid(self) -> None:
        assert is_loopback_host("not-an-ip") is False

    def test_is_loopback_host_ipv4_mapped(self) -> None:
        assert is_loopback_host("::ffff:127.0.0.1") is True

    def test_is_loopback_host_empty_string(self) -> None:
        assert is_loopback_host("") is False

    def test_is_loopback_host_header_with_port(self) -> None:
        assert is_loopback_host_header("127.0.0.1:8787") is True

    def test_is_loopback_host_header_ipv6_bracketed(self) -> None:
        assert is_loopback_host_header("[::1]:8787") is True

    def test_is_loopback_host_header_localhost_port(self) -> None:
        assert is_loopback_host_header("localhost:8787") is True

    def test_is_loopback_host_header_empty(self) -> None:
        assert is_loopback_host_header("") is False

    def test_is_loopback_host_header_none(self) -> None:
        assert is_loopback_host_header(None) is False

    def test_is_loopback_host_header_evil(self) -> None:
        assert is_loopback_host_header("evil.com") is False

    def test_is_loopback_host_header_bare_ipv6_no_bracket(self) -> None:
        # Bare IPv6 without brackets can't have port, so the whole
        # string is treated as the host.
        assert is_loopback_host_header("::1") is True

    def test_is_loopback_host_header_unclosed_bracket(self) -> None:
        assert is_loopback_host_header("[::1") is False

    def test_is_loopback_host_header_whitespace_only(self) -> None:
        assert is_loopback_host_header("   ") is False


# ── forwarded_headers ────────────────────────────────────────────────


class TestForwardedHeaders:
    def test_parse_cidr_list_empty(self) -> None:
        assert _parse_cidr_list("") == ()

    def test_parse_cidr_list_whitespace(self) -> None:
        assert _parse_cidr_list("  ") == ()

    def test_parse_cidr_list_single(self) -> None:
        nets = _parse_cidr_list("10.0.0.0/8")
        assert len(nets) == 1
        assert nets[0] == ipaddress.IPv4Network("10.0.0.0/8")

    def test_parse_cidr_list_multi(self) -> None:
        nets = _parse_cidr_list("10.0.0.0/8, 192.168.0.0/16")
        assert len(nets) == 2

    def test_parse_cidr_list_trailing_comma(self) -> None:
        nets = _parse_cidr_list("10.0.0.0/8,")
        assert len(nets) == 1

    def test_parse_cidr_list_non_strict(self) -> None:
        nets = _parse_cidr_list("10.0.0.1/8")
        assert nets[0] == ipaddress.IPv4Network("10.0.0.0/8")

    def test_parse_cidr_list_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_cidr_list("not-a-cidr")

    def test_parse_cidr_list_ipv6(self) -> None:
        nets = _parse_cidr_list("fd00::/8")
        assert len(nets) == 1
        assert isinstance(nets[0], ipaddress.IPv6Network)

    def test_load_trusted_gateway_cidrs_from_env(self) -> None:
        with patch.dict(os.environ, {TRUSTED_GATEWAY_CIDRS_ENV: "10.0.0.0/8"}):
            nets = load_trusted_gateway_cidrs()
            assert len(nets) == 1

    def test_load_trusted_gateway_cidrs_explicit(self) -> None:
        nets = load_trusted_gateway_cidrs(raw="10.0.0.0/8, 172.16.0.0/12")
        assert len(nets) == 2

    def test_load_trusted_gateway_cidrs_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            nets = load_trusted_gateway_cidrs()
            assert nets == ()

    def test_normalize_ip_ipv4(self) -> None:
        addr = _normalize_ip("10.0.0.1")
        assert addr == ipaddress.IPv4Address("10.0.0.1")

    def test_normalize_ip_ipv4_mapped(self) -> None:
        addr = _normalize_ip("::ffff:10.0.0.1")
        assert addr == ipaddress.IPv4Address("10.0.0.1")

    def test_normalize_ip_invalid(self) -> None:
        assert _normalize_ip("bogus") is None

    def test_peer_is_trusted_gateway_empty_cidrs(self) -> None:
        assert peer_is_trusted_gateway("10.0.0.1", ()) is False

    def test_peer_is_trusted_gateway_none_peer(self) -> None:
        nets = (ipaddress.IPv4Network("10.0.0.0/8"),)
        assert peer_is_trusted_gateway(None, nets) is False

    def test_peer_is_trusted_gateway_match(self) -> None:
        nets = (ipaddress.IPv4Network("10.0.0.0/8"),)
        assert peer_is_trusted_gateway("10.0.0.1", nets) is True

    def test_peer_is_trusted_gateway_no_match(self) -> None:
        nets = (ipaddress.IPv4Network("10.0.0.0/8"),)
        assert peer_is_trusted_gateway("192.168.1.1", nets) is False

    def test_peer_is_trusted_gateway_family_mismatch(self) -> None:
        v4_nets = (ipaddress.IPv4Network("10.0.0.0/8"),)
        assert peer_is_trusted_gateway("fd00::1", v4_nets) is False

    def test_header_first_empty(self) -> None:
        assert _header_first("") == ""

    def test_header_first_single(self) -> None:
        assert _header_first("10.0.0.1") == "10.0.0.1"

    def test_header_first_chain(self) -> None:
        assert _header_first("10.0.0.1, 192.168.1.1, 172.16.0.1") == "10.0.0.1"

    def test_read_header_missing(self) -> None:
        request = MagicMock(headers={})
        assert _read_header(request, "x-forwarded-for") == ""

    def test_read_header_present(self) -> None:
        request = MagicMock(headers={"x-forwarded-for": "10.0.0.1"})
        assert _read_header(request, "x-forwarded-for") == "10.0.0.1"

    def test_read_header_no_headers_attr(self) -> None:
        request = MagicMock(spec=[])  # no headers attribute
        assert _read_header(request, "x-forwarded-for") == ""

    def test_read_header_bytes_value(self) -> None:
        request = MagicMock(headers={"x-forwarded-for": b"10.0.0.1"})
        assert _read_header(request, "x-forwarded-for") == "10.0.0.1"

    def test_peer_host_none(self) -> None:
        request = MagicMock(client=None)
        assert _peer_host(request) is None

    def test_peer_host_present(self) -> None:
        request = MagicMock()
        request.client.host = "10.0.0.1"
        assert _peer_host(request) == "10.0.0.1"

    def test_peer_host_no_client(self) -> None:
        request = MagicMock(spec=[])  # no client attribute
        assert _peer_host(request) is None

    def test_resolve_client_ip_no_forwarded(self) -> None:
        request = MagicMock()
        request.client.host = "10.0.0.1"
        request.state = MagicMock()
        request.state.client_ip = None
        request.state.forwarded = None
        request.headers = {}
        ip = resolve_client_ip(request)
        assert ip == "10.0.0.1"

    def test_resolve_client_ip_with_cache(self) -> None:
        request = MagicMock()
        request.state.client_ip = "cached-ip"
        request.state.forwarded = {"for": "cached"}
        ip = resolve_client_ip(request)
        assert ip == "cached-ip"

    def test_trusted_forwarded_headers_no_forwarded(self) -> None:
        request = MagicMock()
        request.client.host = "10.0.0.1"
        request.state = MagicMock()
        request.state.client_ip = None
        request.state.forwarded = None
        request.headers = {}
        fwd = trusted_forwarded_headers(request)
        assert fwd == {"for": "", "proto": "", "host": ""}


# ── modes ─────────────────────────────────────────────────────────────


class TestModes:
    def test_constants(self) -> None:
        assert PROXY_MODE_TOKEN == "token"
        assert PROXY_MODE_CACHE == "cache"

    def test_normalize_none(self) -> None:
        assert normalize_proxy_mode(None) == "token"

    def test_normalize_empty(self) -> None:
        assert normalize_proxy_mode("") == "token"

    def test_normalize_token(self) -> None:
        assert normalize_proxy_mode("token") == "token"

    def test_normalize_cache(self) -> None:
        assert normalize_proxy_mode("cache") == "cache"

    def test_normalize_token_mode_alias(self) -> None:
        assert normalize_proxy_mode("token_mode") == "token"

    def test_normalize_cost_savings_alias(self) -> None:
        assert normalize_proxy_mode("cost_savings") == "cache"

    def test_normalize_case_insensitive(self) -> None:
        assert normalize_proxy_mode("TOKEN") == "token"

    def test_normalize_unknown(self, caplog) -> None:
        result = normalize_proxy_mode("unknown_mode")
        assert result == "token"
        assert "Unknown HEADROOM_MODE" in caplog.text

    def test_normalize_custom_default(self) -> None:
        assert normalize_proxy_mode(None, default="cache") == "cache"

    def test_is_token_mode_true(self) -> None:
        assert is_token_mode("token") is True

    def test_is_token_mode_false(self) -> None:
        assert is_token_mode("cache") is False

    def test_is_cache_mode_true(self) -> None:
        assert is_cache_mode("cache") is True

    def test_is_cache_mode_false(self) -> None:
        assert is_cache_mode("token") is False


# ── runtime_env ───────────────────────────────────────────────────────


class TestRuntimeEnv:
    def test_knobs_registry(self) -> None:
        envs = [k.env for k in RUNTIME_ENV_KNOBS]
        assert "HEADROOM_OUTPUT_SHAPER" in envs
        assert "HEADROOM_VERBOSITY_LEVEL" in envs
        assert all(isinstance(k.summary, str) for k in RUNTIME_ENV_KNOBS)

    def setup_method(self) -> None:
        clear_overrides()

    def test_getenv_no_override_no_env(self) -> None:
        assert getenv("HEADROOM_OUTPUT_SHAPER") is None

    def test_getenv_no_override_with_default(self) -> None:
        assert getenv("HEADROOM_OUTPUT_SHAPER", "1") == "1"

    def test_getenv_override_wins(self) -> None:
        set_overrides({"HEADROOM_OUTPUT_SHAPER": "0"})
        assert getenv("HEADROOM_OUTPUT_SHAPER") == "0"

    def test_getenv_env_var_fallback(self) -> None:
        with patch.dict(os.environ, {"HEADROOM_OUTPUT_SHAPER": "1"}, clear=True):
            clear_overrides()
            assert getenv("HEADROOM_OUTPUT_SHAPER") == "1"

    def test_set_overrides_known_only(self) -> None:
        result = set_overrides({"HEADROOM_OUTPUT_SHAPER": "0", "UNKNOWN_KEY": "val"})
        assert result == {"HEADROOM_OUTPUT_SHAPER": "0"}
        assert getenv("UNKNOWN_KEY") is None

    def test_set_overrides_non_string_ignored(self) -> None:
        result = set_overrides({"HEADROOM_OUTPUT_SHAPER": 123})
        assert result == {}

    def test_clear_overrides(self) -> None:
        set_overrides({"HEADROOM_OUTPUT_SHAPER": "0"})
        clear_overrides()
        assert getenv("HEADROOM_OUTPUT_SHAPER") is None

    def test_explicit_env_returns_only_set_knobs(self) -> None:
        env = {"HEADROOM_OUTPUT_SHAPER": "1", "OTHER_VAR": "x"}
        result = explicit_env(env)
        assert "HEADROOM_OUTPUT_SHAPER" in result
        assert "OTHER_VAR" not in result

    def test_explicit_env_skips_empty(self) -> None:
        env = {"HEADROOM_OUTPUT_SHAPER": "", "HEADROOM_VERBOSITY_LEVEL": "3"}
        result = explicit_env(env)
        assert "HEADROOM_OUTPUT_SHAPER" not in result
        assert "HEADROOM_VERBOSITY_LEVEL" in result

    def test_effective_runtime_env(self) -> None:
        with patch.dict(os.environ, {"HEADROOM_OUTPUT_SHAPER": "1"}, clear=True):
            env = effective_runtime_env()
            assert env["HEADROOM_OUTPUT_SHAPER"] == "1"
            assert env["HEADROOM_VERBOSITY_LEVEL"] is None


# ── verbosity_controller ──────────────────────────────────────────────


class TestVerbosityController:
    def test_controller_state_defaults(self) -> None:
        s = ControllerState(level=2)
        assert s.level == 2
        assert s.up_streak == 0
        assert s.cooldown == 0

    def test_controller_state_to_dict(self) -> None:
        s = ControllerState(level=3, up_streak=2, cooldown=1)
        assert s.to_dict() == {"level": 3, "up_streak": 2, "cooldown": 1}

    def test_controller_state_from_dict(self) -> None:
        s = ControllerState.from_dict({"level": 4, "up_streak": 1, "cooldown": 3})
        assert s.level == 4
        assert s.up_streak == 1
        assert s.cooldown == 3

    def test_controller_state_from_dict_missing_keys(self) -> None:
        s = ControllerState.from_dict({})
        assert s.level == 2
        assert s.up_streak == 0
        assert s.cooldown == 0

    def test_observe_neutral_resets_streak(self) -> None:
        ctrl = VerbosityController()
        state = ControllerState(level=2, up_streak=3)
        new = ctrl.observe(state, Signal.NEUTRAL)
        assert new.level == 2
        assert new.up_streak == 0

    def test_observe_neutral_decrements_cooldown(self) -> None:
        ctrl = VerbosityController()
        state = ControllerState(level=2, cooldown=3)
        new = ctrl.observe(state, Signal.NEUTRAL)
        assert new.cooldown == 2

    def test_observe_too_little_drops_level_immediately(self) -> None:
        ctrl = VerbosityController()
        state = ControllerState(level=3)
        new = ctrl.observe(state, Signal.TOO_LITTLE)
        assert new.level == 2
        assert new.up_streak == 0

    def test_observe_too_little_respects_floor(self) -> None:
        ctrl = VerbosityController(floor=1)
        state = ControllerState(level=1)
        new = ctrl.observe(state, Signal.TOO_LITTLE)
        assert new.level == 1

    def test_observe_too_little_sets_cooldown(self) -> None:
        ctrl = VerbosityController(cooldown_turns=5)
        state = ControllerState(level=3)
        new = ctrl.observe(state, Signal.TOO_LITTLE)
        assert new.cooldown == 5

    def test_observe_too_much_increments_streak(self) -> None:
        ctrl = VerbosityController()
        state = ControllerState(level=2)
        new = ctrl.observe(state, Signal.TOO_MUCH)
        assert new.up_streak == 1
        assert new.level == 2

    def test_observe_too_much_probes_up_at_threshold(self) -> None:
        ctrl = VerbosityController(probe_threshold=2)
        state = ControllerState(level=2)
        s1 = ctrl.observe(state, Signal.TOO_MUCH)
        assert s1.level == 2
        s2 = ctrl.observe(s1, Signal.TOO_MUCH)
        assert s2.level == 3
        assert s2.up_streak == 0

    def test_observe_too_much_respects_ceil(self) -> None:
        ctrl = VerbosityController(probe_threshold=1, ceil=4)
        state = ControllerState(level=4)
        new = ctrl.observe(state, Signal.TOO_MUCH)
        assert new.level == 4

    def test_observe_too_much_during_cooldown(self) -> None:
        ctrl = VerbosityController(cooldown_turns=3)
        state = ControllerState(level=2, cooldown=3)
        new = ctrl.observe(state, Signal.TOO_MUCH)
        assert new.level == 2
        assert new.up_streak == 0
        assert new.cooldown == 2

    def test_load_state_from_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"level": 4, "up_streak": 2, "cooldown": 1}, f)
            path = f.name
        try:
            state = load_state(Path(path), default_level=2, floor=1, ceil=5)
            assert state.level == 4
            assert state.up_streak == 2
        finally:
            os.unlink(path)

    def test_load_state_file_not_found(self) -> None:
        state = load_state(Path("/nonexistent/state.json"), default_level=2, floor=1, ceil=5)
        assert state.level == 2
        assert state.up_streak == 0

    def test_load_state_clamps_above_ceil(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"level": 10}, f)
            path = f.name
        try:
            state = load_state(Path(path), default_level=2, floor=1, ceil=5)
            assert state.level == 5
        finally:
            os.unlink(path)

    def test_load_state_clamps_below_floor(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"level": 0}, f)
            path = f.name
        try:
            state = load_state(Path(path), default_level=2, floor=1, ceil=5)
            assert state.level == 1
        finally:
            os.unlink(path)

    def test_load_state_invalid_json(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid")
            path = f.name
        try:
            state = load_state(Path(path), default_level=3, floor=1, ceil=5)
            assert state.level == 3
        finally:
            os.unlink(path)

    def test_save_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "state.json"
            state = ControllerState(level=3, up_streak=1, cooldown=0)
            save_state(path, state)
            assert path.exists()
            data = json.loads(path.read_text())
            assert data == {"level": 3, "up_streak": 1, "cooldown": 0}

    def test_signal_enum_values(self) -> None:
        assert Signal.TOO_MUCH.value == "too_much"
        assert Signal.TOO_LITTLE.value == "too_little"
        assert Signal.NEUTRAL.value == "neutral"
