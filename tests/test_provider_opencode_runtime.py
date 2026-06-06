"""Unit tests for headroom.providers.opencode build_launch_env."""

from __future__ import annotations

import json

from headroom.providers.opencode import build_launch_env, proxy_base_url


def test_proxy_base_url_returns_v1_path() -> None:
    assert proxy_base_url(8787) == "http://127.0.0.1:8787/v1"
    assert proxy_base_url(9999) == "http://127.0.0.1:9999/v1"


def test_build_launch_env_sets_opencode_config_content() -> None:
    env, lines = build_launch_env(8787, {"ANTHROPIC_API_KEY": "sk-test"})

    assert "OPENCODE_CONFIG_CONTENT" in env
    assert env["ANTHROPIC_API_KEY"] == "sk-test"

    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    assert config["provider"]["headroom"]["npm"] == "@ai-sdk/openai-compatible"
    assert config["model"] == "headroom/claude-sonnet-4-6"


def test_build_launch_env_sets_anthropic_base_url() -> None:
    env, lines = build_launch_env(8787, {})

    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"


def test_build_launch_env_port_in_base_url() -> None:
    env, lines = build_launch_env(9999, {})

    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    assert config["provider"]["headroom"]["options"]["baseURL"] == "http://127.0.0.1:9999/v1"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9999"


def test_build_launch_env_preserves_existing_env() -> None:
    env, lines = build_launch_env(8787, {"MY_VAR": "hello"})

    assert env["MY_VAR"] == "hello"
    assert "OPENCODE_CONFIG_CONTENT" in env
    assert "ANTHROPIC_BASE_URL" in env


def test_build_launch_env_uses_provided_environ() -> None:
    env, lines = build_launch_env(8787, {"CUSTOM_API_KEY": "k123"})

    assert env["CUSTOM_API_KEY"] == "k123"
    assert "ANTHROPIC_API_KEY" not in env


def test_build_launch_env_returns_display_lines() -> None:
    env, lines = build_launch_env(8787, {})

    assert len(lines) == 2
    assert lines[0] == "OPENCODE_CONFIG_CONTENT=<headroom provider config>"
    assert lines[1] == "ANTHROPIC_BASE_URL=http://127.0.0.1:8787"


def test_build_launch_env_config_has_all_models() -> None:
    env, lines = build_launch_env(8787, {})

    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    models = config["provider"]["headroom"]["models"]
    assert set(models.keys()) == {
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-haiku-4-5-20251001",
        "gpt-4o",
        "gpt-4.1",
    }


def test_build_launch_env_config_has_model_limits() -> None:
    env, lines = build_launch_env(8787, {})

    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    sonnet = config["provider"]["headroom"]["models"]["claude-sonnet-4-6"]
    assert sonnet["limit"]["context"] == 200000
    assert sonnet["limit"]["output"] == 16384
    gpt41 = config["provider"]["headroom"]["models"]["gpt-4.1"]
    assert gpt41["limit"]["context"] == 1048576
    assert gpt41["limit"]["output"] == 32768
