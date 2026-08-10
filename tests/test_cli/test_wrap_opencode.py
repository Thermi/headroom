"""Tests for `headroom wrap opencode` command.

Covers:
  - Provider env-injection (OPENCODE_CONFIG_CONTENT + ANTHROPIC_BASE_URL)
  - Binary-not-found error
  - Extra-arg passthrough
  --no-proxy, --no-rtk, --port, --learn, --memory, --backend, --verbose
  - RTK context-tool injection path
  - lean-ctx context-tool injection path
  - Cleanup / shutdown via SystemExit
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_mod
from headroom.cli.main import main


@pytest.fixture(autouse=True)
def _enable_rtk(monkeypatch: pytest.MonkeyPatch) -> None:
    # RTK is opt-in (off by default); these tests exercise the RTK-on injection path.
    monkeypatch.setenv("HEADROOM_RTK", "1")


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)


def _capture_launch(runner, args, monkeypatch, tmp_path):
    """Helper: invoke wrap opencode, capturing _launch_tool kwargs."""
    captured: dict[str, object] = {}
    monkeypatch.chdir(tmp_path)

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(main, ["wrap", "opencode", *args])

    assert result.exit_code == 0, result.output
    return captured, result


# ---------------------------------------------------------------------------
# Provider env-injection
# ---------------------------------------------------------------------------


class TestEnvInjection:
    """OPENCODE_CONFIG_CONTENT and ANTHROPIC_BASE_URL must be set on launch."""

    def test_sets_both_env_vars(self, runner, tmp_path, monkeypatch) -> None:
        captured, _ = _capture_launch(runner, ["--port", "8787", "--no-rtk"], monkeypatch, tmp_path)
        env = captured["env"]
        assert isinstance(env, dict)
        assert "OPENCODE_CONFIG_CONTENT" in env
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"

    def test_config_content_has_headroom_provider(self, runner, tmp_path, monkeypatch) -> None:
        captured, _ = _capture_launch(runner, ["--port", "9999", "--no-rtk"], monkeypatch, tmp_path)
        env = captured["env"]
        config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        provider = config["provider"]["headroom"]
        assert provider["npm"] == "@ai-sdk/openai-compatible"
        assert provider["options"]["baseURL"] == "http://127.0.0.1:9999/v1"
        assert config["model"] == "headroom/claude-sonnet-4-6"

    def test_config_content_has_all_models(self, runner, tmp_path, monkeypatch) -> None:
        captured, _ = _capture_launch(runner, ["--no-rtk"], monkeypatch, tmp_path)
        env = captured["env"]
        config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        models = config["provider"]["headroom"]["models"]
        assert set(models.keys()) == {
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            "claude-haiku-4-5-20251001",
            "gpt-4o",
            "gpt-4.1",
        }

    def test_tool_label_and_agent_type(self, runner, tmp_path, monkeypatch) -> None:
        captured, _ = _capture_launch(runner, ["--no-rtk"], monkeypatch, tmp_path)
        assert captured["tool_label"] == "OPENCODE"
        assert captured["agent_type"] == "opencode"

    def test_default_port_is_8787(self, runner, tmp_path, monkeypatch) -> None:
        captured, _ = _capture_launch(runner, ["--no-rtk"], monkeypatch, tmp_path)
        assert captured["port"] == 8787


# ---------------------------------------------------------------------------
# Binary-not-found error
# ---------------------------------------------------------------------------


class TestMissingBinary:
    def test_errors_clearly(self, runner, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        with patch.object(wrap_mod.shutil, "which", return_value=None):
            result = runner.invoke(main, ["wrap", "opencode", "--no-rtk"])

        assert result.exit_code == 1
        assert "'opencode' not found in PATH" in result.output
        assert "opencode.ai" in result.output


# ---------------------------------------------------------------------------
# Extra-arg passthrough
# ---------------------------------------------------------------------------


class TestExtraArgs:
    def test_passes_through_model_arg(self, runner, tmp_path, monkeypatch) -> None:
        captured, _ = _capture_launch(
            runner,
            ["--no-rtk", "--", "--model", "headroom/gpt-4o"],
            monkeypatch,
            tmp_path,
        )
        assert captured["args"] == ("--model", "headroom/gpt-4o")

    def test_passes_multiple_args(self, runner, tmp_path, monkeypatch) -> None:
        captured, _ = _capture_launch(
            runner,
            ["--no-rtk", "--", "--model", "headroom/gpt-4o", "--verbose"],
            monkeypatch,
            tmp_path,
        )
        assert captured["args"] == ("--model", "headroom/gpt-4o", "--verbose")


# ---------------------------------------------------------------------------
# Flag forwarding
# ---------------------------------------------------------------------------


class TestFlagForwarding:
    def test_custom_port_propagates(self, runner, tmp_path, monkeypatch) -> None:
        captured, _ = _capture_launch(runner, ["--port", "1234", "--no-rtk"], monkeypatch, tmp_path)
        assert captured["port"] == 1234
        env = captured["env"]
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:1234"
        config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        assert config["provider"]["headroom"]["options"]["baseURL"] == "http://127.0.0.1:1234/v1"

    def test_no_proxy_flag(self, runner, tmp_path, monkeypatch) -> None:
        captured, _ = _capture_launch(runner, ["--no-proxy", "--no-rtk"], monkeypatch, tmp_path)
        assert captured["no_proxy"] is True

    def test_learn_flag(self, runner, tmp_path, monkeypatch) -> None:
        captured, _ = _capture_launch(runner, ["--learn", "--no-rtk"], monkeypatch, tmp_path)
        assert captured["learn"] is True

    def test_memory_flag(self, runner, tmp_path, monkeypatch) -> None:
        captured, _ = _capture_launch(runner, ["--memory", "--no-rtk"], monkeypatch, tmp_path)
        assert captured["memory"] is True

    def test_backend_option(self, runner, tmp_path, monkeypatch) -> None:
        captured, _ = _capture_launch(
            runner, ["--backend", "anyllm", "--no-rtk"], monkeypatch, tmp_path
        )
        assert captured["backend"] == "anyllm"

    def test_verbose_flag(self, runner, tmp_path, monkeypatch) -> None:
        captured, result = _capture_launch(runner, ["--verbose", "--no-rtk"], monkeypatch, tmp_path)
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# RTK context-tool path (default, no --no-rtk)
# ---------------------------------------------------------------------------


class TestRtkContextTool:
    """RTK setup is no longer managed by the upstream wrapper."""

    def test_injects_rtk_instructions_when_rtk_present(self, runner, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        inject_calls: list[Path] = []

        def track_inject(fp: Path, **kw: object) -> None:
            inject_calls.append(fp)

        with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                with patch.object(wrap_mod, "_inject_rtk_instructions", side_effect=track_inject):
                    with patch.object(wrap_mod, "_launch_tool"):
                        result = runner.invoke(main, ["wrap", "opencode"])

        assert result.exit_code == 0, result.output
        assert inject_calls == []

    def test_no_rtk_skips_context_tool_setup(self, runner, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
            with patch.object(wrap_mod, "_ensure_rtk_binary") as mock_rtk:
                with patch.object(wrap_mod, "_launch_tool"):
                    result = runner.invoke(main, ["wrap", "opencode", "--no-rtk"])

        assert result.exit_code == 0, result.output
        mock_rtk.assert_not_called()


# ---------------------------------------------------------------------------
# lean-ctx context-tool path
# ---------------------------------------------------------------------------


class TestLeanCtxContextTool:
    def test_uses_lean_ctx_when_configured(self, runner, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("HEADROOM_CONTEXT_TOOL", "lean-ctx")
        monkeypatch.chdir(tmp_path)

        with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
            with patch.object(wrap_mod, "_setup_lean_ctx_agent") as mock_lean_ctx:
                with patch.object(wrap_mod, "_launch_tool"):
                    result = runner.invoke(main, ["wrap", "opencode"])

        assert result.exit_code != 0
        assert "CLI context tools" in result.output
        mock_lean_ctx.assert_not_called()


# ---------------------------------------------------------------------------
# Cleanup / shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    """Verify the command exits cleanly."""

    def test_exits_with_zero_on_success(self, runner, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
            with patch.object(wrap_mod, "_launch_tool"):
                result = runner.invoke(main, ["wrap", "opencode", "--no-rtk"])

        assert result.exit_code == 0
