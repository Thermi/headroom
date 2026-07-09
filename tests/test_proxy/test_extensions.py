"""Tests for headroom.proxy.extensions — third-party proxy extension point."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from headroom.proxy.extensions import (
    ENV_VAR,
    _resolve_enabled,
    discover,
    install_all,
)


class TestDiscover:
    """Verify extension discovery via entry points."""

    def test_no_entry_points_returns_nothing(self):
        with patch("importlib.metadata.entry_points", return_value=[]):
            result = list(discover())
        assert result == []

    def test_entry_point_load_failure_is_skipped(self):
        broken_ep = MagicMock()
        broken_ep.name = "broken"
        broken_ep.load.side_effect = ImportError("missing dep")
        with patch("importlib.metadata.entry_points", return_value=[broken_ep]):
            result = list(discover())
        assert result == []

    def test_entry_point_enumeration_failure_returns_nothing(self):
        with patch("importlib.metadata.entry_points", side_effect=Exception("boom")):
            result = list(discover())
        assert result == []

    def test_discovery_yields_name_and_callable(self):
        healthy_ep = MagicMock()
        healthy_ep.name = "my_ext"
        install_fn = MagicMock()
        healthy_ep.load.return_value = install_fn
        with patch("importlib.metadata.entry_points", return_value=[healthy_ep]):
            result = list(discover())
        assert len(result) == 1
        assert result[0][0] == "my_ext"
        assert result[0][1] is install_fn


class TestResolveEnabled:
    """Verify enabled-set resolution logic."""

    def test_explicit_enabled_overrides_env(self):
        with patch.dict("os.environ", {ENV_VAR: "from_env"}):
            result = _resolve_enabled(["explicit"])
        assert result == {"explicit"}

    def test_env_var_parsed_when_no_explicit_arg(self):
        with patch.dict("os.environ", {ENV_VAR: "ext_a,ext_b"}):
            result = _resolve_enabled(None)
        assert result == {"ext_a", "ext_b"}

    def test_empty_when_no_env_and_no_explicit(self):
        with patch.dict("os.environ", {}, clear=True):
            result = _resolve_enabled(None)
        assert result == set()

    def test_strips_whitespace(self):
        with patch.dict("os.environ", {ENV_VAR: " ext_a , ext_b "}):
            result = _resolve_enabled(None)
        assert result == {"ext_a", "ext_b"}

    def test_skips_empty_strings(self):
        result = _resolve_enabled(["ext_a", "", "ext_b"])
        assert result == {"ext_a", "ext_b"}


class TestInstallAll:
    """Verify install_all orchestration."""

    def test_no_enabled_returns_empty(self):
        with patch("headroom.proxy.extensions.discover", return_value=[]):
            installed = install_all(MagicMock(), MagicMock(), enabled=set())
        assert installed == []

    def test_wildcard_enables_all(self):
        ext_a = MagicMock()
        ext_b = MagicMock()
        with patch(
            "headroom.proxy.extensions.discover",
            return_value=[("ext_a", ext_a), ("ext_b", ext_b)],
        ):
            installed = install_all(MagicMock(), MagicMock(), enabled={"*"})
        assert installed == ["ext_a", "ext_b"]
        ext_a.assert_called_once()
        ext_b.assert_called_once()

    def test_selective_enable(self):
        ext_a = MagicMock()
        ext_b = MagicMock()
        with patch(
            "headroom.proxy.extensions.discover",
            return_value=[("ext_a", ext_a), ("ext_b", ext_b)],
        ):
            installed = install_all(MagicMock(), MagicMock(), enabled={"ext_a"})
        assert installed == ["ext_a"]
        ext_a.assert_called_once()
        ext_b.assert_not_called()

    def test_install_exception_propagates(self):
        failing_ext = MagicMock()
        failing_ext.side_effect = RuntimeError("license check failed")
        with patch(
            "headroom.proxy.extensions.discover",
            return_value=[("failing", failing_ext)],
        ):
            with pytest.raises(RuntimeError, match="license check failed"):
                install_all(MagicMock(), MagicMock(), enabled={"*"})

    def test_logs_warning_for_missing_extensions(self):
        with patch("headroom.proxy.extensions.discover", return_value=[("ext_a", MagicMock())]):
            with patch("headroom.proxy.extensions.log.warning") as mock_warn:
                install_all(MagicMock(), MagicMock(), enabled={"ext_a", "ext_b"})
                mock_warn.assert_called_once()
                args = mock_warn.call_args[0]
                assert any("ext_b" in str(a) for a in args)
