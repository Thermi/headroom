"""Tests for headroom.proxy.interceptors.astgrep."""

#  Copyright (c) 2026 Noel Kuntze

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

from headroom.proxy.interceptors.astgrep import (
    OUTLINE_MARKER,
    AstGrepReadOutline,
    _build_outline,
    _detect_lang_from_input,
    _min_chars_to_rewrite,
    _path_from_input,
    _run_ast_grep,
)
from headroom.proxy.interceptors.base import ToolResultInterceptor

# ---------------------------------------------------------------------------
# AstGrepReadOutline — protocol compliance
# ---------------------------------------------------------------------------


class TestAstGrepReadOutlineIsInterceptor:
    def test_is_tool_result_interceptor(self) -> None:
        assert isinstance(AstGrepReadOutline(), ToolResultInterceptor)

    def test_name(self) -> None:
        assert AstGrepReadOutline.name == "ast-grep"

    def test_has_all_required_methods(self) -> None:
        inst = AstGrepReadOutline()
        assert callable(inst.matches)
        assert callable(inst.transform)
        assert callable(inst.progressive_disclosure_key)


# ---------------------------------------------------------------------------
# matches()
# ---------------------------------------------------------------------------


class TestMatches:
    def setup_method(self) -> None:
        self.inst = AstGrepReadOutline()

    def test_matches_read_with_python_file(self) -> None:
        assert self.inst.matches("Read", {"file_path": "foo.py"}, "a" * 600)

    def test_rejects_non_read_tools(self) -> None:
        assert not self.inst.matches("Write", {"file_path": "foo.py"}, "a" * 600)
        assert not self.inst.matches("Edit", {"file_path": "foo.py"}, "a" * 600)
        assert not self.inst.matches(None, {}, "a" * 600)

    def test_rejects_short_output(self) -> None:
        assert not self.inst.matches("Read", {"file_path": "foo.py"}, "short")

    def test_respects_offset_range_key(self) -> None:
        assert not self.inst.matches("Read", {"file_path": "foo.py", "offset": 10}, "a" * 600)

    def test_respects_limit_range_key(self) -> None:
        assert not self.inst.matches("Read", {"file_path": "foo.py", "limit": 50}, "a" * 600)

    def test_respects_line_range_key(self) -> None:
        assert not self.inst.matches(
            "Read", {"file_path": "foo.py", "line_range": "10-20"}, "a" * 600
        )

    def test_respects_start_line_key(self) -> None:
        assert not self.inst.matches("Read", {"file_path": "foo.py", "start_line": 10}, "a" * 600)

    def test_respects_end_line_key(self) -> None:
        assert not self.inst.matches("Read", {"file_path": "foo.py", "end_line": 20}, "a" * 600)

    def test_respects_ranges_key(self) -> None:
        assert not self.inst.matches(
            "Read", {"file_path": "foo.py", "ranges": [[1, 10]]}, "a" * 600
        )

    def test_rejects_unsupported_extension(self) -> None:
        assert not self.inst.matches("Read", {"file_path": "foo.xyz"}, "a" * 600)

    def test_rejects_missing_file_path(self) -> None:
        assert not self.inst.matches("Read", {}, "a" * 600)

    def test_accepts_read_file_tool(self) -> None:
        assert self.inst.matches("read_file", {"file_path": "foo.py"}, "a" * 600)

    def test_accepts_view_tool(self) -> None:
        assert self.inst.matches("view", {"file_path": "foo.py"}, "a" * 600)

    def test_accepts_cat_tool(self) -> None:
        assert self.inst.matches("cat", {"file_path": "foo.py"}, "a" * 600)

    def test_supports_go_files(self) -> None:
        assert self.inst.matches("Read", {"file_path": "main.go"}, "a" * 600)

    def test_supports_rust_files(self) -> None:
        assert self.inst.matches("Read", {"file_path": "lib.rs"}, "a" * 600)

    def test_supports_java_files(self) -> None:
        assert self.inst.matches("Read", {"file_path": "App.java"}, "a" * 600)

    def test_min_chars_configurable(self) -> None:
        with patch("headroom.proxy.interceptors.astgrep.runtime_env.getenv", return_value="10"):
            assert self.inst.matches("Read", {"file_path": "foo.py"}, "a" * 15)
            assert not self.inst.matches("Read", {"file_path": "foo.py"}, "a" * 5)


# ---------------------------------------------------------------------------
# progressive_disclosure_key()
# ---------------------------------------------------------------------------


class TestProgressiveDisclosureKey:
    def setup_method(self) -> None:
        self.inst = AstGrepReadOutline()

    def test_returns_file_path(self) -> None:
        assert (
            self.inst.progressive_disclosure_key("Read", {"file_path": "/src/main.py"})
            == "/src/main.py"
        )

    def test_returns_path_alternative(self) -> None:
        assert self.inst.progressive_disclosure_key("Read", {"path": "foo/bar.ts"}) == "foo/bar.ts"

    def test_returns_filePath_alternative(self) -> None:
        assert self.inst.progressive_disclosure_key("Read", {"filePath": "baz.rs"}) == "baz.rs"

    def test_returns_filename_alternative(self) -> None:
        assert self.inst.progressive_disclosure_key("Read", {"filename": "test.go"}) == "test.go"

    def test_returns_none_when_no_path(self) -> None:
        assert self.inst.progressive_disclosure_key("Read", {}) is None

    def test_returns_none_for_empty_string_path(self) -> None:
        assert self.inst.progressive_disclosure_key("Read", {"file_path": ""}) is None

    def test_file_path_takes_priority(self) -> None:
        result = self.inst.progressive_disclosure_key(
            "Read", {"file_path": "/a.py", "path": "/b.py"}
        )
        assert result == "/a.py"

    def test_ignores_tool_name(self) -> None:
        key = self.inst.progressive_disclosure_key("Read", {"file_path": "f.py"})
        key2 = self.inst.progressive_disclosure_key("view", {"file_path": "f.py"})
        assert key == key2


# ---------------------------------------------------------------------------
# _detect_lang_from_input
# ---------------------------------------------------------------------------


class TestDetectLangFromInput:
    def test_python(self) -> None:
        assert _detect_lang_from_input({"file_path": "script.py"}) == "python"

    def test_typescript(self) -> None:
        assert _detect_lang_from_input({"file_path": "app.ts"}) == "typescript"

    def test_tsx(self) -> None:
        assert _detect_lang_from_input({"file_path": "component.tsx"}) == "tsx"

    def test_rust(self) -> None:
        assert _detect_lang_from_input({"file_path": "mod.rs"}) == "rust"

    def test_unknown_extension(self) -> None:
        assert _detect_lang_from_input({"file_path": "data.json"}) is None

    def test_missing_path(self) -> None:
        assert _detect_lang_from_input({}) is None

    def test_case_insensitive_extension(self) -> None:
        assert _detect_lang_from_input({"file_path": "SCRIPT.PY"}) == "python"

    def test_c_header(self) -> None:
        assert _detect_lang_from_input({"file_path": "header.h"}) == "c"

    def test_cpp_source(self) -> None:
        assert _detect_lang_from_input({"file_path": "main.cc"}) == "cpp"


# ---------------------------------------------------------------------------
# _path_from_input
# ---------------------------------------------------------------------------


class TestPathFromInput:
    def test_file_path_key(self) -> None:
        assert _path_from_input({"file_path": "/a/b.py"}) == "/a/b.py"

    def test_path_key(self) -> None:
        assert _path_from_input({"path": "relative.py"}) == "relative.py"

    def test_filePath_key(self) -> None:
        assert _path_from_input({"filePath": "doc.go"}) == "doc.go"

    def test_filename_key(self) -> None:
        assert _path_from_input({"filename": "test.js"}) == "test.js"

    def test_prefers_file_path(self) -> None:
        result = _path_from_input({"file_path": "first.py", "path": "second.py"})
        assert result == "first.py"

    def test_non_string_value_returns_none(self) -> None:
        assert _path_from_input({"file_path": 123}) is None

    def test_empty_dict_returns_none(self) -> None:
        assert _path_from_input({}) is None


# ---------------------------------------------------------------------------
# _build_outline
# ---------------------------------------------------------------------------


class TestBuildOutline:
    def test_no_matches_returns_none(self) -> None:
        assert _build_outline([], "some source code") is None

    def test_single_match(self) -> None:
        source = "def foo():\n    pass\n"
        matches = [
            {
                "range": {
                    "start": {"line": 0},
                    "byteOffset": {"start": 0},
                },
                "text": "def foo():",
            }
        ]
        result = _build_outline(matches, source)
        assert result is not None
        assert "def foo():" in result
        assert OUTLINE_MARKER in result

    def test_multiple_matches(self) -> None:
        source = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        matches = [
            {
                "range": {"start": {"line": 0}, "byteOffset": {"start": 0}},
                "text": "def foo():",
            },
            {
                "range": {"start": {"line": 3}, "byteOffset": {"start": 14}},
                "text": "def bar():",
            },
        ]
        result = _build_outline(matches, source)
        assert result is not None
        assert result.count("def ") == 2
        assert OUTLINE_MARKER in result

    def test_deduplicates_same_line(self) -> None:
        source = "def foo():\n    pass\n"
        matches = [
            {
                "range": {"start": {"line": 0}, "byteOffset": {"start": 0}},
                "text": "def foo():",
            },
            {
                "range": {"start": {"line": 0}, "byteOffset": {"start": 0}},
                "text": "def foo():",
            },
        ]
        result = _build_outline(matches, source)
        assert result is not None
        assert result.count("def foo():") == 1

    def test_includes_header(self) -> None:
        source = "def foo():\n    pass\n"
        matches = [
            {
                "range": {"start": {"line": 0}, "byteOffset": {"start": 0}},
                "text": "def foo():",
            }
        ]
        result = _build_outline(matches, source)
        assert result is not None
        assert result.startswith("[headroom: outlined by ast-grep")

    def test_includes_definition_count_in_header(self) -> None:
        source = "def a():\n    pass\ndef b():\n    pass\n"
        matches = [
            {
                "range": {"start": {"line": 0}, "byteOffset": {"start": 0}},
                "text": "def a():",
            },
            {
                "range": {"start": {"line": 2}, "byteOffset": {"start": 14}},
                "text": "def b():",
            },
        ]
        result = _build_outline(matches, source) or ""
        assert "2 definition" in result

    def test_preserves_docstring_after_signature(self) -> None:
        source = 'def foo():\n    """My docstring."""\n    pass\n'
        matches = [
            {
                "range": {"start": {"line": 0}, "byteOffset": {"start": 0}},
                "text": "def foo():",
            }
        ]
        result = _build_outline(matches, source) or ""
        assert '"""My docstring."""' in result

    def test_skips_line_out_of_range(self) -> None:
        matches = [
            {
                "range": {"start": {"line": 999}, "byteOffset": {"start": 0}},
                "text": "bad",
            }
        ]
        assert _build_outline(matches, "short") is None


# ---------------------------------------------------------------------------
# _run_ast_grep
# ---------------------------------------------------------------------------


class TestRunAstGrep:
    @patch("headroom.proxy.interceptors.astgrep.subprocess.run")
    def test_no_patterns_for_lang(self, mock_run) -> None:
        result = _run_ast_grep("sg", "unknown_lang", "source")
        assert result == []
        mock_run.assert_not_called()

    def _run_with_mock_subprocess(self, return_value=None, side_effect=None) -> list:
        """Run _run_ast_grep with subprocess.run mocked and a real temp dir."""
        real_tmp = tempfile.mkdtemp(prefix="test-astgrep-")
        try:
            with patch("headroom.proxy.interceptors.astgrep.subprocess.run") as mock_run:
                if side_effect is not None:
                    mock_run.side_effect = side_effect
                else:
                    mock_run.return_value = return_value
                with patch(
                    "headroom.proxy.interceptors.astgrep.tempfile.mkdtemp",
                    return_value=real_tmp,
                ):
                    return _run_ast_grep("sg", "python", "def foo():\n    pass\n")
        finally:
            import shutil

            shutil.rmtree(real_tmp, ignore_errors=True)

    def test_rc_1_no_matches(self) -> None:
        result = self._run_with_mock_subprocess(
            return_value=MagicMock(returncode=1, stdout="", stderr="")
        )
        assert result == []

    def test_rc_2_error(self) -> None:
        result = self._run_with_mock_subprocess(
            return_value=MagicMock(returncode=2, stdout="", stderr="syntax error")
        )
        assert result == []

    def test_parses_json_stream(self) -> None:
        result = self._run_with_mock_subprocess(
            return_value=MagicMock(
                returncode=0,
                stdout='{"range":{"start":{"line":0}},"text":"def foo():"}\n',
                stderr="",
            )
        )
        assert result
        assert result[0]["text"] == "def foo():"

    def test_timeout_is_caught(self) -> None:
        import subprocess

        result = self._run_with_mock_subprocess(
            side_effect=subprocess.TimeoutExpired(cmd="sg", timeout=5)
        )
        assert result == []

    def test_oserror_is_caught(self) -> None:
        result = self._run_with_mock_subprocess(side_effect=OSError("not found"))
        assert result == []


# ---------------------------------------------------------------------------
# _min_chars_to_rewrite
# ---------------------------------------------------------------------------


class TestMinCharsToRewrite:
    def test_default_value(self) -> None:
        with patch("headroom.proxy.interceptors.astgrep.runtime_env.getenv", return_value="500"):
            assert _min_chars_to_rewrite() == 500

    def test_custom_value(self) -> None:
        with patch("headroom.proxy.interceptors.astgrep.runtime_env.getenv", return_value="1000"):
            assert _min_chars_to_rewrite() == 1000

    def test_invalid_value_falls_back(self) -> None:
        with patch(
            "headroom.proxy.interceptors.astgrep.runtime_env.getenv", return_value="not-a-number"
        ):
            assert _min_chars_to_rewrite() == 500

    def test_empty_value(self) -> None:
        with patch("headroom.proxy.interceptors.astgrep.runtime_env.getenv", return_value=""):
            assert _min_chars_to_rewrite() == 500


# ---------------------------------------------------------------------------
# transform() — with mocked ast-grep
# ---------------------------------------------------------------------------


class TestTransform:
    def setup_method(self) -> None:
        self.inst = AstGrepReadOutline()

    @patch("headroom.proxy.interceptors.astgrep.binaries.resolve")
    def test_transform_returns_none_when_binary_unavailable(self, mock_resolve) -> None:
        from headroom.binaries import BinaryError

        mock_resolve.side_effect = BinaryError("not found")
        result = self.inst.transform("Read", {"file_path": "foo.py"}, "a" * 600)
        assert result is None

    @patch("headroom.proxy.interceptors.astgrep.binaries.resolve")
    def test_transform_returns_none_for_unsupported_lang(self, mock_resolve) -> None:
        result = self.inst.transform("Read", {"file_path": "foo.xyz"}, "a" * 600)
        assert result is None

    @patch("headroom.proxy.interceptors.astgrep._run_ast_grep")
    @patch("headroom.proxy.interceptors.astgrep.binaries.resolve")
    def test_transform_returns_none_when_no_matches(self, mock_resolve, mock_run) -> None:
        mock_run.return_value = []
        result = self.inst.transform("Read", {"file_path": "foo.py"}, "a" * 600)
        assert result is None

    @patch("headroom.proxy.interceptors.astgrep._run_ast_grep")
    @patch("headroom.proxy.interceptors.astgrep.binaries.resolve")
    def test_transform_returns_outline(self, mock_resolve, mock_run) -> None:
        mock_run.return_value = [
            {
                "range": {"start": {"line": 0}, "byteOffset": {"start": 0}},
                "text": "def foo():",
            }
        ]
        source = "def foo():\n    pass\n"
        result = self.inst.transform("Read", {"file_path": "foo.py"}, source)
        assert result is not None
        assert "def foo():" in result
        assert OUTLINE_MARKER in result
