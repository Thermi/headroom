"""Tests for headroom.proxy.project_context — per-request project attribution."""

from __future__ import annotations

import pytest

from headroom.proxy.project_context import (
    PROJECT_HEADER,
    classify_project,
    get_current_project,
    set_current_project,
    split_project_path,
    strip_project_path_prefix,
    with_project_prefix,
)


@pytest.fixture(autouse=True)
def _reset_project_var() -> None:  # type: ignore[misc]
    set_current_project(None)
    yield
    set_current_project(None)


# ── classify_project ──────────────────────────────────────────────────


class TestClassifyProject:
    def test_reads_x_headroom_project(self) -> None:
        headers = {PROJECT_HEADER: "my-project"}
        assert classify_project(headers) == "my-project"

    def test_reads_x_headroom_project_case_fallback(self) -> None:
        headers = {"X-Headroom-Project": "fallback"}
        assert classify_project(headers) == "fallback"

    def test_returns_none_when_absent(self) -> None:
        assert classify_project({}) is None

    def test_sanitizes_value(self) -> None:
        headers = {PROJECT_HEADER: "  my-project  "}
        assert classify_project(headers) == "my-project"

    def test_returns_none_for_non_string(self) -> None:
        headers = {PROJECT_HEADER: 123}
        assert classify_project(headers) is None

    def test_returns_none_for_empty_string(self) -> None:
        headers = {PROJECT_HEADER: ""}
        assert classify_project(headers) is None

    def test_object_without_get(self) -> None:
        assert classify_project(42) is None


# ── set_current_project / get_current_project ─────────────────────────


class TestProjectContextVar:
    def test_set_and_get(self) -> None:
        set_current_project("my-proj")
        assert get_current_project() == "my-proj"

    def test_none(self) -> None:
        set_current_project(None)
        assert get_current_project() is None

    def test_sanitizes_on_set(self) -> None:
        set_current_project("  trimmed  ")
        assert get_current_project() == "trimmed"

    def test_set_none_clears(self) -> None:
        set_current_project("something")
        set_current_project(None)
        assert get_current_project() is None


# ── split_project_path ───────────────────────────────────────────────


class TestSplitProjectPath:
    def test_name_and_rest(self) -> None:
        project, rest = split_project_path("/p/my-project/v1/messages")
        assert project == "my-project"
        assert rest == "/v1/messages"

    def test_name_only(self) -> None:
        project, rest = split_project_path("/p/my-project")
        assert project == "my-project"
        assert rest == "/"

    def test_name_with_trailing_slash(self) -> None:
        project, rest = split_project_path("/p/my-project/")
        assert project == "my-project"
        assert rest == "/"

    def test_no_prefix(self) -> None:
        project, rest = split_project_path("/v1/messages")
        assert project is None
        assert rest == "/v1/messages"

    def test_empty_segment(self) -> None:
        project, rest = split_project_path("/p/")
        assert project is None
        assert rest == "/p/"

    def test_encoded_name(self) -> None:
        project, rest = split_project_path("/p/my%20project/rest")
        assert project == "my project"
        assert rest == "/rest"

    def test_empty_string(self) -> None:
        project, rest = split_project_path("")
        assert project is None
        assert rest == ""

    def test_only_prefix(self) -> None:
        project, rest = split_project_path("/p")
        assert project is None
        assert rest == "/p"


# ── strip_project_path_prefix ─────────────────────────────────────────


class TestStripProjectPathPrefix:
    def test_strips_and_returns_name(self) -> None:
        scope: dict[str, object] = {"path": "/p/my-proj/v1/messages"}
        assert strip_project_path_prefix(scope) == "my-proj"
        assert scope["path"] == "/v1/messages"

    def test_strips_raw_path(self) -> None:
        scope: dict[str, object] = {
            "path": "/p/my-proj/v1/messages",
            "raw_path": b"/p/my-proj/v1/messages",
        }
        strip_project_path_prefix(scope)
        assert isinstance(scope["raw_path"], bytes)
        assert b"/v1/messages" in scope["raw_path"]

    def test_returns_none_when_no_prefix(self) -> None:
        scope: dict[str, object] = {"path": "/v1/messages"}
        assert strip_project_path_prefix(scope) is None
        assert scope["path"] == "/v1/messages"

    def test_name_only(self) -> None:
        scope: dict[str, object] = {"path": "/p/my-proj"}
        assert strip_project_path_prefix(scope) == "my-proj"
        assert scope["path"] == "/"


# ── with_project_prefix ──────────────────────────────────────────────


class TestWithProjectPrefix:
    def test_basic(self) -> None:
        result = with_project_prefix("http://localhost:8080", "my-proj")
        assert result == "http://localhost:8080/p/my-proj"

    def test_existing_path(self) -> None:
        result = with_project_prefix("http://localhost:8080/v1/messages", "proj")
        assert "/p/proj/v1/messages" in result

    def test_none_project(self) -> None:
        base = "http://localhost:8080"
        assert with_project_prefix(base, None) == base

    def test_empty_project(self) -> None:
        base = "http://localhost:8080"
        assert with_project_prefix(base, "") == base

    def test_special_chars_encoded(self) -> None:
        result = with_project_prefix("http://localhost:8080", "my proj!")
        assert "my%20proj" in result
