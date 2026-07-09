"""Tests for headroom.proxy.native_memory_file_ops — NativeFileToolHandler."""

from __future__ import annotations

from pathlib import Path

import pytest

from headroom.proxy.native_memory_file_ops import NativeFileToolHandler


@pytest.fixture
def handler(tmp_path: Path) -> NativeFileToolHandler:
    return NativeFileToolHandler(tmp_path / "memory")


def make_file(handler: NativeFileToolHandler, path: str, text: str, user: str = "alice") -> None:
    handler.create({"path": path, "file_text": text}, user)


class TestResolvePath:
    """Verify path resolution and traversal protection."""

    def test_resolves_normal_path(self, handler: NativeFileToolHandler, tmp_path: Path):
        resolved = handler.resolve_path("/memories/topic.txt", "alice")
        assert resolved.suffix == ".txt"
        assert resolved.stem == "topic"
        assert "alice" in resolved.parts
        assert "topic.txt" in resolved.parts
        assert resolved.exists() is False

    def test_strips_memories_prefix(self, handler: NativeFileToolHandler, tmp_path: Path):
        resolved = handler.resolve_path("/memories/sub/dir/file.txt", "alice")
        assert "file.txt" in resolved.parts
        assert "dir" in resolved.parts
        assert "sub" in resolved.parts
        assert "alice" in resolved.parts

    def test_raises_on_traversal(self, handler: NativeFileToolHandler):
        with pytest.raises(ValueError, match="Path traversal"):
            handler.resolve_path("/memories/../../../etc/passwd", "alice")

    def test_scopes_by_user(self, handler: NativeFileToolHandler):
        alice_path = handler.resolve_path("/memories/doc.txt", "alice")
        bob_path = handler.resolve_path("/memories/doc.txt", "bob")
        assert alice_path != bob_path
        assert "alice" in str(alice_path)
        assert "bob" in str(bob_path)


class TestCreate:
    """Verify file creation."""

    def test_creates_file(self, handler: NativeFileToolHandler):
        result = handler.create({"path": "/memories/test.txt", "file_text": "hello"}, "alice")
        assert "created" in result.lower()
        resolved = handler.resolve_path("/memories/test.txt", "alice")
        assert resolved.read_text() == "hello"

    def test_creates_parent_dirs(self, handler: NativeFileToolHandler):
        handler.create({"path": "/memories/a/b/c.txt", "file_text": "deep"}, "alice")
        resolved = handler.resolve_path("/memories/a/b/c.txt", "alice")
        assert resolved.read_text() == "deep"

    def test_errors_on_existing_file(self, handler: NativeFileToolHandler):
        make_file(handler, "/memories/exists.txt", "content")
        result = handler.create({"path": "/memories/exists.txt", "file_text": "again"}, "alice")
        assert "already exists" in result

    def test_errors_on_missing_path(self, handler: NativeFileToolHandler):
        result = handler.create({"path": "", "file_text": "content"}, "alice")
        assert "path is required" in result


class TestView:
    """Verify file/directory viewing."""

    def test_view_file_content(self, handler: NativeFileToolHandler):
        make_file(handler, "/memories/notes.txt", "line1\nline2\nline3")
        result = handler.view({"path": "/memories/notes.txt"}, "alice")
        assert "line1" in result
        assert "line numbers" in result

    def test_view_file_with_range(self, handler: NativeFileToolHandler):
        text = "\n".join(f"line{i}" for i in range(10))
        make_file(handler, "/memories/lines.txt", text)
        result = handler.view({"path": "/memories/lines.txt", "view_range": [2, 4]}, "alice")
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result
        assert "line0" not in result
        assert "line4" not in result

    def test_view_nonexistent_path(self, handler: NativeFileToolHandler):
        result = handler.view({"path": "/memories/nonexistent"}, "alice")
        assert "does not exist" in result

    def test_view_directory_listing(self, handler: NativeFileToolHandler):
        make_file(handler, "/memories/a.txt", "aaa")
        make_file(handler, "/memories/sub/b.txt", "bbb")
        result = handler.view({"path": "/memories"}, "alice")
        assert "a.txt" in result
        assert "b.txt" in result
        assert "directories" in result

    def test_view_directory_at_root(self, handler: NativeFileToolHandler):
        make_file(handler, "/memories/root.txt", "root")
        result = handler.view({}, "alice")
        assert "root.txt" in result


class TestStrReplace:
    """Verify string replacement in files."""

    def test_replaces_unique_string(self, handler: NativeFileToolHandler):
        make_file(handler, "/memories/edit.txt", "Hello world\nFoo bar")
        result = handler.str_replace(
            {"path": "/memories/edit.txt", "old_str": "world", "new_str": "there"}, "alice"
        )
        assert "edited" in result
        resolved = handler.resolve_path("/memories/edit.txt", "alice")
        assert resolved.read_text() == "Hello there\nFoo bar"

    def test_errors_on_missing_old_str(self, handler: NativeFileToolHandler):
        make_file(handler, "/memories/edit.txt", "Hello")
        result = handler.str_replace(
            {"path": "/memories/edit.txt", "old_str": "nope", "new_str": "foo"}, "alice"
        )
        assert "did not appear verbatim" in result

    def test_errors_on_non_unique_old_str(self, handler: NativeFileToolHandler):
        make_file(handler, "/memories/edit.txt", "foo\nbar\nfoo")
        result = handler.str_replace(
            {"path": "/memories/edit.txt", "old_str": "foo", "new_str": "baz"}, "alice"
        )
        assert "Multiple occurrences" in result

    def test_errors_on_missing_path(self, handler: NativeFileToolHandler):
        result = handler.str_replace({}, "alice")
        assert "path is required" in result

    def test_errors_on_missing_old_str_param(self, handler: NativeFileToolHandler):
        result = handler.str_replace({"path": "/memories/x.txt"}, "alice")
        assert "old_str is required" in result

    def test_errors_on_nonexistent_file(self, handler: NativeFileToolHandler):
        result = handler.str_replace(
            {"path": "/memories/nope.txt", "old_str": "x", "new_str": "y"}, "alice"
        )
        assert "does not exist" in result


class TestInsert:
    """Verify line insertion in files."""

    def test_inserts_at_line(self, handler: NativeFileToolHandler):
        make_file(handler, "/memories/insert.txt", "first\nlast")
        result = handler.insert(
            {"path": "/memories/insert.txt", "insert_line": 1, "insert_text": "middle"}, "alice"
        )
        assert "edited" in result
        resolved = handler.resolve_path("/memories/insert.txt", "alice")
        assert resolved.read_text() == "first\nmiddle\nlast"

    def test_inserts_at_beginning(self, handler: NativeFileToolHandler):
        make_file(handler, "/memories/insert.txt", "line")
        handler.insert(
            {"path": "/memories/insert.txt", "insert_line": 0, "insert_text": "first"}, "alice"
        )
        resolved = handler.resolve_path("/memories/insert.txt", "alice")
        assert resolved.read_text() == "first\nline"

    def test_inserts_at_end(self, handler: NativeFileToolHandler):
        make_file(handler, "/memories/insert.txt", "start")
        n_lines = len("start\n".split("\n"))
        handler.insert(
            {"path": "/memories/insert.txt", "insert_line": n_lines - 1, "insert_text": "end"},
            "alice",
        )
        resolved = handler.resolve_path("/memories/insert.txt", "alice")
        assert resolved.read_text() == "start\nend"

    def test_errors_on_invalid_line(self, handler: NativeFileToolHandler):
        make_file(handler, "/memories/insert.txt", "only")
        result = handler.insert(
            {"path": "/memories/insert.txt", "insert_line": 99, "insert_text": "x"}, "alice"
        )
        assert "Invalid" in result

    def test_errors_on_missing_path(self, handler: NativeFileToolHandler):
        result = handler.insert({}, "alice")
        assert "path is required" in result

    def test_errors_on_nonexistent_file(self, handler: NativeFileToolHandler):
        result = handler.insert(
            {"path": "/memories/nope.txt", "insert_line": 0, "insert_text": "x"}, "alice"
        )
        assert "does not exist" in result


class TestDelete:
    """Verify file deletion."""

    def test_deletes_file(self, handler: NativeFileToolHandler):
        make_file(handler, "/memories/del.txt", "delete me")
        result = handler.delete_file({"path": "/memories/del.txt"}, "alice")
        assert "deleted" in result
        assert not handler.resolve_path("/memories/del.txt", "alice").exists()

    def test_deletes_directory(self, handler: NativeFileToolHandler):
        make_file(handler, "/memories/dir/sub/file.txt", "nested")
        result = handler.delete_file({"path": "/memories/dir"}, "alice")
        assert "deleted" in result
        assert not handler.resolve_path("/memories/dir", "alice").exists()

    def test_errors_on_nonexistent_path(self, handler: NativeFileToolHandler):
        result = handler.delete_file({"path": "/memories/nope.txt"}, "alice")
        assert "does not exist" in result

    def test_errors_on_missing_path_param(self, handler: NativeFileToolHandler):
        result = handler.delete_file({}, "alice")
        assert "path is required" in result


class TestRename:
    """Verify rename/move operations."""

    def test_renames_file(self, handler: NativeFileToolHandler):
        make_file(handler, "/memories/old.txt", "content")
        result = handler.rename(
            {"old_path": "/memories/old.txt", "new_path": "/memories/new.txt"}, "alice"
        )
        assert "renamed" in result
        assert not handler.resolve_path("/memories/old.txt", "alice").exists()
        assert handler.resolve_path("/memories/new.txt", "alice").read_text() == "content"

    def test_moves_to_subdirectory(self, handler: NativeFileToolHandler):
        make_file(handler, "/memories/file.txt", "move me")
        result = handler.rename(
            {"old_path": "/memories/file.txt", "new_path": "/memories/sub/file.txt"}, "alice"
        )
        assert "renamed" in result
        assert handler.resolve_path("/memories/sub/file.txt", "alice").read_text() == "move me"

    def test_errors_on_missing_old_path(self, handler: NativeFileToolHandler):
        result = handler.rename({}, "alice")
        assert "old_path is required" in result

    def test_errors_on_missing_new_path(self, handler: NativeFileToolHandler):
        result = handler.rename({"old_path": "/memories/x.txt"}, "alice")
        assert "new_path is required" in result

    def test_errors_on_nonexistent_old_path(self, handler: NativeFileToolHandler):
        result = handler.rename(
            {"old_path": "/memories/nope.txt", "new_path": "/memories/yes.txt"}, "alice"
        )
        assert "does not exist" in result

    def test_errors_on_existing_destination(self, handler: NativeFileToolHandler):
        make_file(handler, "/memories/a.txt", "aaa")
        make_file(handler, "/memories/b.txt", "bbb")
        result = handler.rename(
            {"old_path": "/memories/a.txt", "new_path": "/memories/b.txt"}, "alice"
        )
        assert "already exists" in result
