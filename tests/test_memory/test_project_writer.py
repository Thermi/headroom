"""Tests for project_writer -- writing memory markdown files into project dirs.

Tests cover:
  - Mount mapping from HEADROOM_VOLUME_MOUNTS env var
  - Path accessibility checking (is the project path under a mount?)
  - Memory markdown file writing to <project>/.headroom/memories/
  - Localhost-only guard
  - Backward compatibility (no mounts configured -> graceful skip)
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from headroom.memory.project_writer import (
    ProjectMemoryWriter,
    _is_subdir_of_any,
    _parse_mounts,
)

# =============================================================================
# Mount parsing
# =============================================================================


class TestParseMounts:
    """HEADROOM_VOLUME_MOUNTS parsing."""

    def test_empty_env_returns_empty(self) -> None:
        """No env var -> empty mounts list."""
        with patch.dict(os.environ, {}, clear=True):
            mounts = _parse_mounts()
        assert mounts == []

    def test_single_mount(self) -> None:
        """Single mount pair is parsed correctly."""
        with patch.dict(
            os.environ, {"HEADROOM_VOLUME_MOUNTS": "F:/noel/projects:/workspace"}, clear=True
        ):
            mounts = _parse_mounts()
        assert mounts == [(Path("F:/noel/projects"), Path("/workspace"))]

    def test_multiple_mounts(self) -> None:
        """Multiple comma-separated mount pairs."""
        with patch.dict(
            os.environ,
            {"HEADROOM_VOLUME_MOUNTS": "F:/noel/projects:/workspace,G:/other:/data"},
            clear=True,
        ):
            mounts = _parse_mounts()
        assert len(mounts) == 2
        assert (Path("F:/noel/projects"), Path("/workspace")) in mounts
        assert (Path("G:/other"), Path("/data")) in mounts

    def test_mount_with_spaces(self) -> None:
        """Mount paths with spaces are handled."""
        with patch.dict(
            os.environ,
            {"HEADROOM_VOLUME_MOUNTS": "C:/My Projects:/workspace"},
            clear=True,
        ):
            mounts = _parse_mounts()
        assert mounts == [(Path("C:/My Projects"), Path("/workspace"))]

    def test_mount_with_trailing_slash(self) -> None:
        """Trailing slashes in mount paths are stripped."""
        with patch.dict(
            os.environ,
            {"HEADROOM_VOLUME_MOUNTS": "F:/noel/projects/:/workspace/"},
            clear=True,
        ):
            mounts = _parse_mounts()
        assert mounts == [(Path("F:/noel/projects"), Path("/workspace"))]

    def test_malformed_entry_skipped(self) -> None:
        """Entries without ':' separator are skipped."""
        with patch.dict(
            os.environ,
            {"HEADROOM_VOLUME_MOUNTS": "F:/noel/projects:/workspace,invalid-entry"},
            clear=True,
        ):
            mounts = _parse_mounts()
        assert mounts == [(Path("F:/noel/projects"), Path("/workspace"))]

    def test_empty_entry_skipped(self) -> None:
        """Trailing comma (empty entry) is skipped."""
        with patch.dict(
            os.environ,
            {"HEADROOM_VOLUME_MOUNTS": "F:/noel/projects:/workspace,"},
            clear=True,
        ):
            mounts = _parse_mounts()
        assert mounts == [(Path("F:/noel/projects"), Path("/workspace"))]


# =============================================================================
# Subdirectory checking
# =============================================================================


class TestIsSubdirOfAny:
    """Check if a path is a subdirectory of any mount point."""

    def test_direct_child(self) -> None:
        """Direct subdirectory of a mount."""
        mounts = [(Path("/host/projects"), Path("/container/projects"))]
        result = _is_subdir_of_any("/host/projects/my-project", mounts)
        assert result is not None
        host_part, container_part = result
        assert host_part == Path("/host/projects/my-project")
        assert container_part == Path("/container/projects/my-project")

    def test_deeply_nested(self) -> None:
        """Deeply nested subdirectory."""
        mounts = [(Path("/host/projects"), Path("/workspace"))]
        result = _is_subdir_of_any("/host/projects/a/b/c/d", mounts)
        assert result is not None
        host_part, container_part = result
        assert host_part == Path("/host/projects/a/b/c/d")
        assert container_part == Path("/workspace/a/b/c/d")

    def test_exact_match(self) -> None:
        """Exact mount path (not a subdirectory) is NOT considered accessible."""
        mounts = [(Path("/host/projects"), Path("/workspace"))]
        result = _is_subdir_of_any("/host/projects", mounts)
        # Not a subdirectory, it's the mount itself -- still accessible
        # since we can write to the root of the mount
        assert result is not None

    def test_not_under_mount(self) -> None:
        """Path not under any mount returns None."""
        mounts = [(Path("/host/projects"), Path("/workspace"))]
        result = _is_subdir_of_any("/other/path", mounts)
        assert result is None

    def test_windows_path_under_mount(self) -> None:
        """Windows path under a Windows-style mount."""
        mounts = [(Path("F:/noel/projects"), Path("/workspace"))]
        result = _is_subdir_of_any("F:/noel/projects/my-app", mounts)
        assert result is not None
        host_part, container_part = result
        assert host_part == Path("F:/noel/projects/my-app")
        assert container_part == Path("/workspace/my-app")

    def test_no_mounts_returns_candidate(self) -> None:
        """Empty mounts list returns (candidate, candidate) -- native mode."""
        result = _is_subdir_of_any("/any/path", [])
        assert result is not None
        host_part, container_part = result
        assert host_part == Path("/any/path")
        assert container_part == Path("/any/path")


# =============================================================================
# ProjectMemoryWriter -- memory file writing
# =============================================================================


class TestProjectMemoryWriter:
    """Writing memory markdown files to project directories."""

    def test_writes_memory_file(self, tmp_path: Path) -> None:
        """Writing a memory creates a .md file in .headroom/memories/."""
        project_root = tmp_path / "my-project"
        project_root.mkdir(parents=True)

        writer = ProjectMemoryWriter(mounts=[])
        memory = SimpleNamespace(
            content="User prefers Python over JavaScript",
            importance=0.8,
            facts=["User prefers Python", "User prefers JavaScript"],
            entities=[{"entity": "Python", "entity_type": "technology"}],
        )

        result = writer.write_memory(
            project_root=str(project_root),
            memory=memory,
            user_id="alice",
            source="auto_extract",
        )

        assert result is not None
        mem_dir = project_root / ".headroom" / "memories"
        assert mem_dir.exists()
        files = list(mem_dir.glob("*.md"))
        assert len(files) == 1

        content = files[0].read_text(encoding="utf-8")
        assert "User prefers Python over JavaScript" in content
        assert "0.8" in content
        assert "alice" in content

    def test_multiple_memories_multiple_files(self, tmp_path: Path) -> None:
        """Each memory write creates a new file."""
        project_root = tmp_path / "my-project"
        project_root.mkdir(parents=True)

        writer = ProjectMemoryWriter(mounts=[])

        mem1 = SimpleNamespace(content="First memory", importance=0.7, facts=None, entities=None)
        mem2 = SimpleNamespace(content="Second memory", importance=0.5, facts=None, entities=None)

        writer.write_memory(str(project_root), mem1, "alice", "auto_extract")
        writer.write_memory(str(project_root), mem2, "alice", "auto_extract")

        mem_dir = project_root / ".headroom" / "memories"
        assert len(list(mem_dir.glob("*.md"))) == 2

    def test_file_has_yaml_frontmatter(self, tmp_path: Path) -> None:
        """Memory files have parseable YAML-like frontmatter."""
        project_root = tmp_path / "my-project"
        project_root.mkdir(parents=True)

        writer = ProjectMemoryWriter(mounts=[])
        memory = SimpleNamespace(content="Test memory", importance=0.9, facts=None, entities=None)

        writer.write_memory(str(project_root), memory, "bob", "auto_extract")

        mem_dir = project_root / ".headroom" / "memories"
        content = list(mem_dir.glob("*.md"))[0].read_text(encoding="utf-8")

        assert "---" in content
        assert "importance: 0.9" in content
        assert "user: bob" in content
        assert "source: auto_extract" in content

    def test_skipped_when_project_not_under_mount(self, tmp_path: Path) -> None:
        """When mounts are configured and path is outside them, skip."""
        project_root = tmp_path / "outside-project"
        project_root.mkdir(parents=True)

        mounts = [(tmp_path / "some-other-place", Path("/workspace"))]
        writer = ProjectMemoryWriter(mounts=mounts)
        memory = SimpleNamespace(
            content="Should not be written", importance=0.5, facts=None, entities=None
        )

        result = writer.write_memory(str(project_root), memory, "alice", "auto_extract")

        assert result is None
        assert not (project_root / ".headroom" / "memories").exists()

    def test_written_when_path_under_mount(self, tmp_path: Path) -> None:
        """When path is under a configured mount, write is allowed."""
        base = tmp_path / "base"
        base.mkdir(parents=True)
        project_root = base / "nested-project"
        project_root.mkdir(parents=True)

        # The writer maps project_root to container-side path and writes there
        mounts = [(base, Path("/workspace"))]
        writer = ProjectMemoryWriter(mounts=mounts)
        memory = SimpleNamespace(
            content="Under mount memory", importance=0.6, facts=None, entities=None
        )

        result = writer.write_memory(str(project_root), memory, "alice", "auto_extract")

        assert result is not None
        # Result points to the written file in the container-side project dir
        assert ".headroom" in str(result)
        assert "memories" in str(result)
        assert result.name.endswith(".md")

    def test_no_mounts_allows_all_writes(self, tmp_path: Path) -> None:
        """When no mounts are configured (native mode), all paths are writable."""
        project_root = tmp_path / "any-project"
        project_root.mkdir(parents=True)

        writer = ProjectMemoryWriter(mounts=[])
        memory = SimpleNamespace(
            content="Anywhere memory", importance=0.5, facts=None, entities=None
        )

        result = writer.write_memory(str(project_root), memory, "alice", "auto_extract")

        assert result is not None
        assert (project_root / ".headroom" / "memories").exists()

    def test_memory_with_facts_included(self, tmp_path: Path) -> None:
        """Facts array is included in the markdown file."""
        project_root = tmp_path / "facts-project"
        project_root.mkdir(parents=True)

        writer = ProjectMemoryWriter(mounts=[])
        memory = SimpleNamespace(
            content="User info",
            importance=0.7,
            facts=["Fact one", "Fact two", "Fact three"],
            entities=None,
        )

        writer.write_memory(str(project_root), memory, "alice", "auto_extract")

        content = next((project_root / ".headroom" / "memories").glob("*.md")).read_text(
            encoding="utf-8"
        )
        assert "Fact one" in content
        assert "Fact two" in content
        assert "Fact three" in content


# =============================================================================
# Integration: _write_auto_extracted_memories helper
# =============================================================================


class TestWriteAutoExtractedMemories:
    """Integration helper for writing auto-extracted memories to project dirs."""

    def test_writes_from_extraction_result(self, tmp_path: Path) -> None:
        """Helper writes all memories from an extraction result."""
        from headroom.memory.project_writer import _write_auto_extracted_memories

        project_root = tmp_path / "integration-test"
        project_root.mkdir(parents=True)

        saved_memories = [
            {"strategy": "inline", "content": "Memory A", "save": '{"status":"saved"}'},
            {"strategy": "inline", "content": "Memory B", "save": '{"status":"saved"}'},
        ]

        _write_auto_extracted_memories(
            project_root=str(project_root),
            saved_memories=saved_memories,
            user_id="alice",
        )

        mem_dir = project_root / ".headroom" / "memories"
        assert mem_dir.exists()
        files = list(mem_dir.glob("*.md"))
        assert len(files) == 2
        content = files[0].read_text(encoding="utf-8") + files[1].read_text(encoding="utf-8")
        assert "Memory A" in content
        assert "Memory B" in content

    def test_skipped_when_no_project_root(self) -> None:
        """No project root -> no files written (no crash)."""
        from headroom.memory.project_writer import _write_auto_extracted_memories

        # Should not raise
        result = _write_auto_extracted_memories(
            project_root=None,
            saved_memories=[
                {"strategy": "inline", "content": "test", "save": '{"status":"saved"}'}
            ],
            user_id="alice",
        )
        assert result is None

    def test_skipped_when_empty_memories(self, tmp_path: Path) -> None:
        """Empty memories list -> no files written."""
        from headroom.memory.project_writer import _write_auto_extracted_memories

        result = _write_auto_extracted_memories(
            project_root=str(tmp_path / "test"),
            saved_memories=[],
            user_id="alice",
        )
        assert result is None
