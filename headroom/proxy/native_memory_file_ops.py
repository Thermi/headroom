"""Pure filesystem operations for native memory tool support.

Encapsulates the file/directory operations that emulate file-based memory
persistence. These operations mirror Claude Code's native file editing
protocol (view, create, str_replace, insert, delete, rename) but operate
on files in a user-scoped memory directory.

This module has no dependency on the memory backend; it is pure filesystem
I/O with path-traversal protection.

Extracted from ``MemoryHandler`` to separate the filesystem concern from
the vector-store semantic adapter and the main handler orchestration.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class NativeFileToolHandler:
    """Pure filesystem operations for the native memory tool protocol.

    Provides the file-level interface that emulates Claude Code's native
    ``memory_20250818`` tool commands on a local filesystem directory.

    Attributes:
        memory_dir: The root directory under which user-scoped memory
            directories are created.
    """

    def __init__(self, memory_dir: Path) -> None:
        self._memory_dir = memory_dir

    def resolve_path(self, path: str, user_id: str) -> Path:
        """Resolve a virtual path within a user's memory directory safely.

        Prevents path traversal attacks by verifying the resolved path
        stays inside the user's memory directory.

        Args:
            path: Virtual path (e.g. ``/memories/topic.txt``).
            user_id: User identifier for scoping.

        Returns:
            Absolute resolved ``Path``.

        Raises:
            ValueError: If path traversal is detected.
        """
        user_dir = self._memory_dir / user_id

        if path.startswith("/memories"):
            path = path[len("/memories") :]
        if path.startswith("/"):
            path = path[1:]

        resolved = (user_dir / path).resolve()

        try:
            resolved.relative_to(user_dir.resolve())
        except ValueError:
            raise ValueError(f"Path traversal detected: {path}") from None

        return resolved

    def view(self, input_data: dict[str, Any], user_id: str) -> str:
        """View directory contents or file contents."""
        path = input_data.get("path", "/memories")
        view_range = input_data.get("view_range")

        resolved = self.resolve_path(path, user_id)

        if not resolved.exists():
            return f"The path {path} does not exist. Please provide a valid path."

        if resolved.is_dir():
            lines = [
                f"Here're the files and directories up to 2 levels deep in {path}, "
                "excluding hidden items and node_modules:"
            ]

            def get_size(p: Path) -> str:
                if p.is_file():
                    size = p.stat().st_size
                    if size < 1024:
                        return f"{size}B"
                    elif size < 1024 * 1024:
                        return f"{size / 1024:.1f}K"
                    else:
                        return f"{size / (1024 * 1024):.1f}M"
                return "4.0K"

            def list_recursive(p: Path, rel_path: str, depth: int) -> None:
                if depth > 2:
                    return
                if p.name.startswith(".") or p.name == "node_modules":
                    return

                lines.append(f"{get_size(p)}\t{rel_path}")

                if p.is_dir() and depth < 2:
                    try:
                        for child in sorted(p.iterdir()):
                            child_rel = (
                                f"{rel_path}/{child.name}"
                                if rel_path != path
                                else f"{path}/{child.name}"
                            )
                            list_recursive(child, child_rel, depth + 1)
                    except PermissionError:
                        pass

            list_recursive(resolved, path, 0)
            return "\n".join(lines)

        else:
            try:
                content = resolved.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = resolved.read_text(encoding="latin-1")

            lines_content = content.split("\n")

            if len(lines_content) > 999999:
                return f"File {path} exceeds maximum line limit of 999,999 lines."

            start_line = 1
            end_line = len(lines_content)
            if view_range and len(view_range) >= 2:
                start_line = max(1, view_range[0])
                end_line = min(len(lines_content), view_range[1])

            result_lines = [f"Here's the content of {path} with line numbers:"]
            for i, line in enumerate(lines_content[start_line - 1 : end_line], start=start_line):
                result_lines.append(f"{i:6d}\t{line}")

            return "\n".join(result_lines)

    def create(self, input_data: dict[str, Any], user_id: str) -> str:
        """Create a new file."""
        path = input_data.get("path", "")
        file_text = input_data.get("file_text", "")

        if not path:
            return "Error: path is required"

        resolved = self.resolve_path(path, user_id)

        if resolved.exists():
            return f"Error: File {path} already exists"

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(file_text, encoding="utf-8")
        logger.info("Memory: Native create: %s for user %s", path, user_id)

        return f"File created successfully at: {path}"

    def str_replace(self, input_data: dict[str, Any], user_id: str) -> str:
        """Replace text in a file."""
        path = input_data.get("path", "")
        old_str = input_data.get("old_str", "")
        new_str = input_data.get("new_str", "")

        if not path:
            return "Error: path is required"
        if not old_str:
            return "Error: old_str is required"

        resolved = self.resolve_path(path, user_id)

        if not resolved.exists():
            return f"Error: The path {path} does not exist. Please provide a valid path."
        if resolved.is_dir():
            return f"Error: The path {path} does not exist. Please provide a valid path."

        content = resolved.read_text(encoding="utf-8")

        occurrences = content.count(old_str)
        if occurrences == 0:
            return f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."
        if occurrences > 1:
            lines = content.split("\n")
            found_lines = []
            for i, line in enumerate(lines, 1):
                if old_str in line:
                    found_lines.append(str(i))
            return (
                f"No replacement was performed. Multiple occurrences of old_str `{old_str}` "
                f"in lines: {', '.join(found_lines)}. Please ensure it is unique"
            )

        new_content = content.replace(old_str, new_str, 1)
        resolved.write_text(new_content, encoding="utf-8")

        lines = new_content.split("\n")
        for i, line in enumerate(lines):
            if new_str in line:
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                snippet_lines = ["The memory file has been edited."]
                for j in range(start, end):
                    snippet_lines.append(f"{j + 1:6d}\t{lines[j]}")
                return "\n".join(snippet_lines)

        return "The memory file has been edited."

    def insert(self, input_data: dict[str, Any], user_id: str) -> str:
        """Insert text at a specific line."""
        path = input_data.get("path", "")
        insert_line = input_data.get("insert_line", 0)
        insert_text = input_data.get("insert_text", "")

        if not path:
            return "Error: path is required"

        resolved = self.resolve_path(path, user_id)

        if not resolved.exists():
            return f"Error: The path {path} does not exist"
        if resolved.is_dir():
            return f"Error: The path {path} does not exist"

        content = resolved.read_text(encoding="utf-8")
        lines = content.split("\n")
        n_lines = len(lines)

        if insert_line < 0 or insert_line > n_lines:
            return (
                f"Error: Invalid `insert_line` parameter: {insert_line}. "
                f"It should be within the range of lines of the file: [0, {n_lines}]"
            )

        lines.insert(insert_line, insert_text.rstrip("\n"))
        resolved.write_text("\n".join(lines), encoding="utf-8")

        return f"The file {path} has been edited."

    def delete_file(self, input_data: dict[str, Any], user_id: str) -> str:
        """Delete a file or directory."""
        path = input_data.get("path", "")

        if not path:
            return "Error: path is required"

        resolved = self.resolve_path(path, user_id)

        if not resolved.exists():
            return f"Error: The path {path} does not exist"

        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()

        logger.info("Memory: Native delete: %s for user %s", path, user_id)
        return f"Successfully deleted {path}"

    def rename(self, input_data: dict[str, Any], user_id: str) -> str:
        """Rename or move a file/directory."""
        old_path = input_data.get("old_path", "")
        new_path = input_data.get("new_path", "")

        if not old_path:
            return "Error: old_path is required"
        if not new_path:
            return "Error: new_path is required"

        resolved_old = self.resolve_path(old_path, user_id)
        resolved_new = self.resolve_path(new_path, user_id)

        if not resolved_old.exists():
            return f"Error: The path {old_path} does not exist"
        if resolved_new.exists():
            return f"Error: The destination {new_path} already exists"

        resolved_new.parent.mkdir(parents=True, exist_ok=True)
        resolved_old.rename(resolved_new)

        logger.info("Memory: Native rename: %s -> %s for user %s", old_path, new_path, user_id)
        return f"Successfully renamed {old_path} to {new_path}"
