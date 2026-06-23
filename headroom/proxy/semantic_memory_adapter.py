"""Semantic adapter for the native memory tool protocol.

Translates Claude Code's native ``memory_20250818`` file-oriented commands
(view, create, str_replace, insert, delete, rename) into vector-store
operations on the configured backend.

While Claude thinks it is performing file I/O (e.g. ``view /memories/search/pizza``,
``create /memories/preferences.txt``), this adapter performs semantic search
and vector-store save/update/delete operations behind the scenes.

Extracted from ``MemoryHandler`` (headroom/proxy/memory_handler.py) to separate
the semantic-translation concern from filesystem operations and handler
orchestration.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SemanticNativeToolAdapter:
    """Translates native memory tool commands to vector-store operations.

    Maps file-oriented commands from Anthropic's ``memory_20250818`` tool to
    semantic operations:

    * ``view /memories`` → memory overview / search instructions
    * ``view /memories/search/X`` → semantic search for X
    * ``view /memories/recent`` → recent memories
    * ``create /memories/<path>.txt`` → save to vector store
    * ``str_replace`` → update memory content
    * ``insert`` → append memory
    * ``delete`` → remove from vector store
    * ``rename`` → update metadata path/topic

    Requires a backend implementing ``search_memories``, ``save_memory``,
    ``update_memory`` (optional), and ``delete_memory``.
    """

    def __init__(self, backend: Any, agent_type: str = "unknown") -> None:
        self._backend = backend
        self._agent_type = agent_type

    async def view(self, input_data: dict[str, Any], user_id: str) -> str:
        """Handle VIEW command with semantic search capabilities."""
        path = input_data.get("path", "/memories")

        if path.startswith("/memories"):
            subpath = path[len("/memories") :].lstrip("/")
        else:
            subpath = path.lstrip("/")

        if subpath.startswith("search/"):
            query = subpath[len("search/") :]
            if not query:
                return "Error: Please provide a search query. Example: view /memories/search/food preferences"
            return await self._semantic_search(query, user_id)

        if subpath == "recent":
            return await self._get_recent_memories(user_id, limit=10)

        if subpath == "all":
            return await self._list_all_memories(user_id, limit=20)

        if not subpath or subpath == "":
            return await self._get_memory_overview(user_id)

        return await self._semantic_search(subpath.replace("/", " ").replace("_", " "), user_id)

    async def create(self, input_data: dict[str, Any], user_id: str) -> str:
        """Handle CREATE command - save to semantic vector store."""
        path = input_data.get("path", "")
        file_text = input_data.get("file_text", "")

        if not path:
            return "Error: path is required"
        if not file_text:
            return "Error: file_text is required (the memory content)"
        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            topic = (
                path.replace("/memories/", "")
                .replace("/", "_")
                .replace(".txt", "")
                .replace(".md", "")
            )
            memory = await self._backend.save_memory(
                content=file_text,
                user_id=user_id,
                importance=0.5,
                metadata={"virtual_path": path, "topic": topic},
            )
            logger.info(
                "Memory: Semantic create: %s -> id=%s for user %s", path, memory.id, user_id
            )
            return f"File created successfully at: {path}"
        except Exception as e:
            logger.error("Memory: Semantic create failed: %s", e)
            return f"Error: {e}"

    async def update(self, input_data: dict[str, Any], user_id: str) -> str:
        """Handle STR_REPLACE command - update memory content."""
        path = input_data.get("path", "")
        old_str = input_data.get("old_str", "")
        new_str = input_data.get("new_str", "")

        if not path:
            return "Error: path is required"
        if not old_str:
            return "Error: old_str is required"
        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            results = await self._backend.search_memories(
                query=old_str,
                user_id=user_id,
                top_k=5,
            )

            matching_memory = None
            for r in results:
                if old_str in r.memory.content:
                    matching_memory = r.memory
                    break

            if not matching_memory:
                return f"No replacement was performed, old_str `{old_str}` did not appear verbatim in memories."

            if matching_memory.content.count(old_str) > 1:
                return f"No replacement was performed. Multiple occurrences of old_str `{old_str}`. Please ensure it is unique."

            new_content = matching_memory.content.replace(old_str, new_str, 1)

            if hasattr(self._backend, "update_memory"):
                await self._backend.update_memory(
                    memory_id=matching_memory.id,
                    new_content=new_content,
                    user_id=user_id,
                )
            else:
                await self._backend.delete_memory(matching_memory.id)
                await self._backend.save_memory(
                    content=new_content,
                    user_id=user_id,
                    importance=0.5,
                )

            lines = new_content.split("\n")
            snippet = "\n".join(f"{i + 1:6d}\t{line}" for i, line in enumerate(lines[:5]))
            logger.info("Memory: Semantic update for user %s", user_id)
            return f"The memory file has been edited.\n{snippet}"

        except Exception as e:
            logger.error("Memory: Semantic update failed: %s", e)
            return f"Error: {e}"

    async def append(self, input_data: dict[str, Any], user_id: str) -> str:
        """Handle INSERT command - append to memory."""
        path = input_data.get("path", "")
        insert_text = input_data.get("insert_text", "")
        _ = input_data.get("insert_line", 0)

        if not path:
            return "Error: path is required"
        if not insert_text:
            return "Error: insert_text is required"
        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            topic = path.replace("/memories/", "").replace("/", "_").replace(".txt", "")

            await self._backend.save_memory(
                content=insert_text,
                user_id=user_id,
                importance=0.5,
                metadata={"virtual_path": path, "topic": topic, "appended": True},
            )

            logger.info("Memory: Semantic append: %s for user %s", path, user_id)
            return f"The file {path} has been edited."

        except Exception as e:
            logger.error("Memory: Semantic append failed: %s", e)
            return f"Error: {e}"

    async def delete(self, input_data: dict[str, Any], user_id: str) -> str:
        """Handle DELETE command - remove from vector store."""
        path = input_data.get("path", "")

        if not path:
            return "Error: path is required"
        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            topic = (
                path.replace("/memories/", "")
                .replace("/", " ")
                .replace("_", " ")
                .replace(".txt", "")
            )

            results = await self._backend.search_memories(
                query=topic,
                user_id=user_id,
                top_k=10,
            )

            if not results:
                return f"Error: The path {path} does not exist"

            deleted_count = 0
            for r in results:
                metadata = getattr(r.memory, "metadata", {}) or {}
                if metadata.get("virtual_path") == path or r.score > 0.8:
                    await self._backend.delete_memory(r.memory.id)
                    deleted_count += 1

            if deleted_count == 0:
                return f"Error: The path {path} does not exist"

            logger.info(
                "Memory: Semantic delete: %s (%d memories) for user %s",
                path,
                deleted_count,
                user_id,
            )
            return f"Successfully deleted {path}"

        except Exception as e:
            logger.error("Memory: Semantic delete failed: %s", e)
            return f"Error: {e}"

    async def rename(self, input_data: dict[str, Any], user_id: str) -> str:
        """Handle RENAME command - update memory path/topic."""
        old_path = input_data.get("old_path", "")
        new_path = input_data.get("new_path", "")

        if not old_path:
            return "Error: old_path is required"
        if not new_path:
            return "Error: new_path is required"
        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            old_topic = (
                old_path.replace("/memories/", "")
                .replace("/", " ")
                .replace("_", " ")
                .replace(".txt", "")
            )

            results = await self._backend.search_memories(
                query=old_topic,
                user_id=user_id,
                top_k=10,
            )

            if not results:
                return f"Error: The path {old_path} does not exist"

            new_topic = new_path.replace("/memories/", "").replace("/", "_").replace(".txt", "")
            renamed_count = 0

            for r in results:
                metadata = getattr(r.memory, "metadata", {}) or {}
                if metadata.get("virtual_path") == old_path or r.score > 0.8:
                    await self._backend.delete_memory(r.memory.id)
                    await self._backend.save_memory(
                        content=r.memory.content,
                        user_id=user_id,
                        importance=getattr(r.memory, "importance", 0.5),
                        metadata={"virtual_path": new_path, "topic": new_topic},
                    )
                    renamed_count += 1

            if renamed_count == 0:
                return f"Error: The path {old_path} does not exist"

            logger.info(
                "Memory: Semantic rename: %s -> %s for user %s", old_path, new_path, user_id
            )
            return f"Successfully renamed {old_path} to {new_path}"

        except Exception as e:
            logger.error("Memory: Semantic rename failed: %s", e)
            return f"Error: {e}"

    async def _semantic_search(self, query: str, user_id: str, top_k: int = 5) -> str:
        """Perform semantic search and format results."""
        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            results = await self._backend.search_memories(
                query=query,
                user_id=user_id,
                top_k=top_k,
                include_related=True,
            )

            if not results:
                return (
                    f"No memories found matching '{query}'.\n\n"
                    "Tip: Try a broader search term, or use 'view /memories/recent' "
                    "to see recent memories."
                )

            lines = [f"Found {len(results)} memories matching '{query}':\n"]
            for i, r in enumerate(results, 1):
                score_pct = int(r.score * 100)
                content_preview = r.memory.content[:200]
                if len(r.memory.content) > 200:
                    content_preview += "..."

                lines.append(f"{i:6d}\t[{score_pct}% match] {content_preview}")

                if hasattr(r, "related_entities") and r.related_entities:
                    entities = ", ".join(r.related_entities[:3])
                    lines.append(f"      \t   Related: {entities}")
                lines.append("")

            return "\n".join(lines)

        except Exception as e:
            logger.error("Memory: Semantic search failed: %s", e)
            return f"Error searching memories: {e}"

    async def _get_recent_memories(self, user_id: str, limit: int = 10) -> str:
        """Get most recent memories."""
        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            results = await self._backend.search_memories(
                query="recent memories",
                user_id=user_id,
                top_k=limit,
            )

            if not results:
                return (
                    "No memories stored yet.\n\n"
                    "To save a memory, use: create /memories/<topic>.txt with your content"
                )

            lines = ["Recent memories:\n"]
            for i, r in enumerate(results, 1):
                content_preview = r.memory.content[:150]
                if len(r.memory.content) > 150:
                    content_preview += "..."
                timestamp = ""
                if hasattr(r.memory, "created_at") and r.memory.created_at:
                    timestamp = f" ({r.memory.created_at})"
                lines.append(f"{i:6d}\t{content_preview}{timestamp}")
            lines.append("")

            return "\n".join(lines)

        except Exception as e:
            logger.error("Memory: Get recent failed: %s", e)
            return f"Error getting recent memories: {e}"

    async def _list_all_memories(self, user_id: str, limit: int = 20) -> str:
        """List all memories (paginated)."""
        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            results = await self._backend.search_memories(
                query="*",
                user_id=user_id,
                top_k=limit,
            )

            if not results:
                return "No memories stored yet."

            lines = [f"Showing up to {limit} memories:\n"]
            for i, r in enumerate(results, 1):
                content_preview = r.memory.content[:100]
                if len(r.memory.content) > 100:
                    content_preview += "..."
                lines.append(f"{i:6d}\t{content_preview}")

            if len(results) >= limit:
                lines.append(f"\n(Showing first {limit}. Use search to find specific memories.)")

            return "\n".join(lines)

        except Exception as e:
            logger.error("Memory: List all failed: %s", e)
            return f"Error listing memories: {e}"

    async def _get_memory_overview(self, user_id: str) -> str:
        """Get memory directory overview with search instructions."""
        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            results = await self._backend.search_memories(
                query="*",
                user_id=user_id,
                top_k=100,
            )
            count = len(results) if results else 0

            preview_lines = []
            if results:
                for r in results[:3]:
                    preview = r.memory.content[:60]
                    if len(r.memory.content) > 60:
                        preview += "..."
                    preview_lines.append(f"  \u2022 {preview}")

            overview = (
                f"Here're the files and directories up to 2 levels deep in /memories:\n"
                f"4.0K\t/memories\n\n"
                f"\U0001f4c1 Memory System ({count} memories stored)\n\n"
                f"To SEARCH memories (semantic):\n"
                f"  view /memories/search/<your query>\n"
                f"  Example: view /memories/search/food preferences\n"
                f"  Example: view /memories/search/work projects\n\n"
                f"To see RECENT memories:\n"
                f"  view /memories/recent\n\n"
                f"To see ALL memories:\n"
                f"  view /memories/all\n\n"
                f"To SAVE a new memory:\n"
                f'  create /memories/<topic>.txt "your content here"\n'
                f'  Example: create /memories/preferences.txt "User likes pizza"\n'
            )

            if preview_lines:
                overview += "\nRecent memories:\n" + "\n".join(preview_lines)

            return overview

        except Exception as e:
            logger.error("Memory: Overview failed: %s", e)
            return (
                "\U0001f4c1 Memory System\n\n"
                "To SEARCH memories: view /memories/search/<query>\n"
                "To see RECENT: view /memories/recent\n"
                'To SAVE: create /memories/<topic>.txt "content"\n'
            )
