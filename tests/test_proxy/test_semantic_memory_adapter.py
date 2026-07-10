"""Tests for headroom.proxy.semantic_memory_adapter — SemanticNativeToolAdapter."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from headroom.proxy.semantic_memory_adapter import SemanticNativeToolAdapter


class MockMem:
    def __init__(self, content="test content", id="mem-1", metadata=None):
        self.content = content
        self.id = id
        self.metadata = metadata or {}
        self.created_at = "2024-01-01"


class MockResult:
    def __init__(self, content="test", id="mem-1", score=0.9, metadata=None, entities=None):
        self.memory = MockMem(content, id, metadata)
        self.score = score
        self.related_entities = entities or []


@pytest.fixture
def mock_backend():
    backend = AsyncMock()
    backend.search_memories = AsyncMock()
    backend.save_memory = AsyncMock()
    backend.delete_memory = AsyncMock()
    backend.update_memory = AsyncMock()
    return backend


@pytest.fixture
def adapter(mock_backend):
    return SemanticNativeToolAdapter(mock_backend, agent_type="test")


@pytest.fixture
def no_backend_adapter():
    return SemanticNativeToolAdapter(None, agent_type="test")


class TestView:
    async def test_search(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = [
            MockResult(content="I love pizza", id="m1", score=0.95),
            MockResult(content="Pasta is great", id="m2", score=0.85),
        ]
        result = await adapter.view({"path": "/memories/search/food"}, "alice")
        assert "Found 2 memories" in result
        assert "pizza" in result
        assert "95%" in result
        assert "85%" in result
        mock_backend.search_memories.assert_awaited_once_with(
            query="food", user_id="alice", top_k=5, include_related=True
        )

    async def test_search_empty_query(self, adapter):
        result = await adapter.view({"path": "/memories/search/"}, "alice")
        assert "Error" in result or "query" in result.lower()

    async def test_search_no_results(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = []
        result = await adapter.view({"path": "/memories/search/obscure"}, "alice")
        assert "No memories found" in result

    async def test_search_with_entities(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = [
            MockResult(content="python coding", id="m1", score=0.9, entities=["python", "code"]),
        ]
        result = await adapter.view({"path": "/memories/search/python"}, "alice")
        assert "Related" in result
        assert "python" in result

    async def test_recent(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = [
            MockResult(content="memory one", id="m1", score=0.5),
            MockResult(content="memory two", id="m2", score=0.4),
        ]
        result = await adapter.view({"path": "/memories/recent"}, "alice")
        assert "Recent memories" in result
        assert "memory one" in result
        assert "2024-01-01" in result

    async def test_recent_empty(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = []
        result = await adapter.view({"path": "/memories/recent"}, "alice")
        assert "No memories stored yet" in result

    async def test_all(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = [
            MockResult(content="m1", id="m1", score=0.3),
            MockResult(content="m2", id="m2", score=0.2),
        ]
        result = await adapter.view({"path": "/memories/all"}, "alice")
        assert "Showing up to 20" in result
        assert "m1" in result
        assert "m2" in result

    async def test_all_empty(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = []
        result = await adapter.view({"path": "/memories/all"}, "alice")
        assert "No memories stored yet" in result

    async def test_root_path(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = [
            MockResult(content="memory preview", id="m1", score=0.5),
        ]
        result = await adapter.view({"path": "/memories"}, "alice")
        assert "Memory System" in result
        assert "1 memories stored" in result

    async def test_empty_subpath(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = []
        result = await adapter.view({"path": "/memories/"}, "alice")
        assert "Memory System" in result

    async def test_other_subpath_becomes_search(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = [
            MockResult(content="work project notes", id="m1", score=0.88),
        ]
        result = await adapter.view({"path": "/memories/work/projects"}, "alice")
        assert "work project" in result
        mock_backend.search_memories.assert_awaited_once_with(
            query="work projects", user_id="alice", top_k=5, include_related=True
        )

    async def test_search_no_backend(self, no_backend_adapter):
        result = await no_backend_adapter.view({"path": "/memories/search/foo"}, "alice")
        assert "Error" in result

    async def test_recent_no_backend(self, no_backend_adapter):
        result = await no_backend_adapter.view({"path": "/memories/recent"}, "alice")
        assert "Error" in result

    async def test_root_no_backend(self, no_backend_adapter):
        result = await no_backend_adapter.view({"path": "/memories"}, "alice")
        assert "Error" in result

    async def test_all_no_backend(self, no_backend_adapter):
        result = await no_backend_adapter.view({"path": "/memories/all"}, "alice")
        assert "Error" in result

    async def test_search_exception(self, adapter, mock_backend):
        mock_backend.search_memories.side_effect = RuntimeError("boom")
        result = await adapter.view({"path": "/memories/search/foo"}, "alice")
        assert "Error" in result

    async def test_all_pagination_hint(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = [
            MockResult(content=f"mem{i}", id=f"m{i}", score=0.5) for i in range(20)
        ]
        result = await adapter.view({"path": "/memories/all"}, "alice")
        assert "first 20" in result

    async def test_recent_without_timestamp(self, adapter, mock_backend):
        result_no_ts = MockResult(content="no date", id="m1", score=0.5)
        del result_no_ts.memory.created_at
        mock_backend.search_memories.return_value = [result_no_ts]
        result = await adapter.view({"path": "/memories/recent"}, "alice")
        assert "no date" in result

    async def test_search_long_content_truncated(self, adapter, mock_backend):
        long = "x" * 300
        mock_backend.search_memories.return_value = [
            MockResult(content=long, id="m1", score=0.9),
        ]
        result = await adapter.view({"path": "/memories/search/long"}, "alice")
        assert "..." in result

    async def test_empty_input_path(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = []
        result = await adapter.view({}, "alice")
        assert "Memory System" in result


class TestCreate:
    async def test_saves_memory(self, adapter, mock_backend):
        mock_backend.save_memory.return_value = MockMem(id="new-1")
        result = await adapter.create(
            {"path": "/memories/preferences.txt", "file_text": "User likes pizza"}, "alice"
        )
        assert "created successfully" in result
        mock_backend.save_memory.assert_awaited_once()
        args = mock_backend.save_memory.await_args.kwargs
        assert args["content"] == "User likes pizza"
        assert args["user_id"] == "alice"
        assert args["metadata"]["topic"] == "preferences"
        assert args["metadata"]["virtual_path"] == "/memories/preferences.txt"

    async def test_saves_memory_md(self, adapter, mock_backend):
        mock_backend.save_memory.return_value = MockMem(id="new-1")
        result = await adapter.create(
            {"path": "/memories/notes.md", "file_text": "markdown content"}, "alice"
        )
        assert "created successfully" in result
        args = mock_backend.save_memory.await_args.kwargs
        assert args["metadata"]["topic"] == "notes"

    async def test_nested_path(self, adapter, mock_backend):
        mock_backend.save_memory.return_value = MockMem(id="new-1")
        result = await adapter.create(
            {"path": "/memories/work/ideas.txt", "file_text": "idea content"}, "alice"
        )
        assert "created successfully" in result
        args = mock_backend.save_memory.await_args.kwargs
        assert args["metadata"]["topic"] == "work_ideas"

    async def test_missing_path(self, adapter):
        result = await adapter.create({"path": "", "file_text": "content"}, "alice")
        assert "path is required" in result

    async def test_missing_file_text(self, adapter):
        result = await adapter.create({"path": "/memories/x.txt", "file_text": ""}, "alice")
        assert "file_text is required" in result

    async def test_no_backend(self, no_backend_adapter):
        result = await no_backend_adapter.create(
            {"path": "/memories/x.txt", "file_text": "content"}, "alice"
        )
        assert "Error" in result

    async def test_backend_exception(self, adapter, mock_backend):
        mock_backend.save_memory.side_effect = RuntimeError("save failed")
        result = await adapter.create({"path": "/memories/x.txt", "file_text": "content"}, "alice")
        assert "Error" in result


class TestUpdate:
    async def test_replaces_content(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = [
            MockResult(content="Hello world", id="m1", score=0.9),
        ]
        mock_backend.update_memory.return_value = MockMem(id="m1")
        result = await adapter.update(
            {"path": "/memories/hello.txt", "old_str": "world", "new_str": "there"}, "alice"
        )
        assert "edited" in result
        mock_backend.update_memory.assert_awaited_once_with(
            memory_id="m1", new_content="Hello there", user_id="alice"
        )

    async def test_no_matching_memory(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = [
            MockResult(content="something else", id="m1", score=0.5),
        ]
        result = await adapter.update(
            {"path": "/memories/x.txt", "old_str": "notfound", "new_str": "x"}, "alice"
        )
        assert "did not appear verbatim" in result

    async def test_multiple_occurrences(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = [
            MockResult(content="foo bar foo", id="m1", score=0.9),
        ]
        result = await adapter.update(
            {"path": "/memories/x.txt", "old_str": "foo", "new_str": "baz"}, "alice"
        )
        assert "Multiple occurrences" in result

    async def test_missing_path(self, adapter):
        result = await adapter.update({"old_str": "x", "new_str": "y"}, "alice")
        assert "path is required" in result

    async def test_missing_old_str(self, adapter):
        result = await adapter.update({"path": "/memories/x.txt", "new_str": "y"}, "alice")
        assert "old_str is required" in result

    async def test_no_backend(self, no_backend_adapter):
        result = await no_backend_adapter.update(
            {"path": "/memories/x.txt", "old_str": "x", "new_str": "y"}, "alice"
        )
        assert "Error" in result

    async def test_fallback_delete_and_save(self, adapter, mock_backend):
        backend_no_update = AsyncMock()
        backend_no_update.search_memories = AsyncMock()
        backend_no_update.save_memory = AsyncMock()
        backend_no_update.delete_memory = AsyncMock()
        del backend_no_update.update_memory
        adapter._backend = backend_no_update

        backend_no_update.search_memories.return_value = [
            MockResult(content="Hello world", id="m1", score=0.9),
        ]
        result = await adapter.update(
            {"path": "/memories/hello.txt", "old_str": "world", "new_str": "there"}, "alice"
        )
        assert "edited" in result
        backend_no_update.delete_memory.assert_awaited_once_with("m1")
        backend_no_update.save_memory.assert_awaited_once()

    async def test_backend_exception(self, adapter, mock_backend):
        mock_backend.search_memories.side_effect = RuntimeError("search fail")
        result = await adapter.update(
            {"path": "/memories/x.txt", "old_str": "x", "new_str": "y"}, "alice"
        )
        assert "Error" in result

    async def test_update_content_snippet(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = [
            MockResult(content="line1\nline2\nline3", id="m1", score=0.9),
        ]
        mock_backend.update_memory.return_value = MockMem(id="m1")
        result = await adapter.update(
            {"path": "/memories/x.txt", "old_str": "line2", "new_str": "modified"}, "alice"
        )
        assert "edited" in result
        assert "1\tline1" in result
        assert "2\tmodified" in result
        assert "3\tline3" in result


class TestAppend:
    async def test_appends_memory(self, adapter, mock_backend):
        mock_backend.save_memory.return_value = MockMem(id="new-1")
        result = await adapter.append(
            {"path": "/memories/log.txt", "insert_text": "extra line"}, "alice"
        )
        assert "edited" in result
        args = mock_backend.save_memory.await_args.kwargs
        assert args["content"] == "extra line"
        assert args["metadata"]["appended"] is True
        assert args["metadata"]["topic"] == "log"

    async def test_missing_path(self, adapter):
        result = await adapter.append({"insert_text": "content"}, "alice")
        assert "path is required" in result

    async def test_missing_insert_text(self, adapter):
        result = await adapter.append({"path": "/memories/x.txt", "insert_text": ""}, "alice")
        assert "insert_text is required" in result

    async def test_no_backend(self, no_backend_adapter):
        result = await no_backend_adapter.append(
            {"path": "/memories/x.txt", "insert_text": "content"}, "alice"
        )
        assert "Error" in result

    async def test_backend_exception(self, adapter, mock_backend):
        mock_backend.save_memory.side_effect = RuntimeError("append fail")
        result = await adapter.append(
            {"path": "/memories/x.txt", "insert_text": "content"}, "alice"
        )
        assert "Error" in result


class TestDelete:
    async def test_deletes_by_virtual_path(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = [
            MockResult(
                content="test",
                id="m1",
                score=0.7,
                metadata={"virtual_path": "/memories/todelete.txt"},
            ),
        ]
        result = await adapter.delete({"path": "/memories/todelete.txt"}, "alice")
        assert "Successfully deleted" in result
        mock_backend.delete_memory.assert_awaited_once_with("m1")

    async def test_deletes_by_score(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = [
            MockResult(content="test", id="m1", score=0.95),
        ]
        result = await adapter.delete({"path": "/memories/somefile.txt"}, "alice")
        assert "Successfully deleted" in result
        mock_backend.delete_memory.assert_awaited_once_with("m1")

    async def test_no_matching_results(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = [
            MockResult(content="test", id="m1", score=0.5),
        ]
        result = await adapter.delete({"path": "/memories/nonexistent.txt"}, "alice")
        assert "does not exist" in result

    async def test_no_results_from_search(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = []
        result = await adapter.delete({"path": "/memories/nonexistent.txt"}, "alice")
        assert "does not exist" in result

    async def test_no_backend(self, no_backend_adapter):
        result = await no_backend_adapter.delete({"path": "/memories/x.txt"}, "alice")
        assert "Error" in result

    async def test_missing_path(self, adapter):
        result = await adapter.delete({}, "alice")
        assert "path is required" in result

    async def test_backend_exception(self, adapter, mock_backend):
        mock_backend.search_memories.side_effect = RuntimeError("delete fail")
        result = await adapter.delete({"path": "/memories/x.txt"}, "alice")
        assert "Error" in result


class TestRename:
    async def test_renames_by_virtual_path(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = [
            MockResult(
                content="content",
                id="m1",
                score=0.7,
                metadata={"virtual_path": "/memories/old.txt"},
            ),
        ]
        result = await adapter.rename(
            {"old_path": "/memories/old.txt", "new_path": "/memories/new.txt"}, "alice"
        )
        assert "Successfully renamed" in result
        mock_backend.delete_memory.assert_awaited_once_with("m1")
        save_args = mock_backend.save_memory.await_args.kwargs
        assert save_args["metadata"]["virtual_path"] == "/memories/new.txt"
        assert save_args["metadata"]["topic"] == "new"

    async def test_renames_by_score(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = [
            MockResult(content="old content", id="m1", score=0.95),
        ]
        result = await adapter.rename(
            {"old_path": "/memories/old.txt", "new_path": "/memories/new.txt"}, "alice"
        )
        assert "Successfully renamed" in result
        mock_backend.delete_memory.assert_awaited_once_with("m1")
        save_args = mock_backend.save_memory.await_args.kwargs
        assert save_args["metadata"]["virtual_path"] == "/memories/new.txt"

    async def test_no_results_from_search(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = []
        result = await adapter.rename(
            {"old_path": "/memories/old.txt", "new_path": "/memories/new.txt"}, "alice"
        )
        assert "does not exist" in result

    async def test_no_matching_results(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = [
            MockResult(content="test", id="m1", score=0.5),
        ]
        result = await adapter.rename(
            {"old_path": "/memories/old.txt", "new_path": "/memories/new.txt"}, "alice"
        )
        assert "does not exist" in result

    async def test_missing_old_path(self, adapter):
        result = await adapter.rename({"new_path": "/memories/new.txt"}, "alice")
        assert "old_path is required" in result

    async def test_missing_new_path(self, adapter):
        result = await adapter.rename({"old_path": "/memories/old.txt"}, "alice")
        assert "new_path is required" in result

    async def test_no_backend(self, no_backend_adapter):
        result = await no_backend_adapter.rename(
            {"old_path": "/memories/old.txt", "new_path": "/memories/new.txt"}, "alice"
        )
        assert "Error" in result

    async def test_backend_exception(self, adapter, mock_backend):
        mock_backend.search_memories.side_effect = RuntimeError("rename fail")
        result = await adapter.rename(
            {"old_path": "/memories/old.txt", "new_path": "/memories/new.txt"}, "alice"
        )
        assert "Error" in result


class TestPrivateHelpers:
    async def test_semantic_search_no_backend(self, no_backend_adapter):
        result = await no_backend_adapter._semantic_search("test", "alice")
        assert "Error" in result

    async def test_semantic_search_success(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = [
            MockResult(content="result content", id="m1", score=0.92),
        ]
        result = await adapter._semantic_search("query", "alice")
        assert "Found 1 memories" in result
        assert "92%" in result
        assert "result content" in result

    async def test_semantic_search_no_results(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = []
        result = await adapter._semantic_search("query", "alice")
        assert "No memories found" in result
        assert "Tip" in result

    async def test_semantic_search_exception(self, adapter, mock_backend):
        mock_backend.search_memories.side_effect = RuntimeError("search fail")
        result = await adapter._semantic_search("query", "alice")
        assert "Error searching memories" in result

    async def test_semantic_search_empty_results_list(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = []
        result = await adapter._semantic_search("anything", "alice")
        assert "No memories found" in result

    async def test_get_recent_memories_no_backend(self, no_backend_adapter):
        result = await no_backend_adapter._get_recent_memories("alice")
        assert "Error" in result

    async def test_get_recent_memories_success(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = [
            MockResult(content="recent memory", id="m1", score=0.5),
        ]
        result = await adapter._get_recent_memories("alice")
        assert "Recent memories" in result
        assert "recent memory" in result

    async def test_get_recent_memories_empty(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = []
        result = await adapter._get_recent_memories("alice")
        assert "No memories stored yet" in result

    async def test_get_recent_memories_truncated_content(self, adapter, mock_backend):
        long = "x" * 200
        mock_backend.search_memories.return_value = [
            MockResult(content=long, id="m1", score=0.5),
        ]
        result = await adapter._get_recent_memories("alice")
        assert "..." in result

    async def test_get_recent_memories_exception(self, adapter, mock_backend):
        mock_backend.search_memories.side_effect = RuntimeError("recent fail")
        result = await adapter._get_recent_memories("alice")
        assert "Error getting recent memories" in result

    async def test_list_all_memories_no_backend(self, no_backend_adapter):
        result = await no_backend_adapter._list_all_memories("alice")
        assert "Error" in result

    async def test_list_all_memories_success(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = [
            MockResult(content="mem1", id="m1", score=0.5),
            MockResult(content="mem2", id="m2", score=0.4),
        ]
        result = await adapter._list_all_memories("alice")
        assert "Showing up to 20" in result
        assert "mem1" in result
        assert "mem2" in result

    async def test_list_all_memories_empty(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = []
        result = await adapter._list_all_memories("alice")
        assert "No memories stored yet" in result

    async def test_list_all_memories_truncated(self, adapter, mock_backend):
        long = "x" * 150
        mock_backend.search_memories.return_value = [
            MockResult(content=long, id="m1", score=0.5),
        ]
        result = await adapter._list_all_memories("alice")
        assert "..." in result

    async def test_list_all_memories_custom_limit(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = [
            MockResult(content=f"mem{i}", id=f"m{i}", score=0.5) for i in range(5)
        ]
        result = await adapter._list_all_memories("alice", limit=5)
        assert "Showing up to 5" in result

    async def test_list_all_memories_exception(self, adapter, mock_backend):
        mock_backend.search_memories.side_effect = RuntimeError("list fail")
        result = await adapter._list_all_memories("alice")
        assert "Error listing memories" in result

    async def test_get_memory_overview_no_backend(self, no_backend_adapter):
        result = await no_backend_adapter._get_memory_overview("alice")
        assert "Error" in result

    async def test_get_memory_overview_with_memories(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = [
            MockResult(content="preview content here", id="m1", score=0.5),
        ]
        result = await adapter._get_memory_overview("alice")
        assert "Memory System" in result
        assert "1 memories stored" in result
        assert "preview" in result

    async def test_get_memory_overview_empty(self, adapter, mock_backend):
        mock_backend.search_memories.return_value = []
        result = await adapter._get_memory_overview("alice")
        assert "0 memories stored" in result

    async def test_get_memory_overview_exception(self, adapter, mock_backend):
        mock_backend.search_memories.side_effect = RuntimeError("overview fail")
        result = await adapter._get_memory_overview("alice")
        assert "Memory System" in result

    async def test_get_memory_overview_truncated_preview(self, adapter, mock_backend):
        long = "x" * 100
        mock_backend.search_memories.return_value = [
            MockResult(content=long, id="m1", score=0.5),
        ]
        result = await adapter._get_memory_overview("alice")
        assert "..." in result


class TestAdapterInitialization:
    async def test_stores_backend(self, mock_backend):
        adapter = SemanticNativeToolAdapter(mock_backend, agent_type="test-agent")
        assert adapter._backend is mock_backend
        assert adapter._agent_type == "test-agent"

    async def test_none_backend(self):
        adapter = SemanticNativeToolAdapter(None)
        assert adapter._backend is None
        assert adapter._agent_type == "unknown"

    async def test_default_agent_type(self, mock_backend):
        adapter = SemanticNativeToolAdapter(mock_backend)
        assert adapter._agent_type == "unknown"
