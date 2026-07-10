from __future__ import annotations

import json
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock

import pytest

from headroom.proxy.memory_tool_adapter import (
    ANTHROPIC_BETA_HEADER,
    ANTHROPIC_CUSTOM_TOOLS,
    ANTHROPIC_NATIVE_TOOL,
    GEMINI_TOOLS,
    MEMORY_TOOL_NAMES,
    NATIVE_MEMORY_TOOL_NAME,
    NATIVE_MEMORY_TOOL_TYPE,
    OPENAI_TOOLS,
    MemoryToolAdapter,
    MemoryToolAdapterConfig,
    Provider,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> MemoryToolAdapterConfig:
    return MemoryToolAdapterConfig()


@pytest.fixture
def enabled_config() -> MemoryToolAdapterConfig:
    return MemoryToolAdapterConfig(enabled=True)


@pytest.fixture
def adapter(default_config: MemoryToolAdapterConfig) -> MemoryToolAdapter:
    return MemoryToolAdapter(default_config)


@pytest.fixture
def enabled_adapter(enabled_config: MemoryToolAdapterConfig) -> MemoryToolAdapter:
    return MemoryToolAdapter(enabled_config)


@pytest.fixture
def mock_backend() -> MagicMock:
    """Create a mock backend with async methods."""
    backend = MagicMock()
    backend.save_memory = AsyncMock(return_value=MagicMock(id="mem_123", content="test content"))
    backend.search_memories = AsyncMock(
        return_value=[
            MagicMock(
                score=0.95,
                memory=MagicMock(id="mem_456", content="found memory"),
                related_entities=["entity1"],
            )
        ]
    )
    backend.delete_memory = AsyncMock(return_value=True)
    backend.update_memory = AsyncMock(
        return_value=MagicMock(id="mem_789", content="updated content")
    )
    backend.close = AsyncMock()
    return backend


# =============================================================================
# Section 1: MemoryToolAdapterConfig
# =============================================================================


class TestMemoryToolAdapterConfig:
    def test_default_values(self) -> None:
        config = MemoryToolAdapterConfig()
        assert config.enabled is False
        assert config.use_native_tool is True
        assert config.inject_tools is True
        assert config.inject_context is True
        assert config.db_path == "headroom_memory.db"
        assert config.top_k == 10
        assert config.min_similarity == 0.3

    def test_custom_values(self) -> None:
        config = MemoryToolAdapterConfig(
            enabled=True,
            use_native_tool=False,
            inject_tools=False,
            inject_context=False,
            db_path="/custom/path.db",
            top_k=25,
            min_similarity=0.7,
        )
        assert config.enabled is True
        assert config.use_native_tool is False
        assert config.inject_tools is False
        assert config.inject_context is False
        assert config.db_path == "/custom/path.db"
        assert config.top_k == 25
        assert config.min_similarity == 0.7

    def test_partial_custom_values(self) -> None:
        config = MemoryToolAdapterConfig(enabled=True, top_k=5)
        assert config.enabled is True
        assert config.use_native_tool is True
        assert config.top_k == 5
        assert config.min_similarity == 0.3

    def test_asdict_roundtrip(self) -> None:
        config = MemoryToolAdapterConfig(enabled=True)
        d = asdict(config)
        restored = MemoryToolAdapterConfig(**d)
        assert restored == config


# =============================================================================
# Section 2: Tool Schema Constants
# =============================================================================


class TestToolSchemaConstants:
    def test_anthropic_native_tool(self) -> None:
        assert ANTHROPIC_NATIVE_TOOL == {
            "type": NATIVE_MEMORY_TOOL_TYPE,
            "name": NATIVE_MEMORY_TOOL_NAME,
        }

    def test_anthropic_custom_tools_has_four(self) -> None:
        assert len(ANTHROPIC_CUSTOM_TOOLS) == 4

    def test_anthropic_custom_tools_names(self) -> None:
        names = {t["name"] for t in ANTHROPIC_CUSTOM_TOOLS}
        assert names == MEMORY_TOOL_NAMES

    def test_anthropic_custom_tools_have_input_schema(self) -> None:
        for tool in ANTHROPIC_CUSTOM_TOOLS:
            assert "input_schema" in tool

    def test_openai_tools_has_four(self) -> None:
        assert len(OPENAI_TOOLS) == 4

    def test_openai_tools_function_names(self) -> None:
        names = {t["function"]["name"] for t in OPENAI_TOOLS}
        assert names == MEMORY_TOOL_NAMES

    def test_openai_tools_have_parameters(self) -> None:
        for tool in OPENAI_TOOLS:
            assert tool["type"] == "function"
            assert "parameters" in tool["function"]

    def test_gemini_tools_has_four(self) -> None:
        assert len(GEMINI_TOOLS) == 4

    def test_gemini_tools_names(self) -> None:
        names = {t["name"] for t in GEMINI_TOOLS}
        assert names == MEMORY_TOOL_NAMES

    def test_all_schemas_have_matching_names(self) -> None:
        """All tool schemas should define the same 4 memory tools."""
        anthropic_names = {t["name"] for t in ANTHROPIC_CUSTOM_TOOLS}
        openai_names = {t["function"]["name"] for t in OPENAI_TOOLS}
        gemini_names = {t["name"] for t in GEMINI_TOOLS}
        assert anthropic_names == openai_names == gemini_names == MEMORY_TOOL_NAMES

    def test_memory_tool_names_contains_four(self) -> None:
        assert MEMORY_TOOL_NAMES == {
            "memory_save",
            "memory_search",
            "memory_update",
            "memory_delete",
        }


# =============================================================================
# Section 3: detect_provider
# =============================================================================


class TestDetectProvider:
    @pytest.mark.parametrize(
        ("headers", "model", "expected"),
        [
            ({"x-api-key": "sk-ant-xxx"}, None, "anthropic"),
            ({"anthropic-version": "2023-06-01"}, None, "anthropic"),
            ({}, "claude-3-5-sonnet-20241022", "anthropic"),
            ({}, "claude-opus-4-20250514", "anthropic"),
            ({"authorization": "Bearer sk-xxx"}, None, "openai"),
            ({}, "gpt-4o", "openai"),
            ({}, "gpt-4-turbo", "openai"),
            ({}, "o1-preview", "openai"),
            ({}, "o3-mini", "openai"),
            ({}, "gemini-2.0-flash", "gemini"),
            ({}, "gemini-1.5-pro", "gemini"),
            ({}, "gemma-2-27b-it", "gemini"),
            ({}, "models/gemini-2.0-flash", "generic"),  # "models/" prefix doesn't match gemini*
            ({}, None, "generic"),
            ({}, "", "generic"),
            (None, None, "generic"),
        ],
    )
    def test_detect_provider(
        self,
        adapter: MemoryToolAdapter,
        headers: dict[str, str] | None,
        model: str | None,
        expected: Provider,
    ) -> None:
        assert adapter.detect_provider(headers, model) == expected

    def test_header_precedence_over_model(self, adapter: MemoryToolAdapter) -> None:
        """Headers should take priority over model name."""
        result = adapter.detect_provider({"x-api-key": "sk-ant-xxx"}, "gpt-4o")
        assert result == "anthropic"

    def test_authorization_not_bearer_sk(self, adapter: MemoryToolAdapter) -> None:
        result = adapter.detect_provider({"authorization": "Bearer some-other-format"}, None)
        assert result == "generic"

    def test_empty_headers(self, adapter: MemoryToolAdapter) -> None:
        result = adapter.detect_provider({}, None)
        assert result == "generic"

    def test_unknown_model(self, adapter: MemoryToolAdapter) -> None:
        result = adapter.detect_provider({}, "llama-3-70b")
        assert result == "generic"


# =============================================================================
# Section 4: inject_tools
# =============================================================================


class TestInjectTools:
    def test_disabled_returns_tools_unchanged(
        self, default_config: MemoryToolAdapterConfig
    ) -> None:
        config = MemoryToolAdapterConfig(inject_tools=False)
        adapter = MemoryToolAdapter(config)
        existing = [{"name": "existing_tool"}]
        tools, headers = adapter.inject_tools(existing, "anthropic")
        assert tools == existing
        assert headers == {}

    def test_disabled_with_none_returns_empty_list(
        self, default_config: MemoryToolAdapterConfig
    ) -> None:
        config = MemoryToolAdapterConfig(inject_tools=False)
        adapter = MemoryToolAdapter(config)
        tools, headers = adapter.inject_tools(None, "anthropic")
        assert tools == []
        assert headers == {}

    # --- Anthropic native ---

    def test_anthropic_native_tool_injected(self, enabled_adapter: MemoryToolAdapter) -> None:
        tools, headers = enabled_adapter.inject_tools([], "anthropic")
        assert ANTHROPIC_NATIVE_TOOL in tools
        assert headers == {"anthropic-beta": ANTHROPIC_BETA_HEADER}

    def test_anthropic_native_tool_dedup(self, enabled_adapter: MemoryToolAdapter) -> None:
        existing = [{"name": NATIVE_MEMORY_TOOL_NAME, "type": "other"}]
        tools, headers = enabled_adapter.inject_tools(existing, "anthropic")
        native_count = sum(1 for t in tools if t.get("name") == NATIVE_MEMORY_TOOL_NAME)
        assert native_count == 1

    def test_anthropic_native_disabled_injects_custom(
        self, default_config: MemoryToolAdapterConfig
    ) -> None:
        config = MemoryToolAdapterConfig(enabled=True, use_native_tool=False)
        adapter = MemoryToolAdapter(config)
        tools, headers = adapter.inject_tools([], "anthropic")
        assert ANTHROPIC_NATIVE_TOOL not in tools
        tool_names = {t["name"] for t in tools}
        assert tool_names == MEMORY_TOOL_NAMES

    def test_anthropic_custom_tools_dedup(self, enabled_adapter: MemoryToolAdapter) -> None:
        config = MemoryToolAdapterConfig(enabled=True, use_native_tool=False)
        adapter = MemoryToolAdapter(config)
        existing = [{"name": "memory_save"}]
        tools, headers = adapter.inject_tools(existing, "anthropic")
        tool_names = {t["name"] for t in tools}
        assert tool_names == MEMORY_TOOL_NAMES
        save_count = sum(1 for t in tools if t["name"] == "memory_save")
        assert save_count == 1

    # --- OpenAI ---

    def test_openai_tools_injected(self, enabled_adapter: MemoryToolAdapter) -> None:
        tools, headers = enabled_adapter.inject_tools([], "openai")
        assert len(tools) == 4
        for t in tools:
            assert t["type"] == "function"
            assert "function" in t

    def test_openai_tools_name_used_as_key(self, enabled_adapter: MemoryToolAdapter) -> None:
        existing = [{"type": "function", "function": {"name": "memory_search"}}]
        tools, headers = enabled_adapter.inject_tools(existing, "openai")
        search_count = sum(1 for t in tools if t["function"]["name"] == "memory_search")
        assert search_count == 1

    def test_openai_tools_dedup(self, enabled_adapter: MemoryToolAdapter) -> None:
        existing = [
            {"type": "function", "function": {"name": "memory_save"}},
            {"type": "function", "function": {"name": "memory_search"}},
        ]
        tools, headers = enabled_adapter.inject_tools(existing, "openai")
        assert len(tools) == 4

    # --- Gemini ---

    def test_gemini_tools_injected(self, enabled_adapter: MemoryToolAdapter) -> None:
        tools, headers = enabled_adapter.inject_tools([], "gemini")
        assert len(tools) == 4
        for t in tools:
            assert "parameters" in t

    def test_gemini_tools_dedup(self, enabled_adapter: MemoryToolAdapter) -> None:
        existing = [{"name": "memory_delete"}]
        tools, headers = enabled_adapter.inject_tools(existing, "gemini")
        delete_count = sum(1 for t in tools if t["name"] == "memory_delete")
        assert delete_count == 1

    # --- Generic fallback (OpenAI format) ---

    def test_generic_fallback_uses_openai_format(self, enabled_adapter: MemoryToolAdapter) -> None:
        tools, headers = enabled_adapter.inject_tools([], "generic")
        assert len(tools) == 4
        for t in tools:
            assert t["type"] == "function"
            assert "function" in t

    # --- Edge cases ---

    def test_inject_tools_none(self, enabled_adapter: MemoryToolAdapter) -> None:
        tools, headers = enabled_adapter.inject_tools(None, "openai")
        assert len(tools) == 4

    def test_inject_tools_empty_list(self, enabled_adapter: MemoryToolAdapter) -> None:
        tools, headers = enabled_adapter.inject_tools([], "openai")
        assert len(tools) == 4
        assert headers == {}

    def test_inject_anthropic_native_with_other_tools(
        self, enabled_adapter: MemoryToolAdapter
    ) -> None:
        existing = [{"name": "get_weather", "type": "custom"}]
        tools, headers = enabled_adapter.inject_tools(existing, "anthropic")
        assert len(tools) == 2
        assert ANTHROPIC_NATIVE_TOOL in tools


# =============================================================================
# Section 5: get_beta_headers
# =============================================================================


class TestGetBetaHeaders:
    def test_anthropic_with_native(self, enabled_adapter: MemoryToolAdapter) -> None:
        result = enabled_adapter.get_beta_headers("anthropic")
        assert result == {"anthropic-beta": ANTHROPIC_BETA_HEADER}

    def test_anthropic_without_native(self, default_config: MemoryToolAdapterConfig) -> None:
        config = MemoryToolAdapterConfig(enabled=True, use_native_tool=False)
        adapter = MemoryToolAdapter(config)
        result = adapter.get_beta_headers("anthropic")
        assert result == {}

    def test_openai(self, enabled_adapter: MemoryToolAdapter) -> None:
        result = enabled_adapter.get_beta_headers("openai")
        assert result == {}

    def test_gemini(self, enabled_adapter: MemoryToolAdapter) -> None:
        result = enabled_adapter.get_beta_headers("gemini")
        assert result == {}

    def test_generic(self, enabled_adapter: MemoryToolAdapter) -> None:
        result = enabled_adapter.get_beta_headers("generic")
        assert result == {}

    def test_native_disabled_flag_returns_empty(self) -> None:
        config = MemoryToolAdapterConfig(enabled=True, use_native_tool=False)
        adapter = MemoryToolAdapter(config)
        result = adapter.get_beta_headers("anthropic")
        assert result == {}


# =============================================================================
# Section 6: has_memory_tool_calls
# =============================================================================


class TestHasMemoryToolCalls:
    def test_anthropic_native_memory(self, adapter: MemoryToolAdapter) -> None:
        response = {
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "tool_use", "name": "memory", "id": "tu_1"},
            ]
        }
        assert adapter.has_memory_tool_calls(response, "anthropic") is True

    def test_anthropic_custom_memory(self, adapter: MemoryToolAdapter) -> None:
        response = {
            "content": [
                {"type": "tool_use", "name": "memory_save", "id": "tu_1"},
            ]
        }
        assert adapter.has_memory_tool_calls(response, "anthropic") is True

    def test_anthropic_non_memory_tool(self, adapter: MemoryToolAdapter) -> None:
        response = {
            "content": [
                {"type": "tool_use", "name": "get_weather", "id": "tu_1"},
            ]
        }
        assert adapter.has_memory_tool_calls(response, "anthropic") is False

    def test_openai_memory_tool_call(self, adapter: MemoryToolAdapter) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "memory_search"},
                            }
                        ]
                    }
                }
            ]
        }
        assert adapter.has_memory_tool_calls(response, "openai") is True

    def test_openai_non_memory_tool(self, adapter: MemoryToolAdapter) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "get_weather"},
                            }
                        ]
                    }
                }
            ]
        }
        assert adapter.has_memory_tool_calls(response, "openai") is False

    def test_gemini_memory_tool_call(self, adapter: MemoryToolAdapter) -> None:
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "memory_save"}},
                        ]
                    }
                }
            ]
        }
        assert adapter.has_memory_tool_calls(response, "gemini") is True

    def test_gemini_non_memory_tool(self, adapter: MemoryToolAdapter) -> None:
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "get_weather"}},
                        ]
                    }
                }
            ]
        }
        assert adapter.has_memory_tool_calls(response, "gemini") is False

    def test_generic_anthropic_format(self, adapter: MemoryToolAdapter) -> None:
        response = {
            "content": [
                {"type": "tool_use", "name": "memory_save", "id": "tu_1"},
            ]
        }
        assert adapter.has_memory_tool_calls(response, "generic") is True

    def test_generic_openai_format(self, adapter: MemoryToolAdapter) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "memory_save"},
                            }
                        ]
                    }
                }
            ]
        }
        assert adapter.has_memory_tool_calls(response, "generic") is True

    def test_no_memory_tools_mixed(self, adapter: MemoryToolAdapter) -> None:
        response = {
            "content": [
                {"type": "tool_use", "name": "code_edit", "id": "tu_1"},
                {"type": "tool_use", "name": "read_file", "id": "tu_2"},
            ]
        }
        assert adapter.has_memory_tool_calls(response, "anthropic") is False

    def test_empty_response(self, adapter: MemoryToolAdapter) -> None:
        assert adapter.has_memory_tool_calls({}, "anthropic") is False
        assert adapter.has_memory_tool_calls({}, "openai") is False
        assert adapter.has_memory_tool_calls({}, "gemini") is False
        assert adapter.has_memory_tool_calls({}, "generic") is False

    def test_null_tool_calls_openai(self, adapter: MemoryToolAdapter) -> None:
        """OpenAI may return null/None for tool_calls."""
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": None,
                    }
                }
            ]
        }
        assert adapter.has_memory_tool_calls(response, "openai") is False


# =============================================================================
# Section 7: handle_tool_calls
# =============================================================================


class TestHandleToolCalls:
    @pytest.mark.asyncio
    async def test_native_memory_execution(self, enabled_config: MemoryToolAdapterConfig) -> None:
        adapter = MemoryToolAdapter(enabled_config)
        adapter._backend = MagicMock()
        adapter._initialized = True
        adapter._backend.save_memory = AsyncMock(
            return_value=MagicMock(id="mem_native", content="saved")
        )

        response = {
            "content": [
                {
                    "type": "tool_use",
                    "name": "memory",
                    "id": "tu_native",
                    "input": {
                        "command": "create",
                        "file_text": "hello",
                        "path": "/memories/test.txt",
                    },
                }
            ]
        }
        results = await adapter.handle_tool_calls(response, "user_1", "anthropic")
        assert len(results) == 1
        assert results[0]["type"] == "tool_result"
        assert results[0]["tool_use_id"] == "tu_native"

    @pytest.mark.asyncio
    async def test_custom_memory_save(self, enabled_config: MemoryToolAdapterConfig) -> None:
        adapter = MemoryToolAdapter(enabled_config)
        adapter._backend = MagicMock()
        adapter._initialized = True
        adapter._backend.save_memory = AsyncMock(
            return_value=MagicMock(id="mem_save", content="importance of testing")
        )

        response = {
            "content": [
                {
                    "type": "tool_use",
                    "name": "memory_save",
                    "id": "tu_save",
                    "input": {"content": "importance of testing", "importance": 0.9},
                }
            ]
        }
        results = await adapter.handle_tool_calls(response, "user_1", "anthropic")
        assert len(results) == 1
        result_data = json.loads(results[0]["content"])
        assert result_data["status"] == "saved"
        assert result_data["memory_id"] == "mem_save"

    @pytest.mark.asyncio
    async def test_custom_memory_search(self, enabled_config: MemoryToolAdapterConfig) -> None:
        adapter = MemoryToolAdapter(enabled_config)
        backend = MagicMock()
        backend.search_memories = AsyncMock(
            return_value=[
                MagicMock(
                    score=0.92,
                    memory=MagicMock(id="mem_found", content="relevant info"),
                    related_entities=["python"],
                )
            ]
        )
        adapter._backend = backend
        adapter._initialized = True

        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_search",
                                "function": {
                                    "name": "memory_search",
                                    "arguments": '{"query": "python tips"}',
                                },
                            }
                        ]
                    }
                }
            ]
        }
        results = await adapter.handle_tool_calls(response, "user_1", "openai")
        assert len(results) == 1
        assert results[0]["role"] == "tool"
        assert results[0]["tool_call_id"] == "call_search"
        result_data = json.loads(results[0]["content"])
        assert result_data["status"] == "found"
        assert result_data["count"] == 1

    @pytest.mark.asyncio
    async def test_custom_memory_update(self, enabled_config: MemoryToolAdapterConfig) -> None:
        adapter = MemoryToolAdapter(enabled_config)
        backend = MagicMock()
        adapter._backend = backend
        adapter._initialized = True

        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "memory_update",
                                    "args": {
                                        "memory_id": "mem_old",
                                        "new_content": "updated content",
                                    },
                                }
                            }
                        ]
                    }
                }
            ]
        }

        # update_memory path
        backend.update_memory = AsyncMock(
            return_value=MagicMock(id="mem_old", content="updated content")
        )
        results = await adapter.handle_tool_calls(response, "user_1", "gemini")
        assert len(results) == 1
        assert "functionResponse" in results[0]
        result_data = json.loads(results[0]["functionResponse"]["response"]["result"])
        assert result_data["status"] == "updated"

    @pytest.mark.asyncio
    async def test_custom_memory_delete(self, enabled_config: MemoryToolAdapterConfig) -> None:
        adapter = MemoryToolAdapter(enabled_config)
        backend = MagicMock()
        adapter._backend = backend
        adapter._initialized = True
        backend.delete_memory = AsyncMock(return_value=True)

        response = {
            "content": [
                {
                    "type": "tool_use",
                    "name": "memory_delete",
                    "id": "tu_del",
                    "input": {"memory_id": "mem_to_del"},
                }
            ]
        }
        results = await adapter.handle_tool_calls(response, "user_1", "anthropic")
        assert len(results) == 1
        result_data = json.loads(results[0]["content"])
        assert result_data["status"] == "deleted"
        assert result_data["memory_id"] == "mem_to_del"

    @pytest.mark.asyncio
    async def test_non_memory_tool_skipped(self, enabled_config: MemoryToolAdapterConfig) -> None:
        adapter = MemoryToolAdapter(enabled_config)
        adapter._backend = MagicMock()
        adapter._initialized = True

        response = {
            "content": [
                {
                    "type": "tool_use",
                    "name": "get_weather",
                    "id": "tu_weather",
                    "input": {"location": "NYC"},
                },
                {
                    "type": "tool_use",
                    "name": "memory_save",
                    "id": "tu_save",
                    "input": {"content": "test", "importance": 0.5},
                },
            ]
        }
        adapter._backend.save_memory = AsyncMock(
            return_value=MagicMock(id="mem_saved", content="test")
        )

        results = await adapter.handle_tool_calls(response, "user_1", "anthropic")
        assert len(results) == 1
        result_data = json.loads(results[0]["content"])
        assert result_data["status"] == "saved"

    @pytest.mark.asyncio
    async def test_backend_not_initialized(self, enabled_config: MemoryToolAdapterConfig) -> None:
        adapter = MemoryToolAdapter(enabled_config)
        adapter._ensure_initialized = AsyncMock()  # type: ignore[assignment]
        adapter._backend = None

        response = {
            "content": [
                {
                    "type": "tool_use",
                    "name": "memory",
                    "id": "tu_1",
                    "input": {"command": "view", "path": "/memories"},
                }
            ]
        }
        results = await adapter.handle_tool_calls(response, "user_1", "anthropic")
        assert len(results) == 1
        assert "Error" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_generic_provider_format(self, enabled_config: MemoryToolAdapterConfig) -> None:
        adapter = MemoryToolAdapter(enabled_config)
        backend = MagicMock()
        backend.save_memory = AsyncMock(return_value=MagicMock(id="mem_gen", content="generic"))
        adapter._backend = backend
        adapter._initialized = True

        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_gen",
                                "function": {
                                    "name": "memory_save",
                                    "arguments": '{"content": "generic save", "importance": 0.5}',
                                },
                            }
                        ]
                    }
                }
            ]
        }
        results = await adapter.handle_tool_calls(response, "user_1", "generic")
        assert len(results) == 1
        assert results[0]["role"] == "tool"
        assert results[0]["tool_call_id"] == "call_gen"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self, enabled_config: MemoryToolAdapterConfig) -> None:
        adapter = MemoryToolAdapter(enabled_config)
        backend = MagicMock()
        backend.save_memory = AsyncMock(return_value=MagicMock(id="mem_m1", content="m1"))
        backend.search_memories = AsyncMock(
            return_value=[
                MagicMock(
                    score=0.9,
                    memory=MagicMock(id="mem_m2", content="m2"),
                    related_entities=[],
                )
            ]
        )
        adapter._backend = backend
        adapter._initialized = True

        response = {
            "content": [
                {
                    "type": "tool_use",
                    "name": "memory_save",
                    "id": "tu_s1",
                    "input": {"content": "first", "importance": 0.5},
                },
                {
                    "type": "tool_use",
                    "name": "memory_search",
                    "id": "tu_s2",
                    "input": {"query": "second"},
                },
            ]
        }
        results = await adapter.handle_tool_calls(response, "user_1", "anthropic")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_oai_tool_calls_none_value(self, enabled_config: MemoryToolAdapterConfig) -> None:
        adapter = MemoryToolAdapter(enabled_config)
        adapter._backend = MagicMock()
        adapter._initialized = True

        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": None,
                    }
                }
            ]
        }
        results = await adapter.handle_tool_calls(response, "user_1", "openai")
        assert results == []

    @pytest.mark.asyncio
    async def test_oai_update_no_update_memory_fallback(
        self, enabled_config: MemoryToolAdapterConfig
    ) -> None:
        adapter = MemoryToolAdapter(enabled_config)
        backend = MagicMock()
        backend.delete_memory = AsyncMock(return_value=True)
        backend.save_memory = AsyncMock(return_value=MagicMock(id="mem_new", content="new"))
        adapter._backend = backend
        adapter._initialized = True
        del backend.update_memory  # simulate backend without update_memory

        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_up",
                                "function": {
                                    "name": "memory_update",
                                    "arguments": '{"memory_id": "m1", "new_content": "replacement"}',
                                },
                            }
                        ]
                    }
                }
            ]
        }
        results = await adapter.handle_tool_calls(response, "user_1", "openai")
        assert len(results) == 1
        result_data = json.loads(results[0]["content"])
        assert result_data["status"] == "updated"
        assert "Replaced via delete+save" in result_data.get("note", "")


# =============================================================================
# Section 8: Private Helpers
# =============================================================================


class TestExtractToolCalls:
    def test_anthropic(self, adapter: MemoryToolAdapter) -> None:
        response = {
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "tool_use", "name": "memory_save", "id": "tu_1"},
            ]
        }
        calls = adapter._extract_tool_calls(response, "anthropic")
        assert len(calls) == 1
        assert calls[0]["name"] == "memory_save"

    def test_anthropic_no_content(self, adapter: MemoryToolAdapter) -> None:
        calls = adapter._extract_tool_calls({"content": "not a list"}, "anthropic")
        assert calls == []

    def test_openai(self, adapter: MemoryToolAdapter) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "c1", "function": {"name": "memory_save"}},
                        ]
                    }
                }
            ]
        }
        calls = adapter._extract_tool_calls(response, "openai")
        assert len(calls) == 1
        assert calls[0]["id"] == "c1"

    def test_openai_no_tool_calls(self, adapter: MemoryToolAdapter) -> None:
        response = {"choices": [{"message": {}}]}
        calls = adapter._extract_tool_calls(response, "openai")
        assert calls == []

    def test_openai_no_choices(self, adapter: MemoryToolAdapter) -> None:
        calls = adapter._extract_tool_calls({}, "openai")
        assert calls == []

    def test_gemini(self, adapter: MemoryToolAdapter) -> None:
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "memory_search"}},
                            {"text": "hello"},
                        ]
                    }
                }
            ]
        }
        calls = adapter._extract_tool_calls(response, "gemini")
        assert len(calls) == 1
        assert "functionCall" in calls[0]

    def test_gemini_no_candidates(self, adapter: MemoryToolAdapter) -> None:
        calls = adapter._extract_tool_calls({}, "gemini")
        assert calls == []

    def test_generic_anthropic_format(self, adapter: MemoryToolAdapter) -> None:
        response = {
            "content": [
                {"type": "tool_use", "name": "memory_save", "id": "tu_1"},
            ]
        }
        calls = adapter._extract_tool_calls(response, "generic")
        assert len(calls) == 1

    def test_generic_openai_format(self, adapter: MemoryToolAdapter) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "c1", "function": {"name": "memory_save"}},
                        ]
                    }
                }
            ]
        }
        calls = adapter._extract_tool_calls(response, "generic")
        assert len(calls) == 1

    def test_generic_both_formats(self, adapter: MemoryToolAdapter) -> None:
        """Generic collects from both formats."""
        response = {
            "content": [
                {"type": "tool_use", "name": "memory_save", "id": "tu_1"},
            ],
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "c1", "function": {"name": "memory_search"}},
                        ]
                    }
                }
            ],
        }
        calls = adapter._extract_tool_calls(response, "generic")
        assert len(calls) == 2

    def test_generic_empty(self, adapter: MemoryToolAdapter) -> None:
        calls = adapter._extract_tool_calls({}, "generic")
        assert calls == []


class TestGetToolName:
    def test_anthropic(self, adapter: MemoryToolAdapter) -> None:
        tc = {"name": "memory_save"}
        assert adapter._get_tool_name(tc, "anthropic") == "memory_save"

    def test_anthropic_missing_name(self, adapter: MemoryToolAdapter) -> None:
        assert adapter._get_tool_name({}, "anthropic") == ""

    def test_openai(self, adapter: MemoryToolAdapter) -> None:
        tc = {"function": {"name": "memory_search"}}
        assert adapter._get_tool_name(tc, "openai") == "memory_search"

    def test_openai_missing_function(self, adapter: MemoryToolAdapter) -> None:
        assert adapter._get_tool_name({}, "openai") == ""

    def test_gemini(self, adapter: MemoryToolAdapter) -> None:
        tc = {"functionCall": {"name": "memory_delete"}}
        assert adapter._get_tool_name(tc, "gemini") == "memory_delete"

    def test_gemini_missing_call(self, adapter: MemoryToolAdapter) -> None:
        assert adapter._get_tool_name({}, "gemini") == ""

    def test_generic_anthropic(self, adapter: MemoryToolAdapter) -> None:
        tc = {"name": "memory_save"}
        assert adapter._get_tool_name(tc, "generic") == "memory_save"

    def test_generic_openai(self, adapter: MemoryToolAdapter) -> None:
        tc = {"function": {"name": "memory_save"}}
        assert adapter._get_tool_name(tc, "generic") == "memory_save"

    def test_generic_empty(self, adapter: MemoryToolAdapter) -> None:
        assert adapter._get_tool_name({}, "generic") == ""


class TestGetToolId:
    def test_anthropic(self, adapter: MemoryToolAdapter) -> None:
        tc = {"id": "tu_abc"}
        assert adapter._get_tool_id(tc, "anthropic") == "tu_abc"

    def test_openai(self, adapter: MemoryToolAdapter) -> None:
        tc = {"id": "call_xyz"}
        assert adapter._get_tool_id(tc, "openai") == "call_xyz"

    def test_gemini(self, adapter: MemoryToolAdapter) -> None:
        """Gemini uses functionCall name as the ID."""
        tc = {"functionCall": {"name": "memory_save"}}
        assert adapter._get_tool_id(tc, "gemini") == "memory_save"

    def test_gemini_no_call(self, adapter: MemoryToolAdapter) -> None:
        assert adapter._get_tool_id({}, "gemini") == ""

    def test_generic(self, adapter: MemoryToolAdapter) -> None:
        tc = {"id": "generic_id"}
        assert adapter._get_tool_id(tc, "generic") == "generic_id"

    def test_missing_id(self, adapter: MemoryToolAdapter) -> None:
        assert adapter._get_tool_id({}, "anthropic") == ""


class TestGetToolInput:
    def test_anthropic(self, adapter: MemoryToolAdapter) -> None:
        tc = {"input": {"content": "test", "importance": 0.5}}
        result = adapter._get_tool_input(tc, "anthropic")
        assert result == {"content": "test", "importance": 0.5}

    def test_anthropic_empty(self, adapter: MemoryToolAdapter) -> None:
        assert adapter._get_tool_input({}, "anthropic") == {}

    def test_anthropic_not_dict(self, adapter: MemoryToolAdapter) -> None:
        tc = {"input": "string_value"}
        assert adapter._get_tool_input(tc, "anthropic") == {}

    def test_openai(self, adapter: MemoryToolAdapter) -> None:
        tc = {
            "function": {
                "arguments": '{"query": "python", "top_k": 5}',
            }
        }
        result = adapter._get_tool_input(tc, "openai")
        assert result == {"query": "python", "top_k": 5}

    def test_openai_invalid_json(self, adapter: MemoryToolAdapter) -> None:
        tc = {"function": {"arguments": "not-json"}}
        assert adapter._get_tool_input(tc, "openai") == {}

    def test_openai_missing_function(self, adapter: MemoryToolAdapter) -> None:
        assert adapter._get_tool_input({}, "openai") == {}

    def test_gemini(self, adapter: MemoryToolAdapter) -> None:
        tc = {
            "functionCall": {
                "args": {"memory_id": "m1", "new_content": "updated"},
            }
        }
        result = adapter._get_tool_input(tc, "gemini")
        assert result == {"memory_id": "m1", "new_content": "updated"}

    def test_gemini_empty(self, adapter: MemoryToolAdapter) -> None:
        tc = {"functionCall": {}}
        assert adapter._get_tool_input(tc, "gemini") == {}

    def test_generic_anthropic_input(self, adapter: MemoryToolAdapter) -> None:
        tc = {"input": {"content": "generic"}}
        result = adapter._get_tool_input(tc, "generic")
        assert result == {"content": "generic"}

    def test_generic_openai_arguments(self, adapter: MemoryToolAdapter) -> None:
        tc = {"function": {"arguments": '{"content": "generic"}'}}
        result = adapter._get_tool_input(tc, "generic")
        assert result == {"content": "generic"}

    def test_generic_anthropic_not_dict(self, adapter: MemoryToolAdapter) -> None:
        tc = {"input": "string"}
        assert adapter._get_tool_input(tc, "generic") == {}

    def test_generic_invalid_json(self, adapter: MemoryToolAdapter) -> None:
        tc = {"function": {"arguments": "bad"}}
        assert adapter._get_tool_input(tc, "generic") == {}


class TestFormatToolResult:
    def test_anthropic(self, adapter: MemoryToolAdapter) -> None:
        result = adapter._format_tool_result("tu_1", "some content", "anthropic")
        assert result == {
            "type": "tool_result",
            "tool_use_id": "tu_1",
            "content": "some content",
        }

    def test_openai(self, adapter: MemoryToolAdapter) -> None:
        result = adapter._format_tool_result("call_1", "result", "openai")
        assert result == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "result",
        }

    def test_gemini(self, adapter: MemoryToolAdapter) -> None:
        result = adapter._format_tool_result("memory_save", '{"status":"ok"}', "gemini")
        assert result == {
            "functionResponse": {
                "name": "memory_save",
                "response": {"result": '{"status":"ok"}'},
            }
        }

    def test_generic(self, adapter: MemoryToolAdapter) -> None:
        result = adapter._format_tool_result("g_id", "data", "generic")
        assert result == {
            "role": "tool",
            "tool_call_id": "g_id",
            "content": "data",
        }
