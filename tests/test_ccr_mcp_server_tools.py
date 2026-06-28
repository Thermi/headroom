from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("mcp")

from headroom.ccr import mcp_server
from headroom.ccr.mcp_server import (
    CCR_TOOL_NAME,
    COMPRESS_TOOL_NAME,
    READ_TOOL_NAME,
    STATS_TOOL_NAME,
    HeadroomMCPServer,
    SessionStats,
    _format_session_summary,
)
from headroom.compress import CompressResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_server(**kwargs) -> HeadroomMCPServer:
    kwargs.setdefault("check_proxy", False)
    kwargs.setdefault("proxy_url", "http://test:8787")
    return HeadroomMCPServer(**kwargs)


@pytest.fixture(autouse=True)
def _no_real_compress():
    """Prevent real compression pipeline from being triggered."""
    def fake_compress(*args, **kwargs):
        return CompressResult(
            messages=[{"role": "tool", "content": "compressed content"}],
            tokens_before=100,
            tokens_after=30,
            tokens_saved=70,
            compression_ratio=0.7,
            transforms_applied=["test_transform"],
        )
    patcher = patch("headroom.compress.compress", fake_compress)
    patcher.start()
    yield
    patcher.stop()


# ---------------------------------------------------------------------------
# SessionStats
# ---------------------------------------------------------------------------


class TestSessionStats:
    def test_starts_empty(self):
        stats = SessionStats()
        assert stats.compressions == 0
        assert stats.retrievals == 0
        assert stats.total_input_tokens == 0

    def test_record_compression(self):
        stats = SessionStats()
        stats.record_compression(input_tokens=200, output_tokens=50, strategy="test")
        assert stats.compressions == 1
        assert stats.total_input_tokens == 200
        assert stats.total_output_tokens == 50
        assert stats.total_tokens_saved == 150
        assert len(stats.events) == 1
        assert stats.events[0]["type"] == "compress"
        assert stats.events[0]["strategy"] == "test"

    def test_record_retrieval(self):
        stats = SessionStats()
        stats.record_retrieval(hash_key="abc123")
        assert stats.retrievals == 1
        assert len(stats.events) == 1
        assert stats.events[0]["type"] == "retrieve"
        assert stats.events[0]["hash"] == "abc123"

    def test_to_dict_basic(self):
        stats = SessionStats()
        stats.record_compression(input_tokens=200, output_tokens=50, strategy="t1")
        stats.record_retrieval(hash_key="xyz")
        d = stats.to_dict()
        assert d["compressions"] == 1
        assert d["retrievals"] == 1
        assert d["total_tokens_saved"] == 150
        assert d["savings_percent"] == 75.0
        assert "session_duration_seconds" in d

    def test_events_capped_at_50(self):
        stats = SessionStats()
        for _ in range(60):
            stats.record_compression(input_tokens=100, output_tokens=10, strategy="s")
        assert len(stats.events) == 50

    def test_to_dict_zero_input_no_div_zero(self):
        stats = SessionStats()
        d = stats.to_dict()
        assert d["savings_percent"] == 0


# ---------------------------------------------------------------------------
# _format_session_summary
# ---------------------------------------------------------------------------


class TestFormatSessionSummary:
    def test_empty_summary(self):
        text = _format_session_summary({}, {})
        assert "Headroom Session Summary" in text
        assert "Mode:" in text

    def test_with_compression_data(self):
        summary = {
            "mode": "cache",
            "api_requests": 10,
            "primary_model": "gpt-4o",
            "compression": {
                "requests_compressed": 5,
                "avg_compression_pct": 45,
                "total_tokens_removed": 12000,
            },
            "cost": {
                "without_headroom_usd": 1.0,
                "with_headroom_usd": 0.5,
                "total_saved_usd": 0.5,
                "savings_pct": 50,
            },
        }
        text = _format_session_summary(summary, {"compressions": 3, "total_tokens_saved": 600})
        assert "Compression (5 requests compressed)" in text
        assert "Cost Impact:" in text
        assert "MCP Tool:" in text
        assert "You saved:" in text

    def test_with_uncompressed_reasons(self):
        summary = {
            "uncompressed_requests": {"too_small": 3, "passthrough": 1},
        }
        text = _format_session_summary(summary, {})
        assert "Too small" in text
        assert "Passthrough" in text


# ---------------------------------------------------------------------------
# Shared stats file
# ---------------------------------------------------------------------------


class TestSharedStatsFile:
    def test_append_and_read_shared_event(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mcp_server, "_HAS_FCNTL", False)
        monkeypatch.setattr(mcp_server, "fcntl", None)
        monkeypatch.setattr(mcp_server, "SHARED_STATS_DIR", tmp_path)
        monkeypatch.setattr(mcp_server, "SHARED_STATS_FILE", tmp_path / "stats.jsonl")
        monkeypatch.setattr(mcp_server.os, "getpid", lambda: 999)
        monkeypatch.setattr(mcp_server.time, "time", lambda: 2000.0)

        mcp_server._append_shared_event({"type": "compress", "timestamp": 2000.0})
        mcp_server._append_shared_event({"type": "retrieve", "timestamp": 2000.0})

        events = mcp_server._read_shared_events(window_seconds=3600)
        assert len(events) == 2
        assert events[0]["type"] == "compress"
        assert events[0]["pid"] == 999
        assert events[1]["type"] == "retrieve"

    def test_read_shared_events_prunes_old(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mcp_server, "_HAS_FCNTL", False)
        monkeypatch.setattr(mcp_server, "fcntl", None)
        monkeypatch.setattr(mcp_server, "SHARED_STATS_DIR", tmp_path)
        stats_file = tmp_path / "stats.jsonl"
        monkeypatch.setattr(mcp_server, "SHARED_STATS_FILE", stats_file)

        data = [
            {"type": "compress", "timestamp": 100.0},
            {"type": "compress", "timestamp": 9999.0},
        ]
        with open(stats_file, "w") as f:
            for evt in data:
                f.write(json.dumps(evt) + "\n")

        events = mcp_server._read_shared_events(window_seconds=100)
        assert all(e["timestamp"] >= 9999.0 - 100 for e in events)

    def test_read_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mcp_server, "SHARED_STATS_FILE", tmp_path / "nonexistent.jsonl")
        assert mcp_server._read_shared_events() == []


# ---------------------------------------------------------------------------
# HeadroomMCPServer -- _compress_content (sync, CPU-bound)
# ---------------------------------------------------------------------------


class TestCompressContent:
    def test_compress_content_returns_expected_fields(self, monkeypatch):
        server = _make_server()

        with patch("headroom.compress.compress") as mock_compress:
            mock_compress.return_value = CompressResult(
                messages=[{"role": "tool", "content": "shortened"}],
                tokens_before=100,
                tokens_after=25,
                tokens_saved=75,
                compression_ratio=0.75,
                transforms_applied=["test_kompress"],
            )

            with patch(
                "headroom.cache.compression_store.CompressionStore"
            ) as MockStore:
                mock_store = MockStore.return_value
                mock_store.store.return_value = "fakehash123"

                result = server._compress_content("some large content here")

        assert result["compressed"] == "shortened"
        assert result["hash"] == "fakehash123"
        assert result["original_tokens"] == 100
        assert result["compressed_tokens"] == 25
        assert result["tokens_saved"] == 75
        assert "transforms" in result
        assert server._stats.compressions == 1

    def test_compress_empty_content(self, monkeypatch):
        server = _make_server()

        with patch("headroom.compress.compress") as mock_compress:
            mock_compress.return_value = CompressResult(
                messages=[{"role": "tool", "content": ""}],
                tokens_before=0, tokens_after=0, tokens_saved=0,
                compression_ratio=0.0, transforms_applied=[],
            )
            with patch(
                "headroom.cache.compression_store.CompressionStore"
            ) as MockStore:
                mock_store = MockStore.return_value
                mock_store.store.return_value = "emptyhash"
                result = server._compress_content("")

        assert isinstance(result, dict)
        assert "hash" in result


# ---------------------------------------------------------------------------
# HeadroomMCPServer -- _handle_compress (async handler)
# ---------------------------------------------------------------------------


class TestHandleCompress:
    async def test_compress_with_content(self):
        server = _make_server()
        with (
            patch.object(server, "_compress_content") as mock_cc,
            patch(
                "headroom.cache.compression_store.CompressionStore"
            ),
        ):
            mock_cc.return_value = {"compressed": "x", "hash": "h1"}
            result = await server._handle_compress({"content": "hello world"})

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["compressed"] == "x"
        assert data["hash"] == "h1"

    async def test_compress_missing_content(self):
        server = _make_server()
        result = await server._handle_compress({})
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "error" in data

    async def test_compress_null_content(self):
        server = _make_server()
        result = await server._handle_compress({"content": None})
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "error" in data


# ---------------------------------------------------------------------------
# HeadroomMCPServer -- _handle_retrieve
# ---------------------------------------------------------------------------


class TestHandleRetrieve:
    async def test_retrieve_local_hit(self):
        server = _make_server()
        with patch(
            "headroom.cache.compression_store.CompressionStore"
        ) as MockStore:
            mock_store = MockStore.return_value
            entry = MagicMock()
            entry.original_content = "full original text"
            entry.original_item_count = 10
            entry.compressed_item_count = 3
            entry.retrieval_count = 1
            mock_store.retrieve.return_value = entry

            result = await server._handle_retrieve({"hash": "abc123"})

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["source"] == "local"
        assert data["original_content"] == "full original text"
        assert server._stats.retrievals == 1

    async def test_retrieve_local_hit_with_query(self):
        server = _make_server()
        with patch(
            "headroom.cache.compression_store.CompressionStore"
        ) as MockStore:
            mock_store = MockStore.return_value
            mock_store.search.return_value = [{"item": "matched"}]

            result = await server._handle_retrieve({"hash": "abc", "query": "find me"})

        data = json.loads(result[0].text)
        assert data["source"] == "local"
        assert data["query"] == "find me"
        assert data["results"] == [{"item": "matched"}]

    async def test_retrieve_local_miss_then_proxy_hit(self):
        server = _make_server(check_proxy=True)
        with (
            patch(
                "headroom.cache.compression_store.CompressionStore"
            ) as MockStore,
        ):
            mock_store = MockStore.return_value
            mock_store.retrieve.return_value = None

            mock_client = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"original_content": "from proxy"}
            mock_client.post = AsyncMock(return_value=mock_resp)
            server._http_client = mock_client

            result = await server._handle_retrieve({"hash": "abc"})

        data = json.loads(result[0].text)
        assert data["source"] == "proxy"
        assert data["original_content"] == "from proxy"

    async def test_retrieve_not_found_anywhere(self):
        server = _make_server(check_proxy=False)
        with patch(
            "headroom.cache.compression_store.CompressionStore"
        ) as MockStore:
            MockStore.return_value.retrieve.return_value = None
            result = await server._handle_retrieve({"hash": "nonexistent"})

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["hash"] == "nonexistent"

    async def test_retrieve_missing_hash_param(self):
        server = _make_server()
        result = await server._handle_retrieve({})
        data = json.loads(result[0].text)
        assert "error" in data

    async def test_retrieve_proxy_404(self):
        server = _make_server(check_proxy=True)
        with (
            patch(
                "headroom.cache.compression_store.CompressionStore"
            ) as MockStore,
            patch.object(server, "_http_client") as mock_client,
        ):
            MockStore.return_value.retrieve.return_value = None

            mock_resp = AsyncMock()
            mock_resp.status_code = 404
            mock_client.post.return_value = mock_resp

            result = await server._handle_retrieve({"hash": "abc"})

        data = json.loads(result[0].text)
        assert "error" in data

    async def test_retrieve_proxy_unreachable(self):
        server = _make_server(check_proxy=True)
        with (
            patch(
                "headroom.cache.compression_store.CompressionStore"
            ) as MockStore,
            patch.object(server, "_http_client") as mock_client,
        ):
            MockStore.return_value.retrieve.return_value = None
            mock_client.post.side_effect = Exception("connection refused")

            result = await server._handle_retrieve({"hash": "abc"})

        data = json.loads(result[0].text)
        assert "error" in data


# ---------------------------------------------------------------------------
# HeadroomMCPServer -- _handle_stats
# ---------------------------------------------------------------------------


class TestHandleStats:
    async def test_stats_basic(self):
        server = _make_server(check_proxy=False)
        server._stats.record_compression(100, 20, "test")
        server._stats.record_retrieval("h1")

        mock_store_instance = MagicMock()
        mock_store_instance.get_stats.return_value = {
            "entry_count": 2, "max_entries": 500,
        }
        server._local_store = mock_store_instance

        result = await server._handle_stats()

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["compressions"] == 1
        assert data["retrievals"] == 1
        assert data["store"]["entries"] == 2

    async def test_stats_with_sub_agents(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "_HAS_FCNTL", False)
        monkeypatch.setattr(mcp_server, "fcntl", None)
        monkeypatch.setattr(mcp_server, "SHARED_STATS_DIR", MagicMock())
        monkeypatch.setattr(mcp_server, "SHARED_STATS_FILE", MagicMock())
        monkeypatch.setattr(mcp_server.os, "getpid", lambda: 111)

        fake_events = [
            {"type": "compress", "timestamp": time.time(), "pid": 222,
             "input_tokens": 500, "output_tokens": 100},
        ]
        with (
            patch.object(mcp_server, "_read_shared_events", return_value=fake_events),
            patch(
                "headroom.cache.compression_store.CompressionStore"
            ) as MockStore,
        ):
            MockStore.return_value.get_stats.return_value = {
                "entry_count": 0, "max_entries": 500,
            }
            server = _make_server(check_proxy=False)
            result = await server._handle_stats()

        data = json.loads(result[0].text)
        assert data["sub_agents"]["compressions"] == 1
        assert data["sub_agents"]["tokens_saved"] == 400
        assert "combined" in data

    async def test_stats_includes_proxy_summary(self):
        server = _make_server(check_proxy=True)

        mock_proxy_data = {
            "summary": {
                "mode": "token",
                "api_requests": 5,
                "primary_model": "gpt-4o",
            }
        }

        with (
            patch.object(server, "_fetch_full_proxy_stats",
                         return_value=mock_proxy_data),
            patch(
                "headroom.cache.compression_store.CompressionStore"
            ) as MockStore,
        ):
            MockStore.return_value.get_stats.return_value = {
                "entry_count": 0, "max_entries": 500,
            }
            result = await server._handle_stats()

        assert len(result) == 1
        assert "Headroom Session Summary" in result[0].text

    async def test_stats_proxy_unreachable(self):
        server = _make_server(check_proxy=True)
        with (
            patch.object(server, "_fetch_full_proxy_stats",
                         return_value=None),
            patch(
                "headroom.cache.compression_store.CompressionStore"
            ) as MockStore,
        ):
            MockStore.return_value.get_stats.return_value = {
                "entry_count": 0, "max_entries": 500,
            }
            result = await server._handle_stats()

        data = json.loads(result[0].text)
        assert "proxy" not in data


# ---------------------------------------------------------------------------
# HeadroomMCPServer -- _handle_read
# ---------------------------------------------------------------------------


class TestHandleRead:
    async def test_read_new_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mcp_server, "_READ_ENABLED", True)
        server = _make_server()

        test_file = tmp_path / "test.txt"
        test_file.write_text("line one\nline two\nline three\n")

        with patch(
            "headroom.cache.compression_store.CompressionStore"
        ) as MockStore:
            MockStore.return_value.store.return_value = "readhash"

            result = await server._handle_read({"file_path": str(test_file)})

        assert len(result) == 1
        text = result[0].text
        assert "line one" in text
        assert "1\tline one" in text

    async def test_read_cached_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mcp_server, "_READ_ENABLED", True)
        server = _make_server()

        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world\n")

        import hashlib

        from headroom.proxy.helpers import safe_decode_for_logging
        content = safe_decode_for_logging(test_file.read_bytes())
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:24]

        with patch(
            "headroom.cache.compression_store.CompressionStore"
        ) as MockStore:
            mock_store = MockStore.return_value
            mock_store.store.return_value = "readhash"
            mock_store.exists.return_value = True

            server._file_cache[str(test_file)] = (content_hash, "readhash", 1, 2)
            result = await server._handle_read({"file_path": str(test_file)})

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["status"] == "cached"
        assert data["unchanged"] is True

    async def test_read_file_not_found(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "_READ_ENABLED", True)
        server = _make_server()
        result = await server._handle_read({"file_path": "/nonexistent/path/file.txt"})
        data = json.loads(result[0].text)
        assert "error" in data

    async def test_read_missing_path(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "_READ_ENABLED", True)
        server = _make_server()
        result = await server._handle_read({})
        data = json.loads(result[0].text)
        assert "error" in data


# ---------------------------------------------------------------------------
# call_tool dispatch
# ---------------------------------------------------------------------------


class TestCallTool:
    async def test_dispatch_compress_delegates_to_handle_compress(self):
        server = _make_server()
        with patch.object(server, "_handle_compress") as mock_h:
            mock_h.return_value = [MagicMock(text='{"ok":true}')]
            await server._call_tool_handler(COMPRESS_TOOL_NAME, {"content": "x"})
        mock_h.assert_awaited_once_with({"content": "x"})

    async def test_dispatch_unknown_tool(self):
        server = _make_server()
        result = await server._call_tool_handler("unknown_tool", {})
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "Unknown tool" in data["error"]

    async def test_dispatch_raises_exception(self):
        server = _make_server()
        with patch.object(server, "_handle_compress", side_effect=ValueError("boom")):
            result = await server._call_tool_handler(COMPRESS_TOOL_NAME, {"content": "x"})
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "error" in data
        assert "boom" in data["error"]


# ---------------------------------------------------------------------------
# list_tools
# ---------------------------------------------------------------------------


class TestListTools:
    async def test_list_tools_default(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "_READ_ENABLED", False)
        server = _make_server()
        tools = await server._list_tools_handler()
        names = [t.name for t in tools]
        assert COMPRESS_TOOL_NAME in names
        assert CCR_TOOL_NAME in names
        assert STATS_TOOL_NAME in names
        assert READ_TOOL_NAME not in names

    async def test_list_tools_with_read_enabled(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "_READ_ENABLED", True)
        server = _make_server()
        tools = await server._list_tools_handler()
        names = [t.name for t in tools]
        assert READ_TOOL_NAME in names

    async def test_tool_has_correct_schema(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "_READ_ENABLED", False)
        server = _make_server()
        tools = await server._list_tools_handler()
        compress_tool = next(t for t in tools if t.name == COMPRESS_TOOL_NAME)
        assert compress_tool.inputSchema["required"] == ["content"]

        retrieve_tool = next(t for t in tools if t.name == CCR_TOOL_NAME)
        assert retrieve_tool.inputSchema["required"] == ["hash"]


# ---------------------------------------------------------------------------
# _fetch_full_proxy_stats
# ---------------------------------------------------------------------------


class TestFetchProxyStats:
    async def test_proxy_stats_success(self):
        server = _make_server(check_proxy=True)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"requests_total": 42}
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        server._http_client = mock_client

        result = await server._fetch_full_proxy_stats()
        assert result == {"requests_total": 42}

    async def test_proxy_stats_unreachable(self):
        server = _make_server(check_proxy=True)
        with patch.object(server, "_http_client") as mock_client:
            mock_client.get.side_effect = Exception("timeout")
            result = await server._fetch_full_proxy_stats()
        assert result is None

    async def test_proxy_stats_non_200(self):
        server = _make_server(check_proxy=True)
        with patch.object(server, "_http_client") as mock_client:
            mock_resp = AsyncMock()
            mock_resp.status_code = 503
            mock_client.get.return_value = mock_resp

            result = await server._fetch_full_proxy_stats()
        assert result is None


# ---------------------------------------------------------------------------
# _extract_proxy_stats
# ---------------------------------------------------------------------------


class TestExtractProxyStats:
    def test_extract_requests_total(self):
        result = HeadroomMCPServer._extract_proxy_stats({"requests_total": 10})
        assert result["requests_total"] == 10

    def test_extract_cache_hits(self):
        result = HeadroomMCPServer._extract_proxy_stats({"cache": {"hits": 5, "misses": 2}})
        assert result["cache"]["hits"] == 5

    def test_extract_cache_legacy_key(self):
        result = HeadroomMCPServer._extract_proxy_stats(
            {"caching": {"cache_hits": 3, "cache_misses": 1}}
        )
        assert result["cache"]["hits"] == 3

    def test_extract_cost(self):
        result = HeadroomMCPServer._extract_proxy_stats({"cost": {"total_saved": 0.42}})
        assert result["cost_saved_usd"] == 0.42

    def test_extract_empty(self):
        assert HeadroomMCPServer._extract_proxy_stats({}) is None


# ---------------------------------------------------------------------------
# CompressionStore integration via _get_local_store
# ---------------------------------------------------------------------------


class TestLocalStore:
    def test_get_local_store_lazy_init(self):
        server = _make_server()
        assert server._local_store is None
        store = server._get_local_store()
        assert store is not None
        assert server._local_store is store

    def test_get_local_store_reuses_instance(self):
        server = _make_server()
        s1 = server._get_local_store()
        s2 = server._get_local_store()
        assert s1 is s2


# ---------------------------------------------------------------------------
# Initialization guards
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_raises_without_mcp(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "MCP_AVAILABLE", False)
        monkeypatch.setattr(mcp_server, "Server", None)
        with pytest.raises(ImportError, match="MCP SDK not installed"):
            HeadroomMCPServer()

    def test_create_ccr_mcp_server(self):
        server = mcp_server.create_ccr_mcp_server(proxy_url="http://custom:8787")
        assert isinstance(server, HeadroomMCPServer)
        assert server.proxy_url == "http://custom:8787"

    def test_cleanup_closes_http_client(self):
        server = _make_server()
        server._http_client = AsyncMock()
        asyncio.run(server.cleanup())
        server._http_client.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# Tool names constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_tool_names(self):
        assert COMPRESS_TOOL_NAME == "headroom_compress"
        assert CCR_TOOL_NAME == "headroom_retrieve"
        assert STATS_TOOL_NAME == "headroom_stats"
        assert READ_TOOL_NAME == "headroom_read"


# ---------------------------------------------------------------------------
# Memory tool constants
# ---------------------------------------------------------------------------


class TestMemoryToolConstants:
    def test_memory_search_tool_name(self):
        from headroom.ccr.mcp_server import MEMORY_SEARCH_TOOL_NAME

        assert MEMORY_SEARCH_TOOL_NAME == "memory_search"

    def test_memory_save_tool_name(self):
        from headroom.ccr.mcp_server import MEMORY_SAVE_TOOL_NAME

        assert MEMORY_SAVE_TOOL_NAME == "memory_save"

    def test_memory_analyze_tool_name(self):
        from headroom.ccr.mcp_server import MEMORY_ANALYZE_TOOL_NAME

        assert MEMORY_ANALYZE_TOOL_NAME == "memory_analyze"

    def test_memory_delete_tool_name(self):
        from headroom.ccr.mcp_server import MEMORY_DELETE_TOOL_NAME

        assert MEMORY_DELETE_TOOL_NAME == "memory_delete"

    def test_all_constants_importable(self):
        from headroom.ccr.mcp_server import (
            MEMORY_ANALYZE_TOOL_NAME,
            MEMORY_DELETE_TOOL_NAME,
            MEMORY_SAVE_TOOL_NAME,
            MEMORY_SEARCH_TOOL_NAME,
        )

        assert MEMORY_SEARCH_TOOL_NAME
        assert MEMORY_SAVE_TOOL_NAME
        assert MEMORY_ANALYZE_TOOL_NAME
        assert MEMORY_DELETE_TOOL_NAME


# ---------------------------------------------------------------------------
# Memory tool definitions
# ---------------------------------------------------------------------------


class TestMemoryToolDefinitions:
    def test_returns_four_tools(self):
        server = _make_server()
        tools = server._get_memory_tool_definitions()
        assert len(tools) == 4

    def test_tool_names(self):
        server = _make_server()
        tools = server._get_memory_tool_definitions()
        names = [t.name for t in tools]
        assert "memory_search" in names
        assert "memory_save" in names
        assert "memory_analyze" in names
        assert "memory_delete" in names

    def test_memory_search_requires_query(self):
        server = _make_server()
        tools = server._get_memory_tool_definitions()
        search = next(t for t in tools if t.name == "memory_search")
        assert "query" in search.inputSchema["required"]

    def test_memory_save_accepts_facts_array(self):
        server = _make_server()
        tools = server._get_memory_tool_definitions()
        save = next(t for t in tools if t.name == "memory_save")
        props = save.inputSchema["properties"]
        assert "facts" in props
        assert props["facts"]["type"] == "array"

    def test_memory_delete_requires_memory_id(self):
        server = _make_server()
        tools = server._get_memory_tool_definitions()
        delete = next(t for t in tools if t.name == "memory_delete")
        assert "memory_id" in delete.inputSchema["required"]

    def test_lazy_imported_from_memory_module(self):
        from headroom.memory.mcp_server import _TOOLS as _MEMORY_TOOLS

        server = _make_server()
        tools = server._get_memory_tool_definitions()
        assert len(tools) == len(_MEMORY_TOOLS)
        for t1, t2 in zip(tools, _MEMORY_TOOLS):
            assert t1 is t2


# ---------------------------------------------------------------------------
# HeadroomMCPServer -- memory constructor params
# ---------------------------------------------------------------------------


class TestMemoryMCPServerConstructor:
    def test_default_memory_db_path_is_none(self):
        server = _make_server()
        assert server.memory_db_path is None

    def test_default_memory_user_id(self):
        server = _make_server()
        assert server.memory_user_id == "default"

    def test_custom_memory_params_stored(self):
        server = _make_server(
            memory_db_path="/tmp/test.db",
            memory_user_id="custom-user",
        )
        assert server.memory_db_path == "/tmp/test.db"
        assert server.memory_user_id == "custom-user"

    def test_memory_backend_starts_none(self):
        server = _make_server()
        assert server._memory_backend is None

    async def test_existing_functionality_not_broken(self):
        server = _make_server()
        tools = await server._list_tools_handler()
        names = [t.name for t in tools]
        assert COMPRESS_TOOL_NAME in names
        assert CCR_TOOL_NAME in names
        assert STATS_TOOL_NAME in names


# ---------------------------------------------------------------------------
# list_tools -- conditional memory tool inclusion
# ---------------------------------------------------------------------------


class TestListToolsMemoryConditional:
    async def test_no_memory_tools_when_db_path_none(self):
        server = _make_server()
        tools = await server._list_tools_handler()
        names = [t.name for t in tools]
        assert "memory_search" not in names
        assert "memory_save" not in names
        assert "memory_analyze" not in names
        assert "memory_delete" not in names

    async def test_memory_tools_included_when_db_path_set(self):
        server = _make_server(memory_db_path="/tmp/test.db")
        tools = await server._list_tools_handler()
        names = [t.name for t in tools]
        assert "memory_search" in names
        assert "memory_save" in names
        assert "memory_analyze" in names
        assert "memory_delete" in names

    async def test_existing_tools_always_present_without_memory(self):
        server = _make_server()
        tools = await server._list_tools_handler()
        names = [t.name for t in tools]
        assert COMPRESS_TOOL_NAME in names
        assert CCR_TOOL_NAME in names
        assert STATS_TOOL_NAME in names

    async def test_existing_tools_always_present_with_memory(self):
        server = _make_server(memory_db_path="/tmp/test.db")
        tools = await server._list_tools_handler()
        names = [t.name for t in tools]
        assert COMPRESS_TOOL_NAME in names
        assert CCR_TOOL_NAME in names
        assert STATS_TOOL_NAME in names


# ---------------------------------------------------------------------------
# call_tool -- memory tool dispatch
# ---------------------------------------------------------------------------


class TestCallToolMemoryDispatch:
    async def test_dispatch_memory_search(self):
        server = _make_server()
        with patch.object(server, "_handle_memory_search") as mock_h:
            mock_h.return_value = [MagicMock(text='{"ok":true}')]
            await server._call_tool_handler("memory_search", {"query": "test"})
        mock_h.assert_awaited_once_with({"query": "test"})

    async def test_dispatch_memory_save(self):
        server = _make_server()
        with patch.object(server, "_handle_memory_save") as mock_h:
            mock_h.return_value = [MagicMock(text='{"ok":true}')]
            await server._call_tool_handler("memory_save", {"facts": ["test"]})
        mock_h.assert_awaited_once_with({"facts": ["test"]})

    async def test_dispatch_memory_analyze(self):
        server = _make_server()
        with patch.object(server, "_handle_memory_analyze") as mock_h:
            mock_h.return_value = [MagicMock(text='{"ok":true}')]
            await server._call_tool_handler(
                "memory_analyze", {"messages": [], "response_text": ""}
            )
        mock_h.assert_awaited_once_with({"messages": [], "response_text": ""})

    async def test_dispatch_memory_delete(self):
        server = _make_server()
        with patch.object(server, "_handle_memory_delete") as mock_h:
            mock_h.return_value = [MagicMock(text='{"ok":true}')]
            await server._call_tool_handler(
                "memory_delete", {"memory_id": "mem-123"}
            )
        mock_h.assert_awaited_once_with({"memory_id": "mem-123"})

    async def test_memory_search_no_backend_error(self):
        server = _make_server()
        result = await server._handle_memory_search({"query": "test"})
        data = json.loads(result[0].text)
        assert data["error"] == "Memory backend not available"

    async def test_memory_save_no_backend_error(self):
        server = _make_server()
        result = await server._handle_memory_save({"facts": ["test"]})
        data = json.loads(result[0].text)
        assert data["error"] == "Memory backend not available"

    async def test_memory_analyze_no_backend_error(self):
        server = _make_server()
        result = await server._handle_memory_analyze(
            {"messages": [], "response_text": ""}
        )
        data = json.loads(result[0].text)
        assert data["error"] == "Memory backend not available"

    async def test_memory_delete_no_backend_error(self):
        server = _make_server()
        result = await server._handle_memory_delete({"memory_id": "mem-123"})
        data = json.loads(result[0].text)
        assert data["error"] == "Memory backend not available"

    async def test_unknown_tool_still_returns_error(self):
        server = _make_server()
        result = await server._call_tool_handler("nonexistent_tool", {})
        data = json.loads(result[0].text)
        assert "Unknown tool" in data["error"]
        assert "nonexistent_tool" in data["error"]
