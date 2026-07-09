from __future__ import annotations

import asyncio
import contextlib
import json

import pytest

from headroom.cache.compression_store import (
    get_compression_store,
    reset_compression_store,
)
from tests._mcp_stub import import_module_with_mcp_stub

mcp_server = import_module_with_mcp_stub("headroom.ccr.mcp_server")


def test_shared_stats_work_without_fcntl(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(mcp_server, "_HAS_FCNTL", False)
    monkeypatch.setattr(mcp_server, "fcntl", None)
    monkeypatch.setattr(mcp_server, "SHARED_STATS_DIR", tmp_path)
    monkeypatch.setattr(mcp_server, "SHARED_STATS_FILE", tmp_path / "session_stats.jsonl")
    monkeypatch.setattr(mcp_server.os, "getpid", lambda: 4242)
    monkeypatch.setattr(mcp_server.time, "time", lambda: 1001.0)

    event = {"type": "compress", "timestamp": 1000.0}
    mcp_server._append_shared_event(event)

    raw_lines = mcp_server.SHARED_STATS_FILE.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 1
    assert json.loads(raw_lines[0]) == {"type": "compress", "timestamp": 1000.0, "pid": 4242}

    events = mcp_server._read_shared_events(window_seconds=60)
    assert events == [{"type": "compress", "timestamp": 1000.0, "pid": 4242}]


# --- Shared compression store wiring ---------------------------------------
# MCP's _get_local_store() must return the get_compression_store() singleton --
# the same instance the proxy and response_handler use -- so content compressed
# on either side is retrievable in-process. These pin that wiring so a private
# store can't creep back.


@pytest.fixture
def fresh_store():
    reset_compression_store()
    yield
    reset_compression_store()


def test_mcp_uses_shared_singleton_store(fresh_store) -> None:
    """MCP's store is the global singleton, not a private instance."""
    server = mcp_server.HeadroomMCPServer(check_proxy=False)
    assert server._get_local_store() is get_compression_store()


def test_mcp_retrieves_proxy_stored_content(fresh_store) -> None:
    """Content stored via the singleton (as the proxy does) is retrievable
    through MCP's local-store path. The HTTP fallback is disabled so this
    passes only via the shared store."""
    original = '{"some": "original proxy-compressed content"}'
    hash_key = get_compression_store().store(original, '{"compressed": true}')

    server = mcp_server.HeadroomMCPServer(check_proxy=False)
    result = asyncio.run(server._retrieve_content(hash_key))

    assert result.get("source") == "local"
    assert result["original_content"] == original


def test_compress_savings_percent_tracks_token_counts(fresh_store) -> None:
    """``savings_percent`` must be the *removed* percentage derived from the
    token counts — never the retained percentage. Regression for the inversion
    where ``(1 - compression_ratio)`` reported a no-op (0% saved) as 100%."""
    pytest.importorskip("mcp", reason="MCP SDK required")
    server = mcp_server.HeadroomMCPServer(check_proxy=False)

    # Repetitive JSON array — the shape the engine actually compresses.
    content = json.dumps([{"id": i, "status": "ok", "kind": "run"} for i in range(40)])
    result = server._compress_content(content)

    orig = result["original_tokens"]
    comp = result["compressed_tokens"]
    expected = round((1 - comp / orig) * 100, 1) if orig > 0 else 0

    # Reported savings agrees with the token fields (and with tokens_saved).
    assert result["savings_percent"] == expected
    assert 0.0 <= result["savings_percent"] <= 100.0
    if result["tokens_saved"] == 0:
        assert result["savings_percent"] == 0.0  # not inverted to 100
    else:
        assert result["savings_percent"] > 0.0


def test_mcp_retrieve_returns_full_content(fresh_store) -> None:
    """Retrieval is by hash: a stored, unexpired entry always returns its full
    original content (never empty, never a spurious "not found")."""
    original = "the the the the the the the the the the\n" * 5
    hash_key = get_compression_store().store(original, "<<small>>")

    server = mcp_server.HeadroomMCPServer(check_proxy=False)
    result = asyncio.run(server._retrieve_content(hash_key))

    assert "error" not in result
    assert result.get("source") == "local"
    assert result["original_content"] == original


def test_mcp_retrieve_missing_hash_still_errors(fresh_store) -> None:
    """A genuinely missing hash must still report "Content not found"."""
    server = mcp_server.HeadroomMCPServer(check_proxy=False)
    result = asyncio.run(server._retrieve_content("nonexistent_hash"))
    assert "Content not found" in result.get("error", "")


def test_handle_stats_session_output_is_window_scoped() -> None:
    """window-scoped stats output should be explicitly labeled after this change."""

    async def fetch_stats() -> dict[str, object]:
        return {
            "summary": {
                "mode": "token",
                "api_requests": 3,
                "compression": {},
            }
        }

    server = mcp_server.HeadroomMCPServer(check_proxy=True)
    server._fetch_full_proxy_stats = fetch_stats
    response = asyncio.run(server._handle_stats())
    text = response[0].kwargs["text"]

    assert "Headroom Window-Scoped Session Summary" in text
    assert "Headroom Session Summary" not in text


def test_handle_stats_includes_lifetime_totals_from_persistent_savings() -> None:
    """Lifetime savings are appended from /stats persistent_savings.lifetime."""

    async def fetch_stats() -> dict[str, object]:
        return {
            "summary": {
                "mode": "token",
                "api_requests": 3,
                "compression": {},
            },
            "persistent_savings": {
                "lifetime": {"tokens_saved": 12345, "compression_savings_usd": 7.25}
            },
        }

    server = mcp_server.HeadroomMCPServer(check_proxy=True)
    server._fetch_full_proxy_stats = fetch_stats
    response = asyncio.run(server._handle_stats())
    text = response[0].kwargs["text"]

    assert "Lifetime Savings:" in text
    assert "Tokens saved: 12,345" in text
    assert "Compression savings: $7.25" in text


def test_handle_stats_falls_back_gracefully_without_persistent_lifetime() -> None:
    """Missing lifetime data should still return a valid session summary."""

    async def fetch_stats() -> dict[str, object]:
        return {
            "summary": {
                "mode": "token",
                "api_requests": 3,
                "compression": {},
            },
            "persistent_savings": {"lifetime": None},
        }

    server = mcp_server.HeadroomMCPServer(check_proxy=True)
    server._fetch_full_proxy_stats = fetch_stats
    response = asyncio.run(server._handle_stats())
    text = response[0].kwargs["text"]

    assert "Headroom Window-Scoped Session Summary" in text
    assert "Lifetime Savings:" not in text


def test_handle_stats_shows_zero_lifetime_totals_when_present() -> None:
    """A present lifetime payload should still render explicit zero totals."""

    async def fetch_stats() -> dict[str, object]:
        return {
            "summary": {
                "mode": "token",
                "api_requests": 3,
                "compression": {},
            },
            "persistent_savings": {"lifetime": {"tokens_saved": 0, "compression_savings_usd": 0.0}},
        }

    server = mcp_server.HeadroomMCPServer(check_proxy=True)
    server._fetch_full_proxy_stats = fetch_stats
    response = asyncio.run(server._handle_stats())
    text = response[0].kwargs["text"]

    assert "Lifetime Savings:" in text
    assert "Tokens saved: 0" in text
    assert "Compression savings: $0.00" in text


# --- Parent-death watchdog: reap orphaned `mcp serve` on client death --------
# When the launching MCP client is SIGKILLed, stdin EOF may never arrive and the
# SDK's blocking stdin reader wedges server.run() forever, orphaning this process
# under init/launchd. run_stdio() runs a watchdog that detects the reparent and
# forces shutdown. Refs headroomlabs-ai/headroom#2185 (secondary), #1761.


def test_parent_death_watchdog_fires_when_reparented(monkeypatch) -> None:
    """When ppid changes (client died), the watchdog resolves promptly."""
    server = mcp_server.HeadroomMCPServer(check_proxy=False)
    calls = {"n": 0}

    def fake_getppid() -> int:
        calls["n"] += 1
        return 500 if calls["n"] == 1 else 1  # captured live, then reparented

    monkeypatch.setattr(mcp_server.os, "getppid", fake_getppid)

    async def run() -> None:
        await asyncio.wait_for(server._await_parent_death(0.001), timeout=1.0)

    asyncio.run(run())  # returns => detected reparent; TimeoutError would fail


def test_parent_death_watchdog_stays_quiet_with_live_parent(monkeypatch) -> None:
    """A stable ppid must never trip the watchdog."""
    server = mcp_server.HeadroomMCPServer(check_proxy=False)
    monkeypatch.setattr(mcp_server.os, "getppid", lambda: 500)

    async def run() -> None:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(server._await_parent_death(0.001), timeout=0.05)

    asyncio.run(run())


def test_run_stdio_reaps_process_on_parent_death(monkeypatch) -> None:
    """On reparent, run_stdio cleans up and calls os._exit(0) even though the
    (stubbed) server.run never returns — the orphan-reaper path."""
    server = mcp_server.HeadroomMCPServer(check_proxy=False)

    @contextlib.asynccontextmanager
    async def fake_stdio_server():
        yield (object(), object())

    monkeypatch.setattr(mcp_server, "stdio_server", fake_stdio_server)

    async def never_returns(*_args, **_kwargs) -> None:
        await asyncio.sleep(3600)  # emulate the wedged SDK reader

    # DummyServer (MCP SDK stub) has no `.run`; raising=False lets us add it.
    monkeypatch.setattr(server.server, "run", never_returns, raising=False)

    calls = {"n": 0}

    def fake_getppid() -> int:
        calls["n"] += 1
        return 500 if calls["n"] == 1 else 1

    monkeypatch.setattr(mcp_server.os, "getppid", fake_getppid)

    cleaned = {"done": False}

    async def fake_cleanup() -> None:
        cleaned["done"] = True

    monkeypatch.setattr(server, "cleanup", fake_cleanup)

    class _Exited(Exception):
        pass

    def fake_exit(code: int) -> None:
        raise _Exited(code)  # intercept so pytest survives

    monkeypatch.setattr(mcp_server.os, "_exit", fake_exit)

    with pytest.raises(_Exited) as excinfo:
        asyncio.run(server.run_stdio(parent_death_poll_interval=0.001))

    assert excinfo.value.args[0] == 0
    assert cleaned["done"] is True
