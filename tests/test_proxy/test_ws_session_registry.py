"""Tests for headroom.proxy.ws_session_registry."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from headroom.proxy.ws_session_registry import (
    WebSocketSessionRegistry,
    WSSessionHandle,
)


def make_task_like(name: str = "task-1", done: bool = False) -> MagicMock:
    m = MagicMock()
    m.get_name.return_value = name
    m.done.return_value = done
    m.cancel.return_value = True
    return m


class TestWSSessionHandle:
    def test_creation(self) -> None:
        handle = WSSessionHandle(session_id="sid-1", request_id="rid-1")
        assert handle.session_id == "sid-1"
        assert handle.request_id == "rid-1"
        assert handle.termination_cause is None

    def test_mark_activity_updates_last_activity(self) -> None:
        handle = WSSessionHandle(session_id="sid-1", request_id="rid-1")
        old = handle.last_activity_at
        time.sleep(0.001)
        handle.mark_activity()
        assert handle.last_activity_at > old

    def test_age_seconds_returns_positive(self) -> None:
        handle = WSSessionHandle(session_id="sid-1", request_id="rid-1")
        assert handle.age_seconds() >= 0

    def test_to_snapshot_dict_keys(self) -> None:
        handle = WSSessionHandle(
            session_id="sid-1",
            request_id="rid-1",
            client_addr="127.0.0.1",
            upstream_url="wss://example.com",
        )
        d = handle.to_snapshot_dict()
        assert d["session_id"] == "sid-1"
        assert d["request_id"] == "rid-1"
        assert d["client_addr"] == "127.0.0.1"
        assert d["upstream_url"] == "wss://example.com"
        assert "age_seconds" in d
        assert "idle_seconds" in d
        assert "relay_task_count" in d
        assert "relay_task_names" in d
        assert "termination_cause" in d

    def test_relay_tasks_holds_task_like(self) -> None:
        t = make_task_like("relay-1")
        handle = WSSessionHandle(session_id="sid-1", request_id="rid-1", relay_tasks=[t])
        assert len(handle.relay_tasks) == 1
        assert handle.relay_tasks[0].get_name() == "relay-1"


class TestWebSocketSessionRegistry:
    def test_register_adds_session(self) -> None:
        registry = WebSocketSessionRegistry()
        handle = WSSessionHandle(session_id="sid-1", request_id="rid-1")
        registry.register(handle)
        assert "sid-1" in registry
        assert len(registry) == 1
        assert registry.active_count() == 1
        assert registry.active_relay_task_count() == 0

    def test_register_tracks_relay_tasks(self) -> None:
        registry = WebSocketSessionRegistry()
        t = make_task_like("t1")
        handle = WSSessionHandle(session_id="sid-1", request_id="rid-1", relay_tasks=[t])
        registry.register(handle)
        assert registry.active_relay_task_count() == 1

    def test_re_register_replaces_handle(self) -> None:
        registry = WebSocketSessionRegistry()
        t1 = make_task_like("t1")
        h1 = WSSessionHandle(session_id="sid-1", request_id="rid-1", relay_tasks=[t1])
        registry.register(h1)
        assert registry.active_relay_task_count() == 1

        t2 = make_task_like("t2")
        h2 = WSSessionHandle(session_id="sid-1", request_id="rid-1", relay_tasks=[t2])
        registry.register(h2)
        assert registry.active_count() == 1
        assert registry.active_relay_task_count() == 1
        assert registry.get("sid-1") is h2

    def test_get_returns_handle(self) -> None:
        registry = WebSocketSessionRegistry()
        handle = WSSessionHandle(session_id="sid-1", request_id="rid-1")
        registry.register(handle)
        assert registry.get("sid-1") is handle

    def test_get_returns_none_for_unknown(self) -> None:
        registry = WebSocketSessionRegistry()
        assert registry.get("nonexistent") is None

    def test_deregister_removes_session(self) -> None:
        registry = WebSocketSessionRegistry()
        handle = WSSessionHandle(session_id="sid-1", request_id="rid-1")
        registry.register(handle)
        result = registry.deregister("sid-1")
        assert result is handle
        assert "sid-1" not in registry
        assert registry.active_count() == 0

    def test_deregister_sets_termination_cause(self) -> None:
        registry = WebSocketSessionRegistry()
        handle = WSSessionHandle(session_id="sid-1", request_id="rid-1")
        registry.register(handle)
        registry.deregister("sid-1", cause="client_disconnect")
        assert handle.termination_cause == "client_disconnect"

    def test_deregister_clears_relay_tasks(self) -> None:
        registry = WebSocketSessionRegistry()
        t = make_task_like("t1")
        handle = WSSessionHandle(session_id="sid-1", request_id="rid-1", relay_tasks=[t])
        registry.register(handle)
        registry.deregister("sid-1")
        assert handle.relay_tasks == []
        assert registry.active_relay_task_count() == 0

    def test_deregister_idempotent(self) -> None:
        registry = WebSocketSessionRegistry()
        handle = WSSessionHandle(session_id="sid-1", request_id="rid-1")
        registry.register(handle)
        result1 = registry.deregister("sid-1")
        result2 = registry.deregister("sid-1")
        assert result1 is handle
        assert result2 is None
        assert "sid-1" not in registry

    def test_deregister_and_count_returns_handle_and_count(self) -> None:
        registry = WebSocketSessionRegistry()
        t = make_task_like("t1")
        handle = WSSessionHandle(session_id="sid-1", request_id="rid-1", relay_tasks=[t])
        registry.register(handle)
        result, count = registry.deregister_and_count("sid-1")
        assert result is handle
        assert count == 1
        assert handle.termination_cause is not None
        assert handle.relay_tasks == []

    def test_deregister_and_count_returns_none_zero_for_unknown(self) -> None:
        registry = WebSocketSessionRegistry()
        result, count = registry.deregister_and_count("nonexistent")
        assert result is None
        assert count == 0

    def test_attach_tasks_extends_relay_tasks(self) -> None:
        registry = WebSocketSessionRegistry()
        handle = WSSessionHandle(session_id="sid-1", request_id="rid-1")
        registry.register(handle)
        t = make_task_like("t1")
        registry.attach_tasks("sid-1", [t])
        assert len(handle.relay_tasks) == 1
        assert registry.active_relay_task_count() == 1

    def test_attach_tasks_updates_count_and_marks_activity(self) -> None:
        registry = WebSocketSessionRegistry()
        handle = WSSessionHandle(session_id="sid-1", request_id="rid-1")
        registry.register(handle)
        old_activity = handle.last_activity_at
        time.sleep(0.001)
        t1 = make_task_like("t1")
        t2 = make_task_like("t2")
        registry.attach_tasks("sid-1", [t1, t2])
        assert registry.active_relay_task_count() == 2
        assert handle.last_activity_at > old_activity

    def test_attach_tasks_unknown_noop(self) -> None:
        registry = WebSocketSessionRegistry()
        registry.attach_tasks("nonexistent", [make_task_like("t1")])
        assert registry.active_relay_task_count() == 0
        assert registry.active_count() == 0

    def test_active_count_returns_correct_count(self) -> None:
        registry = WebSocketSessionRegistry()
        registry.register(WSSessionHandle("s1", "r1"))
        registry.register(WSSessionHandle("s2", "r2"))
        assert registry.active_count() == 2

    def test_active_relay_task_count_returns_correct_count(self) -> None:
        registry = WebSocketSessionRegistry()
        h1 = WSSessionHandle(
            session_id="s1",
            request_id="r1",
            relay_tasks=[make_task_like("t1")],
        )
        h2 = WSSessionHandle(
            session_id="s2",
            request_id="r2",
            relay_tasks=[make_task_like("t2"), make_task_like("t3")],
        )
        registry.register(h1)
        registry.register(h2)
        assert registry.active_relay_task_count() == 3

    def test_snapshot_returns_list_of_dicts(self) -> None:
        registry = WebSocketSessionRegistry()
        handle = WSSessionHandle(session_id="sid-1", request_id="rid-1")
        registry.register(handle)
        snap = registry.snapshot()
        assert len(snap) == 1
        assert snap[0]["session_id"] == "sid-1"
        assert snap[0]["relay_task_count"] == 0

    def test_contains_and_len(self) -> None:
        registry = WebSocketSessionRegistry()
        registry.register(WSSessionHandle("s1", "r1"))
        assert "s1" in registry
        assert "s2" not in registry
        assert len(registry) == 1

    def test_multiple_sessions_task_counting(self) -> None:
        registry = WebSocketSessionRegistry()
        h1 = WSSessionHandle(
            session_id="s1",
            request_id="r1",
            relay_tasks=[make_task_like("t1")],
        )
        h2 = WSSessionHandle(
            session_id="s2",
            request_id="r2",
            relay_tasks=[make_task_like("t2"), make_task_like("t3")],
        )
        registry.register(h1)
        registry.register(h2)
        assert registry.active_count() == 2
        assert registry.active_relay_task_count() == 3

        registry.deregister("s1")
        assert registry.active_count() == 1
        assert registry.active_relay_task_count() == 2

        registry.deregister("s2")
        assert registry.active_count() == 0
        assert registry.active_relay_task_count() == 0
