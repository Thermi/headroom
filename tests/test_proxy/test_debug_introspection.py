"""Tests for headroom.proxy.debug_introspection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from headroom.proxy.debug_introspection import collect_tasks


def make_mock_task(
    name: str | None,
    qualname: str | None,
    *,
    done: bool = False,
) -> MagicMock:
    m = MagicMock()
    m.get_name.return_value = name
    coro = MagicMock()
    if qualname is not None:
        coro.cr_code.co_qualname = qualname
        coro.cr_code.co_name = qualname.split(".")[-1] if "." in qualname else qualname
    else:
        coro.cr_code = None
    m.get_coro.return_value = coro
    m.done.return_value = done
    return m


class TestCollectTasks:
    def test_no_running_loop_returns_empty(self) -> None:
        with patch("asyncio.all_tasks", side_effect=RuntimeError("no running loop")):
            result = collect_tasks()
        assert result == []

    def test_with_ws_registry_attaches_ages(self) -> None:
        registry = MagicMock()
        handle = MagicMock()
        handle.started_at = 100.0
        registry.get.return_value = handle

        tasks = [make_mock_task("codex-ws-c2u-sid1", "some_func")]
        with (
            patch("asyncio.all_tasks", return_value=tasks),
            patch("headroom.proxy.debug_introspection.time.perf_counter", return_value=200.0),
        ):
            result = collect_tasks(ws_registry=registry)
        assert len(result) == 1
        assert result[0]["name"] == "codex-ws-c2u-sid1"
        assert result[0]["age_seconds"] == 100.0

    def test_with_u2c_prefix(self) -> None:
        registry = MagicMock()
        handle = MagicMock()
        handle.started_at = 50.0
        registry.get.return_value = handle

        tasks = [make_mock_task("codex-ws-u2c-sid2", "other_func")]
        with (
            patch("asyncio.all_tasks", return_value=tasks),
            patch("headroom.proxy.debug_introspection.time.perf_counter", return_value=100.0),
        ):
            result = collect_tasks(ws_registry=registry)
        assert len(result) == 1
        assert result[0]["age_seconds"] == 50.0

    def test_filters_none_tasks(self) -> None:
        good = make_mock_task("task-1", "qual")
        tasks: list = [None, good]
        with patch("asyncio.all_tasks", return_value=tasks):
            result = collect_tasks()
        assert len(result) == 1
        assert result[0]["name"] == "task-1"

    def test_keeps_task_with_name_but_no_qualname(self) -> None:
        m = MagicMock()
        m.get_name.return_value = "nameless-coro"
        m.get_coro.return_value = None
        m.done.return_value = False
        with patch("asyncio.all_tasks", return_value=[m]):
            result = collect_tasks()
        assert len(result) == 1
        assert result[0]["name"] == "nameless-coro"
        assert result[0]["coro_qualname"] is None

    def test_filters_task_with_no_identity(self) -> None:
        m = MagicMock()
        m.get_name.return_value = None
        m.get_coro.return_value = None
        m.done.return_value = False
        with patch("asyncio.all_tasks", return_value=[m]):
            result = collect_tasks()
        assert len(result) == 0

    def test_filters_task_with_get_name_exception(self) -> None:
        m = MagicMock()
        m.get_name.side_effect = Exception("no name")
        coro = MagicMock()
        coro.cr_code.co_qualname = "some_func"
        m.get_coro.return_value = coro
        m.done.return_value = False
        with patch("asyncio.all_tasks", return_value=[m]):
            result = collect_tasks()
        assert len(result) == 1
        assert result[0]["name"] is None
        assert result[0]["coro_qualname"] == "some_func"

    def test_sorting_by_age_descending(self) -> None:
        registry = MagicMock()

        def get_handle(sid: str) -> MagicMock | None:
            h = MagicMock()
            if sid == "sid-old":
                h.started_at = 50.0
            elif sid == "sid-new":
                h.started_at = 100.0
            else:
                return None
            return h

        registry.get.side_effect = get_handle

        tasks = [
            make_mock_task("codex-ws-c2u-sid-old", "old_fn"),
            make_mock_task("codex-ws-c2u-sid-new", "new_fn"),
        ]
        with (
            patch("asyncio.all_tasks", return_value=tasks),
            patch("headroom.proxy.debug_introspection.time.perf_counter", return_value=200.0),
        ):
            result = collect_tasks(ws_registry=registry)
        assert len(result) == 2
        assert result[0]["name"] == "codex-ws-c2u-sid-old"
        assert result[0]["age_seconds"] == 150.0
        assert result[1]["name"] == "codex-ws-c2u-sid-new"
        assert result[1]["age_seconds"] == 100.0

    def test_none_ages_sort_last(self) -> None:
        registry = MagicMock()
        # Only resolve one named task; the other has no prefix match
        handle = MagicMock()
        handle.started_at = 100.0
        registry.get.side_effect = lambda sid: handle if sid == "sid-known" else None

        tasks = [
            make_mock_task("non_ws_task", "some_fn"),
            make_mock_task("codex-ws-c2u-sid-known", "known_fn"),
        ]
        with (
            patch("asyncio.all_tasks", return_value=tasks),
            patch("headroom.proxy.debug_introspection.time.perf_counter", return_value=200.0),
        ):
            result = collect_tasks(ws_registry=registry)
        assert len(result) == 2
        # Known task (age=100) first, unknown task (None) last
        assert result[0]["age_seconds"] == 100.0
        assert result[0]["name"] == "codex-ws-c2u-sid-known"
        assert result[1]["age_seconds"] is None
        assert result[1]["name"] == "non_ws_task"

    def test_with_stack_depth_true(self) -> None:
        tasks = [make_mock_task("task-1", "fn")]
        with patch("asyncio.all_tasks", return_value=tasks):
            result = collect_tasks(with_stack_depth=True)
        assert len(result) == 1
        assert "stack_depth" in result[0]
        assert result[0]["stack_depth"] is not None

    def test_done_flag(self) -> None:
        done_task = make_mock_task("done", "d_fn", done=True)
        pending_task = make_mock_task("pending", "p_fn", done=False)
        with patch("asyncio.all_tasks", return_value=[done_task, pending_task]):
            result = collect_tasks()
        done_entries = [e for e in result if e["name"] == "done"]
        pending_entries = [e for e in result if e["name"] == "pending"]
        assert len(done_entries) == 1
        assert len(pending_entries) == 1
        assert done_entries[0]["done"] is True
        assert pending_entries[0]["done"] is False
