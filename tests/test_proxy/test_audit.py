"""Tests for headroom.proxy.audit — lightweight audit log."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from headroom.proxy.audit import (
    _client_ip,
    is_auditable_path,
    record_admin_action,
)

# ── is_auditable_path ─────────────────────────────────────────────────


class TestIsAuditablePath:
    def test_admin_prefix(self) -> None:
        assert is_auditable_path("/admin/config") is True

    def test_admin_prefix_root(self) -> None:
        assert is_auditable_path("/admin/") is True

    def test_admin_prefix_deep(self) -> None:
        assert is_auditable_path("/admin/runtime/keys") is True

    def test_cache_clear(self) -> None:
        assert is_auditable_path("/cache/clear") is True

    def test_stats_reset(self) -> None:
        assert is_auditable_path("/stats/reset") is True

    def test_regular_path(self) -> None:
        assert is_auditable_path("/v1/messages") is False

    def test_admin_without_slash(self) -> None:
        assert is_auditable_path("/admin") is False

    def test_similar_not_exact(self) -> None:
        assert is_auditable_path("/cache/clear/extra") is False

    def test_empty_string(self) -> None:
        assert is_auditable_path("") is False


# ── _client_ip ────────────────────────────────────────────────────────


class TestClientIp:
    def test_normal_request(self) -> None:
        req = SimpleNamespace(client=SimpleNamespace(host="1.2.3.4"))
        assert _client_ip(req) == "1.2.3.4"

    def test_no_client(self) -> None:
        req = SimpleNamespace(client=None)
        assert _client_ip(req) is None

    def test_no_client_attr(self) -> None:
        req = SimpleNamespace()
        assert _client_ip(req) is None

    def test_no_host_attr(self) -> None:
        req = SimpleNamespace(client=SimpleNamespace())
        assert _client_ip(req) is None


# ── record_admin_action ───────────────────────────────────────────────


class TestRecordAdminAction:
    def _make_request(
        self,
        *,
        method: str = "POST",
        path: str = "/admin/config",
        client_host: str = "10.0.0.1",
    ) -> SimpleNamespace:
        return SimpleNamespace(
            method=method,
            url=SimpleNamespace(path=path),
            client=SimpleNamespace(host=client_host),
        )

    def test_emits_json_to_audit_logger(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.DEBUG, logger="headroom.audit")
        req = self._make_request()
        record_admin_action(request=req, action="toggle_logging", status_code=200)
        assert len(caplog.records) == 1
        rec = caplog.records[0]
        assert rec.name == "headroom.audit"
        assert rec.levelno == logging.INFO
        payload = json.loads(rec.getMessage())
        assert payload["event"] == "headroom_admin_audit"
        assert payload["action"] == "toggle_logging"
        assert payload["method"] == "POST"
        assert payload["path"] == "/admin/config"
        assert payload["source_ip"] == "10.0.0.1"
        assert payload["status_code"] == 200

    def test_with_details(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.DEBUG, logger="headroom.audit")
        req = self._make_request()
        record_admin_action(
            request=req,
            action="update_model",
            status_code=200,
            details={"model": "claude-3"},
        )
        payload = json.loads(caplog.records[0].getMessage())
        assert payload["details"] == {"model": "claude-3"}

    def test_without_details(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.DEBUG, logger="headroom.audit")
        req = self._make_request()
        record_admin_action(request=req, action="clear_cache", status_code=200)
        payload = json.loads(caplog.records[0].getMessage())
        assert "details" not in payload

    def test_none_details_omitted(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.DEBUG, logger="headroom.audit")
        req = self._make_request()
        record_admin_action(request=req, action="noop", status_code=200, details=None)
        payload = json.loads(caplog.records[0].getMessage())
        assert "details" not in payload

    def test_minimal_request_no_crash(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.DEBUG, logger="headroom.audit")
        record_admin_action(request=object(), action="x", status_code=500)
        payload = json.loads(caplog.records[0].getMessage())
        assert payload["method"] is None
        assert payload["path"] is None
        assert payload["source_ip"] is None

    def test_exception_does_not_propagate(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING, logger="headroom.audit")

        class _BadRequest:
            method = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))

        record_admin_action(request=_BadRequest(), action="fail", status_code=200)
        assert any("audit event emission failed" in r.message for r in caplog.records)
        assert caplog.records[-1].levelno == logging.WARNING
