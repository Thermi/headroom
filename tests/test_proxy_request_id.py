from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from headroom.proxy.server import (
    HeadroomProxy,
    _REQUEST_ID_VAR,
)

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def _core_available() -> bool:
    try:
        from headroom._core import hello as _rust_hello

        _rust_hello()
        return True
    except Exception:
        return False


# Integration tests requiring the full proxy app (create_app) need the
# Rust extension.  Skip the whole class when it's not loadable.
_skip_integration = pytest.mark.skipif(
    not _core_available(),
    reason="headroom._core Rust extension not available; skipping integration tests",
)


class TestRequestIdContextVar:
    """Unit tests for the _REQUEST_ID_VAR context var consumed by _next_request_id."""

    @pytest.mark.asyncio
    async def test_next_request_id_consumes_context_var(self):
        proxy = object.__new__(HeadroomProxy)
        proxy._request_counter = 0
        proxy._request_counter_lock = __import__("asyncio").Lock()

        _REQUEST_ID_VAR.set("test-request-123")
        rid = await proxy._next_request_id()
        assert rid == "test-request-123"

        _REQUEST_ID_VAR.set(None)
        rid2 = await proxy._next_request_id()
        assert rid2.startswith("hr_")

    @pytest.mark.asyncio
    async def test_next_request_id_generates_new_id_when_no_context_var(self):
        proxy = object.__new__(HeadroomProxy)
        proxy._request_counter = 0
        proxy._request_counter_lock = __import__("asyncio").Lock()

        _REQUEST_ID_VAR.set(None)
        rid = await proxy._next_request_id()
        assert rid.startswith("hr_")
        assert "_000001" in rid

    @pytest.mark.asyncio
    async def test_next_request_id_increments_counter_after_consumption(self):
        proxy = object.__new__(HeadroomProxy)
        proxy._request_counter = 0
        proxy._request_counter_lock = __import__("asyncio").Lock()

        _REQUEST_ID_VAR.set("first-consumed")
        rid = await proxy._next_request_id()
        assert rid == "first-consumed"

        _REQUEST_ID_VAR.set(None)
        rid2 = await proxy._next_request_id()
        assert rid2.endswith("_000001")

        _REQUEST_ID_VAR.set(None)
        rid3 = await proxy._next_request_id()
        assert rid3.endswith("_000002")


def test_ojson_is_importable():
    """Verify the ojson dependency is installed."""
    ojson = pytest.importorskip("ojson", reason="ojson not installed (optional dependency)")
    assert hasattr(ojson, "__version__")


@_skip_integration
class TestMiddlewareRequestId:
    """Integration tests for request_id threading through the ASGI middleware."""

    @pytest.fixture
    def client(self):
        from headroom.proxy.server import ProxyConfig, create_app

        config = ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
        )
        app = create_app(config)
        with TestClient(app) as c:
            yield c

    def test_middleware_sets_scope_request_id(self, client, caplog):
        _REQUEST_ID_VAR.set(None)
        with caplog.at_level(logging.INFO):
            response = client.get("/livez")
        assert response.status_code == 200

        inbound_log = None
        for record in caplog.records:
            if "event=proxy_inbound_request" in record.getMessage():
                inbound_log = record.getMessage()
                break

        assert inbound_log is not None, "proxy_inbound_request log not found"
        assert "request_id=hr_" in inbound_log, (
            f"Expected request_id=hr_ in log, got: {inbound_log}"
        )

    def test_all_middleware_logs_share_same_request_id(self, client, caplog):
        _REQUEST_ID_VAR.set(None)
        with caplog.at_level(logging.INFO):
            response = client.get("/livez")
        assert response.status_code == 200

        request_ids = set()
        for record in caplog.records:
            msg = record.getMessage()
            if "event=proxy_inbound_" in msg or "event=proxy_inbound_response" in msg:
                for part in msg.split():
                    if part.startswith("request_id="):
                        request_ids.add(part.split("=", 1)[1])

        assert len(request_ids) == 1, (
            f"Expected all middleware logs to share one request_id, got: {request_ids}"
        )

    def test_handler_and_middleware_share_request_id(self, client, caplog):
        _REQUEST_ID_VAR.set(None)
        with caplog.at_level(logging.INFO):
            response = client.get("/livez")
        assert response.status_code == 200

        middleware_id = None
        for record in caplog.records:
            msg = record.getMessage()
            if "event=proxy_inbound_request" in msg:
                for part in msg.split():
                    if part.startswith("request_id="):
                        middleware_id = part.split("=", 1)[1]

        status_id = None
        for record in caplog.records:
            msg = record.getMessage()
            if "event=proxy_inbound_response" in msg:
                for part in msg.split():
                    if part.startswith("request_id="):
                        status_id = part.split("=", 1)[1]

        assert middleware_id is not None, "no middleware request_id found"
        assert status_id is not None, "no response request_id found"
        assert middleware_id == status_id, (
            f"request_id mismatch: middleware={middleware_id} response={status_id}"
        )

    def test_concurrent_requests_get_different_ids(self, client):
        _REQUEST_ID_VAR.set(None)
        ids = []
        for _ in range(5):
            with client.get("/livez") as response:
                proxy = response.request.app.state.proxy
                ids.append(proxy._request_counter)

        assert len(set(ids)) == 5, (
            f"Expected 5 unique request_ids, got: {ids}"
        )
