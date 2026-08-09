"""Regression test: /v1/mcp must be registered before the provider catch-all.

Starlette matches routes in registration order. The provider router
(``register_provider_routes``) installs a ``/{path:path}`` catch-all that
tunnels unknown paths upstream. If the MCP endpoint is registered after
that catch-all, a request to /v1/mcp is swallowed by the catch-all and
forwarded upstream (which returns an HTML 404 instead of an MCP stream).

See headroom/proxy/server.py create_app().
"""

from __future__ import annotations

import pytest

from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app


def _minimal_config() -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        subscription_tracking_enabled=False,
        ds4_subscription_enabled=False,
        license_key=None,
        periodic_toin_stats_enabled=False,
    )


def test_v1_mcp_registered_before_catchall():
    """The /v1/mcp route must precede the /{path:path} passthrough route."""
    app = create_app(_minimal_config())

    mcp_index = None
    catchall_index = None
    for i, route in enumerate(app.routes):
        path = getattr(route, "path", None)
        if path == "/v1/mcp" and mcp_index is None:
            mcp_index = i
        if path == "/{path:path}":
            catchall_index = i

    assert mcp_index is not None, "/v1/mcp route was not registered"
    assert catchall_index is not None, "/{path:path} catch-all was not registered"
    assert mcp_index < catchall_index, (
        "/v1/mcp must be registered before the /{path:path} catch-all, "
        "otherwise it is tunneled upstream as an HTML 404"
    )


def test_mcp_banner_line_reports_available_and_unavailable_states():
    from headroom.cli.proxy import _mcp_endpoint_banner_line

    assert "/v1/mcp" in _mcp_endpoint_banner_line(True)
    assert "Streamable HTTP MCP tools" in _mcp_endpoint_banner_line(True)
    assert "/v1/mcp" in _mcp_endpoint_banner_line(False)
    assert "unavailable" in _mcp_endpoint_banner_line(False)


@pytest.mark.asyncio
async def test_v1_mcp_initializes_and_lists_builtin_tools():
    pytest.importorskip("mcp.server.streamable_http")
    import httpx
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    app = create_app(_minimal_config())
    transport = httpx.ASGITransport(app=app)

    for handler in app.router.on_startup:
        await handler()
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            async with streamable_http_client(
                "http://testserver/v1/mcp",
                http_client=client,
                terminate_on_close=False,
            ) as (read_stream, write_stream, _get_session_id):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools = await session.list_tools()
    finally:
        for handler in reversed(app.router.on_shutdown):
            await handler()

    assert {tool.name for tool in tools.tools} >= {
        "headroom_compress",
        "headroom_retrieve",
        "headroom_stats",
    }
