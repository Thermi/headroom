"""Regression test: /v1/mcp must be registered before the provider catch-all.

Starlette matches routes in registration order. The provider router
(``register_provider_routes``) installs a ``/{path:path}`` catch-all that
tunnels unknown paths upstream. If the MCP endpoint is registered after
that catch-all, a request to /v1/mcp is swallowed by the catch-all and
forwarded upstream (which returns an HTML 404 instead of an MCP stream).

See headroom/proxy/server.py create_app().
"""

from __future__ import annotations

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
