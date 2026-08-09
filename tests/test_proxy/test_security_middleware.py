"""Regression tests for the proxy security middleware task boundary."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware

from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app


def test_security_gate_is_not_base_http_middleware():
    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            image_optimize=False,
            subscription_tracking_enabled=False,
            ds4_subscription_enabled=False,
            license_key=None,
            periodic_toin_stats_enabled=False,
        )
    )

    assert not any(
        isinstance(middleware.cls, type) and issubclass(middleware.cls, BaseHTTPMiddleware)
        for middleware in app.user_middleware
    )
