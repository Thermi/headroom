"""Shared fixtures for proxy tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app


@pytest.fixture
def proxy_config() -> ProxyConfig:
    """Return a minimal ProxyConfig suitable for unit tests."""
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


@pytest.fixture
async def async_client(proxy_config: ProxyConfig) -> AsyncIterator[AsyncClient]:
    """Create an async ASGI test client that avoids the TestClient hang on Python 3.14.

    Uses httpx.AsyncClient with ASGITransport instead of starlette.testclient.TestClient.
    """
    app = create_app(proxy_config)
    transport = ASGITransport(app=app, client=("127.0.0.1", 12345))
    async with AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        yield client


@pytest.fixture
def tmp_memory_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for native memory file ops tests."""
    d = tmp_path / "memory"
    d.mkdir()
    return d
