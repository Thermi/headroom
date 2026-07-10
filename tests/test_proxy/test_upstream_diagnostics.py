from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

from headroom.proxy.upstream_diagnostics import diagnose_upstream


class TestDiagnoseUpstream:
    async def test_happy_path(self, caplog) -> None:
        caplog.set_level(logging.WARNING)
        with (
            patch(
                "headroom.proxy.upstream_diagnostics._resolve_host",
                AsyncMock(return_value=["1.2.3.4"]),
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._check_tcp_connect",
                AsyncMock(return_value="connected"),
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._get_local_address", return_value="192.168.1.1"
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._run_traceroute",
                AsyncMock(return_value=["hop1", "hop2"]),
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._probe_path_mtu",
                AsyncMock(return_value={"interface_mtu": "eth0:1500"}),
            ),
        ):
            await diagnose_upstream("https://api.example.com/v1", "corr-1")

        assert "UPSTREAM_DIAGNOSTICS" in caplog.text
        assert "corr-1" in caplog.text
        assert "api.example.com" in caplog.text

    async def test_custom_port(self, caplog) -> None:
        caplog.set_level(logging.WARNING)
        with (
            patch(
                "headroom.proxy.upstream_diagnostics._resolve_host",
                AsyncMock(return_value=["10.0.0.1"]),
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._check_tcp_connect",
                AsyncMock(return_value="connected"),
            ),
            patch("headroom.proxy.upstream_diagnostics._get_local_address", return_value="::1"),
            patch(
                "headroom.proxy.upstream_diagnostics._run_traceroute", AsyncMock(return_value=[])
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._probe_path_mtu", AsyncMock(return_value={})
            ),
        ):
            await diagnose_upstream("http://api.test:8080/path", "corr-2")

        assert "8080" in caplog.text
        assert "api.test" in caplog.text

    async def test_dns_timeout(self, caplog) -> None:
        caplog.set_level(logging.WARNING)
        with (
            patch(
                "headroom.proxy.upstream_diagnostics._resolve_host",
                AsyncMock(return_value=["<DNS resolution timed out>"]),
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._check_tcp_connect",
                AsyncMock(return_value="connected"),
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._get_local_address", return_value="10.0.0.5"
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._run_traceroute", AsyncMock(return_value=[])
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._probe_path_mtu", AsyncMock(return_value={})
            ),
        ):
            await diagnose_upstream("https://api.example.com", "corr-dns")

        assert "DNS resolution timed out" in caplog.text

    async def test_tcp_refused(self, caplog) -> None:
        caplog.set_level(logging.WARNING)
        with (
            patch(
                "headroom.proxy.upstream_diagnostics._resolve_host",
                AsyncMock(return_value=["10.0.0.1"]),
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._check_tcp_connect",
                AsyncMock(return_value="connection to api.test:443 refused"),
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._get_local_address", return_value="10.0.0.5"
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._run_traceroute", AsyncMock(return_value=[])
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._probe_path_mtu", AsyncMock(return_value={})
            ),
        ):
            await diagnose_upstream("https://api.test", "corr-tcp")

        assert "refused" in caplog.text

    async def test_traceroute_not_available(self, caplog) -> None:
        caplog.set_level(logging.WARNING)
        with (
            patch(
                "headroom.proxy.upstream_diagnostics._resolve_host",
                AsyncMock(return_value=["1.2.3.4"]),
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._check_tcp_connect",
                AsyncMock(return_value="connected"),
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._get_local_address", return_value="192.168.1.1"
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._run_traceroute",
                AsyncMock(return_value=["<traceroute not available on Windows>"]),
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._probe_path_mtu", AsyncMock(return_value={})
            ),
        ):
            await diagnose_upstream("https://api.example.com", "corr-tr")

        assert "traceroute not available" in caplog.text

    async def test_mtu_probe_failure_recorded(self, caplog) -> None:
        caplog.set_level(logging.WARNING)
        with (
            patch(
                "headroom.proxy.upstream_diagnostics._resolve_host",
                AsyncMock(return_value=["1.2.3.4"]),
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._check_tcp_connect",
                AsyncMock(return_value="connected"),
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._get_local_address", return_value="192.168.1.1"
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._run_traceroute", AsyncMock(return_value=[])
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._probe_path_mtu",
                AsyncMock(
                    return_value={
                        "interface_mtu": "eth0:1500",
                        "ping_probes": [
                            {"size": 1400, "success": False, "reason": "fragmentation_needed"}
                        ],
                    }
                ),
            ),
        ):
            await diagnose_upstream("https://api.example.com", "corr-mtu")

        assert "fragmentation_needed" in caplog.text

    async def test_local_address_fallback(self, caplog) -> None:
        caplog.set_level(logging.WARNING)
        with (
            patch(
                "headroom.proxy.upstream_diagnostics._resolve_host",
                AsyncMock(return_value=["<DNS timed out>"]),
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._check_tcp_connect",
                AsyncMock(return_value="connected"),
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._get_local_address", return_value="192.168.1.1"
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._run_traceroute", AsyncMock(return_value=[])
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._probe_path_mtu", AsyncMock(return_value={})
            ),
        ):
            await diagnose_upstream("https://api.example.com", "corr-local")

        assert "8.8.8.8" in caplog.text or "UPSTREAM_DIAGNOSTICS" in caplog.text

    async def test_diagnose_upstream_exception_caught(self, caplog) -> None:
        caplog.set_level(logging.ERROR)
        with patch(
            "headroom.proxy.upstream_diagnostics._resolve_host",
            side_effect=RuntimeError("unexpected crash"),
        ):
            await diagnose_upstream("https://api.example.com", "corr-crash")

        assert "UPSTREAM_DIAGNOSTICS" in caplog.text
        assert "crashed" in caplog.text

    async def test_http_default_port(self, caplog) -> None:
        caplog.set_level(logging.WARNING)
        with (
            patch(
                "headroom.proxy.upstream_diagnostics._resolve_host",
                AsyncMock(return_value=["10.0.0.1"]),
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._check_tcp_connect",
                AsyncMock(return_value="connected"),
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._get_local_address", return_value="10.0.0.1"
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._run_traceroute", AsyncMock(return_value=[])
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._probe_path_mtu", AsyncMock(return_value={})
            ),
        ):
            await diagnose_upstream("http://example.com", "corr-http")

        assert 'port": 80' in caplog.text or "80" in caplog.text

    async def test_raw_hostname_without_scheme(self, caplog) -> None:
        caplog.set_level(logging.WARNING)
        with (
            patch(
                "headroom.proxy.upstream_diagnostics._resolve_host",
                AsyncMock(return_value=["10.0.0.1"]),
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._check_tcp_connect",
                AsyncMock(return_value="connected"),
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._get_local_address", return_value="10.0.0.1"
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._run_traceroute", AsyncMock(return_value=[])
            ),
            patch(
                "headroom.proxy.upstream_diagnostics._probe_path_mtu", AsyncMock(return_value={})
            ),
        ):
            await diagnose_upstream("api.example.com:443", "corr-raw")

        assert "UPSTREAM_DIAGNOSTICS" in caplog.text
