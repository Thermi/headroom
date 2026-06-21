"""Automatic network diagnostics when upstream connections fail.

Spawned as a background asyncio task when the proxy exhausts its retry
budget on a connection or timeout error.  Runs a quick diagnostic
battery (DNS, TCP, traceroute, path MTU probing, local address) so
operators can diagnose routing, firewall, MTU, or DNS issues without
reproducing the failure.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import shutil
import socket
from urllib.parse import urlparse

from headroom.proxy import _json as json

logger = logging.getLogger("headroom.proxy.diagnostics")

_STEP_TIMEOUT: float = 10.0


async def _resolve_host(host: str) -> list[str]:
    try:
        infos = await asyncio.wait_for(
            asyncio.get_event_loop().getaddrinfo(host, None),
            timeout=_STEP_TIMEOUT,
        )
        return sorted({info[4][0] for info in infos})
    except asyncio.TimeoutError:
        return ["<DNS resolution timed out>"]
    except Exception as exc:
        return [f"<DNS resolution failed: {exc}>"]


async def _check_tcp_connect(host: str, port: int) -> str:
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=_STEP_TIMEOUT,
        )
        writer.close()
        await writer.wait_closed()
        return f"connected to {host}:{port}"
    except asyncio.TimeoutError:
        return f"connection to {host}:{port} timed out"
    except ConnectionRefusedError:
        return f"connection to {host}:{port} refused"
    except OSError as exc:
        return f"connection to {host}:{port} failed: {exc}"


def _get_local_address(host: str) -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(5.0)
        s.connect((host, 80))
        addr: str = s.getsockname()[0]
        s.close()
        return addr
    except Exception as exc:
        return f"<could not determine: {exc}>"


def _traceroute_binary() -> str | None:
    system = platform.system().lower()
    if system == "windows":
        return "tracert"
    if system in ("linux", "darwin"):
        return "traceroute"
    return None


async def _run_traceroute(host: str) -> list[str]:
    binary = _traceroute_binary()
    if binary is None:
        return [f"<traceroute not available on {platform.system()}>"]

    try:
        proc = await asyncio.create_subprocess_exec(
            binary,
            "-n",
            "-w",
            "2",
            "-m",
            "15",
            host,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            proc.kill()
            return ["<traceroute timed out>"]

        if proc.returncode != 0:
            return [f"<traceroute exited with code {proc.returncode}>"]
        lines = stdout.decode(errors="replace").strip().splitlines()
        return lines[:30]
    except FileNotFoundError:
        return [f"<{binary} not found on PATH>"]
    except Exception as exc:
        return [f"<traceroute failed: {exc}>"]


def _ping_mtu_command(host: str, size: int) -> list[str] | None:
    """Build a ping command that probes with a given payload size."""
    system = platform.system().lower()
    if system == "windows":
        # -f = don't fragment, -l = payload size, -n 1 = one probe
        return ["ping", "-f", "-l", str(size), "-n", "1", host]
    if system == "linux":
        # -M do = DF bit set, -s = payload size, -c 1 = one probe
        return ["ping", "-M", "do", "-s", str(size), "-c", "1", "-W", "3", host]
    if system == "darwin":
        # -D = DF bit, -s = payload size, -c 1 = one probe
        return ["ping", "-D", "-s", str(size), "-c", "1", "-t", "3", host]
    return None


async def _probe_path_mtu(host: str) -> dict[str, object]:
    """Probe path MTU by sending pings of increasing size with DF set.

    Returns a dict with the interface MTU, probe results at various
    sizes, and the inferred path MTU where fragmentation begins.
    """
    system = platform.system().lower()
    results: dict[str, object] = {}

    # 1. Interface MTU
    results["interface_mtu"] = _detect_interface_mtu()

    # 2. Ping probes with increasing sizes
    probe_sizes = [100, 500, 1000, 1200, 1300, 1400, 1472]
    if system != "linux":
        probe_sizes = [100, 500, 1000, 1200, 1300, 1400]

    probe_results: list[dict[str, object]] = []
    first_fail: int | None = None
    for size in probe_sizes:
        cmd = _ping_mtu_command(host, size)
        if cmd is None:
            probe_results.append({"size": size, "result": f"<ping not available on {system}>"})
            continue
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=_STEP_TIMEOUT,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_STEP_TIMEOUT)
            success = proc.returncode == 0
            output = stdout.decode(errors="replace").strip()
            entry: dict[str, object] = {"size": size, "success": success}
            if not success:
                err = stderr.decode(errors="replace").strip() or output
                # On Linux "Frag needed" means DF was enforced
                if "frag" in err.lower() or "mtu" in err.lower():
                    entry["reason"] = "fragmentation_needed"
                elif "timed out" in err.lower() or "timeout" in err.lower():
                    entry["reason"] = "timeout"
                else:
                    entry["reason"] = err[:200] if err else "unknown"
                if first_fail is None:
                    first_fail = size
            probe_results.append(entry)
        except (asyncio.TimeoutError, OSError) as exc:
            probe_results.append({"size": size, "success": False, "reason": str(exc)})
            if first_fail is None:
                first_fail = size

    results["ping_probes"] = probe_results
    if first_fail is not None:
        results["inferred_path_mtu"] = first_fail + 28  # +IP(20)+ICMP(8) headers
    else:
        results["inferred_path_mtu"] = ">=1500"

    # 3. tracepath on Linux for full per-hop MTU discovery
    if system == "linux" and shutil.which("tracepath"):
        results["tracepath"] = await _run_tracepath(host)

    return results


def _detect_interface_mtu() -> str:
    """Detect the MTU of the default network interface."""
    system = platform.system().lower()
    try:
        if system == "linux":
            import glob as _glob

            ifaces = _glob.glob("/sys/class/net/*/mtu")
            for mtu_path in sorted(ifaces):
                iface_name = mtu_path.split("/")[4]
                if iface_name == "lo":
                    continue
                mtu = open(mtu_path).read().strip()
                return f"{iface_name}:{mtu}"
            return "<no non-loopback interface found>"
        if system == "windows":
            import subprocess as _sp

            result = _sp.run(
                ["netsh", "interface", "ip", "show", "interface"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[2].isdigit():
                        mtu_val = int(parts[2])
                        iface_name = parts[4]
                        if mtu_val < 9000 and "loopback" not in iface_name.lower():
                            return f"{iface_name}:{mtu_val}"
            return f"<netsh result: {result.returncode}>"
        if system == "darwin":
            import subprocess as _sp

            result = _sp.run(
                ["networksetup", "-getinfo", "Ethernet"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "MTU" in line:
                        return line.strip()
            return "<could not determine MTU on macOS>"
        return f"<MTU detection not implemented for {system}>"
    except Exception as exc:
        return f"<MTU detection failed: {exc}>"


async def _run_tracepath(host: str) -> list[str]:
    """Run tracepath for per-hop MTU discovery (Linux only)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "tracepath",
            "-n",
            host,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        except asyncio.TimeoutError:
            proc.kill()
            return ["<tracepath timed out>"]
        if proc.returncode != 0:
            return [f"<tracepath exited with code {proc.returncode}>"]
        lines = stdout.decode(errors="replace").strip().splitlines()
        return lines[:30]
    except FileNotFoundError:
        return ["<tracepath not found>"]
    except Exception as exc:
        return [f"<tracepath failed: {exc}>"]


async def diagnose_upstream(url: str, correlation_id: str) -> None:
    """Run network diagnostics for an upstream URL and log the results.

    Spawn this as a background task (``asyncio.create_task``) when the
    proxy exhausts its retry budget on a connection or timeout error.
    The diagnostics are best-effort — all exceptions are caught.
    """
    try:
        parsed = urlparse(url)
        host = parsed.hostname or url
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        results: dict[str, object] = {
            "correlation_id": correlation_id,
            "upstream_url": url,
            "host": host,
            "port": port,
        }

        addresses: list[str] = await _resolve_host(host)
        results["dns_resolution"] = addresses

        results["tcp_connect"] = await _check_tcp_connect(host, port)

        probe_host = host if addresses and "timed" not in str(addresses[0]) else "8.8.8.8"
        results["local_address"] = _get_local_address(probe_host)

        traceroute = await _run_traceroute(host)
        results["traceroute"] = traceroute

        path_mtu = await _probe_path_mtu(host)
        results["path_mtu"] = path_mtu

        logger.warning(
            "UPSTREAM_DIAGNOSTICS [%s] %s",
            correlation_id,
            json.dumps(results, indent=2, default=str),
        )
    except Exception as exc:
        logger.error(
            "UPSTREAM_DIAGNOSTICS [%s] crashed: %s",
            correlation_id,
            exc,
            exc_info=True,
        )
