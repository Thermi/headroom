"""Tests for the Kompress ONNX provider startup banner wording.

The banner must attribute a CPU-only run to the governing override
(``HEADROOM_KOMPRESS_BACKEND`` / ``HEADROOM_KOMPRESS_ONNX_PROVIDER``) rather
than reporting it as a passive "no GPU detected" hardware fallback.
"""

from __future__ import annotations

import pytest

from headroom.proxy.server import _kompress_provider_banner

PROVIDER_ENV = "HEADROOM_KOMPRESS_ONNX_PROVIDER"
BACKEND_ENV = "HEADROOM_KOMPRESS_BACKEND"


def _banner(**overrides):
    def _boom() -> list[str]:
        raise AssertionError("probe_gpu should not be called")

    kwargs = {
        "provider_override": "",
        "backend": "auto",
        "backend_raw": "",
        "provider_env": PROVIDER_ENV,
        "backend_env": BACKEND_ENV,
        "probe_gpu": _boom,
    }
    kwargs.update(overrides)
    return _kompress_provider_banner(**kwargs)


def test_provider_override_takes_precedence_without_probing():
    msg = _banner(provider_override="CUDAExecutionProvider", backend="auto")
    assert f"overridden by {PROVIDER_ENV}=CUDAExecutionProvider" in msg
    assert "will use CUDAExecutionProvider" in msg


@pytest.mark.parametrize("backend", ["onnx", "onnx_cpu"])
def test_cpu_pinned_backend_attributes_to_override_and_skips_probe(backend):
    msg = _banner(backend=backend, backend_raw="onnx-cpu")
    assert f"pinned to CPU by {BACKEND_ENV}=onnx-cpu" in msg
    assert "GPU detection skipped" in msg
    # Must not claim "No GPU detected" -- that would be a misleading fallback.
    assert "No GPU detected" not in msg


def test_gpu_pinned_backend_with_device():
    msg = _banner(
        backend="onnx_gpu",
        backend_raw="onnx-gpu",
        probe_gpu=lambda: ["CUDAExecutionProvider"],
    )
    assert f"pinned to GPU by {BACKEND_ENV}=onnx-gpu" in msg
    assert "will use CUDAExecutionProvider" in msg


def test_gpu_pinned_backend_without_device_falls_back():
    msg = _banner(backend="onnx_gpu", backend_raw="onnx-gpu", probe_gpu=lambda: [])
    assert "pinned to GPU" in msg
    assert "no GPU detected" in msg
    assert "fall back to CPUExecutionProvider" in msg


def test_auto_with_gpu_detected():
    msg = _banner(backend="auto", probe_gpu=lambda: ["CUDAExecutionProvider"])
    assert msg == "[bg] GPU detected -- Kompress ONNX will use CUDAExecutionProvider"


def test_auto_without_gpu_suggests_provider_override():
    msg = _banner(backend="auto", probe_gpu=lambda: [])
    assert "No GPU detected" in msg
    assert f"set {PROVIDER_ENV}=<provider> to override" in msg
