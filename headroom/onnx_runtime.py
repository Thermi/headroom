"""ONNX Runtime helpers for long-running Headroom processes."""

from __future__ import annotations

import asyncio
import ctypes
import os
import sys
import threading
from typing import Any

# ── Shared onnxruntime availability / provider detection ──────────────
#
# onnxruntime is a heavy import: on a GPU build the first ``import
# onnxruntime`` dlopens ~1 GB of CUDA libraries and can take several
# seconds. Multiple subsystems probe it independently at startup (the
# proxy GPU banner, the memory embedder backend selector, the Kompress
# compressor). Probing concurrently — or before that slow first import
# finishes — produced inconsistent results: one caller would report
# "onnxruntime not available" / "No GPU detected" while another, a moment
# later, successfully loaded CUDA.
#
# These helpers make detection deterministic and race-free:
#   * the answer is computed once and memoized, so every caller agrees;
#   * only a *successful* probe is cached — a transient early failure is
#     never frozen in, so a later call still gets the real answer;
#   * :func:`warm_onnxruntime` lets startup perform the slow import once,
#     serially, before concurrent consumers probe it.
_ort_lock = threading.Lock()
_ort_available: bool | None = None
_gpu_providers: tuple[str, ...] | None = None

_GPU_PROVIDER_KEYWORDS = ("cuda", "tensorrt", "dml", "directml", "coreml", "rocm")


def onnxruntime_available() -> bool:
    """Return whether ``onnxruntime`` can be imported in this process.

    The result is memoized once the import succeeds. A failed import is not
    cached, so a probe issued before the (slow) first import completes does
    not permanently poison the answer for later callers.
    """
    global _ort_available
    if _ort_available:
        return True
    with _ort_lock:
        if _ort_available:
            return True
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        _ort_available = True
        return True


def available_gpu_providers() -> list[str]:
    """Return ONNX Runtime's available GPU execution providers, CUDA first.

    Uses case-insensitive matching so provider names like
    ``CUDAExecutionProvider``, ``TensorrtExecutionProvider``,
    ``DmlExecutionProvider``, etc. are identified regardless of case
    variance across onnxruntime versions. A provider being *available*
    (compiled into the build) does not guarantee its runtime libraries are
    present; that is resolved gracefully at session-creation time.

    The detected list is memoized after the first successful probe so every
    subsystem reports the same providers for the life of the process.
    """
    global _gpu_providers
    if _gpu_providers is not None:
        return list(_gpu_providers)
    if not onnxruntime_available():
        return []
    with _ort_lock:
        if _gpu_providers is not None:
            return list(_gpu_providers)
        import onnxruntime

        all_providers = onnxruntime.get_available_providers()
        gpu_providers = [
            p for p in all_providers if any(kw in p.lower() for kw in _GPU_PROVIDER_KEYWORDS)
        ]
        if "CUDAExecutionProvider" in gpu_providers:
            gpu_providers.insert(0, gpu_providers.pop(gpu_providers.index("CUDAExecutionProvider")))
        _gpu_providers = tuple(gpu_providers)
        return list(gpu_providers)


def warm_onnxruntime() -> None:
    """Eagerly import onnxruntime once, serially, warming the shared caches.

    Call this from a single-threaded startup path *before* concurrent
    subsystems probe onnxruntime. Doing the slow first import here ensures
    later probes hit a warm ``sys.modules`` entry and the memoized result,
    instead of racing the in-progress import (which previously surfaced as
    spurious "onnxruntime not available" / "No GPU detected" logs).
    """
    if onnxruntime_available():
        available_gpu_providers()



def _resolve_revision(repo_id: str, revision: str | None) -> str | None:
    """Resolve the HF revision to download: explicit arg wins, else the pinned
    SHA for a known repo, else ``None`` (floating ref)."""
    if revision is not None:
        return revision
    if os.environ.get("HEADROOM_HF_PIN", "").strip().lower() in ("off", "0", "false", "no"):
        return None
    return _PINNED_REVISIONS.get(repo_id)


def hf_hub_download_local_first(
    repo_id: str,
    filename: str,
    *,
    allow_network: bool = True,
    revision: str | None = None,
) -> str:
    """Download a file from HuggingFace Hub, preferring the local cache.

    Tries ``local_files_only=True`` first to avoid a network HEAD request when
    the model is already cached.  Falls back to a normal (network-allowed)
    download on the first cold start.

    Args:
        repo_id: HuggingFace Hub repository identifier (e.g. ``"org/model"``).
        filename: Filename within the repository.
        allow_network: When ``False``, never fall back to a network download —
            a cache miss re-raises the local-lookup error. Used by startup
            preload so a cold cache cannot block (or, via native crashes in the
            download stack, kill) the process before it binds its port.
        revision: Explicit git revision (commit SHA / tag / branch). When
            ``None``, a pinned SHA is applied for known repos (see
            ``_PINNED_REVISIONS``) for supply-chain integrity; unknown repos use
            the floating default ref.

    Returns:
        Absolute path to the local cached file.

    Raises:
        Any exception raised by ``hf_hub_download`` on a genuine download failure,
        or the local-lookup error when ``allow_network`` is ``False`` and the
        file is not cached.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    revision = _resolve_revision(repo_id, revision)

    try:
        return str(hf_hub_download(repo_id, filename, revision=revision, local_files_only=True))
    except (LocalEntryNotFoundError, EntryNotFoundError, OSError):
        if not allow_network:
            raise
        return str(hf_hub_download(repo_id, filename, revision=revision))


def hf_entry_known_absent(repo_id: str, filename: str, *, revision: str | None = None) -> bool:
    """True only if a prior network lookup already confirmed ``filename`` does
    not exist in ``repo_id`` at the resolved revision.

    Backed by ``huggingface_hub``'s own cache of negative lookups (the
    ``.no_exist`` marker written after a real 404), so this never makes a
    network call itself. Returns ``False`` both when the file is cached and
    when nothing is known yet about it, on purpose: callers in a cache-only
    (``allow_network=False``) code path can use this to tell "confirmed
    missing upstream, safe to use a fallback file" apart from "just never
    checked yet, do not guess."
    """
    from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache

    revision = _resolve_revision(repo_id, revision)
    result = try_to_load_from_cache(repo_id, filename, revision=revision)
    return result is _CACHED_NO_EXIST


def create_cpu_session_options(
    ort: Any,
    *,
    intra_op_num_threads: int | None = None,
    inter_op_num_threads: int | None = None,
) -> Any:
    """Create CPU-oriented ONNX Runtime session options.

    Headroom runs as a long-lived proxy process, so we bias toward predictable
    memory usage over peak ONNX throughput. Disabling ORT's CPU arena and memory
    pattern caches reduces retained anonymous RSS after variable-size inference
    workloads, which is especially important on small VMs.
    """
    sess_options = ort.SessionOptions()

    if intra_op_num_threads is not None:
        sess_options.intra_op_num_threads = intra_op_num_threads
    if inter_op_num_threads is not None:
        sess_options.inter_op_num_threads = inter_op_num_threads

    if hasattr(sess_options, "enable_cpu_mem_arena"):
        sess_options.enable_cpu_mem_arena = False
    if hasattr(sess_options, "enable_mem_pattern"):
        sess_options.enable_mem_pattern = False

    return sess_options


def trim_process_heap() -> bool:
    """Ask glibc to return unused heap pages to the OS when available."""
    if not sys.platform.startswith("linux"):
        return False

    try:
        libc = ctypes.CDLL("libc.so.6")
    except OSError:
        return False

    try:
        return bool(libc.malloc_trim(0))
    except Exception:
        return False


class OnnxRuntimeWarmup:
    """Coordinates onnxruntime background warmup with ready signaling.

    Lets concurrent background tasks wait for onnxruntime to be probed
    before making decisions that depend on its availability, without
    forcing sequential initialization.

    Usage in an async context (e.g. proxy startup)::

        warmup = OnnxRuntimeWarmup()

        async def bg_onnx():
            await warmup.warmup()
            # GPU probe ...

        async def bg_memory():
            await warmup.wait_ready()
            # onnxruntime_available() now returns the correct answer
            ...
    """

    def __init__(self) -> None:
        self._ready = asyncio.Event()

    async def warmup(self) -> None:
        """Run :func:`warm_onnxruntime` in a thread and signal readiness.

        Returns once onnxruntime has been successfully imported and GPU
        providers probed (or the 30s retry loop exhausted).
        """
        await asyncio.to_thread(warm_onnxruntime)
        self._ready.set()

    async def wait_ready(self) -> None:
        """Wait indefinitely for the warmup to complete.

        Returns once :func:`warmup` has finished (onnxruntime has been
        successfully imported and GPU providers probed).
        """
        await self._ready.wait()
