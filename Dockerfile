ARG PYTHON_VERSION=3.13
ARG UV_VERSION=0.11.19
ARG DISTROLESS_IMAGE=gcr.io/distroless/python3-debian13
ARG PYTHON_SITE_PACKAGES=/usr/local/lib/python${PYTHON_VERSION}/site-packages


# ---- CUDA runtime libraries stage (GPU support for onnxruntime-gpu) ----
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS cuda-libs

# ---- Rust toolchain stage (rarely changes; cached independently) ----
FROM python:${PYTHON_VERSION}-slim AS rust-toolchain

ARG UV_VERSION
ARG PYTHON_SITE_PACKAGES
ARG HEADROOM_BUILD_VERSION=""

# build-essential / g++ for any C extension wheels uv may need to build
# from source. curl + ca-certificates are required by the rustup
# bootstrap below. patchelf for maturin's wheel-link repair on linux.
# No OpenSSL system deps required: the rustls-everywhere refactor
# eliminated `openssl-sys` from our build tree by switching fastembed
# to `hf-hub-rustls-tls` + `ort-download-binaries-rustls-tls`.
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    curl \
    ca-certificates \
    patchelf \
    git \
  && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install uv==${UV_VERSION}

# Rust toolchain for the headroom._core extension. With single-wheel
# architecture (post-#355), `pip install -e .` invokes maturin via
# pyproject.toml's [build-system], which calls cargo. No more separate
# headroom-core-py package.
ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:${PATH}
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --profile minimal -c rustfmt -c clippy --default-toolchain 1.95.0

ARG GIT_COMMIT=
ARG BUILD_TIME=

# ---- Build stage: compile native extensions, build wheel ----
FROM rust-toolchain AS builder

ARG GIT_COMMIT
ARG BUILD_TIME

WORKDIR /build

# Install build-time system deps (maturin, setuptools-rust) once.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system maturin setuptools-rust patchelf

# Phase 1 — resolve and install all Python dependencies from the lockfile,
# without building the headroom package itself.  Only pyproject.toml and
# the lockfile invalidate this cache; source-code edits do not.
COPY pyproject.toml uv.lock ./
RUN python <<SCRIPT > /tmp/runtime-deps.txt
import tomllib, pathlib
p = tomllib.loads(pathlib.Path('pyproject.toml').read_text())
deps = list(p['project'].get('dependencies', []))
for name, group in p['project'].get('optional-dependencies', {}).items():
    if name not in ('dev', 'test', 'doc'):
        deps.extend(group)
print('\n'.join(deps))
SCRIPT
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /tmp/runtime-deps.txt

#ARG HEADROOM_EXTRAS=code,proxy,memory
ARG HEADROOM_EXTRAS=all

# Phase 2 — copy the Rust workspace + Python source and build the wheel.
# Cache-busted by actual source changes only; dep install stays cached.
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY .git .git/
COPY crates/ crates/
COPY headroom/ headroom/
COPY README.md ./
COPY scripts/export_kompress_onnx.py export_kompress_onnx.py

# Inject build-time metadata (git commit, build timestamp) into _build_info.py.
# When .git is available (build context from a git checkout), pull the short
# commit hash directly.  Otherwise fall back to the GIT_COMMIT / BUILD_TIME
# build args (CI / docker-compose).
RUN GIT="${GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}" \
    BUILD="${BUILD_TIME:-$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)}" && \
    sed -i "s/BUILD_GIT_COMMIT: str = ''/BUILD_GIT_COMMIT: str = '$GIT'/" headroom/_build_info.py && \
    sed -i "s/BUILD_TIME: str = ''/BUILD_TIME: str = '$BUILD'/" headroom/_build_info.py && \
    echo "headroom/_build_info.py injected: commit=$GIT build=$BUILD"
# Remove .git to keep the builder layer lean — it is not needed at runtime.
RUN rm -rf .git

RUN echo "setuptools<82" > /tmp/uv-constraints.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/build/target \
    uv pip install --system --no-build-isolation --no-deps \
        --constraint /tmp/uv-constraints.txt ".[${HEADROOM_EXTRAS}]"

RUN --mount=type=bind,source=.,target=/context,readonly \
    HEADROOM_BUILD_VERSION="${HEADROOM_BUILD_VERSION}" PYTHON_SITE_PACKAGES="${PYTHON_SITE_PACKAGES}" python - <<'PY'
import hashlib
import os
from pathlib import Path


def git_revision(context: Path) -> str | None:
    git_dir = context / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref_name = head.removeprefix("ref: ").strip()
        ref_path = git_dir / ref_name
        if ref_path.exists():
            head = ref_path.read_text(encoding="utf-8").strip()
        else:
            packed_refs = git_dir / "packed-refs"
            if not packed_refs.exists():
                return None
            for line in packed_refs.read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                sha, _, name = line.partition(" ")
                if name.strip() == ref_name:
                    head = sha
                    break
            else:
                return None
    return head[:12] if len(head) >= 7 and all(c in "0123456789abcdef" for c in head.lower()) else None


def source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    inputs = (
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        "crates",
        "headroom",
    )
    for name in inputs:
        path = root / name
        if not path.exists():
            continue
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for file in files:
            digest.update(file.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(file.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()[:12]


build_version = os.environ["HEADROOM_BUILD_VERSION"].strip()
if not build_version:
    print("no Headroom build version override provided; using installed package metadata")
    raise SystemExit(0)
if build_version == "source-build":
    revision = git_revision(Path("/context"))
    build_version = (
        f"source-build+g{revision}"
        if revision
        else f"source-build+sha256.{source_digest(Path('/build'))}"
    )

package_dir = Path(os.environ["PYTHON_SITE_PACKAGES"]) / "headroom"
(package_dir / "_build_info.py").write_text(
    "BUILD_VERSION = " + repr(build_version) + "\n",
    encoding="utf-8",
)
print("baked Headroom build version: " + build_version)
PY

# Build-stage smoke check: verify the extension loads end-to-end inside
# the build image before we copy site-packages into the runtime image.
# If this fails, the runtime image would fail Phase A0's fail-loud
# startup check on every restart. Run from /tmp so cwd doesn't shadow
# site-packages with /build/headroom/ (which has no _core.so since
# maturin installed the .so into site-packages).
RUN cd /tmp && python -c "from headroom._core import DiffCompressor, SmartCrusher; \
    print(f'build-stage rust core verify OK: {DiffCompressor.__name__}, {SmartCrusher.__name__}')"

# Download rtk binary from GitHub releases
RUN python -c "from headroom.rtk.installer import download_rtk; download_rtk()"

# Replace CPU-only onnxruntime with GPU-enabled onnxruntime-gpu
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --force-reinstall "onnxruntime-gpu>=1.16.0"

# Re-export the Kompress ONNX model with a newer opset and bake it into the
# image so inference never waits for a HuggingFace download at cold start.
# Requires [ml] extras (already installed by HEADROOM_EXTRAS=all).
# The baked path is advertised to the runtime via HEADROOM_KOMPRESS_ONNX_PATH.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system onnxscript && \
    mkdir -p /opt/headroom && \
    python export_kompress_onnx.py \
        --output /opt/headroom/kompress-int8.onnx \
        --opset 18 \
        --no-quantize && \
    rm -f export_kompress_onnx.py && \
    rm -rf /root/.cache/huggingface/hub

# ---- Runtime stage (python-slim): supports root/nonroot via build arg ----
FROM python:${PYTHON_VERSION}-slim AS runtime-slim-base

ARG RUNTIME_USER=nonroot
ARG RUNTIME_HOME=/home/nonroot
ARG PYTHON_SITE_PACKAGES

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl htop iputils-ping iputils-tracepath traceroute && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder ${PYTHON_SITE_PACKAGES} ${PYTHON_SITE_PACKAGES}
COPY --from=builder /usr/local/bin/headroom /usr/local/bin/headroom
COPY --from=builder /root/.headroom/bin/rtk /usr/local/bin/rtk
COPY --from=cuda-libs /usr/local/cuda-12.6 /usr/local/cuda-12.6
# cuDNN is installed to system paths (/usr/lib/x86_64-linux-gnu/) in the
# nvidia/cuda image, not inside the CUDA toolkit directory.  Copy it into
# the CUDA lib path so onnxruntime's CUDAExecutionProvider can find it.
COPY --from=cuda-libs /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda-12.6/lib64/

RUN mkdir -p /home/nonroot /data && \
    if [ "$RUNTIME_USER" = "nonroot" ]; then \
      groupadd --gid 1000 nonroot && \
      useradd --uid 1000 --gid nonroot --create-home nonroot && \
      mkdir -p /home/nonroot/.headroom && \
      chown -R nonroot:nonroot /data /home/nonroot; \
    else \
      mkdir -p /root/.headroom; \
    fi

USER ${RUNTIME_USER}
WORKDIR ${RUNTIME_HOME}

ENV HEADROOM_HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-} \
    HEADROOM_KOMPRESS_ONNX_PATH=/opt/headroom/kompress-int8.onnx

# Declare ~/.headroom as a volume so Docker (and ACA) can attach persistent
# storage here.  Bare `docker run` gets an anonymous volume as a fallback so
# state is never silently written to the ephemeral container layer.
# RUNTIME_HOME defaults to /home/nonroot (the published image default); pass
# --build-arg RUNTIME_HOME=/root when building with RUNTIME_USER=root.
VOLUME ${RUNTIME_HOME}/.headroom

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["curl", "--fail", "--silent", "http://127.0.0.1:8787/readyz"]

ENTRYPOINT ["headroom", "proxy"]
CMD ["--host", "0.0.0.0", "--port", "8787"]

FROM ${DISTROLESS_IMAGE} AS runtime-slim

ARG RUNTIME_USER=nonroot
ARG PYTHON_SITE_PACKAGES

COPY --from=builder ${PYTHON_SITE_PACKAGES} ${PYTHON_SITE_PACKAGES}
COPY --from=builder /root/.headroom/bin/rtk /usr/local/bin/rtk
COPY --from=cuda-libs /usr/local/cuda-12.6 /usr/local/cuda-12.6
COPY --from=cuda-libs /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda-12.6/lib64/

USER ${RUNTIME_USER}
WORKDIR /app

ENV HEADROOM_HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=${PYTHON_SITE_PACKAGES} \
    LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH} \
    HEADROOM_KOMPRESS_ONNX_PATH=/opt/headroom/kompress-int8.onnx

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/readyz', timeout=5)"]

ENTRYPOINT ["python3", "-m", "headroom.cli", "proxy"]
CMD ["--host", "0.0.0.0", "--port", "8787"]

# Default published image remains python-slim runtime
FROM runtime-slim-base AS runtime
