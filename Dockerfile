ARG PYTHON_VERSION=3.13
ARG UV_VERSION=0.11.19
ARG DISTROLESS_IMAGE=gcr.io/distroless/python3-debian13
ARG PYTHON_SITE_PACKAGES=/usr/local/lib/python${PYTHON_VERSION}/site-packages


# ---- CUDA runtime libraries stage (GPU support for onnxruntime-gpu) ----
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS cuda-libs

# ---- Rust toolchain stage (rarely changes; cached independently) ----
FROM python:${PYTHON_VERSION}-slim AS rust-toolchain

ARG UV_VERSION

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

ARG GIT_COMMIT
ARG BUILD_TIME
ARG VERSION=unknown

# ---- Build stage: compile native extensions, build wheel ----
FROM rust-toolchain AS builder

ARG GIT_COMMIT
ARG BUILD_TIME

# Skip .pyc generation — they'd be deleted before export anyway.
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Install build-time system deps (maturin, setuptools-rust) once.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system maturin setuptools-rust patchelf

# Phase 1 — resolve and install Python dependencies for the selected extras,
# without building the headroom package itself.  Only pyproject.toml and
# the lockfile invalidate this cache; source-code edits do not.
COPY pyproject.toml uv.lock ./
# Lean extras for the proxy service: proxy + code + memory + relevance.
# Omits ml/voice (torch ~6 GB), anyllm, otel, image, memory-torch, memory-stack,
# langchain, agno, strands, pdf/fulltext/audio, evals, benchmark, html, reports,
# bedrock, proxy-prod, voice-train, pytorch-mps.  GPU onnxruntime is force-installed
# separately below.
ARG HEADROOM_EXTRAS=proxy,code,memory,relevance
RUN --mount=type=cache,target=/root/.cache/uv \
    export HEADROOM_EXTRAS="${HEADROOM_EXTRAS}" && \
    python <<SCRIPT \
    | uv pip install --system -r /dev/stdin
import os, tomllib, pathlib
p = tomllib.loads(pathlib.Path('pyproject.toml').read_text())
deps = list(p['project'].get('dependencies', []))
wanted = [e.strip() for e in os.environ.get('HEADROOM_EXTRAS', 'all').split(',')]
if 'all' in wanted:
    for name, group in p['project'].get('optional-dependencies', {}).items():
        if name not in ('dev', 'test', 'doc'):
            deps.extend(group)
else:
    seen = set(wanted)
    for extra in wanted:
        q = [extra]
        while q:
            e = q.pop()
            group = p['project'].get('optional-dependencies', {}).get(e, [])
            for spec in group:
                if spec.startswith('headroom-ai['):
                    inner = spec.split('[')[1].split(']')[0]
                    for sub in inner.split(','):
                        sub = sub.strip()
                        if sub not in seen:
                            seen.add(sub)
                            q.append(sub)
                else:
                    deps.append(spec)
    deps = list(dict.fromkeys(deps))
print('\n'.join(deps))
SCRIPT

# Phase 2 — copy the Rust workspace + Python source and build the wheel.
# Cache-busted by actual source changes only; dep install stays cached.
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY .git .git/
COPY crates/ crates/
COPY headroom/ headroom/
COPY README.md ./

# Inject build-time metadata (git commit, build timestamp) into _build_info.py.
# When .git is available (build context from a git checkout), pull the short
# commit hash directly.  Otherwise fall back to the GIT_COMMIT / BUILD_TIME
# build args (CI / docker-compose).
RUN GIT="${GIT_COMMIT}" && \
    if [ -z "$GIT" ] || [ "$GIT" = "unknown" ]; then \
      GIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown"); \
    fi && \
    BUILD="${BUILD_TIME}" && \
    if [ -z "$BUILD" ] || [ "$BUILD" = "unknown" ]; then \
      BUILD=$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo "unknown"); \
    fi && \
    sed -i "s/BUILD_GIT_COMMIT: str = \"\"/BUILD_GIT_COMMIT: str = \"$GIT\"/" headroom/_build_info.py && \
    sed -i "s/BUILD_TIME: str = \"\"/BUILD_TIME: str = \"$BUILD\"/" headroom/_build_info.py && \
    echo -n "$GIT" > /tmp/.git_commit && \
    echo -n "$BUILD" > /tmp/.build_time && \
    echo "headroom/_build_info.py injected: commit=$GIT build=$BUILD"
# Remove .git to keep the builder layer lean — it is not needed at runtime.
RUN rm -rf .git

# Build and install the headroom package (including the Rust _core extension).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/build/target \
    uv pip install --system --no-build-isolation --no-deps \
        ".[${HEADROOM_EXTRAS}]"

# The proxy uses ojson for ordered JSON support during startup and request
# serialization. Install it in the builder so both runtime stages inherit it.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "ojson==0.1.0"

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

# Replace CPU-only onnxruntime with GPU-enabled onnxruntime-gpu.
# onnxruntime-gpu >=1.27.0 requires libcudart.so.13 (CUDA 13) which is not
# in the nvidia/cuda:12.6.3 base image — cap below that threshold.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --force-reinstall "onnxruntime-gpu>=1.16.0,<1.27.0"

# Strip unnecessary files from site-packages so the runtime-stage COPY
# (and its subsequent export compression) has far less data to move.
# onnxruntime-gpu alone bundles ~1 GB of binaries; __pycache__ and test
# artifacts add hundreds more MB.
# Further reduction: CUDA /usr/local/cuda-12.6/ includes headers, static
# libs, and tools not needed at runtime — only the .so under lib64/ plus
# cuDNN are loaded.  Copying only those saves ~2 GB.
RUN find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
    find /usr/local -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete; \
    find /usr/local/lib/python${PYTHON_VERSION}/site-packages -type d \( -name tests -o -name test -o -name testing \) -exec rm -rf {} + 2>/dev/null; \
    find /usr/local/lib/python${PYTHON_VERSION}/site-packages -type f \( -name "*.a" -o -name "*.la" \) -delete 2>/dev/null; \
    find /usr/local/lib/python${PYTHON_VERSION}/site-packages -type d -name ".libs" -exec rm -rf {} + 2>/dev/null; \
    echo "cleaned"

# ---- Runtime stage (python-slim): supports root/nonroot via build arg ----
FROM python:${PYTHON_VERSION}-slim AS runtime-slim-base

ARG RUNTIME_USER=nonroot
ARG RUNTIME_HOME=/home/nonroot
ARG PYTHON_SITE_PACKAGES
ARG GIT_COMMIT
ARG BUILD_TIME
ARG VERSION

LABEL org.opencontainers.image.title="headroom" \
      org.opencontainers.image.description="Universal prompt engineering toolkit" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.created="${BUILD_TIME}" \
      org.opencontainers.image.source="https://github.com/chopratejas/headroom"

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl htop iputils-ping iputils-tracepath traceroute && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder ${PYTHON_SITE_PACKAGES} ${PYTHON_SITE_PACKAGES}
COPY --from=builder /usr/local/bin/headroom /usr/local/bin/headroom
COPY --from=builder /root/.headroom/bin/rtk /usr/local/bin/rtk
# Copy only the CUDA runtime shared libraries (lib64/), not headers/tools.
# The full toolkit weighs ~3 GB; lib64/ + cuDNN is ~1.5 GB.
COPY --from=cuda-libs /usr/local/cuda-12.6/lib64 /usr/local/cuda-12.6/lib64
# cuDNN is installed to system paths (/usr/lib/x86_64-linux-gnu/) in the
# nvidia/cuda image, not inside the CUDA toolkit directory.  Copy it into
# the CUDA lib path so onnxruntime's CUDAExecutionProvider can find it.
COPY --from=cuda-libs /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda-12.6/lib64/

RUN mkdir -p /opt/headroom && python3 <<SCRIPT
import json, pathlib, headroom._build_info as bi
info = {'git_commit': bi.BUILD_GIT_COMMIT, 'build_time': bi.BUILD_TIME}
pathlib.Path('/opt/headroom/build-info.json').write_text(json.dumps(info))
SCRIPT

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
    HF_HOME=${RUNTIME_HOME}/.headroom/huggingface \
    LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}

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
ARG GIT_COMMIT
ARG BUILD_TIME
ARG VERSION

LABEL org.opencontainers.image.title="headroom" \
      org.opencontainers.image.description="Universal prompt engineering toolkit" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.created="${BUILD_TIME}" \
      org.opencontainers.image.source="https://github.com/chopratejas/headroom"

COPY --from=builder ${PYTHON_SITE_PACKAGES} ${PYTHON_SITE_PACKAGES}
COPY --from=builder /root/.headroom/bin/rtk /usr/local/bin/rtk
# Copy only the CUDA runtime shared libraries, not headers/tools.
COPY --from=cuda-libs /usr/local/cuda-12.6/lib64 /usr/local/cuda-12.6/lib64

RUN mkdir -p /opt/headroom && python3 <<SCRIPT
import json, pathlib, headroom._build_info as bi
info = {'git_commit': bi.BUILD_GIT_COMMIT, 'build_time': bi.BUILD_TIME}
pathlib.Path('/opt/headroom/build-info.json').write_text(json.dumps(info))
SCRIPT

USER ${RUNTIME_USER}
WORKDIR /app

ENV HEADROOM_HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=${PYTHON_SITE_PACKAGES} \
    HF_HOME=/app/.headroom/huggingface \
    LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH}

VOLUME /app/.headroom

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/readyz', timeout=5)"]

ENTRYPOINT ["python3", "-m", "headroom.cli", "proxy"]
CMD ["--host", "0.0.0.0", "--port", "8787"]

# Default published image remains python-slim runtime
FROM runtime-slim-base AS runtime
