# Headroom — agents.md

## Quick start

```bash
pip install -e ".[dev]"
```

## Development commands

| Action | Command |
|--------|---------|
| lint | `ruff check .` |
| format | `ruff format .` |
| typecheck | `mypy headroom --ignore-missing-imports` |
| run tests | `pytest` |
| single test | `pytest tests/test_foo.py -v -k "test_bar"` |
| full CI gate | `make ci-precheck` (Rust fmt+clippy+test + Python tests + commitlint) |
| Rust only | `make ci-precheck-rust` |
| Python only | `make ci-precheck-python` |
| benchmarks | `python -m headroom.evals suite --tier 1` |

Pre-commit runs ruff (lint+fix), ruff-format, mypy, and a `sync-plugin-versions.py` script that mutates plugin manifests harmlessly.

## Rebuild the Rust extension

```bash
make verify-rust-core
```
The proxy silently falls back to Python-only mode if `_core.so` isn't built. Always `cd /tmp` before importing headroom in tests — the repo root's `headroom/` shadows the installed package.

## Architecture notes

**Single-wheel maturin build.** `pyproject.toml` → `[tool.maturin]` builds `crates/headroom-py/` into `headroom/_core.so` (PyO3). Rust workspace under `crates/` (`headroom-core`, `headroom-proxy`, `headroom-py`, `headroom-parity`).

**Active realignment (40‑PR plan, `REALIGNMENT/`).** The project is mid‑refactor: removing the old scoring/ICM/rolling-window/LLMLingua machinery (~25K LOC deleted in Phase B). Key invariants:
- passthrough is sacred — never modify the cache hot-zone
- compress only the live zone (append‑only, position‑preserving)
- byte‑faithful re‑serialization via `serde_json::RawValue`

**CCR fragmentation with `--workers N`.** In‑memory CCR doesn't share state across uvicorn workers. Use `SqliteCcrStore` or `RedisCcrStore` for multi‑worker.

## Proxy and environment

- Entry point: `headroom proxy` → `headroom/cli/proxy.py` → `headroom/proxy/server.py:create_app()`
- Async tests run automatically via `asyncio_mode = "auto"`
- CI uses 4‑way pytest sharding (`pytest-split`)

## Testing

- Tests in `tests/`, async enabled by default
- Rust parity tests: `make test-parity` — compares Rust transform output against recorded Python fixtures in `tests/parity/fixtures/`
- Record new fixtures: `scripts/record_fixtures.py`
- Heavy tests marked `@pytest.mark.slow`, `@pytest.mark.real_llm`, `@pytest.mark.live`

## Style and commits

- Google‑style docstrings in Python
- Conventional commits (`feat:`, `fix:`, `chore:`, `docs:`, `perf:`, `test:`)
- Realignment/rust‑migration PRs use `fix:` prefix (not `feat:`) to avoid inflating semantic‑release version
