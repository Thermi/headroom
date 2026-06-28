# Headroom — agents.md

- if you can not find the reason why something is not working, make a simple code that shows the issue. Keep this simple code as sparse as possible and use it to prove that the issue exists. If the issue exists, then verify it is not a software bug but a hardware bug. It is nearly 99,99999% likely it is a software bug in the software you wrote.
- commit as you work along.
- always commit your changes.
- never undo existing commits.
- if you are thinking of circumventing user directives, don't.
- If you are thinking of breaking containment measures, don't.
- if you are thinking of using docker to run code that you would otherwise be unable to, don't.
- always make sure the code you are running does not cause out of memory conditions.
- always make sure the code you are running does not interfere with the operating system or other processes.
- make use of the available tools to help you.
- use sequential thinking tool to help you think through the problem step by step.
- use memories via headroom mcp to remember large pieces of information.
- only read as little as possible.
- keep your context clean
- IMPORTANT: When applicable, prefer using pycharm-index MCP tools for code navigation and refactoring.

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

## Known warnings on Windows

`PytestUnhandledThreadExceptionWarning` is a benign artefact of
`subprocess.Popen` output-capture on Windows.  The background reader
thread decodes captured bytes with the default system encoding
(`cp1252`).  Any non-ASCII Unicode character in test output can trigger
a `UnicodeDecodeError` in the reader (e.g. `0x8f` for em dash `—`).

This warning is harmless — it appears **after** the test run has
already collected and reported the output.  It can be safely ignored.
The project's `pyproject.toml` already suppresses
`PytestUnraisableExceptionWarning`; adding the same for this class is
appropriate.

## Style and commits

- Google‑style docstrings in Python
- Conventional commits (`feat:`, `fix:`, `chore:`, `docs:`, `perf:`, `test:`)
- Realignment/rust‑migration PRs use `fix:` prefix (not `feat:`) to avoid inflating semantic‑release version

## Pre-commit hooks

Hooks (ruff lint+fix, ruff-format, mypy, sync-plugin-versions, commitlint) run
via pre-commit, installed in the project venv at `.venv/Scripts/pre-commit`.

**Reinstall after venv rebuild:**
```bash
.venv/Scripts/python -m pre_commit install --hook-type pre-commit --hook-type commit-msg
```

**Hook behaviour:**
1. Prepends `.venv/Scripts` to `PATH` so hooks calling `python3` resolve to the
   venv interpreter (works around Windows missing `python3.exe`).
2. If the venv does not exist or is broken, prints a warning and exits 0 —
   commits proceed without hooks.
3. The `commit-msg` hook runs `commitlint` (requires Node.js / npx on PATH).
   Without it, this hook is skipped and commits still go through.

**Windows‑specific:** `.venv/Scripts/python3.bat` is a thin wrapper that
forwards `python3 <args>` → `python.exe <args>`, placed there by `install-git-hooks`
or created manually when the hook first runs. Recreate it after venv rebuild if
missing:
```bat
@echo off
"%~dp0python.exe" %*
```
