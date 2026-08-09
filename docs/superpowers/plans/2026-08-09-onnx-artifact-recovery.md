# ONNX Artifact Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh a malformed Kompress ONNX artifact once when its local path is replaceable, then retry loading it without modifying mounted or protected files.

**Architecture:** Keep candidate selection and runtime fallback in `_create_onnx_session`. Add a forced-download option to the shared Hugging Face helper so pinned revisions remain centralized. Refresh through a temporary sibling file and atomic `os.replace`; refresh only after ONNX session construction fails, not after smoke-run failure caused by an unsupported runtime operator.

**Tech Stack:** Python, `huggingface_hub`, ONNX Runtime, `pytest`, `ruff`.

## Global Constraints

- Preserve `allow_download=False` as cache-only and never download during recovery.
- Never replace a path that is not a regular writable file or is a mount point.
- Preserve the original artifact when the fresh download or replacement fails.
- Use the existing pinned Hugging Face revision resolution.
- Keep the existing fallback from an unusable candidate to the next ONNX candidate.

---

### Task 1: Add forced ONNX artifact download support

**Files:**
- Modify: `headroom/onnx_runtime.py:262-306`
- Test: `tests/test_onnx_runtime.py`

**Interfaces:**
- Consumes: `repo_id`, repository-relative `filename`, optional `revision`.
- Produces: `hf_hub_download_local_first(..., force_download=False)`; when true, it skips the local lookup and calls Hugging Face with `force_download=True` while still applying `_resolve_revision`.

- [ ] **Step 1: Write the failing test**

Add a test that replaces the `huggingface_hub` module with a fake `hf_hub_download`, calls `hf_hub_download_local_first("acme/widget", "onnx/model.onnx", force_download=True)`, and asserts the call receives the resolved pinned/explicit revision and `force_download=True` without `local_files_only=True`.

```python
def test_hf_download_force_download_bypasses_local_lookup(monkeypatch):
    import types

    calls = []
    fake_hub = types.SimpleNamespace(
        hf_hub_download=lambda *args, **kwargs: calls.append((args, kwargs)) or "/tmp/fresh.onnx"
    )
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_hub)

    from headroom.onnx_runtime import hf_hub_download_local_first

    assert hf_hub_download_local_first(
        "acme/widget", "onnx/model.onnx", force_download=True, revision="abc123"
    ) == "/tmp/fresh.onnx"
    assert calls == [
        (("acme/widget", "onnx/model.onnx"), {"revision": "abc123", "force_download": True})
    ]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_onnx_runtime.py -q -k force_download`

Expected: FAIL because `hf_hub_download_local_first` does not accept `force_download`.

- [ ] **Step 3: Implement the minimal helper change**

Extend the function signature with `force_download: bool = False`. After resolving the revision and before the local lookup, add:

```python
if force_download:
    return str(
        hf_hub_download(
            repo_id,
            filename,
            revision=revision,
            force_download=True,
        )
    )
```

Keep the current local-first branch unchanged when `force_download` is false.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_onnx_runtime.py -q -k force_download`

Expected: PASS.

- [ ] **Step 5: Run the helper test module**

Run: `.venv/Scripts/python.exe -m pytest tests/test_onnx_runtime.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit the helper change**

```bash
git add headroom/onnx_runtime.py tests/test_onnx_runtime.py
git commit -m "fix: support forced onnx artifact downloads"
```

### Task 2: Refresh replaceable ONNX artifacts after load failure

**Files:**
- Modify: `headroom/transforms/kompress_compressor.py:721-834`
- Test: `tests/test_kompress_failsafe.py`

**Interfaces:**
- Consumes: resolved artifact path, model ID, repository filename, and `allow_download`.
- Produces: private `_refresh_onnx_artifact(model_id, source_filename, target_path, allow_download) -> bool`; it returns true only when a fresh artifact is available at the target path and false for cache-only, ambiguous, mounted, unwritable, failed-download, or failed-replacement cases.

- [ ] **Step 1: Write the failing safety and retry tests**

Add tests covering these exact behaviors:

```python
def test_load_failure_refreshes_replaceable_local_artifact(monkeypatch, tmp_path):
    target = tmp_path / "kompress-int8-wo.onnx"
    target.write_bytes(b"old")
    downloads = []
    attempts = []

    monkeypatch.setenv("HEADROOM_KOMPRESS_ONNX_PATH", str(target))
    monkeypatch.setenv("HEADROOM_KOMPRESS_ONNX_FILENAME", "onnx/kompress-int8-wo.onnx")
    monkeypatch.setattr(
        kc,
        "hf_hub_download_local_first",
        lambda repo, filename, **kwargs: downloads.append((repo, filename, kwargs))
        or (tmp_path / "fresh.onnx").as_posix(),
    )
    (tmp_path / "fresh.onnx").write_bytes(b"new")

    class FakeOrt:
        @staticmethod
        def SessionOptions():
            return object()

        @staticmethod
        def InferenceSession(path, options=None, providers=None):
            attempts.append(path)
            if len(attempts) == 1:
                raise RuntimeError("invalid graph")
            return _FakeOrtSession(path, fails_at_run=False)

    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", FakeOrt)
    monkeypatch.setattr(kc, "_onnx_session_options", lambda _ort: object())

    session = kc._create_onnx_session("org/model", ["CPUExecutionProvider"])

    assert session.path == str(target)
    assert target.read_bytes() == b"new"
    assert downloads[0][1] == "onnx/kompress-int8-wo.onnx"
    assert downloads[0][2]["force_download"] is True


def test_load_failure_does_not_replace_mounted_artifact(monkeypatch, tmp_path):
    target = tmp_path / "mounted.onnx"
    target.write_bytes(b"old")
    monkeypatch.setenv("HEADROOM_KOMPRESS_ONNX_PATH", str(target))
    monkeypatch.setenv("HEADROOM_KOMPRESS_ONNX_FILENAME", "onnx/kompress-fp32.onnx")
    monkeypatch.setattr(kc.os.path, "ismount", lambda path: True)
    monkeypatch.setattr(kc, "hf_hub_download_local_first", lambda *args, **kwargs: pytest.fail())

    _install_fake_ort(monkeypatch, run_fails_for={"onnx/"})
    with pytest.raises(FileNotFoundError, match="No loadable ONNX artifact"):
        kc._create_onnx_session("org/model", ["CPUExecutionProvider"])

    assert target.read_bytes() == b"old"


def test_load_failure_does_not_replace_unwritable_artifact(monkeypatch, tmp_path):
    target = tmp_path / "readonly.onnx"
    target.write_bytes(b"old")
    monkeypatch.setenv("HEADROOM_KOMPRESS_ONNX_PATH", str(target))
    monkeypatch.setenv("HEADROOM_KOMPRESS_ONNX_FILENAME", "onnx/kompress-fp32.onnx")
    monkeypatch.setattr(kc.os, "access", lambda path, mode: False)
    monkeypatch.setattr(kc, "hf_hub_download_local_first", lambda *args, **kwargs: pytest.fail())

    _install_fake_ort(monkeypatch, run_fails_for={"onnx/"})
    with pytest.raises(FileNotFoundError, match="No loadable ONNX artifact"):
        kc._create_onnx_session("org/model", ["CPUExecutionProvider"])

    assert target.read_bytes() == b"old"


def test_cache_only_load_failure_does_not_download(monkeypatch, tmp_path):
    target = tmp_path / "local.onnx"
    target.write_bytes(b"old")
    monkeypatch.setenv("HEADROOM_KOMPRESS_ONNX_PATH", str(target))
    monkeypatch.setattr(kc, "hf_hub_download_local_first", lambda *args, **kwargs: pytest.fail())

    _install_fake_ort(monkeypatch, run_fails_for={"onnx/"})
    with pytest.raises(FileNotFoundError, match="No loadable ONNX artifact"):
        kc._create_onnx_session("org/model", ["CPUExecutionProvider"], allow_download=False)

    assert target.read_bytes() == b"old"


def test_failed_refresh_preserves_original_artifact(monkeypatch, tmp_path):
    target = tmp_path / "kompress-fp32.onnx"
    target.write_bytes(b"old")
    monkeypatch.setenv("HEADROOM_KOMPRESS_ONNX_PATH", str(target))
    monkeypatch.setenv("HEADROOM_KOMPRESS_ONNX_FILENAME", "onnx/kompress-fp32.onnx")
    monkeypatch.setattr(
        kc,
        "hf_hub_download_local_first",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("download failed")),
    )

    _install_fake_ort(monkeypatch, run_fails_for={"onnx/"})
    with pytest.raises(FileNotFoundError, match="No loadable ONNX artifact"):
        kc._create_onnx_session("org/model", ["CPUExecutionProvider"])

    assert target.read_bytes() == b"old"
```

The tests must also assert that a failed forced download leaves `target.read_bytes()` equal to `b"old"`, and that an unwritable target is not refreshed. Use monkeypatches for `os.access` and `os.path.ismount`; do not require root privileges or an actual mount in the test environment.

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_kompress_failsafe.py -q -k "refreshes_replaceable or mounted_artifact or cache_only_load_failure or failed_download or unwritable"`

Expected: FAIL because `_refresh_onnx_artifact` does not exist and the candidate loop does not retry a refreshed artifact.

- [ ] **Step 3: Implement safe source resolution and replacement**

Add these imports near the existing standard-library imports:

```python
import shutil
import tempfile
from pathlib import Path
```

Add a helper that resolves the remote source for an explicit path:

```python
def _onnx_refresh_source(filename: str) -> str | None:
    if not os.path.isabs(filename):
        return filename
    override = os.environ.get(KOMPRESS_ONNX_FILENAME_ENV, "").strip()
    if override:
        return override
    basename = Path(filename).name
    for candidate in _DEFAULT_ONNX_FILENAMES:
        if Path(candidate).name == basename:
            return candidate
    return None
```

Add the safe replacement helper. It must check `allow_download`, `Path.is_file()`, `os.access(target, os.W_OK)`, `os.access(target.parent, os.W_OK)`, and `os.path.ismount(target)` before downloading. Use `tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False)`, copy the fresh download with `shutil.copyfile`, preserve permissions with `shutil.copymode`, atomically replace with `os.replace`, and delete the temporary file in a `finally` block. Catch download and filesystem exceptions, log a warning, and return false without touching the original unless `os.replace` succeeds.

If the forced download returns the same path as the target, treat the forced download as refreshed and return true without copying the file onto itself.

- [ ] **Step 4: Retry only session construction failures**

Refactor the candidate loop so `InferenceSession(...)` is in its own `try` block. On its first exception, call `_refresh_onnx_artifact` and retry the same path once if it returns true. Keep `_smoke_run(session)` in a separate `try` block; smoke-run failures continue directly to the next candidate without forcing a download, because they can represent a valid artifact unsupported by the installed ONNX Runtime.

The retry shape should be equivalent to:

```python
refreshed = False
while True:
    try:
        session = ort.InferenceSession(
            onnx_path,
            _onnx_session_options(ort),
            providers=providers,
        )
    except Exception as exc:
        last_err = exc
        if not refreshed and _refresh_onnx_artifact(
            model_id,
            _onnx_refresh_source(filename),
            onnx_path,
            allow_download=allow_download,
        ):
            refreshed = True
            continue
        logger.warning("ONNX artifact %r from %s is unusable (%s); trying next candidate", filename, model_id, exc)
        break
    try:
        _smoke_run(session)
    except Exception as exc:
        last_err = exc
        logger.warning("ONNX artifact %r from %s is unusable (%s); trying next candidate", filename, model_id, exc)
        break
    return session
```

Update the function docstring to document the one-time replacement policy and the cache-only restriction.

- [ ] **Step 5: Run the focused ONNX tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_kompress_failsafe.py tests/test_transforms/test_kompress_compressor.py -q`

Expected: all selected tests pass.

- [ ] **Step 6: Run lint and whitespace checks**

Run: `.venv/Scripts/ruff.exe check headroom/onnx_runtime.py headroom/transforms/kompress_compressor.py tests/test_onnx_runtime.py tests/test_kompress_failsafe.py`

Run: `rtk git diff --check`

Expected: Ruff reports no violations and `git diff --check` emits no output.

- [ ] **Step 7: Commit the ONNX recovery implementation**

```bash
git add headroom/onnx_runtime.py headroom/transforms/kompress_compressor.py tests/test_onnx_runtime.py tests/test_kompress_failsafe.py
git commit -m "fix: refresh replaceable onnx artifacts"
```

### Task 3: Run the complete relevant verification suite

**Files:**
- No source changes.

- [ ] **Step 1: Run all Kompress tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_kompress_failsafe.py tests/test_kompress_preload_deferral.py tests/test_transforms/test_kompress_compressor.py tests/test_onnx_runtime.py -q`

Expected: all tests pass; pre-existing dependency deprecation warnings are acceptable if no test fails.

- [ ] **Step 2: Inspect the final intended diff**

Run: `rtk git diff HEAD~1 --stat` and `rtk git status --short`

Expected: the implementation commit contains only the ONNX helper, loader, and their tests; unrelated pre-existing worktree modifications remain unstaged.
