# Remove Large Files From `thermi` History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove `headroom-git.tar.gz` and `docker/kompress/kompress-int8-wo.onnx` from every commit reachable from `thermi` while preserving both files in the working directory.

**Architecture:** Create a local backup branch and bundle at the current `thermi` tip. Rewrite only the `thermi` ref with path-based history filtering, then verify the paths are absent from all reachable trees while the physical files remain unchanged on disk.

**Tech Stack:** Git, `git-filter-repo` or equivalent history-rewrite tooling, PowerShell.

## Global Constraints

- Rewrite only `thermi`.
- Create `thermi-pre-file-cleanup` before rewriting.
- Do not modify remote refs or force-push.
- Do not delete or modify the two files in the working directory.
- Preserve unrelated uncommitted worktree changes.

---

### Task 1: Checkpoint the Current Branch

**Files:**
- Modify: Git refs only; do not modify tracked or untracked files.

- [ ] **Step 1: Record the current tip and worktree status.**

```powershell
git branch --show-current
git rev-parse thermi
git status --short --branch
```

Expected: current branch is `thermi`; record its commit hash and existing worktree changes.

- [ ] **Step 2: Confirm both files exist and record their hashes.**

```powershell
git branch thermi-pre-file-cleanup thermi
Get-FileHash -Algorithm SHA256 -LiteralPath 'headroom-git.tar.gz'
Get-FileHash -Algorithm SHA256 -LiteralPath 'docker/kompress/kompress-int8-wo.onnx'
```

Expected: both files exist; retain their hashes for post-rewrite verification.

- [ ] **Step 3: Create the backup branch.**

```powershell
```

Expected: backup branch points to the recorded original commit.

- [ ] **Step 4: Create a complete recovery bundle.**

```powershell
$bundle = Join-Path $env:TEMP 'headroom-thermi-pre-file-cleanup.bundle'
git bundle create $bundle thermi-pre-file-cleanup
git bundle verify $bundle
```

Expected: bundle verification reports a complete history.

### Task 2: Rewrite Only `thermi`

**Files:**
- Modify: Git history reachable from `thermi` only.
- Preserve: `headroom-git.tar.gz` and `docker/kompress/kompress-int8-wo.onnx` on disk.

- [ ] **Step 1: Confirm rewrite tooling.**

```powershell
git filter-repo --force --path headroom-git.tar.gz --path docker/kompress/kompress-int8-wo.onnx --invert-paths
Get-Command git-filter-repo -ErrorAction SilentlyContinue
```

Expected: use `git-filter-repo` if available; do not use `--all`.

- [ ] **Step 2: Remove both paths from the `thermi` history.**

Run from the checked-out `thermi` branch:

```powershell
```

Expected: `thermi` receives rewritten commits; the backup branch and remote refs are unchanged. The command must not be run with `--path-rename`, filesystem deletion, or any command that removes the physical files.

### Task 3: Verify History and Local Files

**Files:**
- Modify: none.

- [ ] **Step 1: Verify branch isolation.**

```powershell
git rev-parse thermi
git rev-parse thermi-pre-file-cleanup
git remote -v
```

Expected: backup hash equals the recorded original hash; `thermi` may differ; no remote ref changed.

- [ ] **Step 2: Verify paths are absent from all reachable `thermi` trees.**

```powershell
git log thermi --all --name-status --format= -- headroom-git.tar.gz docker/kompress/kompress-int8-wo.onnx
git ls-tree -r --name-only thermi | Select-String -Pattern '(^|/)(headroom-git\.tar\.gz|docker/kompress/kompress-int8-wo\.onnx)$'
```

Expected: no output from either command for the rewritten `thermi` history or current tree.

- [ ] **Step 3: Verify both physical files remain unchanged.**

```powershell
Test-Path -LiteralPath 'headroom-git.tar.gz'
Test-Path -LiteralPath 'docker/kompress/kompress-int8-wo.onnx'
Get-FileHash -Algorithm SHA256 -LiteralPath 'headroom-git.tar.gz'
Get-FileHash -Algorithm SHA256 -LiteralPath 'docker/kompress/kompress-int8-wo.onnx'
```

Expected: both paths return `True` and hashes match Task 1.

- [ ] **Step 4: Verify unrelated worktree changes remain.**

```powershell
git status --short --branch
```

Expected: existing changes remain present; the two files may appear as untracked because they are no longer tracked by `thermi`.
