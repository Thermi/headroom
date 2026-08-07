# Shareable Git History Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create an untouched backup of `thermi` and produce a separately rewritten `thermi-shareable` branch with confirmed secrets removed, without changing remote refs or unrelated worktree changes.

**Architecture:** Record the current `thermi` commit, create two local branches at that commit, and perform all rewriting on `thermi-shareable`. Use an isolated temporary clone or installed `git-filter-repo` tooling, an explicit replacement/removal map, and post-rewrite scans before reporting the result.

**Tech Stack:** Git, PowerShell, `git-filter-repo` if needed, repository secret-pattern scans.

## Global Constraints

- Leave `thermi` unchanged.
- Create `thermi-pre-secret-cleanup` and `thermi-shareable` from the current `thermi` commit.
- Rewrite only history reachable from `thermi-shareable`.
- Do not force-push or rewrite remote branches.
- Preserve unrelated uncommitted worktree changes.
- Do not delete broad filename classes or alter synthetic fixtures without confirming a real credential.

---

### Task 1: Capture State and Create Branch Checkpoints

**Files:**
- Modify: Git refs only; do not modify worktree files.

- [ ] **Step 1: Record the current branch and commit.**

Run:

```powershell
git branch --show-current
git rev-parse thermi
git status --short --branch
```

Expected: current branch is `thermi`, and the worktree changes are documented without staging or reverting them.

- [ ] **Step 2: Create the immutable local backup branch.**

Run:

```powershell
git branch thermi-pre-secret-cleanup thermi
```

Expected: the branch is created at the recorded `thermi` commit.

- [ ] **Step 3: Create the rewrite branch.**

Run:

```powershell
git branch thermi-shareable thermi
git switch thermi-shareable
```

Expected: `thermi-shareable` is checked out and points to the same commit as `thermi`.

- [ ] **Step 4: Verify the checkpoints.**

Run:

```powershell
git rev-parse thermi
git rev-parse thermi-pre-secret-cleanup
git rev-parse thermi-shareable
git status --short --branch
```

Expected: all three branch hashes match; only pre-existing worktree changes are present.

### Task 2: Inventory Confirmed Secret Content

**Files:**
- Create: temporary scan output outside the repository.
- Modify: none.

- [ ] **Step 1: Check available rewrite tooling.**

Run:

```powershell
Get-Command git-filter-repo -ErrorAction SilentlyContinue
Get-Command gitleaks -ErrorAction SilentlyContinue
```

Expected: record which tools are available; do not install or modify project dependencies yet.

- [ ] **Step 2: List sensitive historical paths reachable from the rewrite branch.**

Run:

```powershell
git log thermi-shareable --name-only --format= -- | Select-String -Pattern '(^|/)(\.env($|\.)|.*(secret|credential|password|private.*key).*|.*\.(pem|key|p12|pfx))' | Sort-Object -Unique
```

Expected: inspect every result and distinguish ignored/local files, source modules, and actual credential artifacts.

- [ ] **Step 3: Scan historical text for credential-shaped values.**

Run targeted searches over historical blobs and current tracked files, excluding documented synthetic fixtures and public placeholders only after verifying their purpose. Save the exact confirmed values and their paths to a temporary replacement map outside the repository.

Expected: no secret is rewritten based solely on a generic word such as `token`; only verified credentials are included.

### Task 3: Rewrite Only `thermi-shareable`

**Files:**
- Modify: Git history reachable from `thermi-shareable` only.
- Create: temporary replacement map outside the repository.

- [ ] **Step 1: Preserve a recoverable bundle before rewriting.**

Run:

```powershell
git bundle create "$env:TEMP\headroom-thermi-pre-secret-cleanup.bundle" thermi-pre-secret-cleanup
```

Expected: the bundle is created successfully and contains the untouched backup branch.

- [ ] **Step 2: Run `git-filter-repo` with the explicit map.**

Use the replacement map for confirmed secret values and remove only confirmed secret-only files. Run against the checked-out `thermi-shareable` branch and do not include `--all` or any remote-tracking ref.

Expected: `thermi-shareable` receives new commit IDs where affected history changed; `thermi` and `thermi-pre-secret-cleanup` remain unchanged.

- [ ] **Step 3: Restore or preserve the unrelated worktree changes if the rewrite tool moved them.**

Run:

```powershell
git status --short --branch
```

Expected: pre-existing changes remain present and no unrelated file is deleted or overwritten.

### Task 4: Verify the Shareable Branch

**Files:**
- Modify: none.

- [ ] **Step 1: Verify branch isolation.**

Run:

```powershell
git rev-parse thermi
git rev-parse thermi-pre-secret-cleanup
git rev-parse thermi-shareable
git branch --contains thermi-shareable
```

Expected: `thermi` and `thermi-pre-secret-cleanup` still match the recorded original hash; `thermi-shareable` is the only rewritten branch.

- [ ] **Step 2: Rescan rewritten history.**

Run the same historical path and content scans from Task 2 against `thermi-shareable` and confirm each identified secret is absent while synthetic fixtures remain unless specifically confirmed as credentials.

Expected: no confirmed secret remains reachable from `thermi-shareable`.

- [ ] **Step 3: Verify remote refs and worktree status.**

Run:

```powershell
git remote -v
git status --short --branch
git log --oneline -3 thermi-shareable
```

Expected: no remote ref has been changed; only intended local branch/history changes exist.

- [ ] **Step 4: Report credential rotation requirements.**

If any real credential was found, report its service and required revocation/rotation without printing the value. A successful rewrite does not invalidate credentials or old clones.
