# Remove Superpowers Docs From `thermi` History

## Goal

Remove the tracked `docs/superpowers/` directory from every commit reachable
from `thermi` while preserving the directory and its files in the working
directory.

## Scope

- Create `thermi-pre-superpowers-cleanup` at the current `thermi` tip.
- Rewrite only `thermi`.
- Do not modify remote refs.
- Preserve unrelated uncommitted worktree changes.
- Do not delete or modify local `docs/superpowers/` files.

## Method

Create a complete backup bundle, clone the `thermi` branch into a temporary
workspace, and use `git-filter-repo` with `--path docs/superpowers/ --invert-paths`.
Verify the rewritten result before updating the real `thermi` ref. Refresh the
working-tree index without deleting the physical directory.

## Verification

- The backup branch remains at the original `thermi` commit.
- No `docs/superpowers/` path is reachable from rewritten `thermi` history.
- Local `docs/superpowers/` files remain present and unchanged.
- Existing unrelated worktree changes remain present.
- Remote refs are unchanged.
