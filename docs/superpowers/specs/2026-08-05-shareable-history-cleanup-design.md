# Shareable Git History Cleanup Design

## Goal

Prepare a shareable version of the repository with no committed secrets while
preserving the current `thermi` branch as an unchanged recovery point.

## Scope and Branches

- Leave `thermi` unchanged.
- Create `thermi-pre-secret-cleanup` at the current `thermi` commit.
- Create `thermi-shareable` from the same commit.
- Rewrite only the history reachable from `thermi-shareable`.
- Do not rewrite or force-push remote branches.
- Do not modify unrelated uncommitted worktree changes.

## Cleanup Method

Scan the reachable history of the rewrite branch for secret-shaped content and
identify confirmed secrets separately from synthetic test fixtures and public
example values. Use `git-filter-repo` with an explicit, auditable replacement
or removal map for confirmed secrets only. Do not delete broad filename classes
or alter unrelated source and documentation.

## Verification

- Confirm `thermi` still points to its original commit.
- Confirm both backup and rewrite branches were created from that commit.
- Scan the rewritten branch's reachable history for the identified secret
  values and sensitive file paths.
- Confirm synthetic fixtures remain intact unless they contain a confirmed
  credential.
- Confirm no remote refs were changed.
- Report any real credentials found as requiring revocation or rotation; a Git
  rewrite does not invalidate credentials or copies of the old history.

## User-Visible Result

The cleaned branch is `thermi-shareable`. The untouched backup branch is
`thermi-pre-secret-cleanup`. No force-push is performed.
