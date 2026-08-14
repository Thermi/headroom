# MR Preflight: `fix/memory-storage-safety`

Local branch: `fix/memory-storage-safety`
Target repository: `Thermi/headroom`
Target base: `main`
Upstream reference checked: `headroomlabs-ai/headroom`

## Template

The upstream PR template was fetched and will be completed in full before any
MR is submitted.

## Open-MR Check

- [#2899](https://github.com/headroomlabs-ai/headroom/pull/2899) optional Cognee backend is open but does not cover this branch's storage safety changes.
- [#2637](https://github.com/headroomlabs-ai/headroom/pull/2637) fail-closed project operations overlaps project isolation.
- [#2951](https://github.com/headroomlabs-ai/headroom/pull/2951) is merged and covers malformed `entity_refs` normalization.
- [#2579](https://github.com/headroomlabs-ai/headroom/pull/2579) is merged and covers TrafficLearner bounds.

## Required Comment

> Open PR #2899 is an optional-backend feature, not the storage-safety scope
> here. Merged PRs #2951 and #2579 cover two related subproblems, while open
> PR #2637 overlaps project isolation. This branch remains broader and should
> be coordinated with #2637 rather than submitted as a duplicate.
