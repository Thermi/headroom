# MR Preflight: `fix/ccr-cache-integrity`

Local branch: `fix/ccr-cache-integrity`
Target repository: `Thermi/headroom`
Target base: `main`
Upstream reference checked: `headroomlabs-ai/headroom`

## Template

The upstream template is available at
`https://github.com/headroomlabs-ai/headroom/blob/main/.github/PULL_REQUEST_TEMPLATE.md`.
All required sections will be completed before submission.

## Open-MR Check

- [#2607](https://github.com/headroomlabs-ai/headroom/pull/2607) CCR TTL idle-window behavior overlaps cache TTL handling.
- [#2706](https://github.com/headroomlabs-ai/headroom/pull/2706) proactive-expansion eligibility overlaps expansion behavior.
- [#2707](https://github.com/headroomlabs-ai/headroom/pull/2707) expansion-query extraction overlaps CCR response handling.
- [#2876](https://github.com/headroomlabs-ai/headroom/pull/2876) `headroom_retrieve` history repair overlaps retrieval history handling.

## Required Comment

> Open PRs #2607, #2706, #2707, and #2876 overlap individual CCR/cache
> behaviors in this branch, but none covers the complete prefix lineage,
> marker integrity, retrieval injection, streaming, and cache-store changes.
> This MR is broader and should be coordinated with those open PRs.

No MR will be opened until overlap ownership is clarified.
