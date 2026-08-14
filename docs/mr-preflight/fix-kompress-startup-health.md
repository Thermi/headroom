# MR Preflight: `fix/kompress-startup-health`

Local branch: `fix/kompress-startup-health`
Target repository: `Thermi/headroom`
Target base: `main`
Upstream reference checked: `headroomlabs-ai/headroom`

## Template

The upstream PR template was fetched and all required sections will be filled:
description, change type, changes, testing, behavior proof, rollout safety,
review readiness, checklist, screenshots, and notes.

## Open-MR Check

- [#2799](https://github.com/headroomlabs-ai/headroom/pull/2799) `fix(proxy): warm deferred kompress model so /health stops reporting backend=null` directly overlaps deferred warmup and health promotion.
- [#2835](https://github.com/headroomlabs-ai/headroom/pull/2835) remote Kompress batching overlaps the remote-compression portion.
- [#2831](https://github.com/headroomlabs-ai/headroom/pull/2831) structurally tests bounded pre-upstream concurrency, not the full Kompress branch.

## Required Comment

> Open PRs #2799 and #2835 overlap material parts of this branch, but they do
> not cover the complete startup/health reconciliation, cache-promotion,
> ONNX-backend, and transform compatibility changes assembled here. This MR
> should be coordinated with those PRs rather than treated as a duplicate.

No MR will be opened until maintainers choose whether to split or supersede
these overlapping upstream changes.
