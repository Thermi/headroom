# MR Preflight: `fix/proxy-runtime-accounting`

Local branch: `fix/proxy-runtime-accounting`
Target repository: `Thermi/headroom`
Target base: `main`
Upstream reference checked: `headroomlabs-ai/headroom`

## Template

The upstream PR template was fetched. The eventual MR must fill every required
section, including actual test output and runtime rollout safety.

## Open-MR Check

- [#2921](https://github.com/headroomlabs-ai/headroom/pull/2921) cumulative output-token history overlaps savings history.
- [#2891](https://github.com/headroomlabs-ai/headroom/pull/2891) TOIN persistence overlaps telemetry/ledger work but is closed with conflicts.
- [#2688](https://github.com/headroomlabs-ai/headroom/pull/2688) streaming usage alignment overlaps accounting.
- [#2756](https://github.com/headroomlabs-ai/headroom/pull/2756) is merged and covers tokenizer-scale accounting.

## Required Comment

> Open PRs #2921, #2891, and #2688 overlap isolated accounting/history or
> telemetry pieces, but they do not solve the complete runtime-accounting,
> savings-history, retry, batch, dashboard, and tracker compatibility scope of
> this branch. The local MR should not be opened as an uncoordinated duplicate.
