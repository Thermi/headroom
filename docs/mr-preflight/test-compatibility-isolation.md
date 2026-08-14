# MR Preflight: `test/compatibility-isolation`

Local branch: `test/compatibility-isolation`
Target repository: `Thermi/headroom`
Target base: `main`
Upstream reference checked: `headroomlabs-ai/headroom`

## Template

The upstream template is available and must be filled completely. This MR must
mark the change as test/compatibility work and include the exact focused test
commands.

## Open-MR Check

- [#2831](https://github.com/headroomlabs-ai/headroom/pull/2831) adds structural proxy concurrency tests but does not cover this branch's compatibility isolation set.
- [#2266](https://github.com/headroomlabs-ai/headroom/pull/2266) is closed/superseded and concerned stale CI failures.

## Result

No open MR was found that solves the same complete compatibility/isolation
problem. This branch is eligible for an MR after the focused tests and template
fields are prepared.

## Implementation Parity

- `#2831`: **DIFFERENT**. Structural concurrency tests only.
- `#2266`: **CLOSED / superseded**, related stale-CI intent but not this branch.

No open MR solves the same complete problem.
