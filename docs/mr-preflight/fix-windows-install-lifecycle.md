# MR Preflight: `fix/windows-install-lifecycle`

Local branch: `fix/windows-install-lifecycle`
Target repository: `Thermi/headroom`
Target base: `main`
Upstream reference checked: `headroomlabs-ai/headroom`

## Template

The upstream template requires real behavior proof and rollout safety. Those
sections must explicitly document Windows validation and cleanup scope.

## Open-MR Check

- [#2980](https://github.com/headroomlabs-ai/headroom/pull/2980) directly overlaps Windows install fallback and cleanup safety.
- [#2972](https://github.com/headroomlabs-ai/headroom/pull/2972) overlaps Windows installer PATH handling.
- [#2907](https://github.com/headroomlabs-ai/headroom/pull/2907) overlaps inaccessible Windows command launchers.
- [#2522](https://github.com/headroomlabs-ai/headroom/pull/2522) overlaps Windows learn-path permission handling.
- [#2676](https://github.com/headroomlabs-ai/headroom/pull/2676) is merged and covers Serena config creation.

## Required Comment

> Open PRs #2980, #2972, #2907, and #2522 overlap material Windows/install
> changes in this branch, while merged PR #2676 covers one Serena subproblem.
> This branch still contains additional SQLite lifecycle, RTK, path, and test
> portability fixes. Please coordinate ownership before opening a duplicate MR.
