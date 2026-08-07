# Pricing Dashboard Pane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose cached pricing entries and retrieval timestamps through `/pricing` and a dashboard Pricing pane.

**Architecture:** Reuse the existing in-memory pricing cache and add a sanitized snapshot function. The proxy exposes that snapshot through a read-only `/pricing` endpoint. The existing Alpine dashboard fetches `/pricing` alongside `/stats` and renders a responsive table/card pane.

**Tech Stack:** Python, FastAPI, Alpine.js, Tailwind-style existing dashboard CSS, pytest.

## Global Constraints

- Never expose authorization headers, API keys, or request credentials.
- Keep pricing data in memory only.
- Show source, route/provider/model, prices, cache rates, retrieval time, and TTL status.
- Preserve existing `/stats` and `/stats-history` response shapes.
- Existing unrelated worktree changes remain untouched.

---

### Task 1: Pricing Snapshot API

**Files:**
- Modify: `headroom/proxy/pricing_detect.py`
- Modify: `headroom/proxy/server.py`
- Test: `tests/test_pricing_detect.py`
- Test: `tests/test_proxy_settings_endpoints.py`

**Interfaces:**
- Add `pricing_snapshot() -> list[dict[str, Any]]` returning sanitized entries only.
- Add `GET /pricing` returning `{ "entries": [...], "ttl_seconds": number, "generated_at": string }`.

- [ ] Write failing tests for snapshot serialization and `/pricing` response shape.
- [ ] Run the focused tests and verify failure.
- [ ] Implement the snapshot under the existing pricing cache lock, converting monotonic timestamps to age/status fields without exposing headers.
- [ ] Register the endpoint beside existing stats/metrics endpoints.
- [ ] Run focused API tests and verify pass.
- [ ] Commit with `feat: expose pricing snapshot endpoint`.

### Task 2: Dashboard Pricing Pane

**Files:**
- Modify: `headroom/dashboard/templates/dashboard.html`
- Test: `tests/test_dashboard_cache_net_playwright.py` or the existing dashboard test fixture.

**Interfaces:**
- Fetch `/pricing` during dashboard refresh.
- Render a Pricing pane with desktop table and mobile stacked rows.

- [ ] Add a dashboard test fixture asserting the pane title and representative pricing fields.
- [ ] Run the dashboard test to establish failure.
- [ ] Add the pane using existing dashboard typography, colors, cards, and Alpine state conventions.
- [ ] Show `source`, `route`, `model`, input/output prices, cache rates, retrieved age, and stale/valid status.
- [ ] Add empty/error states without breaking the existing dashboard when `/pricing` is unavailable.
- [ ] Run the dashboard test and verify pass.
- [ ] Commit with `feat: add pricing dashboard pane`.

### Task 3: Verification

- [ ] Run `pytest tests/test_pricing_detect.py tests/test_proxy_settings_endpoints.py -q`.
- [ ] Run the relevant dashboard test command.
- [ ] Run Python compilation for modified backend files.
- [ ] Run `git diff --check`.
- [ ] Confirm unrelated `docker/.env2`, `docker/docker-compose.native.yml`, `.worktrees/`, and tokenizer cache files remain untouched.
