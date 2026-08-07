# Provider-Aware Net-Cost Pricing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the net-cost compression gate use provider/model-specific cache economics and pricing from headers or synchronous OpenRouter metadata lookup with TTL refresh.

**Architecture:** Add a small pricing resolver/cache in `headroom.proxy.pricing_detect`. Response pricing is authoritative and refreshes the cache immediately when changed. OpenRouter metadata is queried synchronously before request handling when the cached entry is missing or expired. `ContentRouter` receives resolved input price and cache multipliers for each request, while retaining token-unit fallback when no price is available.

**Tech Stack:** Python, `httpx`, LiteLLM model pricing, pytest, existing `CompressionPolicy` net-cost formula.

## Global Constraints

- Header pricing takes precedence over OpenRouter metadata.
- Pricing cache entries are keyed by route/provider/model.
- Pricing metadata uses a configurable TTL with a safe default.
- Changed pricing from a request refreshes the matching cache entry immediately.
- OpenRouter metadata lookup occurs synchronously before request handling.
- Metadata lookup failures never fail a request; use the last cached value or token-unit fallback.
- Existing unrelated worktree changes remain untouched.

---

### Task 1: Pricing Resolver Cache

**Files:**
- Modify: `headroom/proxy/pricing_detect.py`
- Test: `tests/test_pricing_detect.py`

**Interfaces:**
- Produce `resolve_pricing(model: str, provider: str, route: str, headers: Mapping[str, str] | None = None) -> PricingInfo | None`.
- Produce `PricingInfo` fields for input/output per-token price, source, and refresh timestamp.
- Preserve `feed_response()` behavior for post-response header/body detection.

- [ ] Write failing tests for header precedence, changed-header refresh, fresh TTL reuse, expired TTL refresh, and metadata failure fallback.
- [ ] Run `pytest tests/test_pricing_detect.py -q` and verify the new tests fail.
- [ ] Implement the bounded in-memory cache, TTL environment parsing, header extraction, and synchronous OpenRouter metadata request.
- [ ] Cache OpenRouter metadata using the route and model identity; avoid logging credentials.
- [ ] Run `pytest tests/test_pricing_detect.py -q` and verify all tests pass.
- [ ] Commit with `fix: add ttl pricing resolution for net cost`.

### Task 2: Provider-Aware Net-Cost Formula

**Files:**
- Modify: `headroom/transforms/content_router.py`
- Modify: `headroom/transforms/compression_policy.py`
- Test: `tests/test_netcost_gate.py`

**Interfaces:**
- Extend `_net_cost_allows` with provider/model/route/header pricing inputs without breaking existing callers.
- Use resolved economics: read multiplier, write multiplier, and input price.
- Preserve token-unit behavior when no price is available.

- [ ] Add failing tests proving identical token counts can produce different decisions for Anthropic versus OpenAI economics.
- [ ] Add a failing test proving resolved header pricing changes the gain calculation.
- [ ] Run the focused tests and confirm failure under the hardcoded constants.
- [ ] Implement provider-aware gain calculation while preserving `p_alive`, expected reads, and batch reclaim behavior.
- [ ] Thread request model/provider/route/header context from `ContentRouter.apply` into `_net_cost_allows`.
- [ ] Run `pytest tests/test_netcost_gate.py -q` and verify all tests pass.
- [ ] Commit with `fix: apply provider pricing to net cost policy`.

### Task 3: Pre-Request Resolution Wiring

**Files:**
- Modify: `headroom/proxy/handlers/openai.py`
- Modify: `headroom/proxy/handlers/anthropic.py`
- Modify: `headroom/proxy/handlers/gemini.py`
- Modify: `headroom/proxy/handlers/streaming.py`
- Test: `tests/test_netcost_gate.py`

**Interfaces:**
- Resolve pricing before compression/net-cost decisions using request model, route, provider, and request headers.
- Pass the resulting pricing context into the router/pipeline request state.

- [ ] Add a failing integration-style test showing request headers are available to the net-cost resolver before compression.
- [ ] Add a failing OpenRouter test using mocked `/api/v1/models` metadata when response headers have no pricing.
- [ ] Run the tests and verify they fail before wiring.
- [ ] Add synchronous pre-request resolution in each handler path, with safe exception handling and cached fallback.
- [ ] Ensure post-response `feed_response` updates the same cache and immediately replaces changed values.
- [ ] Run focused handler, pricing, and net-cost tests.
- [ ] Commit with `fix: wire pre-request pricing into net cost policy`.

### Task 4: Full Verification

**Files:**
- No additional source changes expected.

- [ ] Run `pytest tests/test_pricing_detect.py tests/test_netcost_gate.py tests/test_proxy/test_cost_outcome.py -q`.
- [ ] Run `ruff check headroom/proxy/pricing_detect.py headroom/transforms/content_router.py headroom/transforms/compression_policy.py`.
- [ ] Run `git diff --check`.
- [ ] Confirm unrelated `docker/.env2`, `docker/docker-compose.native.yml`, and `.worktrees/` changes remain untouched.
- [ ] Commit any required formatting-only fix separately with `style: format pricing policy changes`.
