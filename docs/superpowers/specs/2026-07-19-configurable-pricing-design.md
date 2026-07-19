# Configurable Model Pricing with Smart Resolution

**Status:** draft
**Date:** 2026-07-19
**Branch:** (TBD)

## Problem

Model pricing is scattered across hardcoded injection functions, hand-maintained
provider remaps, and inconsistent resolution logic. New or obscure models (e.g.
`moonshotai/kimi-k3`, `xai/grok-4`) fail pricing resolution because:

1. **Provider prefix synonyms** (`moonshotai/` vs `moonshot/`) are hardcoded in
   `provider_remap` dicts that diverge between `litellm_pricing.py` and
   `savings_tracker.py`.
2. **New model pricing** must be committed to the codebase (import-time
   injection into `litellm.model_cost`) — no user-facing configuration.
3. **Error messages are useless:** `Failed to get pricing for model
   moonshotai/kimi-k3: BadRequestError` tells the user nothing about how to fix
   it.

## Design

Three layers, progressively more automatic and less hardcoded.

### Layer 1: Dynamic provider prefix resolution

Replace the hardcoded prefix maps with logic that queries litellm's internal
provider registry.

**How it works:**
- litellm's `LlmProviders` enum lists all known providers.
- For a model like `moonshotai/kimi-k3`:
  1. Strip the unknown prefix (`moonshotai/`) → bare model `kimi-k3`.
  2. Try `{provider}/{bare_model}` with every known provider prefix.
  3. First match wins → cache the mapping.
- Existing prefix rules (`claude-` → `anthropic/`, `gpt-` → `openai/`,
  `deepseek-` → `deepseek/`) become the fast-path cache for common cases.
  The dynamic fallback only fires when the fast-path misses.

**Files changed:**
- `headroom/pricing/litellm_pricing.py` — `_resolve_litellm_model_uncached()`
- `headroom/proxy/savings_tracker.py` — `_resolve_litellm_model()`
- `headroom/pricing/litellm_model_resolution.py` — `MODEL_PREFIX_RULES` and
  related helpers (add dynamic provider iteration)

**After this layer:**
- `moonshotai/kimi-k3` → resolved to `moonshot/kimi-k3` (dynamic).
- But `kimi-k3` still has no pricing entry in `litellm.model_cost`.

### Layer 2: `pricing_overrides` in models.json

Extend `~/.headroom/config/models.json` with a `pricing_overrides` key that
injects entries directly into `litellm.model_cost` at proxy startup.

**Format** (litellm-native per-token pricing, same shape as `litellm.model_cost`
entries):

```json
{
  "pricing_overrides": {
    "kimi-k3": {
      "input_cost_per_token": 0.000003,
      "output_cost_per_token": 0.000015,
      "cache_read_input_token_cost": 0.0000003,
      "litellm_provider": "moonshot",
      "max_input_tokens": 1048576,
      "max_output_tokens": 1048576
    },
    "moonshot/kimi-k3": {
      "input_cost_per_token": 0.000003,
      "output_cost_per_token": 0.000015,
      "cache_read_input_token_cost": 0.0000003,
      "litellm_provider": "moonshot",
      "max_input_tokens": 1048576,
      "max_output_tokens": 1048576
    }
  }
}
```

**Resolution order** (highest priority first):
1. `HEADROOM_MODEL_COST_MAP` env var (already exists)
2. `--model-cost-map` CLI flag (already exists)
3. `pricing_overrides` in `models.json`
4. Hardcoded import-time injections (DeepSeek V4, MiniMax, etc.) — now
   **overridable** by any of the above
5. litellm's built-in `model_prices_and_context_window.json`

**Injection timing:**
- At proxy startup, after loading `models.json`, merge `pricing_overrides` into
  `litellm.model_cost`.
- Entries loaded from `pricing_overrides` take precedence over hardcoded
  injections (write after the hardcoded injections run).
- Resolution happens **before** any request is served, so there is no
  per-request overhead.

**File:** `headroom/proxy/server.py` — `HeadroomProxy._inject_builtin_costs()`
or a new `_apply_pricing_overrides()` called after config loading.

### Layer 3: Actionable diagnostics + CLI helper

**Proxy warning** when pricing is unknown:

Instead of the current generic:
```
WARNING - Failed to get pricing for model moonshotai/kimi-k3: BadRequestError
```

Log an actionable message:
```
WARNING - Unknown pricing for model 'moonshotai/kimi-k3' (resolved to 'moonshot/kimi-k3').
  No cost data in litellm's database or pricing_overrides.
  To add, paste this into ~/.headroom/config/models.json:

  {"pricing_overrides":{"kimi-k3":{"input_cost_per_token":"???",
   "output_cost_per_token":"???","litellm_provider":"moonshot"}}}

  See https://platform.kimi.ai/docs/pricing/chat-k3 for pricing.
```

**CLI helper:** `headroom pricing suggest <model>`

```
$ headroom pricing suggest kimi-k3
---
Model: kimi-k3
Provider: moonshot (resolved from moonshotai)
Suggested pricing_overrides entry:
  "kimi-k3": {
    "input_cost_per_token":    "???",  # $3.00/M → 0.000003
    "output_cost_per_token":   "???",  # $15.00/M → 0.000015
    "cache_read_input_token_cost": "???",  # $0.30/M → 0.0000003
    "litellm_provider": "moonshot",
    "max_input_tokens": 1048576
  }

Pricing reference: https://platform.kimi.ai/docs/pricing/chat-k3
Add this entry to ~/.headroom/config/models.json under "pricing_overrides".
```

The `suggest` command does NOT scrape web pages — it's a helper that outputs
the template with known metadata (provider, context window from litellm limits).
The user fills in dollar amounts.

### What stays unchanged

| Component | Status |
|-----------|--------|
| Hardcoded injections (DeepSeek V4, MiniMax) | **Kept** as fast-path. But `pricing_overrides` takes precedence at startup. |
| `HEADROOM_MODEL_COST_MAP` env var | **Kept.** Already exists, just undocumented. |
| `--model-cost-map` CLI flag | **Kept.** |
| Rust-side `model_prices_and_context_window.json` | **Unchanged.** Only used for model limits, not dollar cost. |
| Provider-layer `models.json` (`context_limits`, provider `pricing`) | **Unchanged.** Separate concern — context limits and tier inference, not dollar cost. |
| `DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN` in savings_tracker | **Kept.** ($3.00/1M, Sonnet-tier) |

### What gets removed

- Just-added Kimi K3 hardcoded injection (`headroom/pricing/litellm_pricing.py`
  lines 303-346) — replaced by `pricing_overrides` mechanism.
- Hardcoded `provider_remap = {"moonshotai/": "moonshot/"}` in both resolvers —
  replaced by dynamic provider lookup.

### Implementation plan

**Phase A: Dynamic provider prefix resolution** (new)

1. Add `_find_provider_for_bare_model(bare_model)` to litellm_pricing — iterates
   known litellm providers, tries `{provider}/{bare_model}`.
2. Add dynamic fallback to `_resolve_litellm_model_uncached()` — when a model
   has an unknown `prefix/model` shape, strip prefix and call the dynamic
   resolver.
3. Add same dynamic fallback to `savings_tracker._resolve_litellm_model()`.
4. Cache the resolved mapping in `_resolved_model_cache`.

**Phase B: `pricing_overrides` in models.json** (new)

1. Define `PricingOverride` schema (pydantic `model_cost` entry shape).
2. Add `pricing_overrides: dict[str, dict]` to proxy config loading
   (`headroom/proxy/models.py` or server startup).
3. Inject overrides into `litellm.model_cost` after built-in injections.
4. Remove Kimi K3 hardcoded injection.

**Phase C: Actionable diagnostics** (new)

1. Change `Failed to get pricing` warning in `cost.py:713` to include the
   actionable snippet.
2. Add `headroom pricing suggest <model>` CLI command.

**Phase D: Tests**

1. Unit tests for dynamic provider prefix resolution.
2. Unit tests for `pricing_overrides` precedence.
3. Integration test: models.json override beats hardcoded injection.
4. E2E test: unknown model produces actionable warning.

## Trade-offs

**Pro: Dynamic prefix resolution** — Never need to add `moonshotai/`,
`xai/`, `nvidia/` etc. remaps. Works for any future provider.

**Con: Dynamic prefix resolution** — Extra litellm calls on cold cache
(~1-3 `cost_per_token` calls per unknown prefix, each ~500μs). Acceptable
because the first call caches the result permanently.

**Pro: `pricing_overrides`** — No code change needed to add new model
pricing. Survives docker rebuilds (bind-mounted `models.json`).

**Con: `pricing_overrides`** — User must manually look up pricing and
maintain the file. Layer 3 mitigates this with the actionable snippet.

**Pro: Keeping hardcoded injections** — No regression for existing models.
DeepSeek V4 and MiniMax continue to work without user configuration.

**Risk: litellm registry changes** — If litellm drops a provider from
`LlmProviders`, the dynamic lookup silently fails. Mitigation: the
fast-path prefix rules handle the common cases; dynamic is fallback only.

## References

- `models.json` format: `wiki/configuration.md` lines 280-341
- `HEADROOM_MODEL_COST_MAP`: `headroom/proxy/server.py:5195-5201`,
  `headroom/cli/proxy.py:925-931`, `headroom/proxy/models.py:316-320`
- Current hardcoded injections: `headroom/pricing/litellm_pricing.py:258-346`,
  `headroom/proxy/server.py:2225-2301`
- litellm `model_cost` entry shape: `crates/headroom-proxy/data/model_prices_and_context_window.json`
- Provider prefix rules: `headroom/pricing/litellm_model_resolution.py:1-88`
