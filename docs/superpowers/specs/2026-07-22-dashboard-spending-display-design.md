# Design: Dashboard Spending Display

**Date:** 2026-07-22
**Status:** Approved

## Goal

Replace the "Proxy $ Saved" hero card on the dashboard with a spending comparison card that shows actual cost vs. what it would have cost without Headroom, making the value proposition immediately visible.

## Scope

Dashboard-only change. No backend modifications needed — all required data is already in the `/stats` payload.

## Design

### Card Replacement

Replace the "Proxy $ Saved" hero card (lines 187-205 in `dashboard.html`) with a "Spending (session)" card.

### Layout

```
Spending (session)
$1.23  vs  $4.56
saved 73.0%

1,234 proxy tokens compressed; CLI filtering excluded from $
```

- **Label:** `Spending (session)` — uppercase, xs, gray-500 (same as other hero cards)
- **Primary number:** Actual spend (`$1.23`) — `text-3xl font-light tabular-nums text-emerald-400`
- **"vs" separator:** `text-gray-500 text-sm`
- **Counterfactual:** Without Headroom (`$4.56`) — `text-3xl font-light tabular-nums text-gray-400` (dimmed)
- **Savings %:** `saved 73.0%` — `text-sm text-emerald-400`
- **Sub-line:** Token detail — `text-xs text-gray-500`

### Data Sources

All from `GET /stats` response (no new endpoints):

| Display field | Stats path | Notes |
|---|---|---|
| Actual spend | `stats.persistent_savings.display_session.total_input_cost_usd` | Current session input cost |
| Without Headroom | Computed: `total_input_cost_usd + compression_savings_usd` | What it would have cost |
| Savings % | `stats.summary.cost.savings_pct` | Already computed |
| Token detail | `stats.tokens.proxy_compression_saved` | Proxy tokens only |

### Edge Cases

- **Zero cost (`total_input_cost_usd == 0`):** Show "—" for both dollar amounts, sub-line shows request count
- **LiteLLM unavailable (`stats.litellm_available === false`):** Same fallback message as current card explaining LiteLLM is needed for pricing
- **No session data:** Show "—" with "0 requests processed" sub-line

### Visual Style

Same card dimensions and container classes as current card. No layout disruption to the 4-column hero grid.

## Files to Modify

- `headroom/dashboard/templates/dashboard.html` — Replace hero card HTML (lines 187-205)

## Testing

- Manual: Open `/dashboard`, verify the spending card renders with correct numbers
- Existing dashboard tests in `tests/test_dashboard_*.py` should continue to pass
