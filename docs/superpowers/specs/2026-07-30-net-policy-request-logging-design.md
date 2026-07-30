# Net Policy Request Logging Design

## Goal

Reduce net-policy log noise while retaining one useful cost/model summary per
HTTP request, and log each unknown LiteLLM model only once per process.

## Behavior

- Net-policy candidate evaluation continues to run for every eligible message
  part with no policy decision changes.
- Candidate-level `INFO` logs containing model, token, or cost inputs are
  removed.
- One `INFO` summary is emitted after the request-level net-policy analysis,
  containing the model, candidate count, aggregate token/cost telemetry, and
  aggregate decision counts.
- Unknown-pricing warnings are deduplicated by normalized model name for the
  lifetime of the process, rather than expiring after five minutes.
- Different unknown models each receive one warning.

## Components

- `headroom/transforms/content_router.py`: collect net-policy telemetry during
  `apply()` and emit one request summary; `_net_cost_allows()` remains focused
  on the policy decision.
- `headroom/proxy/cost.py`: replace the expiring warning timestamp map with a
  process-lifetime warned-model set.
- Existing focused test modules receive regression coverage for both behaviors.

## Testing

- Verify repeated net-policy candidates generate one `INFO` summary and no
  candidate-level `INFO` records.
- Verify repeated unknown-pricing lookups for one model warn once, while a
  second unknown model warns once independently.
- Run focused tests, lint, and the relevant broader test modules.
