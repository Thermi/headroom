# MR Preflight: `fix/tokenizer-model-routing`

Local branch: `fix/tokenizer-model-routing`
Target repository: `Thermi/headroom`
Target base: `main`
Upstream reference checked: `headroomlabs-ai/headroom`

## Template

`headroomlabs-ai/headroom/.github/PULL_REQUEST_TEMPLATE.md` was fetched and
reviewed. Required sections are: Description, Type of Change, Changes Made,
Testing, Real Behavior Proof, Runtime Rollout Safety, Review Readiness,
Checklist, Screenshots, and Additional Notes. No template exists in the
`Thermi/headroom` fork.

## Open-MR Check

- [#2681](https://github.com/headroomlabs-ai/headroom/pull/2681) `fix(providers): count Anthropic tool_result/tool_use content blocks in OpenAI token counter` is an open upstream match for one part of this branch.
- [#2935](https://github.com/headroomlabs-ai/headroom/pull/2935) Copilot live model routing is open but is not the tokenizer-routing problem here.

## Required Comment

> Upstream PR #2681 overlaps the Anthropic-block counting portion of this
> branch, but it does not cover the full model-aware tokenizer routing,
> DeepSeek cold-download avoidance, gateway aliases, or `/v1/compress`
> per-model behavior. This branch remains a separate broader fix.

No MR will be opened until this overlap is reviewed.

## Implementation Parity

- `#2681`: **PARTIAL**. Its changed files are limited to one OpenAI provider test; it does not contain this branch's tokenizer registry, DeepSeek, or `/v1/compress` routing solution.
- `#2935`: **NO MATCH**. It is Copilot model-catalog routing.

No open MR provides the same complete solution.
