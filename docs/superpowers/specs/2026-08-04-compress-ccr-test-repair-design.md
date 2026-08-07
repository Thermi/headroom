# Compress and CCR Test Repair Design

## Goal

Restore the existing `/v1/compress` behavior expected by the endpoint contract
without weakening tests: compression must run off the event loop, enforce its
bounded timeout, return the established failure responses, and record request
outcomes. CCR stats tests must observe an isolated global store.

## Approach

Use the existing `_run_compression_in_executor` helper around the synchronous
OpenAI pipeline call. Handle timeout separately as a fail-open response with
zero savings and `skip_reason="compression_timeout"`; record compression
failure metrics and a zero-savings `RequestOutcome`. For ordinary exceptions,
record failed metrics and return the existing `compression_error` 503 body.
On success, record a `RequestOutcome` using the pipeline token and transform
metadata before returning the current response shape.

Keep the endpoint's input validation, bypass behavior, and response fields
unchanged. Avoid unrelated refactors.

## CCR Isolation

Update the CCR test fixture/setup so each test starts and ends with a reset
store, including tests that run in arbitrary selection order. Do not change
production stats semantics because the endpoint correctly reports the global
store contents.

## Verification

Run the four previously failing compression tests, the full
`tests/test_proxy_compress_endpoint.py` file, the full
`tests/test_proxy_ccr.py` file, and inspect the resulting diff.
