# Text and MCP Startup Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the Kompress text adapter type error, demote net-cost diagnostics, advertise the MCP endpoint, and prove the proxy MCP interface works in-process.

**Architecture:** Preserve the existing compressor registry contract by normalizing the Kompress adapter at its boundary. Change only logging levels in `_net_cost_allows`. Derive the banner’s MCP status from the same `MCP_STREAMABLE_HTTP_AVAILABLE` flag used by route registration, and test the actual `/v1/mcp` endpoint through the MCP client transport.

**Tech Stack:** Python, FastAPI/Starlette ASGI transport, MCP Streamable HTTP client, pytest, Click logging.

## Global Constraints

- Both net-cost diagnostic records remain available at `DEBUG`; no net-cost calculation or admission behavior changes.
- The startup banner always shows `/v1/mcp`, including an unavailable/dependency note when MCP is not installed.
- The MCP integration test must use the in-process ASGI app and must not require a provider API or network service.

---

### Task 1: Normalize text adapter output and reduce net-cost log noise

**Files:**
- Modify: `headroom/transforms/content_router.py:344-357,4514-4532`
- Test: `tests/test_transforms_content_router.py`
- Test: `tests/test_transforms/test_content_router.py`

**Interfaces:**
- `_invoke_kompress(router, inp) -> str | None` returns only text.
- `_net_cost_allows(...)` keeps both diagnostic messages but emits them with `logger.debug`.

- [ ] **Step 1: Add the failing text adapter regression test**

Add a test that registers/uses the built-in Kompress adapter with a router whose `_try_ml_compressor` returns `("compressed text", 2)`, then asserts the registry output content is exactly the string and never a tuple.

```python
def test_kompress_registry_adapter_returns_text_not_result_tuple(monkeypatch):
    router = ContentRouter(ContentRouterConfig(enable_code_aware=False))
    monkeypatch.setattr(router, "_try_ml_compressor", lambda *args: ("compressed text", 2))

    output = router._registry_compress(
        "kompress",
        CompressionStrategy.TEXT,
        "original text",
        "",
        1.0,
    )

    assert output is not None
    assert output.content == "compressed text"
    assert isinstance(output.content, str)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_transforms_content_router.py -q -k kompress_registry_adapter_returns_text_not_result_tuple`

Expected: FAIL because `_invoke_kompress` currently returns the full `(content, tokens)` tuple as `CompressOutput.content`.

- [ ] **Step 3: Add the net-cost logging regression test**

Add a test in the existing ContentRouter tests that enables `HEADROOM_NET_COST_POLICY`, calls `_net_cost_allows` with deterministic inputs, and checks the records:

```python
    router = ContentRouter()
    monkeypatch.setenv("HEADROOM_NET_COST_POLICY", "1")

    with caplog.at_level(logging.DEBUG, logger=content_router_module.logger.name):
        router._net_cost_allows(
            slot_idx=0,
            original_tokens=100,
            compressed_tokens=50,
            suffix_tokens=[0, 20],
            route_counts={},
            transforms_applied=[],
        )

    pre_calc = [r for r in caplog.records if "NetCostPolicy pre-calc" in r.message]
    slot = [r for r in caplog.records if "NetCostPolicy slot=" in r.message]
    assert pre_calc and slot
    assert all(r.levelno == logging.DEBUG for r in pre_calc + slot)
```

- [ ] **Step 4: Run the new logging test to verify the level assertion fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_transforms/test_content_router.py -q -k net_cost_diagnostics_are_debug_only`

Expected: FAIL because both records currently use `logger.info`.

- [ ] **Step 5: Implement the minimal fixes**

Change the adapter from:

```python
compressed, _tokens = router._try_ml_compressor(inp.content, inp.query, question)
return compressed
```

and leave only that string return. Change both `logger.info(` calls surrounding `NetCostPolicy pre-calc` and `NetCostPolicy slot=` to `logger.debug(`. Do not alter the arguments, formula, counters, or transforms.

- [ ] **Step 6: Run the focused tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_transforms_content_router.py tests/test_transforms/test_content_router.py tests/test_builtin_compressor_adapters.py -q -k "kompress_registry_adapter_returns_text_not_result_tuple or net_cost_diagnostics_are_debug_only or force_kompress or log_adapter or builtin_registry"`

Expected: all selected tests pass.

- [ ] **Step 7: Commit the task**

```bash
git add headroom/transforms/content_router.py tests/test_transforms_content_router.py tests/test_transforms/test_content_router.py
git commit -m "fix: normalize text adapter and quiet netcost logs"
```

### Task 2: Advertise and test the MCP endpoint

**Files:**
- Modify: `headroom/cli/proxy.py:1075-1082,1590-1608`
- Test: `tests/test_proxy/test_mcp_route_order.py`

**Interfaces:**
- The banner imports `MCP_STREAMABLE_HTTP_AVAILABLE` from `headroom.proxy.server` and prints an availability-specific `/v1/mcp` endpoint line.
- The MCP test uses `create_app`, `httpx.ASGITransport`, `streamable_http_client`, and `ClientSession` to initialize and list tools.

- [ ] **Step 1: Add the failing MCP interface test**

Extend `tests/test_proxy/test_mcp_route_order.py` with an async test:

```python
@pytest.mark.anyio
async def test_v1_mcp_initializes_and_lists_builtin_tools():
    pytest.importorskip("mcp")
    import httpx
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    app = create_app(_minimal_config())
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        async with streamable_http_client(
            "http://testserver/v1/mcp",
            http_client=client,
            terminate_on_close=False,
        ) as (read_stream, write_stream, _get_session_id):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()

    assert {tool.name for tool in tools.tools} >= {
        "headroom_compress",
        "headroom_retrieve",
        "headroom_stats",
    }
```

- [ ] **Step 2: Run the test to verify the current route contract**

Run: `.venv/Scripts/python.exe -m pytest tests/test_proxy/test_mcp_route_order.py -q`

Expected: the new handshake test either passes if the route is already functional or fails with the exact transport/registration defect; preserve the failure output as the implementation target.

- [ ] **Step 3: Add the startup banner regression assertion**

Add a source-level or CLI banner test in the same test module that verifies the banner template contains `/v1/mcp` and both availability branches. The assertion must cover the exact endpoint text emitted for installed and missing MCP dependencies.

- [ ] **Step 4: Implement the banner output**

Add `MCP_STREAMABLE_HTTP_AVAILABLE` to the lazy import tuple in `headroom/cli/proxy.py`. Before the banner `click.echo`, construct:

```python
if MCP_STREAMABLE_HTTP_AVAILABLE:
    mcp_endpoint_line = "  POST /v1/mcp   Streamable HTTP MCP tools"
else:
    mcp_endpoint_line = "  /v1/mcp       MCP tools unavailable (install headroom-ai[proxy])"
```

Insert `{mcp_endpoint_line}` under `Endpoints:`. Keep the existing provider routing lines unchanged.

- [ ] **Step 5: Run MCP and banner tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_proxy/test_mcp_route_order.py tests/test_ccr_mcp_http.py -q`

Expected: MCP initialization, tool discovery, route ordering, and banner assertions pass; tests requiring a missing optional dependency skip according to existing project conventions.

- [ ] **Step 6: Commit the task**

```bash
git add headroom/cli/proxy.py tests/test_proxy/test_mcp_route_order.py
git commit -m "fix: expose and test proxy mcp endpoint"
```

### Task 3: Run final verification

**Files:**
- No source changes.

- [ ] **Step 1: Run the combined relevant suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_transforms_content_router.py tests/test_transforms/test_content_router.py tests/test_builtin_compressor_adapters.py tests/test_proxy/test_mcp_route_order.py tests/test_ccr_mcp_http.py -q`

Expected: all relevant tests pass or optional MCP tests skip without failures.

- [ ] **Step 2: Run syntax/lint and diff checks**

Run: `.venv/Scripts/ruff.exe check --select E9,F63,F7,F82 headroom/cli/proxy.py headroom/transforms/content_router.py tests/test_proxy/test_mcp_route_order.py`

Run: `rtk git diff --check`

Expected: both commands complete without errors.
