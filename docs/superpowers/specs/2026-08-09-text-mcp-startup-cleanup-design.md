# Text and MCP Startup Cleanup

## Scope

Fix the text compressor registry contract, reduce net-cost log noise, expose the
MCP endpoint in the startup banner, and verify the internal Streamable HTTP MCP
interface end to end.

## Design

The Kompress registry adapter will unpack the two-value result returned by
`_try_ml_compressor` and return only the compressed string required by the
`CompressOutput.content` contract.

Both net-cost diagnostic records will remain available, but both will use the
logger's `DEBUG` level. Net-cost calculation and admission behavior will not
change.

The CLI startup banner will always include an MCP endpoint line under
`Endpoints:`. When the Streamable HTTP MCP dependency is available, it will
identify `/v1/mcp` as the built-in MCP tools endpoint. When unavailable, it
will state that `/v1/mcp` requires the MCP dependency rather than silently
omitting the route.

The proxy MCP route test will use the MCP client Streamable HTTP transport over
an in-process ASGI transport. It will initialize a session and list tools,
verifying the built-in compression, retrieval, and stats tools are exposed.

## Testing

- Reproduce and prevent the tuple-to-text failure through the existing router
  compression tests.
- Assert net-cost records are absent from `INFO` and present at `DEBUG`.
- Assert the banner contains `/v1/mcp` and its availability status.
- Assert an in-process MCP client can initialize and discover the built-in tools.
