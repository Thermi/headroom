# Headroom MCP Configurations

The proxy runs inside Docker and exposes a built-in MCP SSE endpoint.
Point your MCP-compatible agent at it with the `url` field — no subprocess needed.

## Where to place the file

| Agent        | Config file                                  | Copy from                     |
|-------------|----------------------------------------------|-------------------------------|
| Claude Code | `~/.claude/mcp.json`                         | `claude/mcp.json`             |
| Cursor      | `~/.cursor/mcp.json`                         | `cursor/mcp.json`             |
| Codex       | `~/.codex/config.toml`                       | `codex/config.toml`           |
| Windsurf    | `~/.codeium/windsurf/mcp_config.json`        | `windsurf/mcp_config.json`    |

Or register via the CLI (Claude Code only):

```bash
headroom mcp install
```

## What you get

A single `headroom` MCP server at `http://localhost:10001/v1/mcp`
exposing:

- **headroom_retrieve** — get original uncompressed content by hash
- **headroom_compress** — compress content on demand
- **headroom_stats** — session compression statistics

The transport uses **Streamable HTTP** (RFC 2119, `mcp.server.streamable_http`)
instead of plain SSE.  GET establishes an SSE stream for server-to-client
messages, POST carries JSON-RPC requests, DELETE tears down the session.
From the agent's perspective it behaves identically to an SSE-based server.

## Docker networking

If the agent runs inside a container alongside the proxy, use the
Docker service name `proxy` instead of `localhost`:

```json
{"mcpServers": {"headroom": {"url": "http://proxy:10001/v1/mcp"}}}
```
