"""Runtime helpers for OpenCode integrations."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping

from headroom.providers.claude import proxy_base_url as claude_proxy_base_url


_OPENCODE_CONFIG_TEMPLATE = {
    "provider": {
        "headroom": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Headroom Proxy",
            "options": {
                "baseURL": "http://127.0.0.1:PORT/v1",
            },
            "models": {
                "claude-sonnet-4-6": {
                    "name": "Claude Sonnet 4.6",
                    "limit": {"context": 200000, "output": 16384},
                },
                "claude-opus-4-6": {
                    "name": "Claude Opus 4.6",
                    "limit": {"context": 200000, "output": 16384},
                },
                "claude-haiku-4-5-20251001": {
                    "name": "Claude Haiku 4.5",
                    "limit": {"context": 200000, "output": 8192},
                },
                "gpt-4o": {
                    "name": "GPT-4o",
                    "limit": {"context": 128000, "output": 16384},
                },
                "gpt-4.1": {
                    "name": "GPT-4.1",
                    "limit": {"context": 1048576, "output": 32768},
                },
            },
        },
    },
    "model": "headroom/claude-sonnet-4-6",
}


def proxy_base_url(port: int) -> str:
    """Return the local proxy base URL used by OpenAI-compatible integrations."""
    return f"http://127.0.0.1:{port}/v1"


def build_launch_env(
    port: int, environ: Mapping[str, str] | None = None
) -> tuple[dict[str, str], list[str]]:
    """Build environment variables for OpenCode through the local proxy.

    Injects a ``headroom`` provider via ``OPENCODE_CONFIG_CONTENT`` (the
    highest-priority config source in OpenCode, merged with the user's
    existing ``opencode.json``). Also sets ``ANTHROPIC_BASE_URL`` as a
    fallback for OpenCode versions that support it.
    """
    env = dict(environ or os.environ)
    display_lines: list[str] = []

    config = json.loads(json.dumps(_OPENCODE_CONFIG_TEMPLATE))
    config["provider"]["headroom"]["options"]["baseURL"] = f"http://127.0.0.1:{port}/v1"
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
    display_lines.append("OPENCODE_CONFIG_CONTENT=<headroom provider config>")

    anthropic_url = claude_proxy_base_url(port)
    env["ANTHROPIC_BASE_URL"] = anthropic_url
    display_lines.append(f"ANTHROPIC_BASE_URL={anthropic_url}")

    return env, display_lines
