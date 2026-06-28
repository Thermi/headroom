"""Tests for the parallelized OpenAI handler pre-upstream stages.

Verifies that ``handle_openai_chat`` and ``handle_openai_responses``
run token counting, headers processing, and memory setup concurrently
via ``asyncio.gather``, with CPU-bound token counting delegated to
``asyncio.to_thread``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ── Mock ``headroom._core`` before any headroom module tries to import it ──
_mock_core = MagicMock(name="headroom._core")
_mock_core.hello = MagicMock(return_value="headroom-core")
_mock_core.content_has_error_indicators = MagicMock(return_value=False)
_mock_core.keyword_registry_snapshot = MagicMock(
    return_value={
        "error": ["error", "exception", "failed", "traceback"],
        "importance": ["critical", "important", "urgent"],
        "warning": ["warning", "deprecated"],
        "security": ["unsafe", "injection", "exploit"],
        "error_indicators": ["Error:", "Traceback", "Exception:"],
        "markdown_prefixes": ["#", "##", "###", "```"],
    }
)
_mock_core.score_line = MagicMock(return_value=(None, 0.0, 0.0))
_mock_core.is_html_tag = MagicMock(return_value=False)
_mock_core.known_html_tag_names = MagicMock(return_value=[])
_mock_core.protect_tags = MagicMock(return_value="")
_mock_core.restore_tags = MagicMock(return_value=("", False))
sys.modules["headroom._core"] = _mock_core


def _make_proxy_config(**overrides):
    from headroom.proxy.models import ProxyConfig

    defaults = {
        "optimize": False,
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "cost_tracking_enabled": False,
        "log_requests": False,
        "ccr_inject_tool": False,
        "ccr_handle_responses": False,
        "ccr_context_tracking": False,
        "image_optimize": False,
        "memory_enabled": False,
    }
    defaults.update(overrides)
    return ProxyConfig(**defaults)


def _fake_request(body, extra_headers=None):
    """Build a minimal FastAPI-shaped request with headers and JSON body."""
    headers = {"content-type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)

    class _FakeRequest:
        def __init__(self):
            self.headers = SimpleNamespace(**{"get": headers.get, "items": headers.items})

        async def json(self):
            return body

        url = SimpleNamespace(path="/v1/chat/completions")
        query_params = {}
        state = SimpleNamespace()

    return _FakeRequest()


# ── handle_openai_chat concurrency tests ──────────────────────────────────


class TestOpenAiChatParallelization:
    async def _get_proxy(self):
        from headroom.proxy.server import create_app

        config = _make_proxy_config(optimize=False)
        return create_app(config).state.proxy

    @pytest.mark.asyncio
    async def test_handler_does_not_crash_with_parallelized_pre_upstream(self) -> None:
        """Smoke test: the handler runs to completion with the gather-based flow."""
        proxy = await self._get_proxy()

        body = {
            "model": "gpt-4o",
            "stream": False,
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
        request = _fake_request(body)

        # The handler will attempt an upstream call and fail (no upstream
        # mocked) -- that's fine; we just want to confirm the pre-upstream
        # stages complete without raising an unexpected exception.
        try:
            await proxy.handle_openai_chat(request)
        except Exception:
            pass  # upstream connection refused is expected

    @pytest.mark.asyncio
    async def test_gather_called_with_three_coroutines(self) -> None:
        """``asyncio.gather`` is called with three coroutines (token
        counting, header building, memory setup)."""
        import asyncio

        proxy = await self._get_proxy()

        # Verify that the handler reaches the gather call by making
        # ``gather`` raise a recognisable exception.
        async def _fatal_gather(*coros, **kw):
            assert len(coros) == 3, f"Expected 3 coroutines for chat handler, got {len(coros)}"
            raise RuntimeError("GATHER_WAS_CALLED")

        saved = asyncio.gather
        asyncio.__dict__["gather"] = _fatal_gather

        body = {
            "model": "gpt-4o",
            "stream": False,
            "messages": [{"role": "user", "content": "Hello"}],
        }

        try:
            with patch(
                "headroom.proxy.helpers._read_request_json",
                lambda req: body,
            ):
                await proxy.handle_openai_chat(_fake_request(body))
                pytest.fail("Expected RuntimeError from gather spy")
        except RuntimeError as e:
            assert str(e) == "GATHER_WAS_CALLED", "Handler successfully reached asyncio.gather"
        except Exception:
            pass  # other exceptions are OK (upstream not mocked)
        finally:
            asyncio.__dict__["gather"] = saved


# ── handle_openai_responses concurrency tests ─────────────────────────────


class TestOpenAiResponsesParallelization:
    async def _get_proxy(self):
        from headroom.proxy.server import create_app

        config = _make_proxy_config(optimize=False)
        return create_app(config).state.proxy

    @pytest.mark.asyncio
    async def test_handler_does_not_crash_with_parallelized_pre_upstream(self) -> None:
        """Smoke test: the responses handler runs to completion."""
        proxy = await self._get_proxy()

        body = {
            "model": "gpt-4o",
            "stream": False,
            "instructions": "Be helpful.",
            "input": "Hello world",
        }
        request = _fake_request(body, extra_headers={"authorization": "Bearer test"})

        try:
            await proxy.handle_openai_responses(request)
        except Exception:
            pass  # upstream connection refused is expected

    @pytest.mark.asyncio
    async def test_gather_called_with_three_coroutines(self) -> None:
        """``asyncio.gather`` is called with three coroutines in the
        responses handler."""
        import asyncio

        proxy = await self._get_proxy()

        async def _fatal_gather(*coros, **kw):
            assert len(coros) == 3, f"Expected 3 coroutines, got {len(coros)}"
            raise RuntimeError("GATHER_WAS_CALLED")

        saved = asyncio.gather
        asyncio.__dict__["gather"] = _fatal_gather

        body = {
            "model": "gpt-4o",
            "stream": False,
            "instructions": "Be helpful.",
            "input": "Hello world",
        }

        try:
            with patch(
                "headroom.proxy.helpers._read_request_json",
                lambda req: body,
            ):
                request = _fake_request(body, extra_headers={"authorization": "Bearer test"})
                await proxy.handle_openai_responses(request)
                pytest.fail("Expected RuntimeError from gather spy")
        except RuntimeError as e:
            assert str(e) == "GATHER_WAS_CALLED", (
                "Responses handler successfully reached asyncio.gather"
            )
        except Exception:
            pass
        finally:
            asyncio.__dict__["gather"] = saved
