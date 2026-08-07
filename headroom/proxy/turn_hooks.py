"""Optional extension hooks around one buffered model turn."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)
CallModel = Callable[[list[dict[str, Any]]], Awaitable[dict[str, Any]]]


@dataclass
class TurnContext:
    provider: str
    model: str
    messages: list[dict[str, Any]]
    tools: Any = None
    config: Any = None


@runtime_checkable
class TurnHook(Protocol):
    name: str

    def on_request(self, ctx: TurnContext) -> None: ...

    async def on_response(
        self, ctx: TurnContext, response: dict[str, Any], call_model: CallModel
    ) -> dict[str, Any] | None: ...


_hooks: list[TurnHook] = []


def register_turn_hook(hook: TurnHook) -> None:
    _hooks.append(hook)


def registered_turn_hooks() -> list[TurnHook]:
    return list(_hooks)


def clear_turn_hooks() -> None:
    _hooks.clear()


def run_request_hooks(ctx: TurnContext) -> None:
    for hook in _hooks:
        callback = getattr(hook, "on_request", None)
        if callback is None:
            continue
        try:
            callback(ctx)
        except Exception:
            logger.exception("turn hook %r on_request failed", hook)


async def run_response_hooks(
    ctx: TurnContext, response: dict[str, Any], call_model: CallModel
) -> dict[str, Any]:
    current = response
    for hook in _hooks:
        callback = getattr(hook, "on_response", None)
        if callback is None:
            continue
        try:
            replacement = await callback(ctx, current, call_model)
        except Exception:
            logger.exception("turn hook %r on_response failed", hook)
            continue
        if replacement is not None:
            current = replacement
    return current
