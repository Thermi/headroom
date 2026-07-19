"""Auto-detect model pricing from upstream API responses.

Providers like OpenRouter return per-request pricing in response headers
or body fields. This module inspects responses and caches detected pricing
into litellm.model_cost so future requests for the same model skip the
lookup.

The pricing can vary per backend even for the same model name (e.g.
OpenRouter routes 'moonshotai/kimi-k3' to different providers with
different prices), so detected pricing is always the most authoritative
source.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("headroom.proxy.pricing_detect")


def feed_response(
    model: str,
    *,
    response_body: dict[str, Any] | None = None,
    response_headers: dict[str, str] | None = None,
    provider: str = "",
) -> None:
    """Feed a response into the auto-detection pipeline.

    Call from handler paths after the upstream response is received.
    Detected pricing is injected into litellm.model_cost.
    Best-effort; never raises.
    """
    try:
        entry = _extract_pricing(model, response_body, response_headers, provider)
        if entry is None:
            logger.debug(
                "No pricing found in response for model=%s (body=%s, headers=%s)",
                model,
                bool(response_body),
                bool(response_headers),
            )
            return
        _inject_pricing(model, entry, provider)
    except Exception:
        logger.debug("Auto-detect pricing failed for model=%s", model, exc_info=True)


def _extract_pricing(
    model: str,
    response_body: dict[str, Any] | None,
    response_headers: dict[str, str] | None,
    provider: str,
) -> dict[str, Any] | None:
    """Extract litellm-compatible pricing entry from response data.

    Tries multiple known patterns:
    1. OpenRouter body usage.prompt_tokens_details / usage.cost
    2. Response headers (x-pricing, x-cost, etc.)
    3. Generic usage block with cost field
    """
    if response_body:
        usage = response_body.get("usage")
        if isinstance(usage, dict):
            entry = _try_usage_block(model, usage, provider)
            if entry:
                return entry

    if response_headers:
        entry = _try_pricing_headers(model, response_headers, provider)
        if entry:
            return entry

    return None


def _try_usage_block(
    model: str,
    usage: dict[str, Any],
    provider: str,
) -> dict[str, Any] | None:
    """Try to derive pricing from a usage block with cost info.

    Some providers include explicit cost in the usage block:
    {"prompt_tokens": 1000, "completion_tokens": 500, "cost": 0.0045}

    From this we can compute per-token costs.
    """
    prompt_tokens = _as_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
    completion_tokens = _as_int(usage.get("completion_tokens") or usage.get("output_tokens"))
    cost = _as_float(usage.get("cost") or usage.get("total_cost"))

    logger.debug(
        "Trying usage block for model=%s: prompt_tokens=%s, completion_tokens=%s, cost=%s",
        model,
        prompt_tokens,
        completion_tokens,
        cost,
    )

    if not (prompt_tokens and completion_tokens and cost and cost > 0):
        return None

    # Assume cost is split proportionally by token count
    total_tokens = prompt_tokens + completion_tokens
    if total_tokens <= 0:
        return None

    input_cost_per_token = (cost * prompt_tokens / total_tokens) / prompt_tokens
    output_cost_per_token = (cost * completion_tokens / total_tokens) / completion_tokens

    logger.info(
        "Decoded pricing from usage block for model=%s: input=$%.8f/tok, output=$%.8f/tok "
        "(total cost=$%s)",
        model,
        input_cost_per_token,
        output_cost_per_token,
        cost,
    )

    return {
        "input_cost_per_token": input_cost_per_token,
        "output_cost_per_token": output_cost_per_token,
        "litellm_provider": provider or "openrouter",
        "mode": "chat",
    }


def _try_pricing_headers(
    model: str,
    headers: dict[str, str],
    provider: str,
) -> dict[str, Any] | None:
    """Try known pricing header patterns.

    Known patterns:
    - x-openrouter-pricing: JSON with input/output per-token costs
    - x-pricing-input / x-pricing-output: per-1M prices
    - x-cost-input / x-cost-output: per-1M prices
    """
    import json

    # OpenRouter-style: single JSON header
    for header_name in ("x-openrouter-pricing", "x-pricing", "x-cost"):
        raw = headers.get(header_name)
        if not raw:
            continue
        logger.debug(
            "Trying pricing header=%s raw=%s for model=%s",
            header_name,
            str(raw)[:200],
            model,
        )
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(data, dict):
                input_cost_per_1m = _as_float(data.get("input", data.get("input_cost", 0))) or 0.0
                output_cost_per_1m = _as_float(data.get("output", data.get("output_cost", 0))) or 0.0
                result : dict[str, Any] = {
                    "input_cost_per_token": input_cost_per_1m / 1_000_000,
                    "output_cost_per_token": output_cost_per_1m / 1_000_000,
                    "litellm_provider": provider or "openrouter",
                    "mode": "chat",
                }
                logger.info(
                    "Decoded pricing header %s for model=%s: input=$%s/M, output=$%s/M",
                    header_name,
                    model,
                    input_cost_per_1m,
                    output_cost_per_1m,
                )
                return result
        except (json.JSONDecodeError, TypeError):
            logger.debug("Failed to parse %s header as JSON for model=%s", header_name, model)
            continue

    # Per-header price signals (per-1M prices in USD)
    input_price = _as_float(headers.get("x-pricing-input") or headers.get("x-cost-input"))
    output_price = _as_float(headers.get("x-pricing-output") or headers.get("x-cost-output"))

    if input_price and output_price:
        return {
            "input_cost_per_token": float(input_price) / 1_000_000,
            "output_cost_per_token": float(output_price) / 1_000_000,
            "litellm_provider": provider or "openrouter",
            "mode": "chat",
        }

    return None


def _as_int(value: Any) -> int | None:
    """Coerce value to int, return None if invalid."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    """Coerce value to float, return None if invalid."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Cache of (model, provider) -> already injected, to avoid re-injection
# on every request for the same model.
_injected_models: set[str] = set()


def _inject_pricing(model: str, entry: dict[str, Any], provider: str) -> None:
    """Inject detected pricing into litellm.model_cost if not already cached."""
    cache_key = f"{model}::{provider or 'any'}"
    if cache_key in _injected_models:
        return
    try:
        import litellm as _litellm
    except ImportError:
        return

    # Write both bare and provider-prefixed keys so both resolution paths
    # find the pricing regardless of how the model name arrives.
    if model not in _litellm.model_cost:
        _litellm.model_cost[model] = entry
    if "/" in model:
        prefixed_key = model
    elif provider:
        prefixed_key = f"{provider}/{model}"
    else:
        prefixed_key = ""
    if prefixed_key and prefixed_key not in _litellm.model_cost:
        _litellm.model_cost[prefixed_key] = entry

    _injected_models.add(cache_key)
    logger.info(
        "Auto-detected pricing for model=%s (input=%s/M, output=%s/M)",
        model,
        f"${entry.get('input_cost_per_token', 0) * 1_000_000:.2f}",
        f"${entry.get('output_cost_per_token', 0) * 1_000_000:.2f}",
    )
