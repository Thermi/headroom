"""Provider model metadata route helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlparse

from fastapi import Request
from fastapi.responses import Response

from headroom.providers.codex.model_metadata import handle_chatgpt_model_metadata


@dataclass(frozen=True, slots=True)
class ModelMetadataEndpoint:
    """OpenAI-compatible model metadata endpoint shape."""

    route_path: str
    upstream_path: str
    passthrough_sub_path: str = "models"


MODEL_METADATA_LIST_ENDPOINT = ModelMetadataEndpoint("/v1/models", "/backend-api/models")

_OPENROUTER_MODEL_INFO: dict[str, dict[str, Any]] = {}


def _is_openrouter_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


def _openrouter_model_payload(model: dict[str, Any]) -> dict[str, Any]:
    """Convert one OpenRouter model entry to an OpenAI-compatible entry."""
    payload = dict(model)
    model_id = payload.get("id")
    payload["object"] = "model"
    payload.setdefault("owned_by", "openrouter")
    if not isinstance(model_id, str) or not model_id:
        payload.pop("id", None)
    return payload


def openrouter_model_info(model: dict[str, Any]) -> dict[str, Any]:
    """Extract Headroom's internal metadata from an OpenRouter model entry."""
    info: dict[str, Any] = {}
    context_length = model.get("context_length")
    if isinstance(context_length, int | float) and context_length > 0:
        info["context_limit"] = int(context_length)

    architecture = model.get("architecture")
    if isinstance(architecture, dict):
        for source, target in (
            ("tokenizer", "tokenizer"),
            ("input_modalities", "input_modalities"),
            ("output_modalities", "output_modalities"),
        ):
            value = architecture.get(source)
            if value is not None:
                info[target] = value

    top_provider = model.get("top_provider")
    if isinstance(top_provider, dict):
        max_output_tokens = top_provider.get("max_completion_tokens")
        if isinstance(max_output_tokens, int | float) and max_output_tokens > 0:
            info["max_output_tokens"] = int(max_output_tokens)

    pricing = model.get("pricing")
    if isinstance(pricing, dict):
        for source, target in (
            ("prompt", "input_cost_per_token"),
            ("completion", "output_cost_per_token"),
            ("cache_read", "cache_read_input_token_cost"),
        ):
            value = pricing.get(source)
            if isinstance(value, str | int | float):
                try:
                    info[target] = float(value)
                except (TypeError, ValueError):
                    pass
    return info


def register_openrouter_model_info(data: Any) -> None:
    """Register model metadata from an OpenRouter catalogue response."""
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return
    for model in models:
        if isinstance(model, dict) and isinstance(model.get("id"), str):
            info = openrouter_model_info(model)
            if info:
                _OPENROUTER_MODEL_INFO[model["id"]] = info


def get_openrouter_model_info(model: str) -> dict[str, Any] | None:
    """Return metadata previously discovered for an OpenRouter model."""
    return _OPENROUTER_MODEL_INFO.get(model)


def translate_openrouter_models_response(data: Any) -> dict[str, Any] | None:
    """Translate OpenRouter's ``data[]`` catalogue to an OpenAI model list."""
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return None
    register_openrouter_model_info(data)
    translated = [
        _openrouter_model_payload(model)
        for model in models
        if isinstance(model, dict) and isinstance(model.get("id"), str) and model["id"]
    ]
    return {"object": "list", "data": translated}


def clear_openrouter_model_info() -> None:
    """Clear discovered metadata, primarily for isolated tests and reloads."""
    _OPENROUTER_MODEL_INFO.clear()


async def _fetch_openrouter_models(proxy: Any, request: Request, base_url: str) -> Response | None:
    """Fetch and translate OpenRouter's native model catalogue."""
    assert proxy.http_client is not None
    headers = dict(request.headers.items())
    headers.pop("host", None)
    headers.pop("accept-encoding", None)
    parsed_url = urlparse(base_url)
    url = f"{parsed_url.scheme}://{parsed_url.netloc}/api/v1/models"
    try:
        upstream = await proxy.http_client.get(url, headers=headers, timeout=30.0)
        if upstream.status_code >= 400:
            return None
        upstream_data = upstream.json()
        translated = translate_openrouter_models_response(upstream_data)
        if translated is None:
            return None
        return Response(
            content=json.dumps(translated),
            status_code=upstream.status_code,
            media_type="application/json",
        )
    except Exception:
        return None


def model_metadata_get_endpoint(model_id: str) -> ModelMetadataEndpoint:
    """Return the single-model metadata endpoint for ``model_id``."""
    return ModelMetadataEndpoint(
        "/v1/models/{model_id}",
        f"/backend-api/models/{model_id}",
    )


async def handle_model_metadata_endpoint(
    proxy: Any,
    request: Request,
    *,
    endpoint: ModelMetadataEndpoint,
    provider_api_base_url: str,
    provider_name: str,
) -> Response:
    """Handle OpenAI-compatible model metadata with Codex ChatGPT-auth support."""
    assert proxy.http_client is not None
    chatgpt_response = await handle_chatgpt_model_metadata(
        proxy.http_client,
        request,
        endpoint.upstream_path,
    )
    if chatgpt_response is not None:
        return chatgpt_response

    if endpoint == MODEL_METADATA_LIST_ENDPOINT and _is_openrouter_url(provider_api_base_url):
        openrouter_response = await _fetch_openrouter_models(proxy, request, provider_api_base_url)
        if openrouter_response is not None:
            return openrouter_response

    return cast(
        Response,
        await proxy.handle_passthrough(
            request,
            provider_api_base_url,
            endpoint.passthrough_sub_path,
            provider_name,
        ),
    )
