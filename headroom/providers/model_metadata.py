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


def translate_openrouter_models_response(data: Any) -> dict[str, Any] | None:
    """Translate OpenRouter's ``data[]`` catalogue to an OpenAI model list."""
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return None
    translated = [
        _openrouter_model_payload(model)
        for model in models
        if isinstance(model, dict) and isinstance(model.get("id"), str) and model["id"]
    ]
    return {"object": "list", "data": translated}


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
        translated = translate_openrouter_models_response(upstream.json())
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
