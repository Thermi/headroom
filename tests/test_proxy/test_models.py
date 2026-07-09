"""Tests for headroom.proxy.models."""

from __future__ import annotations

from datetime import datetime

import pytest

from headroom.proxy.models import CacheEntry, ProxyConfig, RateLimitState, RequestLog


class TestRequestLog:
    """Verify RequestLog dataclass construction and defaults."""

    def test_minimal_construction(self):
        log = RequestLog(
            request_id="req_001",
            timestamp="2024-01-01T00:00:00",
            provider="anthropic",
            model="claude-3-opus",
            input_tokens_original=1000,
            input_tokens_optimized=800,
            output_tokens=500,
            tokens_saved=200,
            savings_percent=20.0,
            optimization_latency_ms=150.0,
            total_latency_ms=2500.0,
            tags={"env": "test"},
            cache_hit=False,
            transforms_applied=["router:bash:lossless_search"],
        )
        assert log.request_id == "req_001"
        assert log.provider == "anthropic"
        assert log.tokens_saved == 200
        assert log.savings_percent == 20.0

    def test_optional_fields_default_to_none(self):
        log = RequestLog(
            request_id="req_002",
            timestamp="2024-01-01T00:00:00",
            provider="openai",
            model="gpt-4o",
            input_tokens_original=500,
            input_tokens_optimized=500,
            output_tokens=None,
            tokens_saved=0,
            savings_percent=0.0,
            optimization_latency_ms=50.0,
            total_latency_ms=None,
            tags={},
            cache_hit=False,
            transforms_applied=[],
        )
        assert log.waste_signals is None
        assert log.request_messages is None
        assert log.compressed_messages is None
        assert log.response_content is None
        assert log.error is None
        assert log.turn_id is None

    def test_full_construction_with_all_fields(self):
        log = RequestLog(
            request_id="req_003",
            timestamp="2024-01-01T00:00:00",
            provider="gemini",
            model="gemini-1.5-pro",
            input_tokens_original=2000,
            input_tokens_optimized=1000,
            output_tokens=800,
            tokens_saved=1000,
            savings_percent=50.0,
            optimization_latency_ms=200.0,
            total_latency_ms=3000.0,
            tags={"user": "alice"},
            cache_hit=True,
            transforms_applied=["router:protected:system_message"],
            waste_signals={"redundant_system_prompt": 2},
            request_messages=[{"role": "user", "content": "hi"}],
            compressed_messages=[{"role": "user", "content": "hi"}],
            response_content="Hello!",
            error=None,
            turn_id="turn_abc123",
        )
        assert log.cache_hit is True
        assert log.waste_signals == {"redundant_system_prompt": 2}
        assert log.turn_id == "turn_abc123"
        assert len(log.request_messages) == 1


class TestCacheEntry:
    """Verify CacheEntry dataclass."""

    def test_minimal_construction(self):
        now = datetime.now()
        entry = CacheEntry(
            response_body=b'{"ok": true}',
            response_headers={"content-type": "application/json"},
            created_at=now,
            ttl_seconds=3600,
        )
        assert entry.response_body == b'{"ok": true}'
        assert entry.ttl_seconds == 3600
        assert entry.hit_count == 0
        assert entry.tokens_saved_per_hit == 0

    def test_with_custom_hit_count(self):
        now = datetime.now()
        entry = CacheEntry(
            response_body=b"data",
            response_headers={},
            created_at=now,
            ttl_seconds=600,
            hit_count=5,
            tokens_saved_per_hit=100,
        )
        assert entry.hit_count == 5
        assert entry.tokens_saved_per_hit == 100


class TestRateLimitState:
    """Verify RateLimitState dataclass."""

    def test_construction(self):
        state = RateLimitState(tokens=100.0, last_update=1234567890.0)
        assert state.tokens == 100.0
        assert state.last_update == 1234567890.0


class TestProxyConfig:
    """Verify ProxyConfig construction, defaults, and validation."""

    def test_defaults(self):
        config = ProxyConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 8787
        assert config.backend == "anthropic"
        assert config.optimize is True
        assert config.mode == "token"

    def test_custom_values(self):
        config = ProxyConfig(
            host="0.0.0.0",
            port=8080,
            backend="litellm",
            optimize=False,
            mode="cache",
        )
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.backend == "litellm"
        assert config.optimize is False
        assert config.mode == "cache"

    def test_provider_api_overrides(self):
        config = ProxyConfig(
            anthropic_api_url="https://anthropic.internal",
            openai_api_url="https://openai.internal",
            gemini_api_url="https://gemini.internal",
        )
        overrides = config.provider_api_overrides
        assert overrides.anthropic == "https://anthropic.internal"
        assert overrides.openai == "https://openai.internal"
        assert overrides.gemini == "https://gemini.internal"
        assert overrides.cloudcode is None
        assert overrides.vertex is None

    def test_provider_api_overrides_defaults(self):
        config = ProxyConfig()
        overrides = config.provider_api_overrides
        assert overrides.anthropic is None
        assert overrides.openai is None
        assert overrides.gemini is None
        assert overrides.cloudcode is None
        assert overrides.vertex is None

    def test_retry_enabled_validates_max_attempts(self):
        with pytest.raises(ValueError, match="retry_max_attempts must be >= 1"):
            ProxyConfig(retry_enabled=True, retry_max_attempts=0)

    def test_retry_disabled_accepts_zero_attempts(self):
        config = ProxyConfig(retry_enabled=False, retry_max_attempts=0)
        assert config.retry_enabled is False

    def test_smart_routing_deprecated_initvar(self):
        config = ProxyConfig(smart_routing=True)
        assert config.smart_routing is None

    def test_memory_defaults(self):
        config = ProxyConfig()
        assert config.memory_backend == "local"
        assert config.memory_storage_mode == "project"
        assert config.memory_mode == "auto_tail"

    def test_rate_limits_defaults(self):
        config = ProxyConfig()
        assert config.rate_limit_enabled is True
        assert config.rate_limit_requests_per_minute == 60
        assert config.rate_limit_tokens_per_minute == 100000

    def test_cache_settings(self):
        config = ProxyConfig(cache_enabled=False, cache_ttl_seconds=7200, cache_max_entries=500)
        assert config.cache_enabled is False
        assert config.cache_ttl_seconds == 7200
        assert config.cache_max_entries == 500
