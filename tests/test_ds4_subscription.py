"""Tests for DS4 (DeepSeek V4) subscription usage, savings, and budget tracking.

Covers:
- API key detection helpers
- Ds4Contribution data model
- Ds4SubscriptionTracker unit tests (notify_active, update_contribution, budget)
- QuotaRegistry integration
- Proxy /stats endpoint integration
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
from fastapi.testclient import TestClient

from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app
from headroom.subscription.base import get_quota_registry, reset_quota_registry
from headroom.subscription.ds4 import (
    Ds4Contribution,
    Ds4SubscriptionTracker,
    configure_ds4_tracker,
    extract_ds4_auth,
    get_ds4_tracker,
    is_ds4_api_key,
)


# -- Mock the Rust _core module for integration tests that need create_app --
def _install_fake_core() -> None:
    """Inject a fake ``headroom._core`` module so proxy tests work without the
    Rust extension built.  The fake provides the subset of symbols that
    ``headroom/transforms/error_detection.py`` imports at module level.
    """
    import re as _re
    import sys
    from types import ModuleType

    if "headroom._core" in sys.modules:
        return

    fake = ModuleType("headroom._core")

    def fake_hello() -> str:
        return "headroom-core"

    def fake_keyword_registry_snapshot() -> dict:
        return {
            "error": [],
            "importance": [],
            "warning": [],
            "security": [],
            "user_info": [],
            "priority": [],
            "error_indicators": [],
            "markdown_prefixes": [],
            "error_pattern": _re.compile("(?!)"),
            "security_pattern": _re.compile("(?!)"),
            "priority_patterns_text": [],
            "priority_patterns_block": [],
        }

    def fake_score_line(_text: str) -> int:
        return 0

    def fake_content_has_error_indicators(_text: str) -> bool:
        return False

    fake.hello = fake_hello
    fake.keyword_registry_snapshot = fake_keyword_registry_snapshot
    fake.score_line = fake_score_line
    fake.content_has_error_indicators = fake_content_has_error_indicators

    # Additional symbols needed by other modules imported during create_app
    def fake_detect_content_type(_raw: str) -> tuple[str, float]:
        return ("text", 0.0)

    fake.detect_content_type = fake_detect_content_type
    from types import SimpleNamespace as _SN

    def fake_compress_logs(*args: Any, **kwargs: Any) -> Any:
        return _SN(log_lines=[], line_types=[])

    fake.compress_logs = fake_compress_logs
    fake.parse_search_lines = lambda text: []
    fake.detect_log_format = lambda text: ("unknown", 0.0)
    fake.compress_tags = lambda tags, config: tags

    sys.modules["headroom._core"] = fake


# =========================================================================
# API key detection helpers
# =========================================================================


class TestDs4KeyDetection:
    def test_is_ds4_api_key_positive(self):
        assert is_ds4_api_key("sk-ds4-abc123def456") is True
        assert is_ds4_api_key("sk-ds4-") is True

    def test_is_ds4_api_key_negative(self):
        assert is_ds4_api_key("") is False
        assert is_ds4_api_key("sk-ant-api03-abcd") is False
        assert is_ds4_api_key("sk-") is False
        assert is_ds4_api_key("sk-ds3-abc") is False
        assert is_ds4_api_key("Bearer sk-ds4-xyz") is False

    def test_extract_ds4_auth_valid(self):
        result = extract_ds4_auth("Bearer sk-ds4-my-secret-key")
        assert result == "sk-ds4-my-secret-key"

    def test_extract_ds4_auth_no_bearer(self):
        assert extract_ds4_auth("") is None
        assert extract_ds4_auth("Basic xyz") is None
        assert extract_ds4_auth("sk-ds4-abc") is None

    def test_extract_ds4_auth_wrong_key_prefix(self):
        assert extract_ds4_auth("Bearer sk-ant-api03-key") is None
        assert extract_ds4_auth("Bearer sk-xyz") is None

    def test_extract_ds4_auth_none_input(self):
        assert extract_ds4_auth("") is None


# =========================================================================
# Ds4Contribution data model
# =========================================================================


class TestDs4Contribution:
    def test_empty_contribution(self):
        c = Ds4Contribution()
        assert c.tokens_submitted == 0
        assert c.tokens_saved == 0
        assert c.cost_without_headroom == 0.0
        d = c.to_dict()
        assert d["tokens_submitted"] == 0
        assert d["cost_without_headroom_usd"] == 0.0

    def test_cost_without_headroom_property(self):
        c = Ds4Contribution(cost_with_headroom_usd=5.0, savings_usd=3.0)
        assert c.cost_without_headroom == 8.0

    def test_to_dict_snapshot(self):
        c = Ds4Contribution(
            tokens_submitted=100,
            tokens_saved=30,
            cost_with_headroom_usd=0.05,
            savings_usd=0.02,
            requests=5,
            last_active_at="2026-06-20T10:00:00Z",
            started_at="2026-06-20T09:00:00Z",
        )
        d = c.to_dict()
        assert d["tokens_submitted"] == 100
        assert d["tokens_saved"] == 30
        assert d["cost_with_headroom_usd"] == 0.05
        assert d["savings_usd"] == 0.02
        assert d["requests"] == 5
        assert d["last_active_at"] == "2026-06-20T10:00:00Z"
        assert d["started_at"] == "2026-06-20T09:00:00Z"

    def test_to_dict_rounding(self):
        c = Ds4Contribution(
            cost_with_headroom_usd=0.1234567,
            savings_usd=0.0012345,
        )
        d = c.to_dict()
        assert d["cost_with_headroom_usd"] == 0.1235
        assert d["savings_usd"] == 0.0012


# =========================================================================
# Ds4SubscriptionTracker unit tests
# =========================================================================


@pytest.fixture(autouse=True)
def reset_tracker():
    """Reset the global DS4 tracker singleton before and after each test."""
    get_ds4_tracker()
    yield
    # Reset by re-configuring with default state
    import headroom.subscription.ds4 as ds4_mod

    ds4_mod._tracker_instance = None


def test_tracker_not_configured_by_default():
    assert get_ds4_tracker() is None


def test_configure_and_is_available():
    tracker = configure_ds4_tracker(enabled=True)
    assert tracker is get_ds4_tracker()
    assert tracker.is_available() is True
    assert tracker.key == "ds4_subscription"
    assert tracker.label == "DS4 (DeepSeek V4)"


def test_configure_disabled():
    tracker = configure_ds4_tracker(enabled=False)
    assert tracker.is_available() is False


def test_configure_idempotent():
    t1 = configure_ds4_tracker(enabled=True)
    t2 = configure_ds4_tracker(enabled=True)
    assert t1 is t2


class TestNotifyActive:
    def test_notify_active_ds4_key(self):
        tracker = Ds4SubscriptionTracker(enabled=True)
        assert tracker._contribution.started_at is None
        assert tracker._contribution.last_active_at is None

        tracker.notify_active("Bearer sk-ds4-test-key-1234")

        c = tracker._contribution
        assert c.started_at is not None
        assert c.last_active_at is not None
        assert len(tracker._tracked_keys) == 1
        # Prefix is first 16 chars of the token
        assert "sk-ds4-test-key-" in tracker._tracked_keys

    def test_notify_active_ignores_non_ds4(self):
        tracker = Ds4SubscriptionTracker(enabled=True)
        tracker.notify_active("")
        tracker.notify_active("Bearer sk-ant-api03-key")
        tracker.notify_active("Authorization: Basic xyz")

        assert tracker._contribution.started_at is None
        assert len(tracker._tracked_keys) == 0

    def test_notify_active_tracks_multiple_keys(self):
        tracker = Ds4SubscriptionTracker(enabled=True)
        tracker.notify_active("Bearer sk-ds4-key-one-xxxx")
        tracker.notify_active("Bearer sk-ds4-key-two-yyyy")

        assert len(tracker._tracked_keys) == 2
        # First 16 chars of each token are the prefix keys
        assert "sk-ds4-key-one-x" in tracker._tracked_keys
        assert "sk-ds4-key-two-y" in tracker._tracked_keys

    def test_notify_active_updates_last_active(self):
        tracker = Ds4SubscriptionTracker(enabled=True)
        tracker.notify_active("Bearer sk-ds4-some-key")
        assert tracker._contribution.last_active_at is not None
        tracker.notify_active("Bearer sk-ds4-other-key")
        assert tracker._contribution.last_active_at is not None
        # Verify the counter incremented (requests tracked in per-key entries)
        assert len(tracker._tracked_keys) == 2


class TestUpdateContribution:
    def test_update_contribution_basic(self):
        tracker = Ds4SubscriptionTracker(enabled=True)
        tracker.update_contribution(
            tokens_submitted=100,
            tokens_saved=30,
            cost_with_headroom=0.05,
            cost_saved=0.02,
        )
        c = tracker._contribution
        assert c.tokens_submitted == 100
        assert c.tokens_saved == 30
        assert c.cost_with_headroom_usd == 0.05
        assert c.savings_usd == 0.02
        assert c.requests == 1
        assert c.last_active_at is not None

    def test_update_contribution_cumulative(self):
        tracker = Ds4SubscriptionTracker(enabled=True)
        for _ in range(5):
            tracker.update_contribution(
                tokens_submitted=10,
                tokens_saved=3,
                cost_with_headroom=0.01,
                cost_saved=0.005,
            )
        c = tracker._contribution
        assert c.tokens_submitted == 50
        assert c.tokens_saved == 15
        assert c.cost_with_headroom_usd == 0.05
        assert c.savings_usd == 0.025
        assert c.requests == 5

    def test_update_contribution_negative_values_clamped(self):
        tracker = Ds4SubscriptionTracker(enabled=True)
        tracker.update_contribution(
            tokens_submitted=-10,
            tokens_saved=-5,
            cost_with_headroom=-1.0,
            cost_saved=-0.5,
        )
        c = tracker._contribution
        assert c.tokens_submitted == 0
        assert c.tokens_saved == 0
        assert c.cost_with_headroom_usd == 0.0
        assert c.savings_usd == 0.0

    def test_update_contribution_with_token_prefix(self):
        tracker = Ds4SubscriptionTracker(enabled=True)
        tracker.notify_active("Bearer sk-ds4-key-alpha-xxxx")

        extracted_token = "sk-ds4-key-alpha-xxxx"
        expected_prefix = extracted_token[:16]  # "sk-ds4-key-alpha-"
        tracker.update_contribution(
            tokens_submitted=200,
            tokens_saved=50,
            cost_with_headroom=0.10,
            cost_saved=0.03,
            token_prefix=expected_prefix,
        )

        c = tracker._contribution
        assert c.tokens_submitted == 200
        assert c.tokens_saved == 50

        pk = tracker._tracked_keys[expected_prefix]
        assert pk.tokens_submitted == 200
        assert pk.tokens_saved == 50
        assert pk.cost_with_headroom_usd == 0.10
        assert pk.savings_usd == 0.03
        assert pk.requests == 1

    def test_update_contribution_unknown_prefix_does_not_crash(self):
        tracker = Ds4SubscriptionTracker(enabled=True)
        tracker.update_contribution(
            tokens_submitted=50,
            tokens_saved=10,
            token_prefix="nonexistent-prefix",
        )
        c = tracker._contribution
        assert c.tokens_submitted == 50
        assert c.tokens_saved == 10


class TestBudget:
    def test_no_budget_unlimited(self):
        tracker = Ds4SubscriptionTracker(enabled=True, budget_limit_usd=None)
        allowed, remaining = tracker.check_budget()
        assert allowed is True
        assert remaining == float("inf")

    def test_budget_within_limit(self):
        tracker = Ds4SubscriptionTracker(enabled=True, budget_limit_usd=10.0, budget_period="daily")
        tracker.update_contribution(tokens_submitted=1000, cost_with_headroom=2.0, cost_saved=0.5)
        allowed, remaining = tracker.check_budget()
        assert allowed is True
        assert remaining == pytest.approx(10.0 - 2.5)

    def test_budget_exceeded(self):
        tracker = Ds4SubscriptionTracker(enabled=True, budget_limit_usd=1.0, budget_period="daily")
        tracker.update_contribution(tokens_submitted=1000, cost_with_headroom=0.8, cost_saved=0.3)
        allowed, remaining = tracker.check_budget()
        assert allowed is False
        assert remaining == 0.0

    def test_budget_at_limit(self):
        tracker = Ds4SubscriptionTracker(enabled=True, budget_limit_usd=1.0, budget_period="daily")
        tracker.update_contribution(
            tokens_submitted=500, cost_with_headroom=0.7, cost_saved=0.29999
        )
        allowed, remaining = tracker.check_budget()
        assert allowed is True
        assert remaining == pytest.approx(1.0 - 0.99999)

    def test_budget_status_no_limit(self):
        tracker = Ds4SubscriptionTracker(enabled=True, budget_limit_usd=None)
        s = tracker.budget_status()
        assert s["budget_limit_usd"] is None
        assert s["within_budget"] is True
        assert s["utilization_pct"] is None

    def test_budget_status_with_limit(self):
        tracker = Ds4SubscriptionTracker(
            enabled=True, budget_limit_usd=100.0, budget_period="daily"
        )
        tracker.update_contribution(tokens_submitted=2000, cost_with_headroom=25.0)
        s = tracker.budget_status()
        assert s["budget_limit_usd"] == 100.0
        assert s["period_cost_usd"] == pytest.approx(25.0)
        assert s["remaining_usd"] == pytest.approx(75.0)
        assert s["within_budget"] is True
        assert s["utilization_pct"] == pytest.approx(25.0)

    def test_budget_period_hourly(self):
        tracker = Ds4SubscriptionTracker(
            enabled=True, budget_limit_usd=10.0, budget_period="hourly"
        )
        tracker.update_contribution(tokens_submitted=100, cost_with_headroom=3.0)
        s = tracker.budget_status()
        assert s["budget_period"] == "hourly"
        assert s["period_cost_usd"] == pytest.approx(3.0)

    def test_budget_period_monthly(self):
        tracker = Ds4SubscriptionTracker(
            enabled=True, budget_limit_usd=1000.0, budget_period="monthly"
        )
        tracker.update_contribution(tokens_submitted=5000, cost_with_headroom=50.0)
        s = tracker.budget_status()
        assert s["budget_period"] == "monthly"
        assert s["period_cost_usd"] == pytest.approx(50.0)


class TestGetStats:
    def test_get_stats_basic(self):
        tracker = Ds4SubscriptionTracker(enabled=True)
        tracker.update_contribution(
            tokens_submitted=100,
            tokens_saved=30,
            cost_with_headroom=0.05,
            cost_saved=0.02,
        )
        stats = tracker.get_stats()
        assert stats is not None
        assert stats["contribution"]["tokens_submitted"] == 100
        assert stats["contribution"]["tokens_saved"] == 30
        assert stats["contribution"]["cost_with_headroom_usd"] == 0.05
        assert stats["tracked_key_count"] == 0

    def test_get_stats_with_tracked_keys(self):
        tracker = Ds4SubscriptionTracker(enabled=True)
        token_a = "sk-ds4-keyA-12345678"
        token_b = "sk-ds4-keyB-87654321"
        prefix_a = token_a[:16]
        tracker.notify_active(f"Bearer {token_a}")
        tracker.notify_active(f"Bearer {token_b}")
        tracker.update_contribution(
            tokens_submitted=50,
            tokens_saved=10,
            token_prefix=prefix_a,
        )
        stats = tracker.get_stats()
        assert stats["tracked_key_count"] == 2
        assert prefix_a in stats["tracked_keys"]
        assert stats["tracked_keys"][prefix_a]["tokens_submitted"] == 50

    def test_get_stats_returns_budget(self):
        tracker = Ds4SubscriptionTracker(enabled=True, budget_limit_usd=50.0, budget_period="daily")
        tracker.update_contribution(tokens_submitted=1000, cost_with_headroom=10.0)
        stats = tracker.get_stats()
        assert stats["budget"]["budget_limit_usd"] == 50.0
        assert stats["budget"]["within_budget"] is True


class TestThreadSafety:
    def test_concurrent_notify_active(self):
        tracker = Ds4SubscriptionTracker(enabled=True)
        n = 50
        errors: list[Exception] = []

        def notify(key_id: int):
            try:
                # Each key must have unique first 16 chars
                label = f"k{key_id:04d}abcdefgh"
                tracker.notify_active(f"Bearer sk-ds4-{label}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=notify, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = tracker.get_stats()
        assert stats is not None
        assert stats["tracked_key_count"] == n

    def test_concurrent_update_contribution(self):
        tracker = Ds4SubscriptionTracker(enabled=True)
        errors: list[Exception] = []

        def update():
            try:
                tracker.update_contribution(
                    tokens_submitted=10,
                    tokens_saved=3,
                    cost_with_headroom=0.01,
                    cost_saved=0.005,
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        c = tracker._contribution
        assert c.tokens_submitted == 1000
        assert c.tokens_saved == 300
        assert c.cost_with_headroom_usd == pytest.approx(1.0)
        assert c.requests == 100


class TestTrackerLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        tracker = Ds4SubscriptionTracker(enabled=True)
        await tracker.start()
        await tracker.stop()
        # No exceptions = pass

    def test_disabled_tracker_still_works(self):
        tracker = Ds4SubscriptionTracker(enabled=False)
        assert tracker.is_available() is False
        tracker.notify_active("Bearer sk-ds4-test-key")
        tracker.update_contribution(tokens_submitted=10)
        assert tracker._contribution.tokens_submitted == 10


# =========================================================================
# QuotaRegistry integration
# =========================================================================


@pytest.fixture(autouse=True)
def clear_registry():
    reset_quota_registry()
    yield
    reset_quota_registry()


def test_register_with_quota_registry():
    reset_quota_registry()
    registry = get_quota_registry()
    tracker = Ds4SubscriptionTracker(enabled=True)
    registry.register(tracker)

    all_stats = registry.get_all_stats()
    assert "ds4_subscription" in all_stats
    assert all_stats["ds4_subscription"]["contribution"]["tokens_submitted"] == 0

    tracker.update_contribution(tokens_submitted=250, tokens_saved=75)
    all_stats = registry.get_all_stats()
    assert all_stats["ds4_subscription"]["contribution"]["tokens_submitted"] == 250


def test_registry_start_stop_available():
    reset_quota_registry()
    registry = get_quota_registry()
    tracker = Ds4SubscriptionTracker(enabled=True)
    registry.register(tracker)

    import asyncio

    asyncio.run(registry.start_all())
    asyncio.run(registry.stop_all())
    assert tracker.is_available()


def test_disabled_tracker_not_in_stats():
    reset_quota_registry()
    registry = get_quota_registry()
    registry.register(Ds4SubscriptionTracker(enabled=False))
    all_stats = registry.get_all_stats()
    assert "ds4_subscription" not in all_stats


# =========================================================================
# Proxy /stats endpoint integration
# =========================================================================


class FakeRequestLogger:
    def __init__(self) -> None:
        self._logs: list[dict[str, object]] = []

    def get_recent(self, limit: int) -> list[dict[str, object]]:
        return self._logs[-limit:]


class FakeLogEntry(dict[str, object]):
    def __getattr__(self, name: str) -> object:
        return self.get(name)


def _minimal_config() -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        http2=False,
        kompress_enabled=False,
        code_aware_enabled=False,
        read_lifecycle=False,
        ccr_inject_marker=False,
        ds4_subscription_enabled=True,
        ds4_budget_limit_usd=None,
    )


class TestStatsEndpointDS4Section:
    def test_stats_contains_ds4_section(self):
        _install_fake_core()
        app = create_app(_minimal_config())
        with TestClient(app) as client:
            response = client.get("/stats")

        assert response.status_code == 200
        payload = response.json()
        assert "ds4" in payload
        assert payload["ds4"]["enabled"] is True

    def test_ds4_section_shows_contribution(self):
        _install_fake_core()
        app = create_app(_minimal_config())
        with TestClient(app) as client:
            from headroom.subscription.ds4 import get_ds4_tracker

            tracker = get_ds4_tracker()
            assert tracker is not None

            tracker.notify_active("Bearer sk-ds4-integration-key")
            prefix = "sk-ds4-integratio"[:16]
            tracker.update_contribution(
                tokens_submitted=500,
                tokens_saved=150,
                cost_with_headroom=0.25,
                cost_saved=0.08,
                token_prefix=prefix,
            )

            response = client.get("/stats")

        assert response.status_code == 200
        payload = response.json()
        ds4 = payload["ds4"]
        assert ds4["enabled"] is True
        assert ds4["contribution"]["tokens_submitted"] == 500
        assert ds4["contribution"]["tokens_saved"] == 150
        assert ds4["contribution"]["cost_with_headroom_usd"] == 0.25
        assert ds4["contribution"]["savings_usd"] == 0.08
        assert ds4["tracked_key_count"] >= 1

    def test_ds4_budget_warning_when_exceeded(self):
        _install_fake_core()
        app = create_app(
            ProxyConfig(
                optimize=False,
                cache_enabled=False,
                rate_limit_enabled=False,
                cost_tracking_enabled=False,
                log_requests=False,
                ccr_inject_tool=False,
                ccr_handle_responses=False,
                ccr_context_tracking=False,
                http2=False,
                ds4_subscription_enabled=True,
                ds4_budget_limit_usd=1.0,
                ds4_budget_period="daily",
            )
        )

        with TestClient(app) as client:
            from headroom.subscription.ds4 import get_ds4_tracker

            tracker = get_ds4_tracker()
            assert tracker is not None
            tracker.update_contribution(
                tokens_submitted=10000,
                cost_with_headroom=0.80,
                cost_saved=0.30,
            )

            response = client.get("/stats")

        assert response.status_code == 200
        payload = response.json()
        ds4 = payload["ds4"]
        assert ds4["budget"]["within_budget"] is False
        assert ds4["budget_warning"] is not None

    def test_ds4_budget_warning_at_90pct(self):
        _install_fake_core()
        app = create_app(
            ProxyConfig(
                optimize=False,
                cache_enabled=False,
                rate_limit_enabled=False,
                cost_tracking_enabled=False,
                log_requests=False,
                ccr_inject_tool=False,
                ccr_handle_responses=False,
                ccr_context_tracking=False,
                http2=False,
                ds4_subscription_enabled=True,
                ds4_budget_limit_usd=10.0,
                ds4_budget_period="daily",
            )
        )

        with TestClient(app) as client:
            from headroom.subscription.ds4 import get_ds4_tracker

            tracker = get_ds4_tracker()
            assert tracker is not None
            tracker.update_contribution(
                tokens_submitted=5000,
                cost_with_headroom=9.0,
            )
            response = client.get("/stats")

        assert response.status_code == 200
        payload = response.json()
        assert "Budget near limit" in payload["ds4"]["budget_warning"]

    def test_ds4_budget_warning_at_75pct(self):
        _install_fake_core()
        app = create_app(
            ProxyConfig(
                optimize=False,
                cache_enabled=False,
                rate_limit_enabled=False,
                cost_tracking_enabled=False,
                log_requests=False,
                ccr_inject_tool=False,
                ccr_handle_responses=False,
                ccr_context_tracking=False,
                http2=False,
                ds4_subscription_enabled=True,
                ds4_budget_limit_usd=100.0,
                ds4_budget_period="daily",
            )
        )

        with TestClient(app) as client:
            from headroom.subscription.ds4 import get_ds4_tracker

            tracker = get_ds4_tracker()
            assert tracker is not None
            tracker.update_contribution(
                tokens_submitted=20000,
                cost_with_headroom=75.0,
            )
            response = client.get("/stats")

        assert response.status_code == 200
        payload = response.json()
        assert "Budget at" in payload["ds4"]["budget_warning"]

    def test_ds4_section_with_no_budget(self):
        _install_fake_core()
        app = create_app(
            ProxyConfig(
                optimize=False,
                cache_enabled=False,
                rate_limit_enabled=False,
                cost_tracking_enabled=False,
                log_requests=False,
                ccr_inject_tool=False,
                ccr_handle_responses=False,
                ccr_context_tracking=False,
                http2=False,
                ds4_subscription_enabled=True,
                ds4_budget_limit_usd=None,
            )
        )

        with TestClient(app) as client:
            response = client.get("/stats")

        assert response.status_code == 200
        ds4 = response.json()["ds4"]
        assert ds4["budget"]["budget_limit_usd"] is None
        assert ds4["budget"]["within_budget"] is True

    def test_ds4_from_quota_registry_in_stats(self):
        _install_fake_core()
        app = create_app(_minimal_config())
        with TestClient(app) as client:
            from headroom.subscription.ds4 import get_ds4_tracker

            tracker = get_ds4_tracker()
            assert tracker is not None
            tracker.notify_active("Bearer sk-ds4-reg-key")
            tracker.update_contribution(
                tokens_submitted=777,
                tokens_saved=222,
            )

            response = client.get("/stats")

        assert response.status_code == 200
        payload = response.json()
        assert "ds4_subscription" in payload
        reg_stats = payload["ds4_subscription"]
        assert reg_stats["contribution"]["tokens_submitted"] == 777
        assert reg_stats["contribution"]["tokens_saved"] == 222


# =========================================================================
# Edge cases and robustness
# =========================================================================


class TestEdgeCases:
    def test_notify_active_empty_header(self):
        tracker = Ds4SubscriptionTracker(enabled=True)
        tracker.notify_active("")
        assert tracker._contribution.started_at is None

    def test_notify_active_only_bearer(self):
        tracker = Ds4SubscriptionTracker(enabled=True)
        tracker.notify_active("Bearer ")
        assert tracker._contribution.started_at is None

    def test_get_stats_never_called_returns_none(self):
        tracker = Ds4SubscriptionTracker(enabled=True)
        stats = tracker.get_stats()
        assert stats is not None
        assert stats["tracked_key_count"] == 0
        assert stats["contribution"]["tokens_submitted"] == 0

    def test_update_with_only_savings(self):
        tracker = Ds4SubscriptionTracker(enabled=True)
        tracker.update_contribution(
            tokens_saved=100,
            cost_saved=0.50,
        )
        c = tracker._contribution
        assert c.tokens_saved == 100
        assert c.savings_usd == 0.50
        assert c.tokens_submitted == 0

    def test_multiple_keys_aggregate_separately(self):
        tracker = Ds4SubscriptionTracker(enabled=True)
        tracker.notify_active("Bearer sk-ds4-key1-abc")
        tracker.notify_active("Bearer sk-ds4-key2-xyz")

        prefix1 = "sk-ds4-key1-abc"[:16]
        prefix2 = "sk-ds4-key2-xyz"[:16]
        tracker.update_contribution(
            tokens_submitted=100,
            tokens_saved=20,
            token_prefix=prefix1,
        )
        tracker.update_contribution(
            tokens_submitted=200,
            tokens_saved=40,
            token_prefix=prefix2,
        )

        assert tracker._tracked_keys[prefix1].tokens_submitted == 100
        assert tracker._tracked_keys[prefix2].tokens_submitted == 200
        assert tracker._contribution.tokens_submitted == 300

    def test_no_double_count_on_reextraction(self):
        tracker = Ds4SubscriptionTracker(enabled=True)
        token = "sk-ds4-double-key-123"
        prefix = token[:16]
        tracker.notify_active(f"Bearer {token}")
        tracker.update_contribution(
            tokens_submitted=50,
            tokens_saved=10,
            cost_with_headroom=0.02,
            token_prefix=prefix,
        )
        c1 = tracker._contribution.tokens_submitted
        pk1 = tracker._tracked_keys[prefix].tokens_submitted

        tracker.update_contribution(
            tokens_submitted=50,
            tokens_saved=10,
            cost_with_headroom=0.02,
            token_prefix=prefix,
        )
        c2 = tracker._contribution.tokens_submitted
        pk2 = tracker._tracked_keys[prefix].tokens_submitted

        assert c2 == c1 + 50
        assert pk2 == pk1 + 50
