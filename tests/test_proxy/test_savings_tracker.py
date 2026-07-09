"""Tests for headroom.proxy.savings_tracker."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from headroom.proxy.savings_tracker import (
    HEADROOM_SAVINGS_PATH_ENV_VAR,
    PROJECT_NAME_MAX_LENGTH,
    SCHEMA_VERSION,
    SavingsTracker,
    get_default_savings_storage_path,
    sanitize_project_name,
)


class TestSanitizeProjectName:
    def test_none_input(self) -> None:
        assert sanitize_project_name(None) is None

    def test_non_string_input_int(self) -> None:
        assert sanitize_project_name(42) is None

    def test_non_string_input_list(self) -> None:
        assert sanitize_project_name(["foo"]) is None

    def test_empty_string(self) -> None:
        assert sanitize_project_name("") is None

    def test_whitespace_only(self) -> None:
        assert sanitize_project_name("   ") is None

    def test_strips_non_printable_characters(self) -> None:
        assert sanitize_project_name("foo\x00bar\x01") == "foobar"

    def test_preserves_internal_whitespace(self) -> None:
        assert sanitize_project_name("my project name") == "my project name"

    def test_strips_leading_trailing_whitespace(self) -> None:
        assert sanitize_project_name("  hello world  ") == "hello world"

    def test_truncates_to_max_length(self) -> None:
        long_name = "a" * (PROJECT_NAME_MAX_LENGTH + 50)
        result = sanitize_project_name(long_name)
        assert len(result) == PROJECT_NAME_MAX_LENGTH
        assert result == "a" * PROJECT_NAME_MAX_LENGTH

    def test_passes_through_normal_name(self) -> None:
        name = "my-cool-project_v2"
        assert sanitize_project_name(name) == name

    def test_decodes_percent_encoded(self) -> None:
        assert sanitize_project_name("hello%20world") == "hello world"

    def test_mixed_printable_and_non_printable(self) -> None:
        assert sanitize_project_name("  \x00good\x01project\x02  ") == "goodproject"


class TestGetDefaultSavingsStoragePath:
    def test_returns_env_var_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(HEADROOM_SAVINGS_PATH_ENV_VAR, "/custom/savings/path.json")
        assert get_default_savings_storage_path() == "/custom/savings/path.json"

    def test_falls_back_to_savings_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(HEADROOM_SAVINGS_PATH_ENV_VAR, raising=False)
        expected = str(Path("default") / "savings.json")
        with patch("headroom.paths.savings_path", return_value=Path(expected)):
            result = get_default_savings_storage_path()
        assert result == expected


class TestSavingsTracker:
    # ── Construction ─────────────────────────────────────────────────

    def test_constructor_with_path(self, tmp_path: Path) -> None:
        path = tmp_path / "savings.json"
        tracker = SavingsTracker(path=str(path))
        assert tracker.storage_path == str(path)
        assert not path.exists()

    def test_storage_path_property(self, tmp_path: Path) -> None:
        path = tmp_path / "custom_savings.json"
        tracker = SavingsTracker(path=str(path))
        assert tracker.storage_path == str(path)

    def test_constructor_custom_parameters(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(
            path=str(tmp_path / "savings.json"),
            max_history_points=100,
            max_history_age_days=30,
            max_response_history_points=50,
            display_session_inactivity_minutes=10,
        )
        snap = tracker.snapshot()
        assert snap["retention"]["max_history_points"] == 100
        assert snap["retention"]["max_history_age_days"] == 30
        assert snap["retention"]["max_response_history_points"] == 50
        assert snap["display_session_policy"]["rollover_inactivity_minutes"] == 10

    # ── record_compression_savings ───────────────────────────────────

    def test_record_compression_savings_positive_tokens_returns_true(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        assert tracker.record_compression_savings(model="test-model", tokens_saved=100) is True

    def test_record_compression_savings_zero_tokens_returns_false(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        assert tracker.record_compression_savings(model="test-model", tokens_saved=0) is False

    def test_record_compression_savings_negative_tokens_returns_false(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        assert tracker.record_compression_savings(model="test-model", tokens_saved=-10) is False

    def test_record_compression_savings_updates_lifetime(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        tracker.record_compression_savings(model="test-model", tokens_saved=1000)
        snap = tracker.snapshot()
        assert snap["lifetime"]["tokens_saved"] == 1000
        assert snap["lifetime"]["compression_savings_usd"] == 0.003

    def test_record_compression_savings_appends_to_history(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        tracker.record_compression_savings(model="m1", tokens_saved=500)
        snap = tracker.snapshot()
        assert len(snap["history"]) == 1
        entry = snap["history"][0]
        assert entry["total_tokens_saved"] == 500
        assert entry["model"] == "m1"
        assert entry["provider"] == "unknown"

    def test_record_compression_savings_multiple_calls_accumulate(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        tracker.record_compression_savings(model="m1", tokens_saved=200)
        tracker.record_compression_savings(model="m1", tokens_saved=300)
        assert tracker.snapshot()["lifetime"]["tokens_saved"] == 500

    def test_record_compression_savings_with_provider_and_model(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        tracker.record_compression_savings(model="gpt-4", tokens_saved=1000, provider="openai")
        entry = tracker.snapshot()["history"][0]
        assert entry["provider"] == "openai"
        assert entry["model"] == "gpt-4"

    def test_record_compression_savings_with_total_input_tokens(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        tracker.record_compression_savings(model="m1", tokens_saved=500, total_input_tokens=10000)
        assert tracker.snapshot()["lifetime"]["total_input_tokens"] == 10000

    # ── record_request ───────────────────────────────────────────────

    def test_record_request_returns_true(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        assert tracker.record_request(model="m1", input_tokens=1000, tokens_saved=200) is True

    def test_record_request_updates_lifetime(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        tracker.record_request(model="m1", input_tokens=1000, tokens_saved=200)
        snap = tracker.snapshot()
        assert snap["lifetime"]["requests"] == 1
        assert snap["lifetime"]["tokens_saved"] == 200
        assert snap["lifetime"]["total_input_tokens"] >= 1000

    def test_record_request_updates_display_session(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        tracker.record_request(model="m1", input_tokens=1000, tokens_saved=200)
        session = tracker.snapshot()["display_session"]
        assert session["requests"] == 1
        assert session["tokens_saved"] == 200
        assert session["savings_percent"] > 0

    def test_record_request_with_project(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        tracker.record_request(
            model="m1", input_tokens=1000, tokens_saved=200, project="my-project"
        )
        projects = tracker.snapshot()["projects"]
        assert "my-project" in projects
        assert projects["my-project"]["requests"] == 1
        assert projects["my-project"]["tokens_saved"] == 200

    def test_record_request_without_project_skips_projects(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        tracker.record_request(model="m1", input_tokens=1000, tokens_saved=200)
        assert tracker.snapshot()["projects"] == {}

    def test_record_request_zero_tokens_saved_still_records_request(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        tracker.record_request(model="m1", input_tokens=500, tokens_saved=0)
        snap = tracker.snapshot()
        assert snap["lifetime"]["requests"] == 1
        assert snap["lifetime"]["tokens_saved"] == 0
        assert snap["lifetime"]["total_input_tokens"] >= 500
        assert len(snap["history"]) == 0

    def test_record_request_with_multiple_projects(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        for project in ("proj-a", "proj-b", "proj-a"):
            tracker.record_request(
                model="m1",
                input_tokens=500,
                tokens_saved=100,
                project=project,
            )
        projects = tracker.snapshot()["projects"]
        assert projects["proj-a"]["requests"] == 2
        assert projects["proj-b"]["requests"] == 1

    def test_record_request_with_cache_breakdown(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        tracker.record_request(
            model="m1",
            input_tokens=1000,
            tokens_saved=200,
            cache_read_tokens=100,
            cache_write_tokens=50,
            uncached_input_tokens=850,
        )
        snap = tracker.snapshot()
        assert snap["lifetime"]["total_input_cost_usd"] > 0

    # ── snapshot ─────────────────────────────────────────────────────

    def test_snapshot_structure(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        snap = tracker.snapshot()
        assert set(snap) == {
            "schema_version",
            "storage_path",
            "lifetime",
            "display_session",
            "display_session_policy",
            "history",
            "retention",
            "projects",
        }

    def test_snapshot_schema_version(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        assert tracker.snapshot()["schema_version"] == SCHEMA_VERSION

    def test_snapshot_retention_keys(self, tmp_path: Path) -> None:
        retention = SavingsTracker(path=str(tmp_path / "savings.json")).snapshot()["retention"]
        assert set(retention) == {
            "max_history_points",
            "max_history_age_days",
            "max_response_history_points",
        }

    # ── stats_preview ────────────────────────────────────────────────

    def test_stats_preview_structure(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        tracker.record_compression_savings(model="m1", tokens_saved=100)
        preview = tracker.stats_preview()
        assert set(preview) == {
            "schema_version",
            "storage_path",
            "lifetime",
            "display_session",
            "display_session_policy",
            "history_points",
            "recent_history",
            "retention",
            "projects",
            "projects_limit",
        }

    def test_stats_preview_recent_history_count(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        for i in range(5):
            tracker.record_compression_savings(
                model="m1",
                tokens_saved=100,
                timestamp=datetime.now(timezone.utc) + timedelta(seconds=i),
            )
        preview = tracker.stats_preview(recent_points=3)
        assert preview["history_points"] == 5
        assert len(preview["recent_history"]) == 3

    # ── history_response ─────────────────────────────────────────────

    def test_history_response_structure(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        tracker.record_compression_savings(model="m1", tokens_saved=100)
        resp = tracker.history_response()
        assert set(resp) == {
            "schema_version",
            "generated_at",
            "storage_path",
            "lifetime",
            "display_session",
            "display_session_policy",
            "history",
            "series",
            "exports",
            "retention",
            "projects",
            "history_summary",
        }

    def test_history_response_mode_none(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        tracker.record_compression_savings(model="m1", tokens_saved=100)
        assert tracker.history_response(history_mode="none")["history"] == []

    def test_history_response_mode_full(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        tracker.record_compression_savings(model="m1", tokens_saved=100)
        assert len(tracker.history_response(history_mode="full")["history"]) == 1

    def test_history_response_series_keys(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        tracker.record_compression_savings(model="m1", tokens_saved=100)
        series = tracker.history_response()["series"]
        assert set(series) == {"hourly", "daily", "weekly", "monthly"}

    # ── export ────────────────────────────────────────────────────────

    def test_export_rows_history(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        tracker.record_compression_savings(model="m1", tokens_saved=100)
        rows = tracker.export_rows(series="history")
        assert isinstance(rows, list)
        assert len(rows) > 0
        assert "timestamp" in rows[0]
        assert "total_tokens_saved" in rows[0]

    def test_export_csv_returns_string(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        tracker.record_compression_savings(model="m1", tokens_saved=100)
        csv = tracker.export_csv(series="history")
        assert isinstance(csv, str)
        assert "timestamp" in csv
        assert "total_tokens_saved" in csv

    # ── Persistence ──────────────────────────────────────────────────

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "savings.json"
        tracker = SavingsTracker(path=str(path))
        tracker.record_compression_savings(model="m1", tokens_saved=1000)

        tracker2 = SavingsTracker(path=str(path))
        s1 = tracker.snapshot()
        s2 = tracker2.snapshot()
        assert s2["lifetime"]["tokens_saved"] == s1["lifetime"]["tokens_saved"]
        assert (
            s2["lifetime"]["compression_savings_usd"] == s1["lifetime"]["compression_savings_usd"]
        )
        assert len(s2["history"]) == len(s1["history"])
        assert s2["schema_version"] == s1["schema_version"]

    def test_load_corrupted_file_uses_default_state(self, tmp_path: Path) -> None:
        path = tmp_path / "savings.json"
        path.write_text("not valid json", encoding="utf-8")
        tracker = SavingsTracker(path=str(path))
        assert tracker.snapshot()["lifetime"]["tokens_saved"] == 0

    def test_empty_tracker_initial_state(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        snap = tracker.snapshot()
        assert snap["lifetime"] == {
            "requests": 0,
            "tokens_saved": 0,
            "compression_savings_usd": 0.0,
            "total_input_tokens": 0,
            "total_input_cost_usd": 0.0,
        }
        assert snap["display_session"]["requests"] == 0
        assert snap["history"] == []
        assert snap["projects"] == {}

    # ── Edge cases ────────────────────────────────────────────────────

    def test_max_history_points_enforced(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(
            path=str(tmp_path / "savings.json"),
            max_history_points=3,
        )
        for i in range(5):
            tracker.record_compression_savings(
                model="m1",
                tokens_saved=100,
                timestamp=datetime.now(timezone.utc) + timedelta(seconds=i),
            )
        assert len(tracker.snapshot()["history"]) == 3

    def test_display_session_expires_after_inactivity(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(
            path=str(tmp_path / "savings.json"),
            display_session_inactivity_minutes=1,
        )
        base = datetime.now(timezone.utc)
        tracker.record_request(
            model="m1",
            input_tokens=500,
            tokens_saved=100,
            timestamp=base,
        )
        assert tracker.snapshot()["display_session"]["requests"] == 1

        tracker.record_request(
            model="m1",
            input_tokens=500,
            tokens_saved=100,
            timestamp=base + timedelta(minutes=2),
        )
        assert tracker.snapshot()["display_session"]["requests"] == 1

    def test_stateless_mode_keeps_in_memory_state(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(
            path=str(tmp_path / "savings.json"),
            stateless=True,
        )
        tracker.record_compression_savings(model="m1", tokens_saved=100)
        assert tracker.snapshot()["lifetime"]["tokens_saved"] == 100

    def test_schema_version_constant(self) -> None:
        assert SCHEMA_VERSION == 3

    def test_full_lifetime_after_mixed_records(self, tmp_path: Path) -> None:
        tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
        tracker.record_compression_savings(model="m1", tokens_saved=500)
        tracker.record_request(model="m1", input_tokens=2000, tokens_saved=300)
        snap = tracker.snapshot()
        assert snap["lifetime"]["requests"] == 1
        assert snap["lifetime"]["tokens_saved"] == 800
        assert snap["lifetime"]["compression_savings_usd"] > 0
        assert snap["lifetime"]["total_input_tokens"] >= 2000
