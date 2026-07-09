"""Tests for headroom.proxy.probe_recorder."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from headroom.pipeline import PipelineEvent, PipelineStage
from headroom.proxy.probe_recorder import (
    RECORD_DIR_ENV,
    CompressionEventRecorder,
    probe_recorder_from_env,
)


class TestCompressionEventRecorder:
    def test_init_creates_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "probes"
        CompressionEventRecorder(d)
        assert d.is_dir()

    def test_init_sets_directory_mode(self, tmp_path: Path) -> None:
        d = tmp_path / "probes"
        with patch("os.chmod") as mock_chmod:
            CompressionEventRecorder(d)
        mock_chmod.assert_called_once()
        args, _kwargs = mock_chmod.call_args
        assert args[0] == d
        assert args[1] == 0o700

    def test_path_includes_pid(self, tmp_path: Path) -> None:
        d = tmp_path / "probes"
        rec = CompressionEventRecorder(d)
        assert str(rec.path).startswith(str(d / "compression-events-"))
        assert rec.path.suffix == ".jsonl"
        assert str(os.getpid()) in rec.path.name

    def test_ignores_non_input_compressed_event(self, tmp_path: Path) -> None:
        rec = CompressionEventRecorder(tmp_path / "probes")
        event = PipelineEvent(
            stage=PipelineStage.INPUT_RECEIVED,
            operation="test",
            request_id="r1",
            messages=[{"role": "user", "content": "hi"}],
            metadata={
                "original_messages": [{"role": "user", "content": "hello"}],
                "tokens_before": 100,
                "tokens_after": 50,
            },
        )
        rec.on_pipeline_event(event)
        assert not rec.path.exists()

    def test_ignores_event_without_original_messages(self, tmp_path: Path) -> None:
        rec = CompressionEventRecorder(tmp_path / "probes")
        event = PipelineEvent(
            stage=PipelineStage.INPUT_COMPRESSED,
            operation="test",
            request_id="r1",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tokens_before": 100, "tokens_after": 50},
        )
        rec.on_pipeline_event(event)
        assert not rec.path.exists()

    def test_ignores_event_when_messages_is_none(self, tmp_path: Path) -> None:
        rec = CompressionEventRecorder(tmp_path / "probes")
        event = PipelineEvent(
            stage=PipelineStage.INPUT_COMPRESSED,
            operation="test",
            request_id="r1",
            messages=None,
            metadata={
                "original_messages": [{"role": "user", "content": "hello"}],
                "tokens_before": 100,
                "tokens_after": 50,
            },
        )
        rec.on_pipeline_event(event)
        assert not rec.path.exists()

    def test_ignores_event_when_tokens_unchanged(self, tmp_path: Path) -> None:
        rec = CompressionEventRecorder(tmp_path / "probes")
        event = PipelineEvent(
            stage=PipelineStage.INPUT_COMPRESSED,
            operation="test",
            request_id="r1",
            messages=[{"role": "user", "content": "hi"}],
            metadata={
                "original_messages": [{"role": "user", "content": "hello"}],
                "tokens_before": 100,
                "tokens_after": 100,
            },
        )
        rec.on_pipeline_event(event)
        assert not rec.path.exists()

    def test_ignores_event_when_tokens_before_is_none(self, tmp_path: Path) -> None:
        rec = CompressionEventRecorder(tmp_path / "probes")
        event = PipelineEvent(
            stage=PipelineStage.INPUT_COMPRESSED,
            operation="test",
            request_id="r1",
            messages=[{"role": "user", "content": "hi"}],
            metadata={
                "original_messages": [{"role": "user", "content": "hello"}],
                "tokens_after": 50,
            },
        )
        rec.on_pipeline_event(event)
        assert not rec.path.exists()

    def test_records_jsonl_line(self, tmp_path: Path) -> None:
        rec = CompressionEventRecorder(tmp_path / "probes")
        event = PipelineEvent(
            stage=PipelineStage.INPUT_COMPRESSED,
            operation="test",
            request_id="r1",
            provider="anthropic",
            model="claude-3",
            messages=[{"role": "assistant", "content": "compressed"}],
            metadata={
                "original_messages": [{"role": "user", "content": "original long text"}],
                "tokens_before": 100,
                "tokens_after": 50,
                "transforms_applied": ["compress"],
            },
        )
        rec.on_pipeline_event(event)
        assert rec.path.exists()
        text = rec.path.read_text(encoding="utf-8").strip()
        assert '"request_id":"r1"' in text
        assert '"provider":"anthropic"' in text
        assert '"tokens_before":100' in text
        assert '"tokens_after":50' in text
        assert '"original_messages"' in text
        assert '"compressed_messages"' in text
        assert '"transforms_applied"' in text

    def test_uses_lock_for_thread_safety(self, tmp_path: Path) -> None:
        rec = CompressionEventRecorder(tmp_path / "probes")
        with patch.object(rec, "_lock") as mock_lock:
            event = PipelineEvent(
                stage=PipelineStage.INPUT_COMPRESSED,
                operation="test",
                request_id="r1",
                messages=[{"role": "assistant", "content": "c"}],
                metadata={
                    "original_messages": [{"role": "user", "content": "o"}],
                    "tokens_before": 100,
                    "tokens_after": 50,
                },
            )
            rec.on_pipeline_event(event)
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()


class TestProbeRecorderFromEnv:
    def test_returns_none_when_env_not_set(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = probe_recorder_from_env()
        assert result is None

    def test_returns_recorder_when_env_set(self, tmp_path: Path) -> None:
        d = tmp_path / "probes"
        with patch.dict(os.environ, {RECORD_DIR_ENV: str(d)}):
            result = probe_recorder_from_env()
        assert isinstance(result, CompressionEventRecorder)
        assert d.is_dir()

    def test_returns_none_on_creation_error(self) -> None:
        with patch.dict(os.environ, {RECORD_DIR_ENV: "/some/path"}):
            with patch.object(
                CompressionEventRecorder, "__init__", side_effect=PermissionError("denied")
            ):
                result = probe_recorder_from_env()
        assert result is None
