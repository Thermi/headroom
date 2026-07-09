"""Tests for handler-level functions from headroom.proxy.handlers.

The ``should_skip_ccr_request_compression`` closure logic is extracted here
as a pure-function equivalent so it can be tested in isolation.
"""

from __future__ import annotations

from headroom.ccr.tool_injection import CCR_TOOL_NAME
from headroom.proxy.modes import is_token_mode

# ---------------------------------------------------------------------------
# Helpers: reify the closure as a pure function
# ---------------------------------------------------------------------------


def _should_skip_ccr_request_compression(
    mode: str,
    ccr_inject_tool: bool,
    frozen_message_count: int,
    existing_tool_names: set[str],
) -> bool:
    """Reified version of the nested closure in ``handle_anthropic_messages``."""
    if is_token_mode(mode):
        return False
    return ccr_inject_tool and frozen_message_count > 0 and CCR_TOOL_NAME not in existing_tool_names


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestShouldSkipCcrRequestCompression:
    def test_token_mode_never_skips(self) -> None:
        assert not _should_skip_ccr_request_compression(
            mode="token",
            ccr_inject_tool=True,
            frozen_message_count=5,
            existing_tool_names=set(),
        )

    def test_token_mode_with_zero_frozen(self) -> None:
        assert not _should_skip_ccr_request_compression(
            mode="token",
            ccr_inject_tool=True,
            frozen_message_count=0,
            existing_tool_names=set(),
        )

    def test_cache_mode_skips_when_tool_not_present(self) -> None:
        assert _should_skip_ccr_request_compression(
            mode="cache",
            ccr_inject_tool=True,
            frozen_message_count=3,
            existing_tool_names=set(),
        )

    def test_cache_mode_does_not_skip_when_tool_already_present(self) -> None:
        assert not _should_skip_ccr_request_compression(
            mode="cache",
            ccr_inject_tool=True,
            frozen_message_count=3,
            existing_tool_names={CCR_TOOL_NAME},
        )

    def test_cache_mode_does_not_skip_when_inject_tool_is_false(self) -> None:
        assert not _should_skip_ccr_request_compression(
            mode="cache",
            ccr_inject_tool=False,
            frozen_message_count=3,
            existing_tool_names=set(),
        )

    def test_cache_mode_does_not_skip_when_frozen_count_is_zero(self) -> None:
        assert not _should_skip_ccr_request_compression(
            mode="cache",
            ccr_inject_tool=True,
            frozen_message_count=0,
            existing_tool_names=set(),
        )

    def test_all_conditions_must_be_true_for_skip(self) -> None:
        assert not _should_skip_ccr_request_compression(
            mode="cache",
            ccr_inject_tool=False,
            frozen_message_count=0,
            existing_tool_names=set(),
        )

    def test_other_mode_aliases(self) -> None:
        assert not _should_skip_ccr_request_compression(
            mode="token_mode",
            ccr_inject_tool=True,
            frozen_message_count=5,
            existing_tool_names=set(),
        )
        assert not _should_skip_ccr_request_compression(
            mode="token_savings",
            ccr_inject_tool=True,
            frozen_message_count=5,
            existing_tool_names=set(),
        )

    def test_cache_mode_with_alias(self) -> None:
        assert _should_skip_ccr_request_compression(
            mode="cost_savings",
            ccr_inject_tool=True,
            frozen_message_count=2,
            existing_tool_names=set(),
        )

    def test_unknown_mode_falls_back_to_token(self) -> None:
        assert not _should_skip_ccr_request_compression(
            mode="unknown_mode",
            ccr_inject_tool=True,
            frozen_message_count=5,
            existing_tool_names=set(),
        )

    def test_none_mode_falls_back_to_token(self) -> None:
        assert not _should_skip_ccr_request_compression(
            mode=None,
            ccr_inject_tool=True,
            frozen_message_count=5,
            existing_tool_names=set(),
        )
