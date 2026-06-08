"""Tests for fast-path message copy functions (perf optimization)."""

from __future__ import annotations

from typing import Any

from headroom.proxy.helpers import _fast_copy_message, _fast_copy_messages


def _msg(content: str | list | None = None, role: str = "user") -> dict[str, Any]:
    m: dict[str, Any] = {"role": role}
    if content is not None:
        m["content"] = content
    return m


def _block(text: str = "hello", block_type: str = "text") -> dict[str, Any]:
    return {"type": block_type, "text": text}


class TestFastCopyMessage:
    def test_copies_top_level_keys(self) -> None:
        original = _msg("hello")
        copied = _fast_copy_message(original)
        assert copied == original
        assert copied is not original

    def test_preserves_extra_keys(self) -> None:
        original = {"role": "user", "name": "test-user", "content": "hi"}
        copied = _fast_copy_message(original)
        assert copied["name"] == "test-user"

    def test_string_content_is_shallow(self) -> None:
        original = _msg("hello world")
        copied = _fast_copy_message(original)
        assert copied["content"] is original["content"]

    def test_list_content_copies_blocks(self) -> None:
        b1 = _block("a")
        b2 = _block("b")
        original = _msg([b1, b2])
        copied = _fast_copy_message(original)
        assert copied["content"] == original["content"]
        assert copied["content"] is not original["content"]
        assert copied["content"][0] is not b1
        assert copied["content"][1] is not b2
        assert copied["content"][0] == b1
        assert copied["content"][1] == b2

    def test_content_not_present(self) -> None:
        original = {"role": "user"}
        copied = _fast_copy_message(original)
        assert copied == original
        assert "content" not in copied

    def test_content_is_none(self) -> None:
        original = {"role": "user", "content": None}
        copied = _fast_copy_message(original)
        assert copied.get("content") is None

    def test_non_dict_items_in_content_list(self) -> None:
        original = _msg(["just a string", 42, _block("b")])
        copied = _fast_copy_message(original)
        assert copied["content"] is not original["content"]
        assert copied["content"][0] == "just a string"
        assert copied["content"][1] == 42
        assert copied["content"][2] is not original["content"][2]
        assert copied["content"][2] == original["content"][2]

    def test_preserves_internal_refs_across_messages(self) -> None:
        shared = {"type": "text", "text": "shared"}
        original = _msg([shared, shared])
        copied = _fast_copy_message(original)
        assert copied["content"][0] is not copied["content"][1]


class TestFastCopyMessages:
    def test_empty_list(self) -> None:
        assert _fast_copy_messages([]) == []

    def test_copies_all_messages(self) -> None:
        original = [_msg("a"), _msg("b", role="assistant")]
        copied = _fast_copy_messages(original)
        assert copied == original
        assert copied is not original
        assert all(c is not o for c, o in zip(copied, original))

    def test_does_not_mutate_original_when_mutating_copy(self) -> None:
        b1 = _block("original")
        original = [_msg([b1])]
        copied = _fast_copy_messages(original)

        copied[0]["role"] = "assistant"
        assert original[0]["role"] == "user"

        copied_block = copied[0]["content"][0]
        copied_block["text"] = "modified"
        assert b1["text"] == "original"

    def test_mixed_content_types(self) -> None:
        original = [
            _msg("plain string"),
            _msg([_block("block a"), _block("block b")]),
            _msg(None),
            {"role": "tool", "tool_call_id": "tc_1", "content": "result"},
        ]
        copied = _fast_copy_messages(original)
        assert copied == original
        assert len(copied) == 4
        assert copied[1]["content"][1] is not original[1]["content"][1]

    def test_large_message_list_performance(self) -> None:
        messages = [_msg([_block(f"block_{i}")]) for i in range(100)]
        import time

        t0 = time.perf_counter()
        for _ in range(100):
            _fast_copy_messages(messages)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0
