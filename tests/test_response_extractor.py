"""Tests for post-response memory extraction."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from headroom.memory.response_extractor import (
    ExtractedMemory,
    ExtractionResult,
    _extract_text_from_anthropic_response,
    _extract_text_from_openai_response,
    _parse_llm_extraction_output,
    extract_inline,
    extract_memories_from_turn,
)


class TestExtractTextFromResponse:
    """Tests for extracting text from provider-specific response bodies."""

    def test_anthropic_simple_text(self) -> None:
        body = {"content": [{"type": "text", "text": "Hello world"}]}
        assert _extract_text_from_anthropic_response(body) == "Hello world"

    def test_anthropic_multiple_blocks(self) -> None:
        body = {
            "content": [
                {"type": "text", "text": "First block"},
                {"type": "tool_use", "name": "test"},
                {"type": "text", "text": "Second block"},
            ]
        }
        assert _extract_text_from_anthropic_response(body) == "First block\nSecond block"

    def test_anthropic_no_text(self) -> None:
        body = {"content": [{"type": "tool_use", "name": "test"}]}
        assert _extract_text_from_anthropic_response(body) == ""

    def test_openai_simple(self) -> None:
        body = {"choices": [{"message": {"content": "Hello"}}]}
        assert _extract_text_from_openai_response(body) == "Hello"

    def test_openai_null_content(self) -> None:
        body = {"choices": [{"message": {"content": None}}]}
        assert _extract_text_from_openai_response(body) == ""

    def test_openai_no_choices(self) -> None:
        assert _extract_text_from_openai_response({}) == ""


class TestExtractInline:
    """Tests for inline <memory> block parsing."""

    def test_with_memory_block(self) -> None:
        response = 'Here is your answer.\n<memory>{"memories": [{"content": "User likes Python"}]}</memory>'
        result = extract_inline(response)
        assert result.strategy == "inline"
        assert len(result.memories) == 1
        assert result.memories[0].content == "User likes Python"
        assert result.memories[0].importance == 0.5

    def test_with_importance(self) -> None:
        response = '<memory>{"memories": [{"content": "Important fact", "importance": 0.9}]}</memory>'
        result = extract_inline(response)
        assert len(result.memories) == 1
        assert result.memories[0].content == "Important fact"
        assert result.memories[0].importance == 0.9

    def test_empty_memory_block(self) -> None:
        response = 'Just a response.\n<memory>{"memories": []}</memory>'
        result = extract_inline(response)
        assert result.strategy == "none"
        assert result.memories == []

    def test_no_memory_block(self) -> None:
        response = "Just a regular response with no memory block."
        result = extract_inline(response)
        assert result.strategy == "none"
        assert result.memories == []

    def test_multiple_memories(self) -> None:
        response = (
            '<memory>{"memories": [{"content": "Fact 1"}, {"content": "Fact 2"}]}</memory>'
        )
        result = extract_inline(response)
        assert len(result.memories) == 2
        assert result.memories[0].content == "Fact 1"
        assert result.memories[1].content == "Fact 2"


class TestParseLLMOutput:
    """Tests for parsing LLM extraction output."""

    def test_valid_json_object(self) -> None:
        text = '{"memories": [{"content": "Fact", "importance": 0.7}]}'
        result = _parse_llm_extraction_output(text)
        assert len(result) == 1
        assert result[0].content == "Fact"
        assert result[0].importance == 0.7

    def test_json_in_code_block(self) -> None:
        text = 'Some text\n```json\n{"memories": [{"content": "Fact from code block"}]}\n```'
        result = _parse_llm_extraction_output(text)
        assert len(result) == 1
        assert result[0].content == "Fact from code block"

    def test_invalid_json(self) -> None:
        text = "not valid json at all"
        result = _parse_llm_extraction_output(text)
        assert result == []

    def test_empty_memories(self) -> None:
        text = '{"memories": []}'
        result = _parse_llm_extraction_output(text)
        assert result == []

    def test_facts_field_fallback(self) -> None:
        text = '{"facts": [{"content": "Fallback fact"}]}'
        result = _parse_llm_extraction_output(text)
        assert len(result) == 1
        assert result[0].content == "Fallback fact"


class TestExtractedMemory:
    """Tests for the ExtractedMemory dataclass."""

    def test_to_save_input_with_facts(self) -> None:
        mem = ExtractedMemory(content="Test", facts=["Specific fact"])
        inp = mem.to_save_input()
        assert inp["content"] == "Test"
        assert inp["facts"] == ["Specific fact"]
        assert inp["importance"] == 0.5

    def test_to_save_input_empty_facts(self) -> None:
        mem = ExtractedMemory(content="Fallback content", importance=0.8)
        inp = mem.to_save_input()
        assert inp["facts"] == ["Fallback content"]
        assert inp["importance"] == 0.8


class TestExtractMemoriesFromTurn:
    """Tests for the main extraction function."""

    @pytest.mark.asyncio
    async def test_inline_only_no_llm(self) -> None:
        messages = [{"role": "user", "content": "I love Python"}]
        response = '<memory>{"memories": [{"content": "User loves Python"}]}</memory>'
        result = await extract_memories_from_turn(messages, response, use_llm=False)
        assert result.strategy == "inline"
        assert len(result.memories) == 1

    @pytest.mark.asyncio
    async def test_no_inline_does_not_call_llm_when_disabled(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        response = "Hi there!"
        result = await extract_memories_from_turn(messages, response, use_llm=False)
        assert result.strategy == "none"
        assert result.memories == []

    @pytest.mark.asyncio
    async def test_llm_fallback_when_enabled(self) -> None:
        messages = [{"role": "user", "content": "I prefer TypeScript over JavaScript"}]
        response = "TypeScript is a great choice for type safety!"

        with patch(
            "headroom.memory.response_extractor.extract_with_llm",
            AsyncMock(
                return_value=ExtractionResult(
                    memories=[ExtractedMemory(content="User prefers TypeScript", importance=0.7)],
                    strategy="llm",
                )
            ),
        ):
            result = await extract_memories_from_turn(
                messages, response, use_llm=True, api_key="test", base_url="http://test"
            )
            assert result.strategy == "llm"
            assert len(result.memories) == 1
            assert result.memories[0].content == "User prefers TypeScript"

    @pytest.mark.asyncio
    async def test_inline_takes_precedence_over_llm(self) -> None:
        messages = [{"role": "user", "content": "I love Python"}]
        response = '<memory>{"memories": [{"content": "User loves Python"}]}</memory>'

        with patch(
            "headroom.memory.response_extractor.extract_with_llm",
            AsyncMock(),
        ) as mock_llm:
            result = await extract_memories_from_turn(
                messages, response, use_llm=True, api_key="test", base_url="http://test"
            )
            assert result.strategy == "inline"
            mock_llm.assert_not_called()
