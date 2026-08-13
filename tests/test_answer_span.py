"""Unit tests for final-answer span detection.

A fake character-level decoder stands in for the tokenizer, so these run
without downloading a model. The cases mirror real Qwen2.5 output, where
``####`` doubles as a markdown H4 heading.
"""

from __future__ import annotations

from typing import Sequence

from src.model import find_answer_span

MARKER = "####"


def char_decode(token_ids: Sequence[int]) -> str:
    """Decode integer code points as a string, one character per token.

    Args:
        token_ids: Unicode code points standing in for token identifiers.

    Returns:
        The decoded string.
    """
    return "".join(chr(code) for code in token_ids)


def encode(text: str) -> list[int]:
    """Encode a string as one token per character.

    Args:
        text: Source text.

    Returns:
        Unicode code points, one per character.
    """
    return [ord(char) for char in text]


class TestFindAnswerSpan:
    """Marker-anchored span detection."""

    def test_no_marker_returns_none(self) -> None:
        tokens = encode("The answer is seven.")
        assert find_answer_span(tokens, MARKER, char_decode) is None

    def test_simple_marker_at_end(self) -> None:
        text = "Tom has 7 apples. #### 7"
        tokens = encode(text)
        span = find_answer_span(tokens, MARKER, char_decode)

        assert span is not None
        start, end = span
        assert end == len(tokens)
        # The span begins once the marker is complete, so the decoded prefix
        # ends with the marker and the span itself covers " 7".
        assert char_decode(tokens[:start]).endswith(MARKER)
        assert char_decode(tokens[start:end]) == " 7"

    def test_markdown_headings_do_not_capture_the_span(self) -> None:
        # This is the real failure mode: taking the first "####" would anchor
        # the span to the top of the chain of thought.
        text = "#### Step 1\nAdd them.\n#### Step 2\nCheck.\n#### 7"
        tokens = encode(text)
        span = find_answer_span(tokens, MARKER, char_decode)

        assert span is not None
        start, _ = span
        assert char_decode(tokens[start:]) == " 7"

    def test_heading_without_digits_is_ineligible(self) -> None:
        text = "#### Summary\nAll done."
        tokens = encode(text)
        assert find_answer_span(tokens, MARKER, char_decode) is None

    def test_last_numeric_marker_wins(self) -> None:
        text = "#### 1 first pass\nrevised\n#### 42"
        tokens = encode(text)
        span = find_answer_span(tokens, MARKER, char_decode)

        assert span is not None
        assert char_decode(tokens[span[0] :]) == " 42"

    def test_trailing_heading_after_answer_is_skipped(self) -> None:
        # The final marker has no digits after it, so the previous numeric
        # one must be selected instead.
        text = "#### 42\n#### Notes"
        tokens = encode(text)
        span = find_answer_span(tokens, MARKER, char_decode)

        assert span is not None
        assert char_decode(tokens[span[0] :]) == " 42\n#### Notes"

    def test_span_is_within_bounds(self) -> None:
        tokens = encode("blah #### 3")
        span = find_answer_span(tokens, MARKER, char_decode)

        assert span is not None
        start, end = span
        assert 0 <= start <= end == len(tokens)

    def test_empty_generation(self) -> None:
        assert find_answer_span([], MARKER, char_decode) is None
