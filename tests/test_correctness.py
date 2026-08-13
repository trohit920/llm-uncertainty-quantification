"""Unit tests for answer normalisation and correctness scoring."""

from __future__ import annotations

import pytest

from src.correctness import (
    canonical_answer,
    extract_final_number,
    extract_short_answer,
    majority_answer,
    normalize_answer,
    score_generation,
    score_gsm8k,
    score_nq_open,
)


class TestNormalizeAnswer:
    """SQuAD-style normalisation."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("The Beatles", "beatles"),
            ("  BARACK   OBAMA  ", "barack obama"),
            ("St. Louis, Missouri", "st louis missouri"),
            ("a red car", "red car"),
            ("An Apple", "apple"),
            ("!!!", ""),
            ("", ""),
        ],
    )
    def test_normalisation_cases(self, raw: str, expected: str) -> None:
        assert normalize_answer(raw) == expected


class TestExtractShortAnswer:
    """Reduction of a chat generation to its short-answer span."""

    @pytest.mark.parametrize(
        ("generation", "expected"),
        [
            ("Paris", "Paris"),
            ("The answer is Paris.", "Paris"),
            ("Answer: Paris", "Paris"),
            ("A: Paris", "Paris"),
            ("Paris\nIt is the capital of France.", "Paris"),
            ("  Paris.  ", "Paris"),
        ],
    )
    def test_prefix_and_line_handling(
        self, generation: str, expected: str
    ) -> None:
        assert extract_short_answer(generation) == expected


class TestScoreNqOpen:
    """Alias exact match."""

    def test_matches_any_alias(self) -> None:
        aliases = ["Barack Obama", "Obama"]
        assert score_nq_open("Obama", aliases)
        assert score_nq_open("barack obama", aliases)

    def test_ignores_punctuation_and_articles(self) -> None:
        assert score_nq_open("The Beatles!", ["Beatles"])

    def test_rejects_wrong_answer(self) -> None:
        assert not score_nq_open("Paris", ["London"])

    def test_rejects_empty_generation(self) -> None:
        assert not score_nq_open("", ["London"])
        assert not score_nq_open("...", ["London"])

    def test_substring_is_not_a_match(self) -> None:
        # Exact match after normalisation, not containment.
        assert not score_nq_open("New York City", ["New York"])


class TestExtractFinalNumber:
    """Numeric extraction from a chain of thought."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("The total is 18. #### 18", 18.0),
            ("Steps... #### 1,234", 1234.0),
            ("#### -5", -5.0),
            ("#### 3.5", 3.5),
            ("He had 5 apples and ate 2, leaving 3", 3.0),
            ("First 10, then 20, finally 30.", 30.0),
            ("no numbers here", None),
            ("", None),
        ],
    )
    def test_extraction_cases(self, text: str, expected: float | None) -> None:
        assert extract_final_number(text) == expected

    def test_marker_takes_priority_over_later_prose(self) -> None:
        assert extract_final_number("#### 42 (that is 7 times 6)") == 42.0

    def test_last_marker_wins(self) -> None:
        assert extract_final_number("#### 1 then again #### 2") == 2.0

    def test_dangling_marker_falls_back_to_last_number(self) -> None:
        assert extract_final_number("the answer is 7 ####") == 7.0


class TestScoreGsm8k:
    """Numeric comparison against the gold solution."""

    def test_matching_answer(self) -> None:
        assert score_gsm8k("Working... #### 18", "Reasoning...\n#### 18")

    def test_integer_and_decimal_forms_agree(self) -> None:
        assert score_gsm8k("#### 18.0", "#### 18")

    def test_mismatched_answer(self) -> None:
        assert not score_gsm8k("#### 17", "#### 18")

    def test_unparseable_generation_is_incorrect(self) -> None:
        assert not score_gsm8k("I am not sure", "#### 18")

    def test_thousands_separators(self) -> None:
        assert score_gsm8k("#### 1,234", "#### 1234")


class TestCanonicalAnswer:
    """Canonical forms used for clustering and majority voting."""

    def test_nq_open_canonicalises_text(self) -> None:
        assert canonical_answer("nq_open", "The answer is Paris.") == "paris"

    def test_gsm8k_canonicalises_number(self) -> None:
        assert canonical_answer("gsm8k", "so #### 18.0") == "18"
        assert canonical_answer("gsm8k", "so #### 18") == "18"

    def test_gsm8k_keeps_true_decimals(self) -> None:
        assert canonical_answer("gsm8k", "#### 2.5") == "2.5"

    def test_unparseable_is_empty(self) -> None:
        assert canonical_answer("gsm8k", "no idea") == ""

    def test_unknown_task_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown task"):
            canonical_answer("chess", "e4")


class TestScoreGeneration:
    """Task dispatch."""

    def test_dispatches_to_nq_open(self) -> None:
        assert score_generation("nq_open", "Paris", ["Paris"])

    def test_dispatches_to_gsm8k(self) -> None:
        assert score_generation("gsm8k", "#### 18", ["#### 18"])

    def test_unknown_task_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown task"):
            score_generation("chess", "e4", ["e4"])


class TestMajorityAnswer:
    """Modal answer selection for self-consistency decoding."""

    def test_returns_most_frequent(self) -> None:
        assert majority_answer(["18", "18", "17"]) == "18"

    def test_ignores_empty_strings(self) -> None:
        assert majority_answer(["", "", "17"]) == "17"

    def test_all_empty_returns_empty(self) -> None:
        assert majority_answer(["", ""]) == ""

    def test_empty_input_returns_empty(self) -> None:
        assert majority_answer([]) == ""
