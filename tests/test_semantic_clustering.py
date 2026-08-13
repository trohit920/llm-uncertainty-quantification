"""Unit tests for bidirectional-entailment clustering.

The clustering algorithm is exercised against synthetic entailment oracles, so
these tests run without downloading an NLI checkpoint.
"""

from __future__ import annotations

from typing import Sequence

from src.semantic_clustering import (
    cluster_by_entailment,
    cluster_unique_answers,
)


def make_meaning_oracle(meanings: dict[str, str]):
    """Build an oracle that equates answers sharing a meaning key.

    Args:
        meanings: Mapping from answer string to its meaning identifier.

    Returns:
        A batched entailment predicate.
    """

    def oracle(pairs: Sequence[tuple[str, str]]) -> list[bool]:
        return [
            meanings.get(left, left) == meanings.get(right, right)
            for left, right in pairs
        ]

    return oracle


def make_counting_oracle(meanings: dict[str, str], counter: list[int]):
    """Build a meaning oracle that records how many pairs it was asked about.

    Args:
        meanings: Mapping from answer string to its meaning identifier.
        counter: Single-element list accumulating the pair count.

    Returns:
        A batched entailment predicate with a side effect.
    """
    inner = make_meaning_oracle(meanings)

    def oracle(pairs: Sequence[tuple[str, str]]) -> list[bool]:
        counter[0] += len(pairs)
        return inner(pairs)

    return oracle


class TestClusterByEntailment:
    """Greedy single-representative clustering."""

    def test_empty_input(self) -> None:
        result = cluster_by_entailment([], make_meaning_oracle({}))
        assert result.labels == []
        assert result.num_clusters == 0

    def test_single_answer_is_one_cluster(self) -> None:
        result = cluster_by_entailment(["paris"], make_meaning_oracle({}))
        assert result.labels == [0]
        assert result.num_clusters == 1

    def test_all_equivalent_collapse_to_one_cluster(self) -> None:
        meanings = {"paris": "P", "the city of paris": "P", "Paris, France": "P"}
        result = cluster_by_entailment(list(meanings), make_meaning_oracle(meanings))
        assert result.num_clusters == 1
        assert result.labels == [0, 0, 0]

    def test_distinct_meanings_separate(self) -> None:
        meanings = {"paris": "P", "london": "L", "berlin": "B"}
        result = cluster_by_entailment(list(meanings), make_meaning_oracle(meanings))
        assert result.num_clusters == 3
        assert sorted(result.labels) == [0, 1, 2]

    def test_mixed_case_groups_correctly(self) -> None:
        answers = ["paris", "london", "city of paris", "london town"]
        meanings = {
            "paris": "P",
            "city of paris": "P",
            "london": "L",
            "london town": "L",
        }
        result = cluster_by_entailment(answers, make_meaning_oracle(meanings))
        assert result.num_clusters == 2
        assert result.labels[0] == result.labels[2]
        assert result.labels[1] == result.labels[3]
        assert result.labels[0] != result.labels[1]

    def test_labels_are_parallel_to_input(self) -> None:
        answers = ["a", "b", "a"]
        result = cluster_by_entailment(answers, make_meaning_oracle({}))
        assert len(result.labels) == len(answers)
        assert result.labels[0] == result.labels[2]

    def test_first_matching_cluster_wins(self) -> None:
        # An oracle that matches everything must produce a single cluster.
        always = lambda pairs: [True] * len(pairs)  # noqa: E731
        result = cluster_by_entailment(["a", "b", "c"], always)
        assert result.num_clusters == 1


class TestClusterUniqueAnswers:
    """De-duplication before NLI scoring."""

    def test_matches_undeduplicated_result(self) -> None:
        answers = ["paris", "paris", "london", "paris"]
        meanings = {"paris": "P", "london": "L"}
        deduplicated = cluster_unique_answers(answers, make_meaning_oracle(meanings))
        assert deduplicated.num_clusters == 2
        assert deduplicated.labels[0] == deduplicated.labels[1]
        assert deduplicated.labels[0] == deduplicated.labels[3]
        assert deduplicated.labels[2] != deduplicated.labels[0]

    def test_saves_oracle_calls_on_duplicates(self) -> None:
        answers = ["paris"] * 9 + ["london"]
        meanings = {"paris": "P", "london": "L"}

        naive_counter = [0]
        cluster_by_entailment(answers, make_counting_oracle(meanings, naive_counter))

        deduplicated_counter = [0]
        cluster_unique_answers(
            answers, make_counting_oracle(meanings, deduplicated_counter)
        )

        assert deduplicated_counter[0] < naive_counter[0]
        # Two unique strings need exactly one comparison.
        assert deduplicated_counter[0] == 1

    def test_empty_input(self) -> None:
        result = cluster_unique_answers([], make_meaning_oracle({}))
        assert result.labels == []
        assert result.num_clusters == 0

    def test_all_identical_needs_no_comparisons(self) -> None:
        counter = [0]
        result = cluster_unique_answers(
            ["18"] * 10, make_counting_oracle({}, counter)
        )
        assert counter[0] == 0
        assert result.num_clusters == 1
        assert result.labels == [0] * 10
