"""Unit tests for sentence-level uncertainty signals."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.sentence_metrics import (
    cluster_frequency_entropy,
    cluster_likelihood_entropy,
    lexical_entropy,
    self_consistency,
    semantic_self_consistency,
    shannon_entropy,
    summarize_sentence_uncertainty,
)


class TestShannonEntropy:
    """Entropy of a discrete distribution, in nats."""

    def test_uniform_over_two_is_log_two(self) -> None:
        assert shannon_entropy([0.5, 0.5]) == pytest.approx(math.log(2))

    def test_uniform_over_four_is_log_four(self) -> None:
        assert shannon_entropy([1, 1, 1, 1]) == pytest.approx(math.log(4))

    def test_degenerate_distribution_is_zero(self) -> None:
        assert shannon_entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_renormalises_unnormalised_input(self) -> None:
        assert shannon_entropy([3, 3]) == pytest.approx(math.log(2))

    def test_zero_mass_is_nan(self) -> None:
        assert math.isnan(shannon_entropy([0.0, 0.0]))

    def test_empty_is_nan(self) -> None:
        assert math.isnan(shannon_entropy([]))


class TestClusterFrequencyEntropy:
    """Discrete semantic entropy from cluster labels."""

    def test_single_cluster_is_zero(self) -> None:
        assert cluster_frequency_entropy([0, 0, 0, 0]) == pytest.approx(0.0)

    def test_two_equal_clusters_is_log_two(self) -> None:
        assert cluster_frequency_entropy([0, 0, 1, 1]) == pytest.approx(math.log(2))

    def test_all_distinct_is_log_n(self) -> None:
        assert cluster_frequency_entropy([0, 1, 2, 3]) == pytest.approx(math.log(4))

    def test_skewed_is_between(self) -> None:
        value = cluster_frequency_entropy([0, 0, 0, 1])
        assert 0.0 < value < math.log(2)

    def test_empty_is_nan(self) -> None:
        assert math.isnan(cluster_frequency_entropy([]))


class TestClusterLikelihoodEntropy:
    """Likelihood-weighted semantic entropy."""

    def test_single_cluster_is_zero(self) -> None:
        value = cluster_likelihood_entropy([0, 0, 0], [-1.0, -2.0, -3.0])
        assert value == pytest.approx(0.0)

    def test_equal_likelihood_matches_frequency_entropy(self) -> None:
        labels = [0, 0, 1, 1]
        log_probs = [-1.0] * 4
        assert cluster_likelihood_entropy(labels, log_probs) == pytest.approx(
            cluster_frequency_entropy(labels)
        )

    def test_dominant_likelihood_lowers_entropy(self) -> None:
        labels = [0, 1]
        # Cluster 0 carries far more mass than cluster 1.
        skewed = cluster_likelihood_entropy(labels, [-0.1, -10.0])
        balanced = cluster_likelihood_entropy(labels, [-1.0, -1.0])
        assert skewed < balanced

    def test_survives_very_negative_log_probabilities(self) -> None:
        # Naive exponentiation would underflow these to zero.
        labels = [0, 1]
        value = cluster_likelihood_entropy(labels, [-900.0, -900.0])
        assert value == pytest.approx(math.log(2))

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            cluster_likelihood_entropy([0, 1], [-1.0])

    def test_non_finite_input_is_nan(self) -> None:
        assert math.isnan(cluster_likelihood_entropy([0, 1], [-1.0, float("nan")]))

    def test_empty_is_nan(self) -> None:
        assert math.isnan(cluster_likelihood_entropy([], []))


class TestLexicalEntropy:
    """Exact-match answer entropy."""

    def test_identical_answers_is_zero(self) -> None:
        assert lexical_entropy(["paris"] * 5) == pytest.approx(0.0)

    def test_distinct_answers_is_log_n(self) -> None:
        assert lexical_entropy(["a", "b", "c"]) == pytest.approx(math.log(3))

    def test_ignores_semantic_equivalence(self) -> None:
        # This is the point of the ablation: lexical entropy cannot see that
        # these mean the same thing, whereas semantic clustering can.
        assert lexical_entropy(["paris", "the city of paris"]) == pytest.approx(
            math.log(2)
        )


class TestSelfConsistency:
    """Agreement with the modal answer."""

    def test_unanimous_is_one(self) -> None:
        assert self_consistency(["18"] * 10) == pytest.approx(1.0)

    def test_split_vote(self) -> None:
        assert self_consistency(["18", "18", "17"]) == pytest.approx(2 / 3)

    def test_all_distinct_is_one_over_n(self) -> None:
        assert self_consistency(["a", "b", "c", "d"]) == pytest.approx(0.25)

    def test_empty_is_nan(self) -> None:
        assert math.isnan(self_consistency([]))


class TestSemanticSelfConsistency:
    """Share of the largest meaning cluster."""

    def test_single_cluster_is_one(self) -> None:
        assert semantic_self_consistency([0, 0, 0]) == pytest.approx(1.0)

    def test_majority_cluster_share(self) -> None:
        assert semantic_self_consistency([0, 0, 0, 1]) == pytest.approx(0.75)


class TestSummarizeSentenceUncertainty:
    """The aggregate bundle."""

    def test_confident_case_is_all_zero_entropy(self) -> None:
        summary = summarize_sentence_uncertainty(
            answers=["paris"] * 5,
            labels=[0] * 5,
            sequence_log_probabilities=[-0.5] * 5,
        )
        assert summary.discrete_semantic_entropy == pytest.approx(0.0)
        assert summary.weighted_semantic_entropy == pytest.approx(0.0)
        assert summary.lexical_entropy == pytest.approx(0.0)
        assert summary.self_consistency == pytest.approx(1.0)
        assert summary.num_semantic_clusters == 1
        assert summary.num_distinct_answers == 1

    def test_semantic_clustering_collapses_lexical_variation(self) -> None:
        # Four distinct strings, two meanings: lexical entropy is high but
        # semantic entropy is the entropy of a two-cluster split.
        summary = summarize_sentence_uncertainty(
            answers=["paris", "city of paris", "london", "city of london"],
            labels=[0, 0, 1, 1],
            sequence_log_probabilities=[-1.0] * 4,
        )
        assert summary.lexical_entropy == pytest.approx(math.log(4))
        assert summary.discrete_semantic_entropy == pytest.approx(math.log(2))
        assert summary.discrete_semantic_entropy < summary.lexical_entropy

    def test_signal_dict_keys(self) -> None:
        summary = summarize_sentence_uncertainty(["a"], [0], [-1.0])
        keys = set(summary.as_signal_dict())
        assert {
            "semantic_entropy",
            "weighted_semantic_entropy",
            "lexical_entropy",
            "self_consistency",
        } <= keys

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="disagree in length"):
            summarize_sentence_uncertainty(["a", "b"], [0], [-1.0])

    def test_signal_values_are_finite_for_valid_input(self) -> None:
        summary = summarize_sentence_uncertainty(
            ["a", "b", "a"], [0, 1, 0], [-1.0, -2.0, -1.5]
        )
        for name, value in summary.as_signal_dict().items():
            assert np.isfinite(value), f"{name} is not finite"
