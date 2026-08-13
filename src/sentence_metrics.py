"""Sentence-level uncertainty signals computed over a set of sampled answers.

Given N stochastic samples for one prompt, this module produces:

* **Discrete semantic entropy** -- entropy of the meaning-cluster frequency
  distribution (Farquhar et al. 2024). Needs only cluster labels.
* **Likelihood-weighted semantic entropy** -- entropy of cluster probabilities
  formed by summing length-normalised sequence likelihoods within each
  cluster (Kuhn et al. 2023). Nearly free once sample log-probabilities are
  available, and sensitive to *how* the model spread its mass, not just how
  often each meaning was drawn.
* **Lexical entropy** -- the same computation over exact-match answer strings,
  serving as the ablation that isolates what semantic clustering buys.
* **Self-consistency** -- the fraction of samples agreeing with the modal
  answer, the cheapest sentence-level signal and a strong baseline.

Everything here is pure NumPy over cluster labels and log-probabilities, so it
is unit-testable without a model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np
from numpy.typing import NDArray

#: Guards log(0) for clusters that receive no probability mass.
_PROBABILITY_EPSILON: Final[float] = 1e-12

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SentenceUncertainty:
    """Aggregated sentence-level signals for one prompt.

    Attributes:
        discrete_semantic_entropy: Entropy over meaning-cluster frequencies,
            in nats.
        weighted_semantic_entropy: Entropy over likelihood-weighted cluster
            probabilities, in nats.
        lexical_entropy: Entropy over exact-match answer strings, in nats.
        self_consistency: Fraction of samples matching the modal answer.
        semantic_self_consistency: Fraction of samples in the largest meaning
            cluster.
        num_semantic_clusters: Count of distinct meaning classes.
        num_distinct_answers: Count of distinct answer strings.
        num_samples: Number of samples the signals were computed from.
    """

    discrete_semantic_entropy: float
    weighted_semantic_entropy: float
    lexical_entropy: float
    self_consistency: float
    semantic_self_consistency: float
    num_semantic_clusters: int
    num_distinct_answers: int
    num_samples: int

    def as_signal_dict(self) -> dict[str, float]:
        """Flatten to a ``{signal_name: value}`` mapping."""
        return {
            "semantic_entropy": self.discrete_semantic_entropy,
            "weighted_semantic_entropy": self.weighted_semantic_entropy,
            "lexical_entropy": self.lexical_entropy,
            "self_consistency": self.self_consistency,
            "semantic_self_consistency": self.semantic_self_consistency,
            "num_semantic_clusters": float(self.num_semantic_clusters),
            "num_distinct_answers": float(self.num_distinct_answers),
        }


def shannon_entropy(probabilities: Sequence[float]) -> float:
    """Shannon entropy of a discrete distribution, in nats.

    Args:
        probabilities: Non-negative masses. Renormalised if they do not sum
            to one; zero-mass entries contribute nothing.

    Returns:
        The entropy in nats, or ``nan`` when the total mass is zero.
    """
    masses = np.asarray(probabilities, dtype=np.float64)
    if masses.size == 0:
        return float("nan")

    total = masses.sum()
    if total <= _PROBABILITY_EPSILON:
        return float("nan")

    normalized = masses / total
    support = normalized > _PROBABILITY_EPSILON
    return float(-np.sum(normalized[support] * np.log(normalized[support])))


def cluster_frequency_entropy(labels: Sequence[int]) -> float:
    """Entropy of the empirical cluster-frequency distribution.

    Args:
        labels: Cluster index per sample.

    Returns:
        Entropy in nats, or ``nan`` for an empty input.
    """
    if not labels:
        return float("nan")
    counts = Counter(labels)
    return shannon_entropy(list(counts.values()))


def cluster_likelihood_entropy(
    labels: Sequence[int], sequence_log_probabilities: Sequence[float]
) -> float:
    """Entropy of likelihood-weighted cluster probabilities.

    Cluster mass is the sum of sample likelihoods within the cluster. Summing
    is performed in log space via log-sum-exp so that very negative sequence
    log-probabilities -- routine for a 300-token chain of thought -- do not
    underflow to zero.

    Args:
        labels: Cluster index per sample.
        sequence_log_probabilities: Length-normalised log-likelihood per
            sample, in nats.

    Returns:
        Entropy in nats, or ``nan`` when inputs are empty.

    Raises:
        ValueError: If the two inputs have different lengths.
    """
    if len(labels) != len(sequence_log_probabilities):
        raise ValueError(
            f"Length mismatch: {len(labels)} labels vs "
            f"{len(sequence_log_probabilities)} log-probabilities"
        )
    if not labels:
        return float("nan")

    log_probabilities = np.asarray(sequence_log_probabilities, dtype=np.float64)
    if not np.all(np.isfinite(log_probabilities)):
        return float("nan")

    cluster_indices = sorted(set(labels))
    label_array = np.asarray(labels)

    cluster_log_masses = np.asarray(
        [
            _log_sum_exp(log_probabilities[label_array == cluster])
            for cluster in cluster_indices
        ],
        dtype=np.float64,
    )

    # Normalise in log space, then exponentiate: masses are now O(1).
    normalized = np.exp(cluster_log_masses - _log_sum_exp(cluster_log_masses))
    return shannon_entropy(normalized)


def _log_sum_exp(values: FloatArray) -> float:
    """Numerically stable ``log(sum(exp(values)))``.

    Args:
        values: Log-space quantities.

    Returns:
        The log of the summed exponentials, or ``-inf`` for an empty input.
    """
    if values.size == 0:
        return float("-inf")
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return maximum
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


def lexical_entropy(answers: Sequence[str]) -> float:
    """Entropy over exact-match answer strings.

    Args:
        answers: Canonical answer strings, one per sample.

    Returns:
        Entropy in nats, or ``nan`` for an empty input.
    """
    if not answers:
        return float("nan")
    counts = Counter(answers)
    return shannon_entropy(list(counts.values()))


def self_consistency(answers: Sequence[str]) -> float:
    """Fraction of samples agreeing with the modal answer string.

    Args:
        answers: Canonical answer strings, one per sample.

    Returns:
        A value in (0, 1], or ``nan`` for an empty input.
    """
    if not answers:
        return float("nan")
    counts = Counter(answers)
    return counts.most_common(1)[0][1] / len(answers)


def semantic_self_consistency(labels: Sequence[int]) -> float:
    """Fraction of samples falling in the largest meaning cluster.

    Args:
        labels: Cluster index per sample.

    Returns:
        A value in (0, 1], or ``nan`` for an empty input.
    """
    if not labels:
        return float("nan")
    counts = Counter(labels)
    return counts.most_common(1)[0][1] / len(labels)


def summarize_sentence_uncertainty(
    answers: Sequence[str],
    labels: Sequence[int],
    sequence_log_probabilities: Sequence[float],
) -> SentenceUncertainty:
    """Compute every sentence-level signal for one prompt.

    Args:
        answers: Canonical answer string per sample.
        labels: Meaning-cluster index per sample.
        sequence_log_probabilities: Length-normalised log-likelihood per
            sample, in nats.

    Returns:
        The populated signal bundle.

    Raises:
        ValueError: If the three inputs do not share a length.
    """
    lengths = {len(answers), len(labels), len(sequence_log_probabilities)}
    if len(lengths) != 1:
        raise ValueError(f"Sample arrays disagree in length: {sorted(lengths)}")

    return SentenceUncertainty(
        discrete_semantic_entropy=cluster_frequency_entropy(labels),
        weighted_semantic_entropy=cluster_likelihood_entropy(
            labels, sequence_log_probabilities
        ),
        lexical_entropy=lexical_entropy(answers),
        self_consistency=self_consistency(answers),
        semantic_self_consistency=semantic_self_consistency(labels),
        num_semantic_clusters=len(set(labels)),
        num_distinct_answers=len(set(answers)),
        num_samples=len(answers),
    )
