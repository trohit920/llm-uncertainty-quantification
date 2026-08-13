"""Hand-implemented evaluation metrics for uncertainty quality.

Three families of metric, all computed from first principles on NumPy arrays
so their behaviour is transparent and unit-testable:

* **Error-detection AUROC** -- rank-based (Mann-Whitney U) area under the ROC
  curve, treating *incorrect* answers as the positive class and the
  uncertainty score as the ranker. Ties receive average ranks.
* **Expected Calibration Error** -- binned gap between confidence and
  accuracy, offered with both fixed-width and equal-mass bins. Equal-mass
  binning matters at our sample sizes, where fixed-width bins are often empty.
* **Risk-coverage / AURC** -- error rate among the most-confident fraction of
  predictions, swept across all coverage levels.

Every headline metric is also available with a percentile bootstrap
confidence interval, because at 100-200 examples per task a bare point
estimate does not support claims about one method beating another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Final, Sequence

import numpy as np
from numpy.typing import NDArray

from .config import (
    BOOTSTRAP_CONFIDENCE_LEVEL,
    NUM_BOOTSTRAP_RESAMPLES,
    NUM_CALIBRATION_BINS,
    SEED,
)

#: Smallest denominator tolerated before a metric is reported as undefined.
_MIN_DENOMINATOR: Final[int] = 1

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """A point estimate with a bootstrap confidence interval."""

    value: float
    lower: float
    upper: float

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-serialisable view of the interval."""
        return {"value": self.value, "lower": self.lower, "upper": self.upper}

    def __str__(self) -> str:
        return f"{self.value:.3f} [{self.lower:.3f}, {self.upper:.3f}]"


@dataclass(frozen=True)
class ReliabilityBin:
    """One bin of a reliability diagram."""

    lower_edge: float
    upper_edge: float
    count: int
    mean_confidence: float
    accuracy: float


@dataclass(frozen=True)
class ReliabilityDiagram:
    """Binned calibration summary plus its scalar error."""

    expected_calibration_error: float
    maximum_calibration_error: float
    bins: list[ReliabilityBin] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view of the diagram."""
        return {
            "ece": self.expected_calibration_error,
            "mce": self.maximum_calibration_error,
            "bins": [
                {
                    "lower_edge": b.lower_edge,
                    "upper_edge": b.upper_edge,
                    "count": b.count,
                    "mean_confidence": b.mean_confidence,
                    "accuracy": b.accuracy,
                }
                for b in self.bins
            ],
        }


@dataclass(frozen=True)
class RiskCoverageCurve:
    """Selective-prediction risk as a function of coverage."""

    coverage: FloatArray
    risk: FloatArray
    area_under_risk_coverage: float

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view of the curve."""
        return {
            "coverage": self.coverage.tolist(),
            "risk": self.risk.tolist(),
            "aurc": self.area_under_risk_coverage,
        }


# --------------------------------------------------------------------------
# Error-detection AUROC
# --------------------------------------------------------------------------


def _average_ranks(values: FloatArray) -> FloatArray:
    """Rank values in ascending order, assigning average ranks to ties.

    Args:
        values: One-dimensional array of scores.

    Returns:
        Array of 1-based ranks, tied entries sharing their mean rank.
    """
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)

    index = 0
    while index < len(sorted_values):
        stop = index
        while (
            stop + 1 < len(sorted_values)
            and sorted_values[stop + 1] == sorted_values[index]
        ):
            stop += 1
        average_rank = (index + stop) / 2.0 + 1.0
        ranks[order[index : stop + 1]] = average_rank
        index = stop + 1

    return ranks


def error_detection_auroc(
    uncertainty: Sequence[float], is_correct: Sequence[bool]
) -> float:
    """Area under the ROC curve for detecting incorrect answers.

    The positive class is an *incorrect* answer, and the ranker is the
    uncertainty score, so a well-behaved signal scores above 0.5.

    Args:
        uncertainty: Higher values mean the model is less certain.
        is_correct: Ground-truth correctness for each prediction.

    Returns:
        The AUROC, or ``nan`` when either class is empty (every answer
        correct, or every answer wrong), in which case AUROC is undefined.

    Raises:
        ValueError: If the two inputs have different lengths.
    """
    scores = np.asarray(uncertainty, dtype=np.float64)
    correct = np.asarray(is_correct, dtype=bool)
    if scores.shape != correct.shape:
        raise ValueError(
            f"Length mismatch: {scores.shape} uncertainty vs {correct.shape} labels"
        )

    positives = ~correct  # incorrect answers are the positive class
    num_positive = int(positives.sum())
    num_negative = int((~positives).sum())
    if num_positive < _MIN_DENOMINATOR or num_negative < _MIN_DENOMINATOR:
        return float("nan")

    ranks = _average_ranks(scores)
    positive_rank_sum = float(ranks[positives].sum())
    u_statistic = positive_rank_sum - num_positive * (num_positive + 1) / 2.0
    return u_statistic / (num_positive * num_negative)


# --------------------------------------------------------------------------
# Expected calibration error
# --------------------------------------------------------------------------


def _fixed_width_edges(num_bins: int) -> FloatArray:
    """Return ``num_bins + 1`` evenly spaced edges spanning [0, 1]."""
    return np.linspace(0.0, 1.0, num_bins + 1)


def _equal_mass_edges(confidence: FloatArray, num_bins: int) -> FloatArray:
    """Return bin edges placing roughly equal counts in each bin.

    Args:
        confidence: Predicted probabilities of correctness.
        num_bins: Desired number of bins.

    Returns:
        Monotonically non-decreasing edges with the outer edges pinned to
        0 and 1. Duplicate interior edges are removed, which can yield fewer
        than ``num_bins`` bins when the scores are highly concentrated.
    """
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(confidence, quantiles)
    edges[0], edges[-1] = 0.0, 1.0
    return np.unique(edges)


def reliability_diagram(
    confidence: Sequence[float],
    is_correct: Sequence[bool],
    num_bins: int = NUM_CALIBRATION_BINS,
    strategy: str = "fixed_width",
) -> ReliabilityDiagram:
    """Bin predictions by confidence and measure the calibration gap.

    Args:
        confidence: Predicted probability that each answer is correct, in
            [0, 1].
        is_correct: Ground-truth correctness for each prediction.
        num_bins: Number of bins to use.
        strategy: ``"fixed_width"`` for evenly spaced edges, or
            ``"equal_mass"`` for quantile edges.

    Returns:
        The populated diagram, with empty bins omitted.

    Raises:
        ValueError: If the inputs disagree in length or ``strategy`` is not
            recognised.
    """
    scores = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(is_correct, dtype=bool)
    if scores.shape != correct.shape:
        raise ValueError(
            f"Length mismatch: {scores.shape} confidence vs {correct.shape} labels"
        )
    if scores.size == 0:
        return ReliabilityDiagram(float("nan"), float("nan"), [])

    if strategy == "fixed_width":
        edges = _fixed_width_edges(num_bins)
    elif strategy == "equal_mass":
        edges = _equal_mass_edges(scores, num_bins)
    else:
        raise ValueError(f"Unknown binning strategy: {strategy!r}")

    bins: list[ReliabilityBin] = []
    weighted_gap = 0.0
    maximum_gap = 0.0
    total = float(scores.size)

    for lower, upper in zip(edges[:-1], edges[1:]):
        # Include the left edge everywhere, and the right edge in the last bin
        # only, so every point lands in exactly one bin.
        in_bin = (scores >= lower) & (
            scores < upper if upper < edges[-1] else scores <= upper
        )
        count = int(in_bin.sum())
        if count == 0:
            continue

        mean_confidence = float(scores[in_bin].mean())
        accuracy = float(correct[in_bin].mean())
        gap = abs(mean_confidence - accuracy)
        weighted_gap += (count / total) * gap
        maximum_gap = max(maximum_gap, gap)

        bins.append(
            ReliabilityBin(
                lower_edge=float(lower),
                upper_edge=float(upper),
                count=count,
                mean_confidence=mean_confidence,
                accuracy=accuracy,
            )
        )

    return ReliabilityDiagram(weighted_gap, maximum_gap, bins)


def expected_calibration_error(
    confidence: Sequence[float],
    is_correct: Sequence[bool],
    num_bins: int = NUM_CALIBRATION_BINS,
    strategy: str = "fixed_width",
) -> float:
    """Return only the scalar ECE from a reliability diagram.

    Args:
        confidence: Predicted probability of correctness, in [0, 1].
        is_correct: Ground-truth correctness.
        num_bins: Number of bins.
        strategy: ``"fixed_width"`` or ``"equal_mass"``.

    Returns:
        The expected calibration error.
    """
    return reliability_diagram(
        confidence, is_correct, num_bins, strategy
    ).expected_calibration_error


def brier_score(
    confidence: Sequence[float], is_correct: Sequence[bool]
) -> float:
    """Mean squared error between confidence and the correctness indicator.

    Args:
        confidence: Predicted probability of correctness, in [0, 1].
        is_correct: Ground-truth correctness.

    Returns:
        The Brier score; lower is better.
    """
    scores = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(is_correct, dtype=np.float64)
    if scores.size == 0:
        return float("nan")
    return float(np.mean((scores - correct) ** 2))


# --------------------------------------------------------------------------
# Risk-coverage
# --------------------------------------------------------------------------


def risk_coverage_curve(
    uncertainty: Sequence[float], is_correct: Sequence[bool]
) -> RiskCoverageCurve:
    """Sweep the error rate over the most-confident fraction of predictions.

    Predictions are sorted by ascending uncertainty; at coverage ``k/n`` the
    system answers the ``k`` most confident items and abstains on the rest.
    Risk is the error rate among those answered.

    Args:
        uncertainty: Higher values mean the model is less certain.
        is_correct: Ground-truth correctness for each prediction.

    Returns:
        The curve together with its area (AURC); lower area is better.

    Raises:
        ValueError: If the two inputs have different lengths.
    """
    scores = np.asarray(uncertainty, dtype=np.float64)
    correct = np.asarray(is_correct, dtype=bool)
    if scores.shape != correct.shape:
        raise ValueError(
            f"Length mismatch: {scores.shape} uncertainty vs {correct.shape} labels"
        )
    if scores.size == 0:
        empty = np.asarray([], dtype=np.float64)
        return RiskCoverageCurve(empty, empty, float("nan"))

    order = np.argsort(scores, kind="mergesort")
    errors_in_order = (~correct[order]).astype(np.float64)
    counts = np.arange(1, scores.size + 1, dtype=np.float64)

    risk = np.cumsum(errors_in_order) / counts
    coverage = counts / scores.size
    return RiskCoverageCurve(coverage, risk, float(risk.mean()))


def area_under_risk_coverage(
    uncertainty: Sequence[float], is_correct: Sequence[bool]
) -> float:
    """Return only the scalar AURC.

    Args:
        uncertainty: Higher values mean the model is less certain.
        is_correct: Ground-truth correctness.

    Returns:
        The area under the risk-coverage curve; lower is better.
    """
    return risk_coverage_curve(uncertainty, is_correct).area_under_risk_coverage


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------


def bootstrap_interval(
    metric: Callable[[Sequence[float], Sequence[bool]], float],
    scores: Sequence[float],
    is_correct: Sequence[bool],
    num_resamples: int = NUM_BOOTSTRAP_RESAMPLES,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
    seed: int = SEED,
) -> Interval:
    """Percentile bootstrap confidence interval for a paired metric.

    Resampling is done over example indices, keeping each score paired with
    its correctness label.

    Args:
        metric: Callable taking ``(scores, labels)`` and returning a scalar.
        scores: Per-example uncertainty or confidence values.
        is_correct: Ground-truth correctness for each example.
        num_resamples: Number of bootstrap resamples.
        confidence_level: Two-sided coverage, e.g. 0.95.
        seed: Seed for the resampling RNG.

    Returns:
        The point estimate on the full sample plus percentile bounds. Bounds
        are ``nan`` when every resample produced an undefined metric.
    """
    score_array = np.asarray(scores, dtype=np.float64)
    correct_array = np.asarray(is_correct, dtype=bool)
    point_estimate = metric(score_array, correct_array)

    if score_array.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    replicates: list[float] = []
    for _ in range(num_resamples):
        indices = rng.integers(0, score_array.size, size=score_array.size)
        value = metric(score_array[indices], correct_array[indices])
        if not np.isnan(value):
            replicates.append(value)

    if not replicates:
        return Interval(point_estimate, float("nan"), float("nan"))

    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(replicates, [tail, 1.0 - tail])
    return Interval(float(point_estimate), float(lower), float(upper))
