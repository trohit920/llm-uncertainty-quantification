"""Practical applications of uncertainty-aware generation.

Two applications that turn a scalar signal into a decision:

* **Selective answering** -- answer when confident, abstain otherwise. The
  abstention threshold is chosen on the dev split to hit a target accuracy
  among answered items, then applied unchanged to the held-out test split.
  The gap between dev and test accuracy at a fixed threshold is the honest
  measure of whether the operating point transfers.
* **Confidence tags** -- bucket every prediction into High / Medium / Low from
  dev-split quantiles. A user-facing surface cannot show a nat-valued entropy;
  it can show three tiers, and the tiers are only useful if accuracy actually
  separates across them.

Every threshold here is fitted on dev and reported on test. Fitting on test
would let the system pick the threshold that flatters its own numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .config import (
    CONFIDENCE_TAG_NAMES,
    CONFIDENCE_TAG_QUANTILES,
    SELECTIVE_TARGET_ACCURACIES,
)
from .signals import to_uncertainty

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class SelectiveOperatingPoint:
    """One answer-or-abstain operating point.

    Attributes:
        target_accuracy: Accuracy the threshold was fitted to achieve.
        threshold: Maximum uncertainty at which the system still answers.
        dev_coverage: Fraction answered on the dev split.
        dev_accuracy: Accuracy among answered dev items.
        test_coverage: Fraction answered on the held-out split.
        test_accuracy: Accuracy among answered test items.
        test_num_answered: Count of answered test items.
        target_met_on_test: Whether the dev-fitted threshold still reached the
            target once applied to unseen data.
    """

    target_accuracy: float
    threshold: float
    dev_coverage: float
    dev_accuracy: float
    test_coverage: float
    test_accuracy: float
    test_num_answered: int
    target_met_on_test: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the operating point."""
        return {
            "target_accuracy": self.target_accuracy,
            "threshold": self.threshold,
            "dev_coverage": self.dev_coverage,
            "dev_accuracy": self.dev_accuracy,
            "test_coverage": self.test_coverage,
            "test_accuracy": self.test_accuracy,
            "test_num_answered": self.test_num_answered,
            "target_met_on_test": self.target_met_on_test,
        }


@dataclass(frozen=True)
class ConfidenceTierReport:
    """Accuracy and share of one confidence tier.

    Attributes:
        tier: ``"High"``, ``"Medium"`` or ``"Low"``.
        count: Number of test items in the tier.
        share: Fraction of test items in the tier.
        accuracy: Accuracy within the tier.
    """

    tier: str
    count: int
    share: float
    accuracy: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the tier report."""
        return {
            "tier": self.tier,
            "count": self.count,
            "share": self.share,
            "accuracy": self.accuracy,
        }


def _prepare(
    signal: str, values: Sequence[float], is_correct: Sequence[bool]
) -> tuple[FloatArray, BoolArray]:
    """Reorient a signal to uncertainty and drop non-finite entries.

    Args:
        signal: Signal name, used for orientation.
        values: Raw signal values.
        is_correct: Correctness labels.

    Returns:
        A ``(uncertainty, labels)`` pair containing only finite entries.
    """
    uncertainty = to_uncertainty(signal, values)
    labels = np.asarray(is_correct, dtype=bool)
    mask = np.isfinite(uncertainty)
    return uncertainty[mask], labels[mask]


def fit_abstention_threshold(
    uncertainty: FloatArray, is_correct: BoolArray, target_accuracy: float
) -> float:
    """Find the most permissive threshold meeting a target accuracy.

    Sweeps every coverage level and keeps the largest one whose accuracy among
    answered items still clears the target -- maximising coverage subject to
    the accuracy constraint.

    Args:
        uncertainty: Dev-split uncertainty values.
        is_correct: Dev-split correctness labels.
        target_accuracy: Accuracy the answered subset must achieve.

    Returns:
        The threshold, or ``-inf`` when no coverage level meets the target,
        which makes the system abstain on everything.
    """
    if uncertainty.size == 0:
        return float("-inf")

    order = np.argsort(uncertainty, kind="mergesort")
    sorted_uncertainty = uncertainty[order]
    sorted_correct = is_correct[order].astype(np.float64)

    running_accuracy = np.cumsum(sorted_correct) / np.arange(
        1, uncertainty.size + 1
    )
    meets_target = running_accuracy >= target_accuracy
    if not meets_target.any():
        logger.info(
            "No coverage level reaches target accuracy %.2f; abstaining fully",
            target_accuracy,
        )
        return float("-inf")

    best_index = int(np.max(np.flatnonzero(meets_target)))
    return float(sorted_uncertainty[best_index])


def _apply_threshold(
    uncertainty: FloatArray, is_correct: BoolArray, threshold: float
) -> tuple[float, float, int]:
    """Report coverage and accuracy for a fixed threshold.

    Args:
        uncertainty: Uncertainty values.
        is_correct: Correctness labels.
        threshold: Maximum uncertainty at which the system answers.

    Returns:
        A ``(coverage, accuracy, num_answered)`` tuple. Accuracy is ``nan``
        when nothing is answered.
    """
    answered = uncertainty <= threshold
    num_answered = int(answered.sum())
    if num_answered == 0:
        return 0.0, float("nan"), 0
    return (
        num_answered / uncertainty.size,
        float(is_correct[answered].mean()),
        num_answered,
    )


def selective_answering(
    signal: str,
    dev_values: Sequence[float],
    dev_correct: Sequence[bool],
    test_values: Sequence[float],
    test_correct: Sequence[bool],
    target_accuracies: Sequence[float] = SELECTIVE_TARGET_ACCURACIES,
) -> list[SelectiveOperatingPoint]:
    """Build answer-or-abstain operating points across target accuracies.

    Args:
        signal: Signal name driving the abstention decision.
        dev_values: Dev-split raw signal values, used to fit thresholds.
        dev_correct: Dev-split correctness labels.
        test_values: Held-out raw signal values.
        test_correct: Held-out correctness labels.
        target_accuracies: Accuracy targets to fit thresholds for.

    Returns:
        One operating point per target accuracy.
    """
    dev_uncertainty, dev_labels = _prepare(signal, dev_values, dev_correct)
    test_uncertainty, test_labels = _prepare(signal, test_values, test_correct)

    points: list[SelectiveOperatingPoint] = []
    for target in target_accuracies:
        threshold = fit_abstention_threshold(dev_uncertainty, dev_labels, target)
        dev_coverage, dev_accuracy, _ = _apply_threshold(
            dev_uncertainty, dev_labels, threshold
        )
        test_coverage, test_accuracy, num_answered = _apply_threshold(
            test_uncertainty, test_labels, threshold
        )

        points.append(
            SelectiveOperatingPoint(
                target_accuracy=target,
                threshold=threshold,
                dev_coverage=dev_coverage,
                dev_accuracy=dev_accuracy,
                test_coverage=test_coverage,
                test_accuracy=test_accuracy,
                test_num_answered=num_answered,
                target_met_on_test=bool(
                    np.isfinite(test_accuracy) and test_accuracy >= target
                ),
            )
        )

    return points


def fit_confidence_tag_thresholds(
    uncertainty: FloatArray,
    quantiles: tuple[float, float] = CONFIDENCE_TAG_QUANTILES,
) -> tuple[float, float]:
    """Fit the two cut-points separating High / Medium / Low tiers.

    Args:
        uncertainty: Dev-split uncertainty values.
        quantiles: Lower and upper quantiles of the dev distribution.

    Returns:
        A ``(high_cut, medium_cut)`` pair of uncertainty thresholds.
    """
    if uncertainty.size == 0:
        return float("nan"), float("nan")
    low_quantile, high_quantile = quantiles
    return (
        float(np.quantile(uncertainty, low_quantile)),
        float(np.quantile(uncertainty, high_quantile)),
    )


def assign_confidence_tags(
    uncertainty: FloatArray, thresholds: tuple[float, float]
) -> list[str]:
    """Label each prediction High, Medium or Low confidence.

    Args:
        uncertainty: Uncertainty values.
        thresholds: The ``(high_cut, medium_cut)`` pair from the dev split.

    Returns:
        One tier name per input value.
    """
    high_cut, medium_cut = thresholds
    high_name, medium_name, low_name = CONFIDENCE_TAG_NAMES
    return [
        high_name if value <= high_cut else medium_name if value <= medium_cut else low_name
        for value in uncertainty
    ]


def confidence_tag_report(
    signal: str,
    dev_values: Sequence[float],
    dev_correct: Sequence[bool],
    test_values: Sequence[float],
    test_correct: Sequence[bool],
) -> list[ConfidenceTierReport]:
    """Tag held-out predictions and report accuracy per tier.

    Args:
        signal: Signal name driving the tagging.
        dev_values: Dev-split raw signal values, used to fit the cut-points.
        dev_correct: Dev-split correctness labels. Unused by the fit, which
            depends only on the signal distribution, but accepted for a
            uniform call signature.
        test_values: Held-out raw signal values.
        test_correct: Held-out correctness labels.

    Returns:
        One report per tier, ordered most to least confident.
    """
    del dev_correct  # Cut-points are distribution-based, not label-based.

    dev_uncertainty = to_uncertainty(signal, dev_values)
    dev_uncertainty = dev_uncertainty[np.isfinite(dev_uncertainty)]
    test_uncertainty, test_labels = _prepare(signal, test_values, test_correct)

    thresholds = fit_confidence_tag_thresholds(dev_uncertainty)
    tags = np.asarray(assign_confidence_tags(test_uncertainty, thresholds))

    reports: list[ConfidenceTierReport] = []
    for tier in CONFIDENCE_TAG_NAMES:
        in_tier = tags == tier
        count = int(in_tier.sum())
        reports.append(
            ConfidenceTierReport(
                tier=tier,
                count=count,
                share=count / tags.size if tags.size else float("nan"),
                accuracy=(
                    float(test_labels[in_tier].mean()) if count else float("nan")
                ),
            )
        )

    return reports


def tier_separation(reports: Sequence[ConfidenceTierReport]) -> float:
    """Accuracy gap between the High and Low tiers.

    A single number for whether the tags mean anything: if High and Low
    predictions are equally accurate, the tags are decoration.

    Args:
        reports: Tier reports, in any order.

    Returns:
        High-tier accuracy minus Low-tier accuracy, or ``nan`` when either
        tier is empty.
    """
    by_tier = {report.tier: report.accuracy for report in reports}
    high_name, _, low_name = CONFIDENCE_TAG_NAMES
    high, low = by_tier.get(high_name), by_tier.get(low_name)
    if high is None or low is None:
        return float("nan")
    return high - low
