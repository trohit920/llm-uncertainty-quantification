"""Self-consistency decoding and the uncertainty-gated cascade.

Self-consistency -- sample N chains of thought and take the majority answer --
reliably beats greedy decoding on math, at N-times the cost. That flat cost is
the interesting part: it is paid on every question, including the ones greedy
already had right.

The cascade makes the cost conditional. Decode greedily, measure uncertainty
on that single pass, and escalate to sampling only when the greedy answer
looks unreliable. If the uncertainty signal is any good, most of the accuracy
gain is recoverable for a fraction of the compute, and the resulting
accuracy-versus-cost curve is a far more useful artefact than either endpoint
alone.

The escalation threshold is fitted on the dev split and reported on test.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .config import NUM_SAMPLES
from .signals import to_uncertainty

logger = logging.getLogger(__name__)

#: Escalation budgets swept when building the cost-accuracy curve.
_ESCALATION_BUDGETS: tuple[float, ...] = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.8,
    1.0,
)

#: Accuracy shortfall tolerated when choosing the cheapest useful threshold.
_ACCURACY_TOLERANCE: float = 0.01

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class CascadePoint:
    """One point on the cascade's accuracy-versus-cost curve.

    Attributes:
        escalation_budget: Dev-split fraction of items intended to escalate.
        threshold: Uncertainty above which the system escalates to sampling.
        test_escalation_rate: Fraction of held-out items actually escalated.
        test_accuracy: Accuracy of the cascade's chosen answers.
        mean_generations: Average generations per item, greedy counted as 1.
        generation_cost_ratio: Mean generations relative to always sampling.
    """

    escalation_budget: float
    threshold: float
    test_escalation_rate: float
    test_accuracy: float
    mean_generations: float
    generation_cost_ratio: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the point."""
        return {
            "escalation_budget": self.escalation_budget,
            "threshold": self.threshold,
            "test_escalation_rate": self.test_escalation_rate,
            "test_accuracy": self.test_accuracy,
            "mean_generations": self.mean_generations,
            "generation_cost_ratio": self.generation_cost_ratio,
        }


@dataclass(frozen=True)
class CascadeReport:
    """Cascade results alongside both baselines.

    Attributes:
        greedy_accuracy: Accuracy of greedy decoding alone.
        self_consistency_accuracy: Accuracy of always taking the majority
            vote over ``num_samples`` samples.
        curve: Accuracy-versus-cost points across escalation budgets.
        selected: The cheapest point that essentially matches full sampling.
        num_samples: Samples drawn when escalating.
        num_test: Number of held-out items.
    """

    greedy_accuracy: float
    self_consistency_accuracy: float
    curve: list[CascadePoint]
    selected: CascadePoint | None
    num_samples: int
    num_test: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the report."""
        return {
            "greedy_accuracy": self.greedy_accuracy,
            "self_consistency_accuracy": self.self_consistency_accuracy,
            "curve": [point.as_dict() for point in self.curve],
            "selected": self.selected.as_dict() if self.selected else None,
            "num_samples": self.num_samples,
            "num_test": self.num_test,
        }


def cascade_accuracy(
    greedy_correct: BoolArray,
    majority_correct: BoolArray,
    uncertainty: FloatArray,
    threshold: float,
) -> tuple[float, float]:
    """Accuracy and escalation rate of the cascade at a fixed threshold.

    Args:
        greedy_correct: Correctness of the greedy answer per item.
        majority_correct: Correctness of the majority-vote answer per item.
        uncertainty: Greedy-pass uncertainty per item.
        threshold: Escalate when uncertainty exceeds this value.

    Returns:
        A ``(accuracy, escalation_rate)`` pair.
    """
    if uncertainty.size == 0:
        return float("nan"), float("nan")

    escalate = uncertainty > threshold
    chosen = np.where(escalate, majority_correct, greedy_correct)
    return float(chosen.mean()), float(escalate.mean())


def _threshold_for_budget(uncertainty: FloatArray, budget: float) -> float:
    """Return the uncertainty cut-point that escalates a given dev fraction.

    Args:
        uncertainty: Dev-split uncertainty values.
        budget: Fraction of items to escalate, in [0, 1].

    Returns:
        The threshold. A budget of 0 returns ``+inf`` (never escalate) and a
        budget of 1 returns ``-inf`` (always escalate).
    """
    if budget <= 0.0:
        return float("inf")
    if budget >= 1.0:
        return float("-inf")
    if uncertainty.size == 0:
        return float("inf")
    return float(np.quantile(uncertainty, 1.0 - budget))


def _generation_cost(escalation_rate: float, num_samples: int) -> float:
    """Mean generations per item under the cascade.

    Args:
        escalation_rate: Fraction of items escalated to sampling.
        num_samples: Samples drawn when escalating.

    Returns:
        Average number of generations per item, greedy counted as one.
    """
    return 1.0 + escalation_rate * num_samples


def build_cascade_report(
    signal: str,
    dev_values: Sequence[float],
    dev_greedy_correct: Sequence[bool],
    dev_majority_correct: Sequence[bool],
    test_values: Sequence[float],
    test_greedy_correct: Sequence[bool],
    test_majority_correct: Sequence[bool],
    num_samples: int = NUM_SAMPLES,
) -> CascadeReport:
    """Build the cascade's accuracy-cost curve and pick an operating point.

    Thresholds are set from dev-split quantiles so each budget escalates the
    intended fraction of dev items; every reported number is then computed on
    the held-out split.

    Args:
        signal: Signal name gating escalation.
        dev_values: Dev-split raw signal values.
        dev_greedy_correct: Dev-split greedy correctness.
        dev_majority_correct: Dev-split majority-vote correctness.
        test_values: Held-out raw signal values.
        test_greedy_correct: Held-out greedy correctness.
        test_majority_correct: Held-out majority-vote correctness.
        num_samples: Samples drawn when escalating.

    Returns:
        The populated report.
    """
    dev_uncertainty = to_uncertainty(signal, dev_values)
    test_uncertainty = to_uncertainty(signal, test_values)

    dev_mask = np.isfinite(dev_uncertainty)
    test_mask = np.isfinite(test_uncertainty)

    dev_uncertainty = dev_uncertainty[dev_mask]
    dev_greedy = np.asarray(dev_greedy_correct, dtype=bool)[dev_mask]
    dev_majority = np.asarray(dev_majority_correct, dtype=bool)[dev_mask]

    test_uncertainty = test_uncertainty[test_mask]
    test_greedy = np.asarray(test_greedy_correct, dtype=bool)[test_mask]
    test_majority = np.asarray(test_majority_correct, dtype=bool)[test_mask]

    always_sample_cost = _generation_cost(1.0, num_samples)
    curve: list[CascadePoint] = []
    dev_accuracies: list[float] = []

    for budget in _ESCALATION_BUDGETS:
        threshold = _threshold_for_budget(dev_uncertainty, budget)

        dev_accuracy, _ = cascade_accuracy(
            dev_greedy, dev_majority, dev_uncertainty, threshold
        )
        dev_accuracies.append(dev_accuracy)

        test_accuracy, escalation_rate = cascade_accuracy(
            test_greedy, test_majority, test_uncertainty, threshold
        )
        mean_generations = _generation_cost(escalation_rate, num_samples)

        curve.append(
            CascadePoint(
                escalation_budget=budget,
                threshold=threshold,
                test_escalation_rate=escalation_rate,
                test_accuracy=test_accuracy,
                mean_generations=mean_generations,
                generation_cost_ratio=mean_generations / always_sample_cost,
            )
        )

    selected = _select_cheapest_point(curve, dev_accuracies)

    return CascadeReport(
        greedy_accuracy=float(test_greedy.mean()) if test_greedy.size else float("nan"),
        self_consistency_accuracy=(
            float(test_majority.mean()) if test_majority.size else float("nan")
        ),
        curve=curve,
        selected=selected,
        num_samples=num_samples,
        num_test=int(test_greedy.size),
    )


def _select_cheapest_point(
    curve: Sequence[CascadePoint], dev_accuracies: Sequence[float]
) -> CascadePoint | None:
    """Pick the lowest-budget point that matches full escalation on dev.

    Selection uses dev accuracies only; the corresponding test numbers are
    read off afterwards, so the choice never sees held-out data.

    Args:
        curve: Cascade points, ordered by increasing budget.
        dev_accuracies: Dev-split accuracy at each corresponding budget.

    Returns:
        The chosen point, or ``None`` for an empty curve.
    """
    if not curve:
        return None

    finite = [value for value in dev_accuracies if np.isfinite(value)]
    if not finite:
        return curve[0]

    best_dev_accuracy = max(finite)
    for point, dev_accuracy in zip(curve, dev_accuracies):
        if (
            np.isfinite(dev_accuracy)
            and dev_accuracy >= best_dev_accuracy - _ACCURACY_TOLERANCE
        ):
            logger.info(
                "Cascade: budget %.0f%% matches best dev accuracy %.3f",
                point.escalation_budget * 100,
                best_dev_accuracy,
            )
            return point

    return curve[-1]
