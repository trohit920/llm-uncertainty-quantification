"""Evaluation of every signal against ground-truth correctness.

Discrimination metrics (AUROC, AURC) are rank-based and can be computed from
a raw signal directly. Calibration metrics cannot: ECE and Brier both need a
value on a probability scale, and an entropy in nats is not one.

The fix is a one-feature logistic calibrator mapping the signal to
P(correct). It is fitted on the **dev split only** and applied to the held-out
test split, which is the single place in this project where leakage could
creep in. Fitting it on the same data the ECE is reported on would make every
method look far better calibrated than it is.

All headline numbers carry percentile bootstrap intervals; at 100-200
examples per task, differences smaller than the interval width are not
findings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .calibration import (
    Interval,
    ReliabilityDiagram,
    area_under_risk_coverage,
    bootstrap_interval,
    brier_score,
    error_detection_auroc,
    expected_calibration_error,
    reliability_diagram,
    risk_coverage_curve,
)
from .config import NUM_CALIBRATION_BINS, SEED
from .signals import available_signals, get_spec, to_uncertainty

logger = logging.getLogger(__name__)

#: Regularisation strength for the one-feature logistic calibrator. Weak
#: enough not to bias the fit, strong enough to survive separable dev splits.
_CALIBRATOR_REGULARISATION: Final[float] = 1.0

#: Minimum dev examples of each class needed to fit a calibrator.
_MIN_CLASS_COUNT: Final[int] = 2

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class SignalEvaluation:
    """Metrics for one signal on one task.

    Attributes:
        signal: Signal name.
        display_name: Human-readable label.
        granularity: ``"token"`` or ``"sentence"``.
        cost_tier: Cost tier from the signal registry.
        auroc: Error-detection AUROC with bootstrap interval.
        aurc: Area under the risk-coverage curve with bootstrap interval.
        ece_fixed_width: ECE on fixed-width bins, post-calibration.
        ece_equal_mass: ECE on equal-mass bins, post-calibration.
        brier: Brier score of the calibrated probabilities.
        num_test: Number of held-out examples the metrics were computed on.
    """

    signal: str
    display_name: str
    granularity: str
    cost_tier: str
    auroc: Interval
    aurc: Interval
    ece_fixed_width: float
    ece_equal_mass: float
    brier: float
    num_test: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the evaluation."""
        return {
            "signal": self.signal,
            "display_name": self.display_name,
            "granularity": self.granularity,
            "cost_tier": self.cost_tier,
            "auroc": self.auroc.as_dict(),
            "aurc": self.aurc.as_dict(),
            "ece_fixed_width": self.ece_fixed_width,
            "ece_equal_mass": self.ece_equal_mass,
            "brier": self.brier,
            "num_test": self.num_test,
        }


class LogisticCalibrator:
    """Maps a scalar uncertainty onto P(correct) via logistic regression."""

    def __init__(self) -> None:
        """Initialise an unfitted calibrator."""
        self._model: Any | None = None
        self._fallback_rate: float = 0.5

    def fit(self, uncertainty: FloatArray, is_correct: BoolArray) -> "LogisticCalibrator":
        """Fit on the dev split.

        Args:
            uncertainty: Dev-split uncertainty values, higher meaning less
                trustworthy.
            is_correct: Dev-split correctness labels.

        Returns:
            ``self``, fitted. When one class is absent or the split is too
            small, the calibrator degrades to predicting the dev base rate,
            which keeps the pipeline running and is reported honestly rather
            than silently producing a spurious ECE.
        """
        from sklearn.linear_model import LogisticRegression

        self._fallback_rate = (
            float(np.mean(is_correct)) if is_correct.size else 0.5
        )

        num_correct = int(np.sum(is_correct))
        num_wrong = int(np.sum(~is_correct))
        if min(num_correct, num_wrong) < _MIN_CLASS_COUNT:
            logger.warning(
                "Calibrator falling back to base rate: dev split has "
                "%d correct / %d wrong",
                num_correct,
                num_wrong,
            )
            self._model = None
            return self

        model = LogisticRegression(C=_CALIBRATOR_REGULARISATION)
        model.fit(uncertainty.reshape(-1, 1), is_correct.astype(int))
        self._model = model
        return self

    def predict_proba(self, uncertainty: FloatArray) -> FloatArray:
        """Predict P(correct) for held-out uncertainty values.

        Args:
            uncertainty: Test-split uncertainty values.

        Returns:
            Probabilities in [0, 1], one per input.
        """
        if self._model is None:
            return np.full(uncertainty.shape, self._fallback_rate, dtype=np.float64)
        probabilities = self._model.predict_proba(uncertainty.reshape(-1, 1))
        return probabilities[:, 1].astype(np.float64)


def _finite_mask(values: FloatArray) -> BoolArray:
    """Return a mask selecting finite entries."""
    return np.isfinite(values)


def evaluate_signal(
    signal: str,
    dev_values: Sequence[float],
    dev_correct: Sequence[bool],
    test_values: Sequence[float],
    test_correct: Sequence[bool],
    num_bins: int = NUM_CALIBRATION_BINS,
    seed: int = SEED,
) -> SignalEvaluation:
    """Evaluate one signal, fitting calibration on dev and reporting on test.

    Args:
        signal: Signal name, used for orientation and metadata.
        dev_values: Raw signal values on the dev split.
        dev_correct: Correctness labels on the dev split.
        test_values: Raw signal values on the test split.
        test_correct: Correctness labels on the test split.
        num_bins: Bin count for the reliability diagrams.
        seed: Seed for the bootstrap.

    Returns:
        The populated evaluation. Examples whose signal is non-finite are
        dropped from both splits before any metric is computed.
    """
    spec = get_spec(signal)

    dev_uncertainty = to_uncertainty(signal, dev_values)
    test_uncertainty = to_uncertainty(signal, test_values)
    dev_labels = np.asarray(dev_correct, dtype=bool)
    test_labels = np.asarray(test_correct, dtype=bool)

    dev_mask = _finite_mask(dev_uncertainty)
    test_mask = _finite_mask(test_uncertainty)
    dev_uncertainty, dev_labels = dev_uncertainty[dev_mask], dev_labels[dev_mask]
    test_uncertainty, test_labels = (
        test_uncertainty[test_mask],
        test_labels[test_mask],
    )

    if dev_mask.size and not dev_mask.all():
        logger.info(
            "Signal %s: dropped %d non-finite dev values",
            signal,
            int((~dev_mask).sum()),
        )

    auroc = bootstrap_interval(
        error_detection_auroc, test_uncertainty, test_labels, seed=seed
    )
    aurc = bootstrap_interval(
        area_under_risk_coverage, test_uncertainty, test_labels, seed=seed
    )

    calibrator = LogisticCalibrator().fit(dev_uncertainty, dev_labels)
    test_confidence = calibrator.predict_proba(test_uncertainty)

    return SignalEvaluation(
        signal=signal,
        display_name=spec.display_name,
        granularity=spec.granularity,
        cost_tier=spec.cost_tier,
        auroc=auroc,
        aurc=aurc,
        ece_fixed_width=expected_calibration_error(
            test_confidence, test_labels, num_bins, "fixed_width"
        ),
        ece_equal_mass=expected_calibration_error(
            test_confidence, test_labels, num_bins, "equal_mass"
        ),
        brier=brier_score(test_confidence, test_labels),
        num_test=int(test_labels.size),
    )


def evaluate_all_signals(
    dev_rows: Mapping[str, Sequence[float]],
    dev_correct: Sequence[bool],
    test_rows: Mapping[str, Sequence[float]],
    test_correct: Sequence[bool],
    seed: int = SEED,
) -> list[SignalEvaluation]:
    """Evaluate every usable signal on a task.

    Args:
        dev_rows: Mapping from signal name to its dev-split column.
        dev_correct: Dev-split correctness labels.
        test_rows: Mapping from signal name to its test-split column.
        test_correct: Test-split correctness labels.
        seed: Seed for the bootstrap.

    Returns:
        Evaluations sorted by descending AUROC, with undefined AUROCs last.
    """
    usable = available_signals(
        {name: np.asarray(column, dtype=np.float64) for name, column in test_rows.items()}
    )
    skipped = set(test_rows) - set(usable)
    if skipped:
        logger.info(
            "Skipping %d degenerate signals (constant or all-nan): %s",
            len(skipped),
            sorted(skipped),
        )

    evaluations = [
        evaluate_signal(
            name,
            dev_rows[name],
            dev_correct,
            test_rows[name],
            test_correct,
            seed=seed,
        )
        for name in usable
    ]

    return sorted(
        evaluations,
        key=lambda ev: (
            -ev.auroc.value if np.isfinite(ev.auroc.value) else np.inf
        ),
    )


def build_reliability_diagram(
    signal: str,
    dev_values: Sequence[float],
    dev_correct: Sequence[bool],
    test_values: Sequence[float],
    test_correct: Sequence[bool],
    strategy: str = "fixed_width",
    num_bins: int = NUM_CALIBRATION_BINS,
) -> ReliabilityDiagram:
    """Produce the reliability diagram used by the plotting stage.

    Args:
        signal: Signal name.
        dev_values: Dev-split raw signal values, used to fit the calibrator.
        dev_correct: Dev-split correctness labels.
        test_values: Test-split raw signal values.
        test_correct: Test-split correctness labels.
        strategy: ``"fixed_width"`` or ``"equal_mass"``.
        num_bins: Bin count.

    Returns:
        The diagram computed on the held-out split.
    """
    dev_uncertainty = to_uncertainty(signal, dev_values)
    test_uncertainty = to_uncertainty(signal, test_values)
    dev_labels = np.asarray(dev_correct, dtype=bool)
    test_labels = np.asarray(test_correct, dtype=bool)

    dev_mask, test_mask = _finite_mask(dev_uncertainty), _finite_mask(test_uncertainty)
    calibrator = LogisticCalibrator().fit(
        dev_uncertainty[dev_mask], dev_labels[dev_mask]
    )
    confidence = calibrator.predict_proba(test_uncertainty[test_mask])

    return reliability_diagram(
        confidence, test_labels[test_mask], num_bins, strategy
    )


def build_risk_coverage_curve(
    signal: str, values: Sequence[float], is_correct: Sequence[bool]
) -> dict[str, Any]:
    """Produce a risk-coverage curve for plotting.

    Args:
        signal: Signal name, used for orientation.
        values: Raw signal values.
        is_correct: Correctness labels.

    Returns:
        A JSON-serialisable curve description.
    """
    uncertainty = to_uncertainty(signal, values)
    labels = np.asarray(is_correct, dtype=bool)
    mask = _finite_mask(uncertainty)
    curve = risk_coverage_curve(uncertainty[mask], labels[mask])
    return {"signal": signal, **curve.as_dict()}


def accuracy(is_correct: Sequence[bool]) -> float:
    """Return the mean correctness rate.

    Args:
        is_correct: Correctness labels.

    Returns:
        The accuracy, or ``nan`` for an empty input.
    """
    labels = np.asarray(is_correct, dtype=bool)
    return float(labels.mean()) if labels.size else float("nan")
