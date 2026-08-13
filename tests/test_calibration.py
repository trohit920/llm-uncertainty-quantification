"""Unit tests for the hand-implemented evaluation metrics.

Where a reference implementation exists (scikit-learn's ``roc_auc_score``) the
hand-rolled version is cross-checked against it, including on tied scores,
which is where naive rank implementations diverge.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.calibration import (
    area_under_risk_coverage,
    bootstrap_interval,
    brier_score,
    error_detection_auroc,
    expected_calibration_error,
    reliability_diagram,
    risk_coverage_curve,
)


class TestErrorDetectionAuroc:
    """Rank-based AUROC with incorrect answers as the positive class."""

    def test_perfect_separation_scores_one(self) -> None:
        uncertainty = [0.1, 0.2, 0.8, 0.9]
        is_correct = [True, True, False, False]
        assert error_detection_auroc(uncertainty, is_correct) == pytest.approx(1.0)

    def test_inverted_separation_scores_zero(self) -> None:
        uncertainty = [0.9, 0.8, 0.2, 0.1]
        is_correct = [True, True, False, False]
        assert error_detection_auroc(uncertainty, is_correct) == pytest.approx(0.0)

    def test_all_tied_scores_one_half(self) -> None:
        uncertainty = [0.5, 0.5, 0.5, 0.5]
        is_correct = [True, False, True, False]
        assert error_detection_auroc(uncertainty, is_correct) == pytest.approx(0.5)

    def test_single_class_is_undefined(self) -> None:
        assert math.isnan(error_detection_auroc([0.1, 0.2], [True, True]))
        assert math.isnan(error_detection_auroc([0.1, 0.2], [False, False]))

    def test_matches_sklearn_on_random_data(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            uncertainty = rng.normal(size=50)
            is_correct = rng.random(size=50) > 0.5
            expected = roc_auc_score(~is_correct, uncertainty)
            actual = error_detection_auroc(uncertainty, is_correct)
            assert actual == pytest.approx(expected)

    def test_matches_sklearn_with_heavy_ties(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(20):
            # Coarse rounding forces many tied scores.
            uncertainty = np.round(rng.normal(size=60), decimals=0)
            is_correct = rng.random(size=60) > 0.4
            if is_correct.all() or (~is_correct).all():
                continue
            expected = roc_auc_score(~is_correct, uncertainty)
            assert error_detection_auroc(uncertainty, is_correct) == pytest.approx(
                expected
            )

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            error_detection_auroc([0.1, 0.2], [True])


class TestExpectedCalibrationError:
    """Binned confidence-versus-accuracy gap."""

    def test_perfect_confidence_and_accuracy_is_zero(self) -> None:
        confidence = [1.0] * 10
        is_correct = [True] * 10
        assert expected_calibration_error(confidence, is_correct) == pytest.approx(0.0)

    def test_maximally_overconfident_is_one(self) -> None:
        confidence = [1.0] * 10
        is_correct = [False] * 10
        assert expected_calibration_error(confidence, is_correct) == pytest.approx(1.0)

    def test_well_calibrated_half_confidence(self) -> None:
        # Ten items at confidence 0.5, exactly half of them correct.
        confidence = [0.5] * 10
        is_correct = [True] * 5 + [False] * 5
        assert expected_calibration_error(confidence, is_correct) == pytest.approx(0.0)

    def test_equal_mass_binning_populates_every_bin(self) -> None:
        rng = np.random.default_rng(2)
        # Scores concentrated near 1.0 leave most fixed-width bins empty.
        confidence = np.clip(rng.normal(loc=0.95, scale=0.02, size=200), 0.0, 1.0)
        is_correct = rng.random(size=200) < confidence

        fixed = reliability_diagram(confidence, is_correct, strategy="fixed_width")
        equal_mass = reliability_diagram(
            confidence, is_correct, strategy="equal_mass"
        )

        assert len(equal_mass.bins) > len(fixed.bins)
        counts = [b.count for b in equal_mass.bins]
        assert max(counts) - min(counts) <= max(counts) / 2

    def test_bin_counts_sum_to_sample_size(self) -> None:
        rng = np.random.default_rng(3)
        confidence = rng.random(size=137)
        is_correct = rng.random(size=137) > 0.5
        for strategy in ("fixed_width", "equal_mass"):
            diagram = reliability_diagram(confidence, is_correct, strategy=strategy)
            assert sum(b.count for b in diagram.bins) == 137

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown binning strategy"):
            reliability_diagram([0.5], [True], strategy="nonsense")

    def test_empty_input_is_nan(self) -> None:
        assert math.isnan(expected_calibration_error([], []))


class TestBrierScore:
    """Mean squared error of the confidence estimate."""

    def test_perfect_prediction_is_zero(self) -> None:
        assert brier_score([1.0, 0.0], [True, False]) == pytest.approx(0.0)

    def test_worst_prediction_is_one(self) -> None:
        assert brier_score([0.0, 1.0], [True, False]) == pytest.approx(1.0)


class TestRiskCoverage:
    """Selective-prediction risk swept over coverage."""

    def test_known_curve(self) -> None:
        # Sorted by uncertainty the labels are [T, T, F, F], so cumulative
        # risk runs 0, 0, 1/3, 2/4.
        curve = risk_coverage_curve([0, 1, 2, 3], [True, True, False, False])
        assert curve.risk.tolist() == pytest.approx([0.0, 0.0, 1 / 3, 0.5])
        assert curve.coverage.tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0])
        assert curve.area_under_risk_coverage == pytest.approx(
            (0.0 + 0.0 + 1 / 3 + 0.5) / 4
        )

    def test_perfect_ranking_beats_inverted_ranking(self) -> None:
        is_correct = [True, True, True, False, False]
        good = area_under_risk_coverage([0, 1, 2, 3, 4], is_correct)
        bad = area_under_risk_coverage([4, 3, 2, 1, 0], is_correct)
        assert good < bad

    def test_full_coverage_risk_equals_error_rate(self) -> None:
        is_correct = [True, False, False, True]
        curve = risk_coverage_curve([0.1, 0.2, 0.3, 0.4], is_correct)
        assert curve.risk[-1] == pytest.approx(0.5)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            risk_coverage_curve([0.1], [True, False])


class TestBootstrapInterval:
    """Percentile bootstrap over paired (score, label) resamples."""

    def test_interval_brackets_point_estimate(self) -> None:
        rng = np.random.default_rng(4)
        uncertainty = rng.normal(size=100)
        is_correct = uncertainty < 0.0  # strong but imperfect signal

        interval = bootstrap_interval(
            error_detection_auroc, uncertainty, is_correct, num_resamples=200
        )
        assert interval.lower <= interval.value <= interval.upper

    def test_is_deterministic_under_seed(self) -> None:
        rng = np.random.default_rng(5)
        uncertainty = rng.normal(size=60)
        is_correct = rng.random(size=60) > 0.5

        first = bootstrap_interval(
            error_detection_auroc, uncertainty, is_correct, num_resamples=100
        )
        second = bootstrap_interval(
            error_detection_auroc, uncertainty, is_correct, num_resamples=100
        )
        assert first == second

    def test_empty_input_is_nan(self) -> None:
        interval = bootstrap_interval(error_detection_auroc, [], [])
        assert math.isnan(interval.value)
