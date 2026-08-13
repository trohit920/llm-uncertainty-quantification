"""Unit tests for selective answering, confidence tags and the cascade.

Signal orientation matters throughout: ``mean_entropy`` is uncertainty-oriented
(higher is worse) while ``self_consistency`` is confidence-oriented (higher is
better), and these tests exercise both so a sign error cannot pass.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.applications import (
    assign_confidence_tags,
    confidence_tag_report,
    fit_abstention_threshold,
    fit_confidence_tag_thresholds,
    selective_answering,
    tier_separation,
)
from src.self_consistency import (
    build_cascade_report,
    cascade_accuracy,
)


class TestFitAbstentionThreshold:
    """Threshold selection under an accuracy constraint."""

    def test_maximises_coverage_subject_to_target(self) -> None:
        # Sorted by uncertainty the labels are [T, T, T, F, F]. A 100% target
        # can only be met by answering the first three.
        uncertainty = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        is_correct = np.array([True, True, True, False, False])

        threshold = fit_abstention_threshold(uncertainty, is_correct, 1.0)
        assert threshold == pytest.approx(0.3)

    def test_lower_target_allows_more_coverage(self) -> None:
        uncertainty = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        is_correct = np.array([True, True, True, False, False])

        strict = fit_abstention_threshold(uncertainty, is_correct, 1.0)
        lenient = fit_abstention_threshold(uncertainty, is_correct, 0.6)
        assert lenient > strict

    def test_unreachable_target_abstains_entirely(self) -> None:
        uncertainty = np.array([0.1, 0.2, 0.3])
        is_correct = np.array([False, False, False])

        threshold = fit_abstention_threshold(uncertainty, is_correct, 0.9)
        assert threshold == float("-inf")

    def test_empty_input_abstains(self) -> None:
        threshold = fit_abstention_threshold(
            np.array([]), np.array([], dtype=bool), 0.8
        )
        assert threshold == float("-inf")


class TestSelectiveAnswering:
    """End-to-end operating points across targets."""

    def _perfect_signal(self) -> tuple[list[float], list[bool]]:
        """A signal that ranks every error last."""
        values = [0.0, 0.1, 0.2, 0.3, 0.8, 0.9]
        correct = [True, True, True, True, False, False]
        return values, correct

    def test_coverage_decreases_as_target_rises(self) -> None:
        values, correct = self._perfect_signal()
        points = selective_answering(
            "mean_entropy", values, correct, values, correct, (0.5, 0.9)
        )
        assert points[0].test_coverage >= points[1].test_coverage

    def test_perfect_signal_reaches_full_accuracy(self) -> None:
        values, correct = self._perfect_signal()
        points = selective_answering(
            "mean_entropy", values, correct, values, correct, (1.0,)
        )
        assert points[0].test_accuracy == pytest.approx(1.0)
        assert points[0].test_coverage == pytest.approx(4 / 6)

    def test_confidence_oriented_signal_is_flipped(self) -> None:
        # self_consistency is confidence-oriented: high values are GOOD, so
        # the correct answers must carry the high values here.
        values = [0.9, 0.9, 0.8, 0.2, 0.1]
        correct = [True, True, True, False, False]
        points = selective_answering(
            "self_consistency", values, correct, values, correct, (1.0,)
        )
        assert points[0].test_accuracy == pytest.approx(1.0)
        assert points[0].test_num_answered == 3

    def test_nan_values_are_dropped(self) -> None:
        values = [0.1, float("nan"), 0.3, 0.9]
        correct = [True, True, True, False]
        points = selective_answering(
            "mean_entropy", values, correct, values, correct, (0.5,)
        )
        assert points[0].test_num_answered <= 3

    def test_operating_point_serialises(self) -> None:
        values, correct = self._perfect_signal()
        point = selective_answering(
            "mean_entropy", values, correct, values, correct, (0.8,)
        )[0]
        assert set(point.as_dict()) >= {"target_accuracy", "test_coverage"}


class TestConfidenceTags:
    """High / Medium / Low tagging."""

    def test_thresholds_are_ordered(self) -> None:
        uncertainty = np.linspace(0.0, 1.0, 100)
        low_cut, high_cut = fit_confidence_tag_thresholds(uncertainty)
        assert low_cut < high_cut

    def test_assignment_covers_all_three_tiers(self) -> None:
        uncertainty = np.linspace(0.0, 1.0, 99)
        thresholds = fit_confidence_tag_thresholds(uncertainty)
        tags = assign_confidence_tags(uncertainty, thresholds)
        assert set(tags) == {"High", "Medium", "Low"}

    def test_lowest_uncertainty_is_high_confidence(self) -> None:
        thresholds = (0.3, 0.6)
        tags = assign_confidence_tags(np.array([0.1, 0.45, 0.9]), thresholds)
        assert tags == ["High", "Medium", "Low"]

    def test_report_separates_accuracy_by_tier(self) -> None:
        # Uncertainty ascends while correctness descends, so High must be the
        # most accurate tier.
        values = list(np.linspace(0.0, 1.0, 30))
        correct = [i < 15 for i in range(30)]

        reports = confidence_tag_report(
            "mean_entropy", values, correct, values, correct
        )
        by_tier = {report.tier: report.accuracy for report in reports}
        assert by_tier["High"] > by_tier["Low"]
        assert tier_separation(reports) > 0

    def test_shares_sum_to_one(self) -> None:
        values = list(np.linspace(0.0, 1.0, 60))
        correct = [i % 2 == 0 for i in range(60)]
        reports = confidence_tag_report(
            "mean_entropy", values, correct, values, correct
        )
        assert sum(report.share for report in reports) == pytest.approx(1.0)

    def test_separation_is_nan_when_a_tier_is_empty(self) -> None:
        reports = confidence_tag_report(
            "mean_entropy", [0.5], [True], [], []
        )
        assert math.isnan(tier_separation(reports))


class TestCascadeAccuracy:
    """Accuracy of the greedy/sampling switch at a fixed threshold."""

    def test_never_escalating_equals_greedy(self) -> None:
        greedy = np.array([True, False, True])
        majority = np.array([True, True, True])
        uncertainty = np.array([0.1, 0.5, 0.9])

        result, rate = cascade_accuracy(
            greedy, majority, uncertainty, float("inf")
        )
        assert result == pytest.approx(greedy.mean())
        assert rate == pytest.approx(0.0)

    def test_always_escalating_equals_majority(self) -> None:
        greedy = np.array([True, False, True])
        majority = np.array([True, True, True])
        uncertainty = np.array([0.1, 0.5, 0.9])

        result, rate = cascade_accuracy(
            greedy, majority, uncertainty, float("-inf")
        )
        assert result == pytest.approx(majority.mean())
        assert rate == pytest.approx(1.0)

    def test_selective_escalation_beats_both_when_signal_is_good(self) -> None:
        # Majority fixes item 1 but breaks item 2; a signal that flags only
        # item 1 outperforms either fixed policy.
        greedy = np.array([False, True])
        majority = np.array([True, False])
        uncertainty = np.array([0.9, 0.1])

        result, rate = cascade_accuracy(greedy, majority, uncertainty, 0.5)
        assert result == pytest.approx(1.0)
        assert rate == pytest.approx(0.5)


class TestBuildCascadeReport:
    """The full accuracy-versus-cost sweep."""

    def _data(self) -> dict[str, list]:
        """A case where the signal perfectly identifies greedy failures."""
        values = [0.9, 0.8, 0.1, 0.05]
        greedy = [False, False, True, True]
        majority = [True, True, True, True]
        return {"values": values, "greedy": greedy, "majority": majority}

    def test_endpoints_match_the_baselines(self) -> None:
        data = self._data()
        report = build_cascade_report(
            "mean_entropy",
            data["values"],
            data["greedy"],
            data["majority"],
            data["values"],
            data["greedy"],
            data["majority"],
            num_samples=10,
        )

        assert report.curve[0].test_accuracy == pytest.approx(
            report.greedy_accuracy
        )
        assert report.curve[-1].test_accuracy == pytest.approx(
            report.self_consistency_accuracy
        )

    def test_cost_rises_with_escalation(self) -> None:
        data = self._data()
        report = build_cascade_report(
            "mean_entropy",
            data["values"],
            data["greedy"],
            data["majority"],
            data["values"],
            data["greedy"],
            data["majority"],
        )
        costs = [point.mean_generations for point in report.curve]
        assert costs == sorted(costs)
        assert costs[0] == pytest.approx(1.0)

    def test_selected_point_is_cheaper_than_always_sampling(self) -> None:
        data = self._data()
        report = build_cascade_report(
            "mean_entropy",
            data["values"],
            data["greedy"],
            data["majority"],
            data["values"],
            data["greedy"],
            data["majority"],
            num_samples=10,
        )
        assert report.selected is not None
        assert report.selected.generation_cost_ratio < 1.0
        assert report.selected.test_accuracy == pytest.approx(1.0)

    def test_report_serialises(self) -> None:
        data = self._data()
        report = build_cascade_report(
            "mean_entropy",
            data["values"],
            data["greedy"],
            data["majority"],
            data["values"],
            data["greedy"],
            data["majority"],
        )
        payload = report.as_dict()
        assert "curve" in payload and len(payload["curve"]) > 1
