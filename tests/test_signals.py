"""Unit tests for the signal registry and its orientation logic.

A sign error here would silently invert every AUROC, so orientation is tested
explicitly for both families rather than assumed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.sentence_metrics import SentenceUncertainty
from src.signals import (
    SIGNAL_NAMES,
    SIGNAL_REGISTRY,
    available_signals,
    build_signal_row,
    get_spec,
    signals_by_cost_tier,
    signals_by_granularity,
    to_uncertainty,
)
from src.token_metrics import TokenUncertainty


def make_token_uncertainty(value: float = 1.0) -> TokenUncertainty:
    """Build a filled token-uncertainty bundle for tests."""
    return TokenUncertainty(
        mean_entropy=value,
        max_entropy=value,
        log_perplexity=value,
        mean_max_probability=0.5,
        min_max_probability=0.4,
        mean_margin=0.3,
        min_margin=0.2,
        sequence_log_probability=-value,
        sequence_probability=math.exp(-value),
        num_tokens=4,
    )


def make_sentence_uncertainty() -> SentenceUncertainty:
    """Build a filled sentence-uncertainty bundle for tests."""
    return SentenceUncertainty(
        discrete_semantic_entropy=0.6,
        weighted_semantic_entropy=0.5,
        lexical_entropy=0.9,
        self_consistency=0.7,
        semantic_self_consistency=0.8,
        num_semantic_clusters=2,
        num_distinct_answers=3,
        num_samples=10,
    )


class TestRegistry:
    """Structure of the signal registry."""

    def test_every_name_resolves(self) -> None:
        for name in SIGNAL_NAMES:
            assert get_spec(name).name == name

    def test_unknown_signal_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown signal"):
            get_spec("not_a_signal")

    def test_answer_span_mirrors_token_signals(self) -> None:
        token_names = {
            name
            for name in SIGNAL_NAMES
            if get_spec(name).granularity == "token"
            and not name.startswith("answer_")
        }
        for name in token_names:
            assert f"answer_{name}" in SIGNAL_REGISTRY

    def test_answer_span_inherits_orientation(self) -> None:
        for name in SIGNAL_NAMES:
            if not name.startswith("answer_"):
                continue
            base = name.removeprefix("answer_")
            assert (
                get_spec(name).higher_means_more_uncertain
                == get_spec(base).higher_means_more_uncertain
            )

    def test_cost_tiers_partition_the_registry(self) -> None:
        tiers = ("single_pass", "extra_call", "multi_sample")
        covered = {name for tier in tiers for name in signals_by_cost_tier(tier)}
        assert covered == set(SIGNAL_NAMES)

    def test_granularities_partition_the_registry(self) -> None:
        covered = set(signals_by_granularity("token")) | set(
            signals_by_granularity("sentence")
        )
        assert covered == set(SIGNAL_NAMES)


class TestOrientation:
    """Conversion to a uniform uncertainty orientation."""

    def test_entropy_is_left_unchanged(self) -> None:
        values = [0.1, 0.5, 0.9]
        assert np.allclose(to_uncertainty("mean_entropy", values), values)

    def test_probability_is_negated(self) -> None:
        values = [0.1, 0.5, 0.9]
        assert np.allclose(to_uncertainty("mean_max_prob", values), [-0.1, -0.5, -0.9])

    def test_self_consistency_is_negated(self) -> None:
        assert np.allclose(to_uncertainty("self_consistency", [1.0]), [-1.0])

    def test_semantic_entropy_is_not_negated(self) -> None:
        assert np.allclose(to_uncertainty("semantic_entropy", [1.0]), [1.0])

    def test_orientation_preserves_rank_magnitude(self) -> None:
        # Negation must reverse the ordering, since that is the whole point.
        values = np.array([0.1, 0.9])
        oriented = to_uncertainty("mean_max_prob", values)
        assert oriented[0] > oriented[1]

    def test_unknown_signal_raises(self) -> None:
        with pytest.raises(KeyError):
            to_uncertainty("not_a_signal", [1.0])


class TestBuildSignalRow:
    """Assembly of a per-example signal mapping."""

    def test_row_has_every_registered_signal(self) -> None:
        row = build_signal_row(
            make_token_uncertainty(),
            make_token_uncertainty(2.0),
            make_sentence_uncertainty(),
            0.75,
        )
        assert set(row) == set(SIGNAL_NAMES)

    def test_missing_answer_span_yields_nan_not_absence(self) -> None:
        # Rows must stay rectangular so downstream tables line up.
        row = build_signal_row(
            make_token_uncertainty(), None, make_sentence_uncertainty(), None
        )
        assert set(row) == set(SIGNAL_NAMES)
        assert math.isnan(row["answer_mean_entropy"])
        assert math.isnan(row["verbalized_confidence"])

    def test_answer_span_values_are_namespaced(self) -> None:
        row = build_signal_row(
            make_token_uncertainty(1.0),
            make_token_uncertainty(2.0),
            make_sentence_uncertainty(),
            None,
        )
        assert row["mean_entropy"] == pytest.approx(1.0)
        assert row["answer_mean_entropy"] == pytest.approx(2.0)


class TestAvailableSignals:
    """Filtering out signals that cannot rank anything."""

    def test_constant_column_is_excluded(self) -> None:
        rows = {
            "mean_entropy": np.array([1.0, 1.0, 1.0]),
            "max_entropy": np.array([1.0, 2.0, 3.0]),
        }
        assert available_signals(rows) == ("max_entropy",)

    def test_all_nan_column_is_excluded(self) -> None:
        rows = {
            "mean_entropy": np.array([np.nan, np.nan]),
            "max_entropy": np.array([1.0, 2.0]),
        }
        assert available_signals(rows) == ("max_entropy",)

    def test_result_follows_registry_order(self) -> None:
        rows = {
            "max_entropy": np.array([1.0, 2.0]),
            "mean_entropy": np.array([3.0, 4.0]),
        }
        result = available_signals(rows)
        assert result.index("mean_entropy") < result.index("max_entropy")
