"""End-to-end tests of the analysis orchestration on synthetic records.

These exercise the whole chain from a JSONL record file to metrics,
applications and cascade reports without loading a model, so the post-run
stages are validated independently of GPU availability.

The synthetic generator plants a known structure: ``semantic_entropy`` is
genuinely informative about correctness while ``mean_entropy`` is pure noise.
Any regression that scrambles orientation, splitting or ranking shows up as
the wrong signal winning.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.analysis import analyse_task, save_metrics, split_indices, top_signals
from src.config import RECORDS_FILENAME
from src.signals import SIGNAL_NAMES


def make_synthetic_records(
    task: str, num_records: int = 120, seed: int = 0
) -> list[dict]:
    """Generate records with a known signal-to-correctness relationship.

    Args:
        task: Task name written into each record.
        num_records: Number of records to generate.
        seed: Seed for the generator.

    Returns:
        Records in the schema produced by :mod:`src.experiment`.
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    for index in range(num_records):
        is_correct = bool(rng.random() < 0.55)

        # Informative: lower semantic entropy on correct answers.
        semantic_entropy = float(
            rng.normal(loc=0.3 if is_correct else 1.2, scale=0.35)
        )
        # Uninformative: identical distribution regardless of correctness.
        mean_entropy = float(rng.normal(loc=0.8, scale=0.3))

        signals = {name: float(rng.normal()) for name in SIGNAL_NAMES}
        signals["semantic_entropy"] = semantic_entropy
        signals["mean_entropy"] = mean_entropy
        signals["self_consistency"] = float(
            np.clip(1.0 - semantic_entropy / 2.0, 0.05, 1.0)
        )

        # Majority voting rescues some wrong greedy answers.
        majority_correct = is_correct or bool(rng.random() < 0.35)

        records.append(
            {
                "example_id": f"{task}-{index:05d}",
                "task": task,
                "question": f"Synthetic question {index}?",
                "gold_answers": ["42"],
                "metadata": {},
                "greedy_text": "42",
                "greedy_num_tokens": 3,
                "greedy_correct": is_correct,
                "greedy_canonical": "42" if is_correct else "41",
                "answer_span": [1, 3],
                "sample_texts": ["42"] * 10,
                "sample_canonical": ["42"] * 10,
                "majority_answer": "42",
                "majority_correct": majority_correct,
                "sample_correct": [majority_correct] * 10,
                "signals": signals,
                "timings": {
                    "greedy_seconds": 0.5,
                    "sampling_seconds": 4.0,
                    "clustering_seconds": 0.2,
                    "verbalized_seconds": 0.1,
                },
            }
        )

    return records


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    """Write synthetic record files for both tasks into a temp directory."""
    for task in ("nq_open", "gsm8k"):
        path = tmp_path / RECORDS_FILENAME.format(task=task)
        records = make_synthetic_records(task, seed=hash(task) % 1000)
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
    return tmp_path


class TestSplitIndices:
    """Dev/test partitioning."""

    def test_splits_are_disjoint_and_exhaustive(self) -> None:
        dev, test = split_indices(100, dev_fraction=0.5, seed=1)
        assert set(dev.tolist()).isdisjoint(test.tolist())
        assert len(dev) + len(test) == 100

    def test_respects_the_requested_fraction(self) -> None:
        dev, test = split_indices(100, dev_fraction=0.3, seed=1)
        assert len(dev) == 30
        assert len(test) == 70

    def test_is_deterministic_under_seed(self) -> None:
        first, _ = split_indices(50, 0.5, seed=7)
        second, _ = split_indices(50, 0.5, seed=7)
        assert first.tolist() == second.tolist()

    def test_different_seeds_differ(self) -> None:
        first, _ = split_indices(50, 0.5, seed=1)
        second, _ = split_indices(50, 0.5, seed=2)
        assert first.tolist() != second.tolist()


class TestAnalyseTask:
    """The full record-to-results pipeline."""

    def test_recovers_the_informative_signal(self, results_dir: Path) -> None:
        analysis = analyse_task("nq_open", results_dir)
        # semantic_entropy was planted as informative; mean_entropy as noise.
        assert analysis.best_signal in {"semantic_entropy", "self_consistency"}

        by_name = {ev.signal: ev for ev in analysis.evaluations}
        assert by_name["semantic_entropy"].auroc.value > 0.7
        assert by_name["mean_entropy"].auroc.value < 0.65

    def test_splits_are_reported_consistently(self, results_dir: Path) -> None:
        analysis = analyse_task("nq_open", results_dir)
        assert analysis.num_dev + analysis.num_test == analysis.num_examples
        assert analysis.num_test > 0

    def test_evaluations_are_ranked_by_auroc(self, results_dir: Path) -> None:
        analysis = analyse_task("nq_open", results_dir)
        finite = [
            ev.auroc.value
            for ev in analysis.evaluations
            if np.isfinite(ev.auroc.value)
        ]
        assert finite == sorted(finite, reverse=True)

    def test_bootstrap_intervals_bracket_estimates(
        self, results_dir: Path
    ) -> None:
        analysis = analyse_task("nq_open", results_dir)
        for evaluation in analysis.evaluations:
            if np.isfinite(evaluation.auroc.value) and np.isfinite(
                evaluation.auroc.lower
            ):
                assert (
                    evaluation.auroc.lower
                    <= evaluation.auroc.value
                    <= evaluation.auroc.upper
                )

    def test_selective_answering_is_populated(self, results_dir: Path) -> None:
        analysis = analyse_task("nq_open", results_dir)
        assert len(analysis.selective) >= 1
        for point in analysis.selective:
            assert 0.0 <= point.test_coverage <= 1.0

    def test_confidence_tiers_separate(self, results_dir: Path) -> None:
        analysis = analyse_task("nq_open", results_dir)
        assert len(analysis.tiers) == 3
        assert analysis.tier_gap > 0

    def test_cascade_only_for_chain_of_thought_task(
        self, results_dir: Path
    ) -> None:
        assert analyse_task("nq_open", results_dir).cascade is None
        assert analyse_task("gsm8k", results_dir).cascade is not None

    def test_cascade_endpoints_are_the_baselines(
        self, results_dir: Path
    ) -> None:
        cascade = analyse_task("gsm8k", results_dir).cascade
        assert cascade is not None
        assert cascade.curve[0].mean_generations == pytest.approx(1.0)
        assert cascade.self_consistency_accuracy >= cascade.greedy_accuracy

    def test_cost_tiers_are_measured(self, results_dir: Path) -> None:
        analysis = analyse_task("nq_open", results_dir)
        assert analysis.cost_seconds["single_pass"] == pytest.approx(0.5)
        assert analysis.cost_seconds["multi_sample"] == pytest.approx(4.2)
        assert (
            analysis.cost_seconds["multi_sample"]
            > analysis.cost_seconds["single_pass"]
        )

    def test_top_signals_are_ordered(self, results_dir: Path) -> None:
        analysis = analyse_task("nq_open", results_dir)
        leaders = top_signals(analysis, count=3)
        assert len(leaders) == 3
        assert leaders[0] == analysis.best_signal

    def test_metrics_round_trip_to_json(
        self, results_dir: Path
    ) -> None:
        analysis = analyse_task("gsm8k", results_dir)
        path = save_metrics(analysis, results_dir)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["task"] == "gsm8k"
        assert payload["cascade"] is not None
        assert len(payload["evaluations"]) == len(analysis.evaluations)

    def test_missing_records_raise(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="run_experiment"):
            analyse_task("nq_open", tmp_path)


class TestNoLeakage:
    """Thresholds must be fitted on dev, never on test."""

    def test_dev_and_test_metrics_differ(self, results_dir: Path) -> None:
        # If the calibrator were fitted on test, dev-fitted thresholds would
        # transfer perfectly and dev/test accuracy would coincide exactly.
        analysis = analyse_task("nq_open", results_dir)
        gaps = [
            abs(point.dev_accuracy - point.test_accuracy)
            for point in analysis.selective
            if np.isfinite(point.dev_accuracy) and np.isfinite(point.test_accuracy)
        ]
        assert gaps, "no comparable operating points"
        assert max(gaps) > 0.0

    def test_ece_is_finite_on_held_out_data(self, results_dir: Path) -> None:
        analysis = analyse_task("nq_open", results_dir)
        best = analysis.evaluations[0]
        assert not math.isnan(best.ece_fixed_width)
        assert not math.isnan(best.ece_equal_mass)
