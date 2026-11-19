"""Tests that every signal choice feeding a reported result is made on dev only.

Selective answering, confidence tags and the cascade all report held-out
numbers for one chosen signal. If that signal is chosen by its held-out AUROC,
the reported numbers are selected on the test set. These tests pin the rule
down at the unit level and end to end, using records where dev and test
deliberately disagree about which signal is best.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis import (
    FREE_COST_TIER,
    analyse_task,
    selection_candidates,
    split_indices,
)
from src.config import DEV_FRACTION, RECORDS_FILENAME, SEED
from src.evaluate import select_signal_on_dev
from src.signals import SIGNAL_NAMES, get_spec

#: Correctness labels for the unit tests: three right, three wrong.
_DEV_CORRECT = [True, True, True, False, False, False]

#: Dev columns. Higher entropy means less certain; higher log-prob means more
#: certain, so ``seq_log_prob`` is a perfect signal only if orientation is
#: respected.
_DEV_ROWS = {
    "semantic_entropy": [0.1, 0.2, 0.3, 0.9, 1.0, 1.1],
    "seq_log_prob": [-0.1, -0.2, -0.3, -2.0, -3.0, -4.0],
    "mean_entropy": [0.5, 0.9, 0.1, 0.4, 0.8, 0.2],
    "max_entropy": [0.2, float("nan"), 0.1, 1.5, 1.2, float("nan")],
    "log_perplexity": [float("nan")] * 6,
}

#: Size of the synthetic task used by the end-to-end tests.
_NUM_RECORDS = 120
#: Seed for the end-to-end synthetic records.
_RECORD_SEED = 7


class TestSelectSignalOnDev:
    """Unit behaviour of dev-only signal selection."""

    @pytest.mark.parametrize(
        ("candidates", "expected"),
        [
            (["mean_entropy", "semantic_entropy"], "semantic_entropy"),
            (["mean_entropy", "seq_log_prob"], "seq_log_prob"),
            (["seq_log_prob", "semantic_entropy"], "seq_log_prob"),
            (["semantic_entropy", "seq_log_prob"], "semantic_entropy"),
            (["mean_entropy", "max_entropy"], "max_entropy"),
            (["log_perplexity", "mean_entropy"], "mean_entropy"),
        ],
        ids=[
            "informative-beats-noise",
            "respects-orientation",
            "tie-keeps-first-a",
            "tie-keeps-first-b",
            "drops-nan-values",
            "skips-undefined-auroc",
        ],
    )
    def test_selects_highest_dev_auroc(
        self, candidates: list[str], expected: str
    ) -> None:
        assert select_signal_on_dev(_DEV_ROWS, _DEV_CORRECT, candidates) == expected

    def test_raises_when_no_candidate_is_defined(self) -> None:
        with pytest.raises(ValueError, match="No candidate signal"):
            select_signal_on_dev(_DEV_ROWS, _DEV_CORRECT, ["log_perplexity"])


def _record(task: str, index: int, is_correct: bool, signals: dict[str, float]) -> dict:
    """Build one record in the schema produced by :mod:`src.experiment`."""
    return {
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
        "majority_correct": is_correct or index % 3 == 0,
        "sample_correct": [is_correct] * 10,
        "signals": signals,
        "timings": {
            "greedy_seconds": 0.5,
            "sampling_seconds": 4.0,
            "clustering_seconds": 0.2,
            "verbalized_seconds": 0.1,
        },
    }


def make_disagreeing_records(task: str) -> list[dict]:
    """Records where dev and test disagree about the best signal.

    On the dev split ``semantic_entropy`` (sampled) and ``seq_log_prob``
    (free) track correctness. On the test split only ``mean_entropy`` does.
    A pipeline that chooses by test AUROC picks ``mean_entropy``; one that
    chooses on dev picks the dev winners.
    """
    rng = np.random.default_rng(_RECORD_SEED)
    dev_indices, _ = split_indices(_NUM_RECORDS, DEV_FRACTION, SEED)
    dev_set = set(dev_indices.tolist())
    records = []
    for index in range(_NUM_RECORDS):
        is_correct = bool(index % 2 == 0)
        signals = {name: float(rng.normal()) for name in SIGNAL_NAMES}
        shift = -1.0 if is_correct else 1.0
        if index in dev_set:
            signals["semantic_entropy"] = 1.0 + shift + float(rng.normal(scale=0.2))
            signals["seq_log_prob"] = -1.0 - shift + float(rng.normal(scale=0.2))
        else:
            signals["mean_entropy"] = 1.0 + shift + float(rng.normal(scale=0.2))
        records.append(_record(task, index, is_correct, signals))
    return records


@pytest.fixture
def disagreeing_results_dir(tmp_path: Path) -> Path:
    """Write the disagreeing records for both tasks into a temp directory."""
    for task in ("nq_open", "gsm8k"):
        path = tmp_path / RECORDS_FILENAME.format(task=task)
        lines = [json.dumps(record) for record in make_disagreeing_records(task)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


class TestNoTestSetSelection:
    """End to end: the pipeline never chooses a signal by its test score."""

    def test_test_leader_differs_from_dev_choice(
        self, disagreeing_results_dir: Path
    ) -> None:
        analysis = analyse_task("nq_open", disagreeing_results_dir)
        # Guard: the fixture must actually make test and dev disagree,
        # otherwise the assertions below prove nothing.
        assert analysis.evaluations[0].signal == "mean_entropy"
        assert analysis.best_signal in {"semantic_entropy", "seq_log_prob"}

    def test_cascade_gate_is_free_and_chosen_on_dev(
        self, disagreeing_results_dir: Path
    ) -> None:
        analysis = analyse_task("gsm8k", disagreeing_results_dir)
        assert get_spec(analysis.gate_signal).cost_tier == FREE_COST_TIER
        assert analysis.gate_signal == "seq_log_prob"
        assert analysis.cascade is not None

    def test_gate_is_recorded_in_metrics(
        self, disagreeing_results_dir: Path
    ) -> None:
        payload = analyse_task("gsm8k", disagreeing_results_dir).as_dict()
        assert payload["gate_signal"] == "seq_log_prob"
        assert payload["best_signal"] in {"semantic_entropy", "seq_log_prob"}


class TestSelectionCandidates:
    """Candidate order and eligibility must not depend on test results."""

    def test_order_is_registry_order_not_test_ranking(self) -> None:
        # Test ranking lists seq_prob first; the registry lists seq_log_prob first.
        dev_rows = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in _DEV_ROWS.items()
        }
        dev_rows["seq_prob"] = np.exp(dev_rows["seq_log_prob"])
        ranked_by_test = ["seq_prob", "seq_log_prob", "semantic_entropy"]
        candidates = selection_candidates(ranked_by_test, dev_rows)
        assert candidates == ["seq_log_prob", "seq_prob", "semantic_entropy"]

    def test_dev_ties_resolve_by_registry_not_test_order(self) -> None:
        # seq_prob = exp(seq_log_prob), so the two always tie on dev AUROC.
        dev_rows = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in _DEV_ROWS.items()
        }
        dev_rows["seq_prob"] = np.exp(dev_rows["seq_log_prob"])
        candidates = selection_candidates(["seq_prob", "seq_log_prob"], dev_rows)
        assert select_signal_on_dev(dev_rows, _DEV_CORRECT, candidates) == "seq_log_prob"

    def test_signal_degenerate_on_dev_is_excluded(self) -> None:
        dev_rows = {
            "semantic_entropy": np.asarray(_DEV_ROWS["semantic_entropy"]),
            "mean_entropy": np.full(len(_DEV_CORRECT), 0.5),
        }
        candidates = selection_candidates(["mean_entropy", "semantic_entropy"], dev_rows)
        assert candidates == ["semantic_entropy"]


class TestNoFreeSignal:
    """A task with no usable free signal skips the cascade instead of crashing."""

    def test_gate_is_none_and_cascade_skipped(self, tmp_path: Path) -> None:
        free_names = [
            name for name in SIGNAL_NAMES if get_spec(name).cost_tier == FREE_COST_TIER
        ]
        records = make_disagreeing_records("gsm8k")
        for record in records:
            for name in free_names:
                record["signals"][name] = 0.0
        path = tmp_path / RECORDS_FILENAME.format(task="gsm8k")
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        analysis = analyse_task("gsm8k", tmp_path)
        assert analysis.gate_signal is None
        assert analysis.cascade is None
        assert get_spec(analysis.best_signal).cost_tier != FREE_COST_TIER
