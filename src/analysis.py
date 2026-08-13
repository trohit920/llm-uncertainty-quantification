"""Orchestration layer tying records to metrics, applications and figures.

This module sits above :mod:`evaluate`, :mod:`applications` and
:mod:`self_consistency`, composing them into the per-task bundle that the
figure, report and notebook stages all consume. Keeping the composition here
means the scripts stay thin and the same numbers back every deliverable --
there is exactly one place that decides how records become results.

The dev/test split happens once, here, and everything downstream inherits it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np

from .applications import (
    ConfidenceTierReport,
    SelectiveOperatingPoint,
    confidence_tag_report,
    selective_answering,
    tier_separation,
)
from .config import (
    DEV_FRACTION,
    METRICS_FILENAME,
    RESULTS_DIR,
    SEED,
    selective_targets_for_task,
)
from .evaluate import (
    SignalEvaluation,
    accuracy,
    build_reliability_diagram,
    build_risk_coverage_curve,
    evaluate_all_signals,
)
from .experiment import records_for_task, signal_columns
from .self_consistency import CascadeReport, build_cascade_report

logger = logging.getLogger(__name__)

#: Number of top-ranked signals carried into the detailed figures.
TOP_SIGNALS_FOR_FIGURES: Final[int] = 3

#: Cost tiers, mapped to the timing field that measures them.
_TIER_TIMING_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "single_pass": ("greedy_seconds",),
    "extra_call": ("verbalized_seconds",),
    "multi_sample": ("sampling_seconds", "clustering_seconds"),
}


@dataclass
class TaskAnalysis:
    """Everything computed for one task.

    Attributes:
        task: Task name.
        num_examples: Total records analysed.
        num_dev: Size of the threshold-fitting split.
        num_test: Size of the held-out split.
        greedy_accuracy: Greedy-decode accuracy on the held-out split.
        evaluations: Per-signal metrics, ranked by AUROC.
        best_signal: Name of the top-ranked signal.
        selective: Selective-answering operating points for the best signal.
        tiers: Confidence-tier reports for the best signal.
        tier_gap: Accuracy gap between the High and Low tiers.
        cascade: Cascade report, present only for chain-of-thought tasks.
        cost_seconds: Mean seconds per example, by cost tier.
    """

    task: str
    num_examples: int
    num_dev: int
    num_test: int
    greedy_accuracy: float
    evaluations: list[SignalEvaluation]
    best_signal: str
    selective: list[SelectiveOperatingPoint]
    tiers: list[ConfidenceTierReport]
    tier_gap: float
    cascade: CascadeReport | None
    cost_seconds: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the analysis."""
        return {
            "task": self.task,
            "num_examples": self.num_examples,
            "num_dev": self.num_dev,
            "num_test": self.num_test,
            "greedy_accuracy": self.greedy_accuracy,
            "best_signal": self.best_signal,
            "evaluations": [ev.as_dict() for ev in self.evaluations],
            "selective": [point.as_dict() for point in self.selective],
            "tiers": [tier.as_dict() for tier in self.tiers],
            "tier_gap": self.tier_gap,
            "cascade": self.cascade.as_dict() if self.cascade else None,
            "cost_seconds": self.cost_seconds,
        }


def split_indices(
    num_records: int, dev_fraction: float = DEV_FRACTION, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Partition record positions into dev and test index arrays.

    Args:
        num_records: Total number of records.
        dev_fraction: Fraction assigned to the dev split.
        seed: Seed for the permutation.

    Returns:
        A ``(dev_indices, test_indices)`` pair.
    """
    permutation = np.random.default_rng(seed).permutation(num_records)
    split_point = int(round(num_records * dev_fraction))
    return permutation[:split_point], permutation[split_point:]


def _mean_cost_by_tier(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Average the wall-clock cost of each signal tier across records.

    Args:
        records: Parsed records carrying a ``timings`` mapping.

    Returns:
        Mapping from cost tier to mean seconds per example.
    """
    costs: dict[str, float] = {}
    for tier, fields in _TIER_TIMING_FIELDS.items():
        totals = [
            sum(record["timings"].get(field, 0.0) for field in fields)
            for record in records
        ]
        costs[tier] = float(np.mean(totals)) if totals else float("nan")
    return costs


def analyse_task(
    task: str,
    results_dir: Path = RESULTS_DIR,
    dev_fraction: float = DEV_FRACTION,
    seed: int = SEED,
) -> TaskAnalysis:
    """Load a task's records and compute every downstream result.

    Args:
        task: Task name.
        results_dir: Directory holding the record files.
        dev_fraction: Fraction of examples used to fit thresholds.
        seed: Seed for the split and the bootstrap.

    Returns:
        The populated analysis.

    Raises:
        ValueError: If the record file is empty.
    """
    records = records_for_task(task, results_dir)
    if not records:
        raise ValueError(f"No records found for task {task!r}")

    columns = signal_columns(records)
    correct = np.asarray([record["greedy_correct"] for record in records], dtype=bool)
    dev_indices, test_indices = split_indices(len(records), dev_fraction, seed)

    logger.info(
        "Analysing %s: %d records (%d dev / %d test), greedy accuracy %.3f",
        task,
        len(records),
        dev_indices.size,
        test_indices.size,
        float(correct.mean()),
    )

    dev_rows = {name: column[dev_indices] for name, column in columns.items()}
    test_rows = {name: column[test_indices] for name, column in columns.items()}
    dev_correct, test_correct = correct[dev_indices], correct[test_indices]

    evaluations = evaluate_all_signals(
        dev_rows, dev_correct, test_rows, test_correct, seed=seed
    )
    if not evaluations:
        raise ValueError(f"No usable signals for task {task!r}")

    best = evaluations[0].signal
    logger.info(
        "Best signal for %s: %s (AUROC %s)",
        task,
        evaluations[0].display_name,
        evaluations[0].auroc,
    )

    selective = selective_answering(
        best,
        dev_rows[best],
        dev_correct,
        test_rows[best],
        test_correct,
        target_accuracies=selective_targets_for_task(task),
    )
    tiers = confidence_tag_report(
        best, dev_rows[best], dev_correct, test_rows[best], test_correct
    )

    cascade = _build_cascade(task, records, best, columns, dev_indices, test_indices)

    return TaskAnalysis(
        task=task,
        num_examples=len(records),
        num_dev=int(dev_indices.size),
        num_test=int(test_indices.size),
        greedy_accuracy=accuracy(test_correct),
        evaluations=evaluations,
        best_signal=best,
        selective=selective,
        tiers=tiers,
        tier_gap=tier_separation(tiers),
        cascade=cascade,
        cost_seconds=_mean_cost_by_tier(records),
    )


def _build_cascade(
    task: str,
    records: Sequence[dict[str, Any]],
    signal: str,
    columns: dict[str, np.ndarray],
    dev_indices: np.ndarray,
    test_indices: np.ndarray,
) -> CascadeReport | None:
    """Build the cascade report when self-consistency is meaningful.

    The cascade only makes sense where majority voting beats greedy decoding,
    which in this project means the chain-of-thought task. Building it for
    short-form QA would report a compute trade-off against a baseline that
    sampling does not actually improve.

    Args:
        task: Task name.
        records: Parsed records.
        signal: Signal gating escalation.
        columns: Signal columns over all records.
        dev_indices: Dev-split positions.
        test_indices: Test-split positions.

    Returns:
        The cascade report, or ``None`` when the task is not eligible.
    """
    if task != "gsm8k":
        return None

    greedy = np.asarray([record["greedy_correct"] for record in records], dtype=bool)
    majority = np.asarray(
        [record["majority_correct"] for record in records], dtype=bool
    )
    values = columns[signal]

    return build_cascade_report(
        signal,
        values[dev_indices],
        greedy[dev_indices],
        majority[dev_indices],
        values[test_indices],
        greedy[test_indices],
        majority[test_indices],
    )


def top_signals(
    analysis: TaskAnalysis, count: int = TOP_SIGNALS_FOR_FIGURES
) -> list[str]:
    """Return the highest-AUROC signal names.

    Args:
        analysis: A completed task analysis.
        count: Number of names to return.

    Returns:
        Signal names, best first.
    """
    return [
        ev.signal
        for ev in analysis.evaluations[:count]
        if np.isfinite(ev.auroc.value)
    ]


def reliability_for_signals(
    task: str,
    signals: Sequence[str],
    strategy: str,
    results_dir: Path = RESULTS_DIR,
    dev_fraction: float = DEV_FRACTION,
    seed: int = SEED,
) -> dict[str, Any]:
    """Build reliability diagrams for a set of signals.

    Args:
        task: Task name.
        signals: Signal names to diagram.
        strategy: ``"fixed_width"`` or ``"equal_mass"``.
        results_dir: Directory holding the record files.
        dev_fraction: Dev split fraction.
        seed: Split seed.

    Returns:
        Mapping from signal display name to its diagram.
    """
    records = records_for_task(task, results_dir)
    columns = signal_columns(records)
    correct = np.asarray([record["greedy_correct"] for record in records], dtype=bool)
    dev_indices, test_indices = split_indices(len(records), dev_fraction, seed)

    from .signals import get_spec

    return {
        get_spec(name).display_name: build_reliability_diagram(
            name,
            columns[name][dev_indices],
            correct[dev_indices],
            columns[name][test_indices],
            correct[test_indices],
            strategy=strategy,
        )
        for name in signals
    }


def risk_coverage_for_signals(
    task: str,
    signals: Sequence[str],
    results_dir: Path = RESULTS_DIR,
    dev_fraction: float = DEV_FRACTION,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], list[str], float]:
    """Build risk-coverage curves on the held-out split.

    Args:
        task: Task name.
        signals: Signal names to plot.
        results_dir: Directory holding the record files.
        dev_fraction: Dev split fraction.
        seed: Split seed.

    Returns:
        A ``(curves, labels, base_error_rate)`` triple.
    """
    records = records_for_task(task, results_dir)
    columns = signal_columns(records)
    correct = np.asarray([record["greedy_correct"] for record in records], dtype=bool)
    _, test_indices = split_indices(len(records), dev_fraction, seed)

    from .signals import get_spec

    curves = [
        build_risk_coverage_curve(
            name, columns[name][test_indices], correct[test_indices]
        )
        for name in signals
    ]
    labels = [get_spec(name).display_name for name in signals]
    base_error_rate = float(1.0 - correct[test_indices].mean())
    return curves, labels, base_error_rate


def save_metrics(analysis: TaskAnalysis, results_dir: Path = RESULTS_DIR) -> Path:
    """Write a task analysis to JSON.

    Args:
        analysis: The analysis to serialise.
        results_dir: Destination directory.

    Returns:
        Path to the written file.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / METRICS_FILENAME.format(task=analysis.task)
    path.write_text(
        json.dumps(analysis.as_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote %s", path)
    return path
