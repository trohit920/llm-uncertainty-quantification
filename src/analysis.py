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
    select_signal_on_dev,
)
from .experiment import records_for_task, signal_columns
from .self_consistency import CascadeReport, build_cascade_report
from .signals import SIGNAL_NAMES, available_signals, get_spec

logger = logging.getLogger(__name__)

#: Number of top-ranked signals carried into the detailed figures.
TOP_SIGNALS_FOR_FIGURES: Final[int] = 3

#: Cost tier whose signals come free with the greedy decode. The cascade gate
#: must come from this tier: a gate that needs the samples it is deciding
#: whether to draw saves nothing.
FREE_COST_TIER: Final[str] = "single_pass"

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
        evaluations: Per-signal metrics, ranked by held-out AUROC for reporting.
        best_signal: Signal with the highest dev AUROC; drives selective
            answering and confidence tags.
        gate_signal: Free (single-pass) signal with the highest dev AUROC;
            gates the cascade. ``None`` when no free signal is usable, in
            which case no cascade is built.
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
    gate_signal: str | None
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
            "gate_signal": self.gate_signal,
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

    # Choices that feed reported results are made on dev only; the test-AUROC
    # ranking in ``evaluations`` is for reporting, never for selection.
    candidates = selection_candidates([ev.signal for ev in evaluations], dev_rows)
    best = select_signal_on_dev(dev_rows, dev_correct, candidates)
    free = [name for name in candidates if get_spec(name).cost_tier == FREE_COST_TIER]
    gate = select_signal_on_dev(dev_rows, dev_correct, free) if free else None
    logger.info(
        "Signals for %s chosen on dev: best=%s, cascade gate=%s; test-AUROC leader=%s",
        task,
        best,
        gate,
        evaluations[0].signal,
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

    cascade = (
        _build_cascade(task, records, gate, columns, dev_indices, test_indices)
        if gate is not None
        else None
    )

    return TaskAnalysis(
        task=task,
        num_examples=len(records),
        num_dev=int(dev_indices.size),
        num_test=int(test_indices.size),
        greedy_accuracy=accuracy(test_correct),
        evaluations=evaluations,
        best_signal=best,
        gate_signal=gate,
        selective=selective,
        tiers=tiers,
        tier_gap=tier_separation(tiers),
        cascade=cascade,
        cost_seconds=_mean_cost_by_tier(records),
    )


def selection_candidates(
    evaluated: Sequence[str], dev_rows: dict[str, np.ndarray]
) -> list[str]:
    """Signals eligible for selection, in fixed registry order.

    Two properties keep test data out of the choice. The order is the signal
    registry's, not the test-AUROC ranking, because selection breaks dev ties
    by candidate order (and ties are real: ``seq_prob`` is ``exp`` of
    ``seq_log_prob``). Usability is judged on dev columns, so a signal that is
    degenerate on dev is never chosen because it varies on test.

    Args:
        evaluated: Names of signals that were evaluated for reporting.
        dev_rows: Mapping from signal name to its dev-split column.

    Returns:
        Signal names in registry order.
    """
    evaluated_set = set(evaluated)
    usable_on_dev = set(available_signals(dev_rows))
    return [
        name
        for name in SIGNAL_NAMES
        if name in evaluated_set and name in usable_on_dev
    ]


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
