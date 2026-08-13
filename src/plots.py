"""Figures evaluating uncertainty-signal quality.

Signal rankings, calibration, risk-coverage, distributional separation and the
cost-versus-quality trade-off. Application-facing figures live in
:mod:`plots_applications`; shared styling lives in :mod:`plot_style`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

# plot_style selects the Agg backend; import it before pyplot is bound.
from .plot_style import (
    ALL_PAIRS_PALETTE,
    BAR_GAP,
    LINE_STYLES,
    LINE_WIDTH,
    MARKER_SIZE,
    PALETTE,
    REFERENCE_COLOR,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WIDE_FIGURE,
    apply_style,
    finish,
    hide_spines,
)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .calibration import ReliabilityDiagram
from .evaluate import SignalEvaluation
from .config import FIGURES_DIR

logger = logging.getLogger(__name__)


def plot_auroc_by_signal(
    evaluations: Sequence[SignalEvaluation],
    task: str,
    output_dir: Path = FIGURES_DIR,
    top_n: int = 12,
) -> Path:
    """Rank signals by error-detection AUROC with bootstrap intervals.

    Args:
        evaluations: Signal evaluations, any order.
        task: Task name, used in the title and filename.
        output_dir: Directory to write into.
        top_n: Number of signals to show.

    Returns:
        Path to the written figure.
    """
    apply_style()
    ranked = [ev for ev in evaluations if np.isfinite(ev.auroc.value)][:top_n]
    ranked = list(reversed(ranked))

    figure, axis = plt.subplots(figsize=(9.0, 0.42 * len(ranked) + 2.0))
    positions = np.arange(len(ranked))
    values = [ev.auroc.value for ev in ranked]
    lower = [max(0.0, ev.auroc.value - ev.auroc.lower) for ev in ranked]
    upper = [max(0.0, ev.auroc.upper - ev.auroc.value) for ev in ranked]

    colors = [
        PALETTE[0] if ev.granularity == "sentence" else PALETTE[1] for ev in ranked
    ]
    axis.barh(
        positions,
        values,
        color=colors,
        height=1 - BAR_GAP * 20,
        xerr=[lower, upper],
        error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1.2, "capsize": 3},
    )
    axis.axvline(0.5, color=REFERENCE_COLOR, linewidth=1.2, linestyle="--", zorder=0)

    for position, ev in zip(positions, ranked):
        axis.text(
            ev.auroc.upper + 0.012,
            position,
            f"{ev.auroc.value:.3f}",
            va="center",
            fontsize=8.5,
            color=TEXT_SECONDARY,
        )

    axis.set_yticks(positions, [ev.display_name for ev in ranked])
    axis.set_xlim(0.3, 1.02)
    axis.set_xlabel("Error-detection AUROC (95% bootstrap CI)")
    axis.set_title(f"Which signals predict an error?  |  {task}")
    axis.grid(axis="y", visible=False)
    hide_spines(axis)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PALETTE[0]),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE[1]),
        plt.Line2D([0], [0], color=REFERENCE_COLOR, linestyle="--"),
    ]
    axis.legend(
        handles,
        ["Sentence-level", "Token-level", "Chance (0.5)"],
        loc="lower right",
    )

    return finish(figure, output_dir / f"auroc_by_signal_{task}.png")


def plot_reliability_diagrams(
    diagrams: Mapping[str, ReliabilityDiagram],
    task: str,
    strategy: str,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Draw reliability diagrams for several signals side by side.

    Args:
        diagrams: Mapping from signal display name to its diagram.
        task: Task name, used in the title and filename.
        strategy: Binning strategy name, used in the title and filename.
        output_dir: Directory to write into.

    Returns:
        Path to the written figure.
    """
    apply_style()
    names = list(diagrams)
    figure, axes = plt.subplots(
        1, max(len(names), 1), figsize=(4.0 * max(len(names), 1), 4.4), squeeze=False
    )

    for index, name in enumerate(names):
        axis = axes[0][index]
        diagram = diagrams[name]
        axis.plot(
            [0, 1],
            [0, 1],
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1.2,
            zorder=0,
        )

        if diagram.bins:
            confidences = [b.mean_confidence for b in diagram.bins]
            accuracies = [b.accuracy for b in diagram.bins]
            weights = np.asarray([b.count for b in diagram.bins], dtype=float)
            sizes = MARKER_SIZE + 26.0 * weights / weights.max()

            axis.plot(
                confidences,
                accuracies,
                color=PALETTE[0],
                linewidth=LINE_WIDTH,
                zorder=2,
            )
            axis.scatter(
                confidences,
                accuracies,
                s=sizes,
                color=PALETTE[0],
                edgecolor=SURFACE,
                linewidth=2.0,
                zorder=3,
            )

        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_aspect("equal")
        axis.set_xlabel("Predicted confidence")
        if index == 0:
            axis.set_ylabel("Observed accuracy")
        axis.set_title(
            f"{name}\nECE {diagram.expected_calibration_error:.3f}", fontsize=10
        )
        hide_spines(axis)

    figure.suptitle(
        f"Calibration after dev-fitted mapping  |  {task}  |  {strategy} bins",
        fontsize=12,
        color=TEXT_PRIMARY,
    )
    return finish(figure, output_dir / f"reliability_{strategy}_{task}.png")


def plot_risk_coverage(
    curves: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    task: str,
    base_error_rate: float,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Compare risk-coverage curves across signals.

    Args:
        curves: Curve dictionaries with ``coverage`` and ``risk`` keys.
        labels: Display label per curve.
        task: Task name, used in the title and filename.
        base_error_rate: Error rate at full coverage, drawn as the reference.
        output_dir: Directory to write into.

    Returns:
        Path to the written figure.
    """
    apply_style()
    figure, axis = plt.subplots(figsize=WIDE_FIGURE)

    # Distinct dash patterns as well as hues: two signals that rank examples
    # identically produce identical curves, and the later one would otherwise
    # paint over the earlier one invisibly.
    for index, (curve, label) in enumerate(zip(curves, labels)):
        axis.plot(
            curve["coverage"],
            curve["risk"],
            color=PALETTE[index % len(PALETTE)],
            linestyle=LINE_STYLES[index % len(LINE_STYLES)],
            linewidth=LINE_WIDTH,
            label=f"{label}  (AURC {curve['aurc']:.3f})",
        )

    axis.axhline(
        base_error_rate,
        color=REFERENCE_COLOR,
        linestyle="--",
        linewidth=1.2,
        label=f"Answer everything ({base_error_rate:.3f})",
    )

    axis.set_xlabel("Coverage — fraction of questions answered")
    axis.set_ylabel("Risk — error rate among answered")
    axis.set_title(f"Abstaining on the uncertain cases lowers risk  |  {task}")
    axis.set_xlim(0, 1)
    axis.legend(loc="upper left")
    hide_spines(axis)

    return finish(figure, output_dir / f"risk_coverage_{task}.png")


def plot_signal_distribution(
    correct_values: Sequence[float],
    wrong_values: Sequence[float],
    signal_label: str,
    task: str,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Contrast a signal's distribution on correct versus incorrect answers.

    Args:
        correct_values: Signal values where the model was right.
        wrong_values: Signal values where the model was wrong.
        signal_label: Display name of the signal.
        task: Task name, used in the title and filename.
        output_dir: Directory to write into.

    Returns:
        Path to the written figure.
    """
    apply_style()
    figure, axis = plt.subplots(figsize=WIDE_FIGURE)

    correct = np.asarray(correct_values, dtype=float)
    wrong = np.asarray(wrong_values, dtype=float)
    combined = np.concatenate([correct, wrong])
    combined = combined[np.isfinite(combined)]
    bins = np.linspace(combined.min(), combined.max(), 24) if combined.size else 24

    axis.hist(
        correct, bins=bins, color=PALETTE[2], alpha=0.75, label="Answer correct"
    )
    axis.hist(
        wrong, bins=bins, color=PALETTE[1], alpha=0.75, label="Answer wrong"
    )

    for values, color in ((correct, PALETTE[2]), (wrong, PALETTE[1])):
        finite = values[np.isfinite(values)]
        if finite.size:
            axis.axvline(
                float(np.median(finite)), color=color, linewidth=LINE_WIDTH, linestyle=":"
            )

    axis.set_xlabel(signal_label)
    axis.set_ylabel("Examples")
    axis.set_title(
        f"Signal separation between right and wrong answers  |  {task}\n"
        "dotted lines mark medians"
    )
    axis.legend(loc="upper right")
    hide_spines(axis)

    return finish(figure, output_dir / f"distribution_{task}.png")


def plot_cost_versus_quality(
    evaluations: Sequence[SignalEvaluation],
    cost_seconds: Mapping[str, float],
    task: str,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot AUROC against the wall-clock cost of obtaining each signal.

    Args:
        evaluations: Signal evaluations.
        cost_seconds: Mapping from cost tier to mean seconds per example.
        task: Task name, used in the title and filename.
        output_dir: Directory to write into.

    Returns:
        Path to the written figure.
    """
    apply_style()
    figure, axis = plt.subplots(figsize=WIDE_FIGURE)

    # Scatter puts every pair of series side by side, so this form is capped
    # at the three slots validated across all pairs.
    tiers = ("single_pass", "extra_call", "multi_sample")
    tier_labels = {
        "single_pass": "Single greedy pass (free)",
        "extra_call": "One extra short call",
        "multi_sample": "N samples (+ NLI)",
    }

    for index, tier in enumerate(tiers):
        members = [
            ev
            for ev in evaluations
            if ev.cost_tier == tier and np.isfinite(ev.auroc.value)
        ]
        if not members:
            continue
        axis.scatter(
            [cost_seconds.get(tier, np.nan)] * len(members),
            [ev.auroc.value for ev in members],
            s=70,
            color=ALL_PAIRS_PALETTE[index],
            edgecolor=SURFACE,
            linewidth=2.0,
            label=tier_labels[tier],
            zorder=3,
        )

        best = max(members, key=lambda ev: ev.auroc.value)
        axis.annotate(
            best.display_name,
            (cost_seconds.get(tier, np.nan), best.auroc.value),
            textcoords="offset points",
            xytext=(10, 4),
            fontsize=8.5,
            color=TEXT_SECONDARY,
        )

    axis.axhline(0.5, color=REFERENCE_COLOR, linestyle="--", linewidth=1.2, zorder=0)
    axis.set_xscale("log")
    axis.set_xlabel("Mean seconds per example (log scale)")
    axis.set_ylabel("Error-detection AUROC")
    axis.set_title(f"What does better uncertainty actually cost?  |  {task}")
    axis.legend(loc="lower right")
    hide_spines(axis)

    return finish(figure, output_dir / f"cost_quality_{task}.png")


def plot_span_comparison(
    evaluations: Sequence[SignalEvaluation],
    task: str,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Compare whole-sequence token signals against answer-span versions.

    Args:
        evaluations: Signal evaluations for one task.
        task: Task name, used in the title and filename.
        output_dir: Directory to write into.

    Returns:
        Path to the written figure.
    """
    apply_style()
    by_name = {ev.signal: ev for ev in evaluations}
    base_names = [
        name
        for name in by_name
        if not name.startswith("answer_") and f"answer_{name}" in by_name
    ]

    figure, axis = plt.subplots(figsize=WIDE_FIGURE)
    positions = np.arange(len(base_names))
    width = 0.38

    axis.bar(
        positions - width / 2 - BAR_GAP,
        [by_name[name].auroc.value for name in base_names],
        width=width,
        color=PALETTE[1],
        label="Whole generation",
    )
    axis.bar(
        positions + width / 2 + BAR_GAP,
        [by_name[f"answer_{name}"].auroc.value for name in base_names],
        width=width,
        color=PALETTE[0],
        label="Final-answer span only",
    )

    axis.axhline(0.5, color=REFERENCE_COLOR, linestyle="--", linewidth=1.2, zorder=0)
    axis.set_xticks(
        positions,
        [by_name[name].display_name for name in base_names],
        rotation=30,
        ha="right",
    )
    axis.set_ylabel("Error-detection AUROC")
    axis.set_title(
        f"Restricting token signals to the answer span  |  {task}"
    )
    axis.legend(loc="upper left")
    axis.grid(axis="x", visible=False)
    hide_spines(axis)

    return finish(figure, output_dir / f"span_comparison_{task}.png")
