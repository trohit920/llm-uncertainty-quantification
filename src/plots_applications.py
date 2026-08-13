"""Figures for the uncertainty-aware applications.

Selective answering, the self-consistency cascade and confidence tags. Signal
quality figures live in :mod:`plots`; shared styling in :mod:`plot_style`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

# plot_style selects the Agg backend; import it before pyplot is bound.
from .plot_style import (
    BAR_GAP,
    LINE_WIDTH,
    PALETTE,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TIER_RAMP,
    WIDE_FIGURE,
    apply_style,
    finish,
    hide_spines,
)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import FIGURES_DIR

logger = logging.getLogger(__name__)


def plot_selective_answering(
    points: Sequence[Any], task: str, output_dir: Path = FIGURES_DIR
) -> Path:
    """Show dev-fitted operating points as they land on held-out data.

    Args:
        points: Selective operating points.
        task: Task name, used in the title and filename.
        output_dir: Directory to write into.

    Returns:
        Path to the written figure.
    """
    apply_style()
    figure, axis = plt.subplots(figsize=WIDE_FIGURE)

    usable = [p for p in points if np.isfinite(p.test_accuracy)]
    positions = np.arange(len(usable))
    width = 0.38

    axis.bar(
        positions - width / 2 - BAR_GAP,
        [p.target_accuracy for p in usable],
        width=width,
        color=PALETTE[3],
        label="Target accuracy (fitted on dev)",
    )
    axis.bar(
        positions + width / 2 + BAR_GAP,
        [p.test_accuracy for p in usable],
        width=width,
        color=PALETTE[0],
        label="Achieved accuracy (held-out test)",
    )

    for position, point in zip(positions, usable):
        axis.text(
            position,
            max(point.target_accuracy, point.test_accuracy) + 0.02,
            f"{point.test_coverage:.0%} answered",
            ha="center",
            fontsize=8.5,
            color=TEXT_SECONDARY,
        )

    axis.set_xticks(positions, [f"{p.target_accuracy:.0%}" for p in usable])
    axis.set_xlabel("Accuracy target")
    axis.set_ylabel("Accuracy among answered")
    axis.set_ylim(0, 1.12)
    axis.set_title(
        f"Selective answering: do dev-fitted thresholds transfer?  |  {task}"
    )
    axis.legend(loc="upper left")
    axis.grid(axis="x", visible=False)
    hide_spines(axis)

    return finish(figure, output_dir / f"selective_answering_{task}.png")


def plot_cascade(report: Any, task: str, output_dir: Path = FIGURES_DIR) -> Path:
    """Plot the cascade's accuracy against its generation cost.

    Args:
        report: A cascade report.
        task: Task name, used in the title and filename.
        output_dir: Directory to write into.

    Returns:
        Path to the written figure.
    """
    apply_style()
    figure, axis = plt.subplots(figsize=WIDE_FIGURE)

    generations = [point.mean_generations for point in report.curve]
    accuracies = [point.test_accuracy for point in report.curve]

    axis.plot(
        generations, accuracies, color=PALETTE[0], linewidth=LINE_WIDTH, zorder=2
    )
    axis.scatter(
        generations,
        accuracies,
        s=48,
        color=PALETTE[0],
        edgecolor=SURFACE,
        linewidth=2.0,
        zorder=3,
        label="Uncertainty-gated cascade",
    )

    axis.axhline(
        report.greedy_accuracy,
        color=PALETTE[1],
        linestyle="--",
        linewidth=LINE_WIDTH,
        label=f"Greedy only ({report.greedy_accuracy:.3f}, 1 generation)",
    )
    axis.axhline(
        report.self_consistency_accuracy,
        color=PALETTE[2],
        linestyle="--",
        linewidth=LINE_WIDTH,
        label=(
            f"Always self-consistent ({report.self_consistency_accuracy:.3f}, "
            f"{1 + report.num_samples} generations)"
        ),
    )

    if report.selected is not None:
        axis.scatter(
            [report.selected.mean_generations],
            [report.selected.test_accuracy],
            s=180,
            facecolor="none",
            edgecolor=TEXT_PRIMARY,
            linewidth=2.0,
            zorder=4,
            label=(
                f"Dev-selected point "
                f"({report.selected.test_escalation_rate:.0%} escalated)"
            ),
        )

    axis.set_xlabel("Mean generations per question")
    axis.set_ylabel("Accuracy")
    axis.set_title(f"Spending compute only where the model is unsure  |  {task}")
    # Both baselines are horizontal rules spanning the full width, so any
    # in-axes legend placement collides with one of them.
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        markerscale=0.6,
    )
    hide_spines(axis)

    return finish(figure, output_dir / f"cascade_{task}.png")


def plot_confidence_tiers(
    reports: Sequence[Any], task: str, output_dir: Path = FIGURES_DIR
) -> Path:
    """Show accuracy and share for each confidence tier.

    Args:
        reports: Tier reports, ordered most to least confident.
        task: Task name, used in the title and filename.
        output_dir: Directory to write into.

    Returns:
        Path to the written figure.
    """
    apply_style()
    figure, axis = plt.subplots(figsize=(7.5, 5.0))

    positions = np.arange(len(reports))
    accuracies = [report.accuracy for report in reports]

    axis.bar(
        positions,
        accuracies,
        color=list(TIER_RAMP[: len(reports)]),
        width=1 - BAR_GAP * 18,
    )

    for position, report in zip(positions, reports):
        if np.isfinite(report.accuracy):
            axis.text(
                position,
                report.accuracy + 0.02,
                f"{report.accuracy:.0%}\nn={report.count}",
                ha="center",
                fontsize=9,
                color=TEXT_SECONDARY,
            )

    axis.set_xticks(positions, [report.tier for report in reports])
    axis.set_xlabel("Confidence tag (cut-points fitted on dev)")
    axis.set_ylabel("Accuracy within tier")
    axis.set_ylim(0, 1.15)
    axis.set_title(f"Do the confidence tags mean anything?  |  {task}")
    axis.grid(axis="x", visible=False)
    hide_spines(axis)

    return finish(figure, output_dir / f"confidence_tiers_{task}.png")
