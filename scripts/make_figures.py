"""Generate every figure from the recorded runs.

Reads ``results/records_{task}.jsonl``, computes the analysis bundle and
writes PNGs to ``figures/``. Also persists ``results/metrics_{task}.json``, so
the report and notebook read the same numbers the figures were drawn from.

Usage::

    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.analysis import (  # noqa: E402
    analyse_task,
    reliability_for_signals,
    risk_coverage_for_signals,
    save_metrics,
    top_signals,
)
from src.config import (  # noqa: E402
    FIGURES_DIR,
    RESULTS_DIR,
    TASKS,
    ensure_output_dirs,
)
from src.experiment import records_for_task, signal_columns  # noqa: E402
from src.logging_utils import configure_logging  # noqa: E402
from src.plots import (  # noqa: E402
    plot_auroc_by_signal,
    plot_cost_versus_quality,
    plot_reliability_diagrams,
    plot_risk_coverage,
    plot_signal_distribution,
    plot_span_comparison,
)
from src.plots_applications import (  # noqa: E402
    plot_cascade,
    plot_confidence_tiers,
    plot_selective_answering,
)
from src.signals import get_spec  # noqa: E402

logger = logging.getLogger("make_figures")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", nargs="+", default=list(TASKS), choices=list(TASKS)
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    return parser.parse_args()


def figures_for_task(task: str, results_dir: Path, figures_dir: Path) -> list[Path]:
    """Build every figure for one task.

    Args:
        task: Task name.
        results_dir: Directory holding the record files.
        figures_dir: Destination directory for PNGs.

    Returns:
        Paths of the figures written.
    """
    analysis = analyse_task(task, results_dir)
    save_metrics(analysis, results_dir)

    best = analysis.best_signal
    leaders = top_signals(analysis)
    written: list[Path] = [
        plot_auroc_by_signal(analysis.evaluations, task, figures_dir),
        plot_selective_answering(analysis.selective, task, figures_dir),
        plot_confidence_tiers(analysis.tiers, task, figures_dir),
        plot_cost_versus_quality(
            analysis.evaluations, analysis.cost_seconds, task, figures_dir
        ),
    ]

    for strategy in ("fixed_width", "equal_mass"):
        written.append(
            plot_reliability_diagrams(
                reliability_for_signals(task, leaders, strategy, results_dir),
                task,
                strategy,
                figures_dir,
            )
        )

    curves, labels, base_error_rate = risk_coverage_for_signals(
        task, leaders, results_dir
    )
    written.append(
        plot_risk_coverage(curves, labels, task, base_error_rate, figures_dir)
    )

    written.append(_distribution_figure(task, best, results_dir, figures_dir))

    if any(ev.signal.startswith("answer_") for ev in analysis.evaluations):
        written.append(
            plot_span_comparison(analysis.evaluations, task, figures_dir)
        )

    if analysis.cascade is not None:
        written.append(plot_cascade(analysis.cascade, task, figures_dir))

    return written


def _distribution_figure(
    task: str, signal: str, results_dir: Path, figures_dir: Path
) -> Path:
    """Draw the correct-versus-wrong distribution for the best signal.

    Args:
        task: Task name.
        signal: Signal name.
        results_dir: Directory holding the record files.
        figures_dir: Destination directory.

    Returns:
        Path to the written figure.
    """
    records = records_for_task(task, results_dir)
    values = signal_columns(records)[signal]
    correct = np.asarray([record["greedy_correct"] for record in records], dtype=bool)

    return plot_signal_distribution(
        values[correct],
        values[~correct],
        get_spec(signal).display_name,
        task,
        figures_dir,
    )


def main() -> int:
    """Generate figures for every requested task.

    Returns:
        Process exit code.
    """
    args = parse_args()
    configure_logging()
    ensure_output_dirs()

    total = 0
    for task in args.tasks:
        written = figures_for_task(task, args.results_dir, args.figures_dir)
        total += len(written)
        logger.info("Task %s: wrote %d figures", task, len(written))

    logger.info("Wrote %d figures to %s", total, args.figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
