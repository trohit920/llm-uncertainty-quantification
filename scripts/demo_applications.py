"""Demonstrate uncertainty-aware generation on real recorded outputs.

Prints three things per task:

1. **Selective answering** -- the coverage/accuracy trade-off at each target,
   with dev-fitted thresholds applied to held-out data.
2. **Confidence tags** -- accuracy within each High/Medium/Low tier.
3. **Worked examples** -- actual questions the system answered confidently and
   actual questions it abstained on, so the numbers have faces attached.

Usage::

    python scripts/demo_applications.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.analysis import analyse_task, split_indices  # noqa: E402
from src.applications import assign_confidence_tags  # noqa: E402
from src.config import RESULTS_DIR, TASKS  # noqa: E402
from src.experiment import records_for_task, signal_columns  # noqa: E402
from src.logging_utils import configure_logging  # noqa: E402
from src.signals import get_spec, to_uncertainty  # noqa: E402

logger = logging.getLogger("demo_applications")

#: Worked examples shown per confidence tier.
_EXAMPLES_PER_TIER: int = 2

#: Characters of a generation shown in the worked-example table.
_SNIPPET_LENGTH: int = 70


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
    return parser.parse_args()


def _rule(title: str) -> None:
    """Print a section heading.

    Args:
        title: Heading text.
    """
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _snippet(text: str) -> str:
    """Collapse a generation to a single short line.

    Args:
        text: Raw generation text.

    Returns:
        A whitespace-collapsed, truncated snippet.
    """
    flattened = " ".join(text.split())
    if len(flattened) <= _SNIPPET_LENGTH:
        return flattened
    return flattened[: _SNIPPET_LENGTH - 1] + "…"


def print_selective_table(points: Sequence[Any]) -> None:
    """Print the selective-answering operating points.

    Args:
        points: Selective operating points.
    """
    print(
        f"{'Target':>7} | {'Coverage':>9} | {'Accuracy':>9} | "
        f"{'Answered':>9} | Target met?"
    )
    print("-" * 62)
    for point in points:
        accuracy_text = (
            f"{point.test_accuracy:9.3f}"
            if np.isfinite(point.test_accuracy)
            else f"{'n/a':>9}"
        )
        print(
            f"{point.target_accuracy:7.0%} | {point.test_coverage:9.1%} | "
            f"{accuracy_text} | {point.test_num_answered:9d} | "
            f"{'yes' if point.target_met_on_test else 'no'}"
        )


def print_tier_table(tiers: Sequence[Any], gap: float) -> None:
    """Print the confidence-tier report.

    Args:
        tiers: Tier reports.
        gap: Accuracy gap between the High and Low tiers.
    """
    print(f"{'Tier':>8} | {'Share':>7} | {'Count':>6} | {'Accuracy':>9}")
    print("-" * 40)
    for tier in tiers:
        accuracy_text = (
            f"{tier.accuracy:9.3f}" if np.isfinite(tier.accuracy) else f"{'n/a':>9}"
        )
        print(
            f"{tier.tier:>8} | {tier.share:7.1%} | {tier.count:6d} | {accuracy_text}"
        )
    print(f"\nHigh minus Low accuracy gap: {gap:+.3f}")


def print_worked_examples(
    task: str, signal: str, results_dir: Path
) -> None:
    """Show real questions from the most and least confident tiers.

    Args:
        task: Task name.
        signal: Signal driving the tags.
        results_dir: Directory holding the record files.
    """
    records = records_for_task(task, results_dir)
    values = signal_columns(records)[signal]
    uncertainty = to_uncertainty(signal, values)
    dev_indices, test_indices = split_indices(len(records))

    dev_uncertainty = uncertainty[dev_indices]
    dev_uncertainty = dev_uncertainty[np.isfinite(dev_uncertainty)]
    thresholds = (
        float(np.quantile(dev_uncertainty, 0.33)),
        float(np.quantile(dev_uncertainty, 0.66)),
    )

    test_uncertainty = uncertainty[test_indices]
    tags = assign_confidence_tags(test_uncertainty, thresholds)

    for tier in ("High", "Low"):
        print(f"\n--- {tier}-confidence examples ---")
        shown = 0
        for position, tag in enumerate(tags):
            if tag != tier or shown >= _EXAMPLES_PER_TIER:
                continue
            record = records[test_indices[position]]
            verdict = "correct" if record["greedy_correct"] else "WRONG"
            print(f"  Q: {_snippet(record['question'])}")
            print(f"  A: {_snippet(record['greedy_text'])}   [{verdict}]")
            print(
                f"     {get_spec(signal).display_name} = "
                f"{record['signals'][signal]:.4f}\n"
            )
            shown += 1


def main() -> int:
    """Print the application demonstrations.

    Returns:
        Process exit code.
    """
    args = parse_args()
    configure_logging()

    for task in args.tasks:
        analysis = analyse_task(task, args.results_dir)
        best = analysis.best_signal

        _rule(f"{task}  |  gating signal: {get_spec(best).display_name}")
        print(
            f"Held-out examples: {analysis.num_test}   "
            f"Greedy accuracy: {analysis.greedy_accuracy:.3f}\n"
        )

        print("[1] SELECTIVE ANSWERING — answer or abstain")
        print_selective_table(analysis.selective)

        print("\n[2] CONFIDENCE TAGS")
        print_tier_table(analysis.tiers, analysis.tier_gap)

        print("\n[3] WORKED EXAMPLES")
        print_worked_examples(task, best, args.results_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
