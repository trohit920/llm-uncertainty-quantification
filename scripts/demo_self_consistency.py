"""Demonstrate self-consistency decoding and the uncertainty-gated cascade.

Shows, on GSM8K, three decoding policies side by side:

* **Greedy** -- one generation per question.
* **Self-consistency** -- majority vote over N samples, always.
* **Cascade** -- greedy first, escalating to sampling only when the greedy
  pass looks uncertain.

The point of the third is that most of self-consistency's accuracy gain is
concentrated on questions the greedy pass already got wrong, so paying the
sampling cost everywhere is largely wasted.

Usage::

    python scripts/demo_self_consistency.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.analysis import analyse_task, split_indices  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.experiment import records_for_task  # noqa: E402
from src.logging_utils import configure_logging  # noqa: E402
from src.signals import get_spec  # noqa: E402

logger = logging.getLogger("demo_self_consistency")

#: Task the cascade is demonstrated on.
_CASCADE_TASK: str = "gsm8k"

#: Number of flipped examples printed.
_MAX_FLIPS_SHOWN: int = 3

#: Characters of a question shown per flipped example.
_SNIPPET_LENGTH: int = 72


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def _snippet(text: str) -> str:
    """Collapse text to a single short line.

    Args:
        text: Source text.

    Returns:
        A whitespace-collapsed, truncated snippet.
    """
    flattened = " ".join(text.split())
    if len(flattened) <= _SNIPPET_LENGTH:
        return flattened
    return flattened[: _SNIPPET_LENGTH - 1] + "…"


def print_policy_comparison(report) -> None:  # noqa: ANN001 - CascadeReport
    """Print accuracy and cost for each decoding policy.

    Args:
        report: The cascade report.
    """
    selected = report.selected
    always_cost = 1 + report.num_samples

    print(f"{'Policy':<34} | {'Accuracy':>9} | {'Gen/question':>13}")
    print("-" * 64)
    print(
        f"{'Greedy only':<34} | {report.greedy_accuracy:9.3f} | {1.0:13.2f}"
    )
    print(
        f"{'Self-consistency (always)':<34} | "
        f"{report.self_consistency_accuracy:9.3f} | {always_cost:13.2f}"
    )
    if selected is not None:
        print(
            f"{'Uncertainty-gated cascade':<34} | "
            f"{selected.test_accuracy:9.3f} | {selected.mean_generations:13.2f}"
        )

    if selected is None:
        return

    gain = report.self_consistency_accuracy - report.greedy_accuracy
    recovered = selected.test_accuracy - report.greedy_accuracy
    share = recovered / gain if abs(gain) > 1e-9 else float("nan")
    print(
        f"\nSampling gains {gain:+.3f} accuracy at {always_cost:.0f}x cost.\n"
        f"The cascade recovers {recovered:+.3f} of that "
        f"({share:.0%}) while escalating only "
        f"{selected.test_escalation_rate:.0%} of questions, at "
        f"{selected.generation_cost_ratio:.0%} of the always-sample cost."
    )


def print_cost_curve(report) -> None:  # noqa: ANN001 - CascadeReport
    """Print the cascade's accuracy-cost sweep.

    Args:
        report: The cascade report.
    """
    print(
        f"\n{'Budget':>7} | {'Escalated':>10} | {'Accuracy':>9} | "
        f"{'Gen/question':>13}"
    )
    print("-" * 50)
    for point in report.curve:
        marker = " <- selected" if point is report.selected else ""
        print(
            f"{point.escalation_budget:7.0%} | {point.test_escalation_rate:10.1%} | "
            f"{point.test_accuracy:9.3f} | {point.mean_generations:13.2f}{marker}"
        )


def print_flipped_examples(task: str, results_dir: Path) -> None:
    """Show questions where majority voting fixed a wrong greedy answer.

    Args:
        task: Task name.
        results_dir: Directory holding the record files.
    """
    records = records_for_task(task, results_dir)
    _, test_indices = split_indices(len(records))

    print("\nQuestions where majority voting rescued a wrong greedy answer:")
    shown = 0
    for index in test_indices:
        record = records[index]
        if record["greedy_correct"] or not record["majority_correct"]:
            continue
        if shown >= _MAX_FLIPS_SHOWN:
            break

        agreement = np.mean(
            [answer == record["majority_answer"] for answer in record["sample_canonical"]]
        )
        print(f"\n  Q: {_snippet(record['question'])}")
        print(
            f"     greedy said {record['greedy_canonical']!r}, "
            f"majority said {record['majority_answer']!r} "
            f"({agreement:.0%} of samples agreed)"
        )
        shown += 1

    if shown == 0:
        print("  (none in the held-out split)")


def main() -> int:
    """Print the self-consistency demonstration.

    Returns:
        Process exit code.
    """
    args = parse_args()
    configure_logging()

    analysis = analyse_task(_CASCADE_TASK, args.results_dir)
    if analysis.cascade is None:
        logger.error("No cascade report available for %s", _CASCADE_TASK)
        return 1

    print(f"\n{'=' * 64}")
    print(
        f"Self-consistency decoding  |  {_CASCADE_TASK}  |  "
        f"gate: {get_spec(analysis.best_signal).display_name}"
    )
    print(f"{'=' * 64}\n")

    print_policy_comparison(analysis.cascade)
    print_cost_curve(analysis.cascade)
    print_flipped_examples(_CASCADE_TASK, args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
