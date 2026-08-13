"""Generate ``report.md`` from the recorded runs.

Every number in the report is read from ``results/metrics_{task}.json`` and
interpolated, and the narrative claims are *derived* rather than asserted --
which signal leads, whether answer-span restriction helped, whether the
cascade paid off are all computed from the data. A rerun with different
results produces a report that says different things, so the prose cannot
drift away from the numbers.

Usage::

    python scripts/fill_report.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.analysis import TaskAnalysis, analyse_task  # noqa: E402
from src.config import (  # noqa: E402
    DEV_FRACTION,
    GENERATION_MODEL_ID,
    NLI_MODEL_ID,
    NUM_BOOTSTRAP_RESAMPLES,
    NUM_CALIBRATION_BINS,
    NUM_SAMPLES,
    PROJECT_ROOT,
    RESULTS_DIR,
    SAMPLING_TEMPERATURE,
    SAMPLING_TOP_P,
    SEED,
    TASKS,
)
from src.logging_utils import configure_logging  # noqa: E402
from src.report_text import (  # noqa: E402
    discussion_section,
    header_section,
    methods_section,
    reproduction_section,
)
from src.signals import get_spec  # noqa: E402

logger = logging.getLogger("fill_report")

#: Signals listed in the per-task results table.
_TABLE_ROWS: int = 14

#: Human-readable task descriptions.
_TASK_BLURB: dict[str, str] = {
    "nq_open": "open-domain factual QA (short answers, closed book)",
    "gsm8k": "grade-school math (chain of thought, single numeric answer)",
}


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
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "report.md"
    )
    return parser.parse_args()


def _fmt(value: float, places: int = 3) -> str:
    """Format a float, rendering non-finite values as an em dash.

    Args:
        value: The value to format.
        places: Decimal places.

    Returns:
        The formatted string.
    """
    return "—" if not np.isfinite(value) else f"{value:.{places}f}"


def _pct(value: float) -> str:
    """Format a fraction as a percentage.

    Args:
        value: Fraction in [0, 1].

    Returns:
        The formatted string.
    """
    return "—" if not np.isfinite(value) else f"{value:.0%}"


def results_table(analysis: TaskAnalysis) -> str:
    """Build the per-signal results table for one task.

    Args:
        analysis: A completed task analysis.

    Returns:
        Markdown text.
    """
    lines = [
        "| Signal | Granularity | Cost | AUROC (95% CI) | AURC | ECE (fixed) | ECE (equal-mass) |",
        "|---|---|---|---|---|---|---|",
    ]
    for evaluation in analysis.evaluations[:_TABLE_ROWS]:
        lines.append(
            f"| {evaluation.display_name} | {evaluation.granularity} | "
            f"`{evaluation.cost_tier}` | "
            f"{_fmt(evaluation.auroc.value)} "
            f"[{_fmt(evaluation.auroc.lower)}, {_fmt(evaluation.auroc.upper)}] | "
            f"{_fmt(evaluation.aurc.value)} | "
            f"{_fmt(evaluation.ece_fixed_width)} | "
            f"{_fmt(evaluation.ece_equal_mass)} |"
        )
    return "\n".join(lines)


def _span_finding(analysis: TaskAnalysis) -> str:
    """Compare whole-sequence against answer-span token signals.

    Args:
        analysis: A completed task analysis.

    Returns:
        A sentence describing the comparison, derived from the data.
    """
    by_name = {ev.signal: ev for ev in analysis.evaluations}
    pairs = [
        (by_name[name], by_name[f"answer_{name}"])
        for name in by_name
        if not name.startswith("answer_") and f"answer_{name}" in by_name
    ]
    if not pairs:
        return ""

    deltas = [
        span.auroc.value - full.auroc.value
        for full, span in pairs
        if np.isfinite(full.auroc.value) and np.isfinite(span.auroc.value)
    ]
    if not deltas:
        return ""

    mean_delta = float(np.mean(deltas))
    improved = sum(1 for delta in deltas if delta > 0)
    direction = "improves" if mean_delta > 0 else "does not improve"

    return (
        f"Restricting token signals to the final-answer span {direction} error "
        f"detection: mean AUROC change {mean_delta:+.3f} across "
        f"{len(deltas)} signal pairs, with {improved}/{len(deltas)} improving. "
        f"Whole-sequence entropy over a chain of thought is dominated by "
        f"ordinary prose tokens, which carry little information about whether "
        f"the final number is right.\n"
    )


def task_section(analysis: TaskAnalysis) -> str:
    """Build the results section for one task.

    Args:
        analysis: A completed task analysis.

    Returns:
        Markdown text.
    """
    best = analysis.evaluations[0]
    blurb = _TASK_BLURB.get(analysis.task, analysis.task)

    single_pass_best = max(
        (ev for ev in analysis.evaluations if ev.cost_tier == "single_pass"),
        key=lambda ev: ev.auroc.value if np.isfinite(ev.auroc.value) else -1,
        default=None,
    )
    cheap_note = ""
    if single_pass_best is not None and np.isfinite(single_pass_best.auroc.value):
        gap = best.auroc.value - single_pass_best.auroc.value
        overlaps = single_pass_best.auroc.upper >= best.auroc.lower
        cheap_note = (
            f"The best **free** signal is {single_pass_best.display_name} at "
            f"{_fmt(single_pass_best.auroc.value)}, a gap of {gap:+.3f} against "
            f"the best signal overall. The bootstrap intervals "
            f"{'overlap, so this gap is not resolved at this sample size' if overlaps else 'do not overlap'}.\n"
        )

    section = f"""
### {analysis.task} — {blurb}

Greedy accuracy on the held-out split: **{_fmt(analysis.greedy_accuracy)}**
({analysis.num_test} examples; {analysis.num_dev} held out for fitting).

{results_table(analysis)}

**Best signal: {best.display_name}** — AUROC {_fmt(best.auroc.value)}
[{_fmt(best.auroc.lower)}, {_fmt(best.auroc.upper)}], AURC {_fmt(best.aurc.value)}.

{cheap_note}
{_span_finding(analysis)}
#### Selective answering

| Target | Coverage | Accuracy achieved | Target met on test? |
|---|---|---|---|
"""
    for point in analysis.selective:
        section += (
            f"| {point.target_accuracy:.0%} | {_pct(point.test_coverage)} | "
            f"{_fmt(point.test_accuracy)} | "
            f"{'yes' if point.target_met_on_test else 'no'} |\n"
        )

    met = sum(1 for point in analysis.selective if point.target_met_on_test)
    section += (
        f"\nDev-fitted thresholds held on unseen data for {met} of "
        f"{len(analysis.selective)} targets — the honest measure of whether an "
        f"operating point transfers, as opposed to whether it can be fitted.\n"
    )

    section += "\n#### Confidence tags\n\n| Tier | Share | Accuracy |\n|---|---|---|\n"
    for tier in analysis.tiers:
        section += f"| {tier.tier} | {_pct(tier.share)} | {_fmt(tier.accuracy)} |\n"
    section += (
        f"\nHigh-minus-Low accuracy gap: **{_fmt(analysis.tier_gap)}**. "
        f"{'The tags separate.' if analysis.tier_gap > 0.1 else 'The separation is weak — the tags would be close to decoration in a product surface.'}\n"
    )

    if analysis.cascade is not None:
        section += _cascade_section(analysis)

    return section


def _cascade_section(analysis: TaskAnalysis) -> str:
    """Build the self-consistency cascade section.

    Args:
        analysis: A completed task analysis with a cascade report.

    Returns:
        Markdown text.
    """
    cascade = analysis.cascade
    assert cascade is not None
    selected = cascade.selected
    always_cost = 1 + cascade.num_samples

    text = f"""
#### Self-consistency and the uncertainty-gated cascade

| Policy | Accuracy | Generations / question |
|---|---|---|
| Greedy only | {_fmt(cascade.greedy_accuracy)} | 1.00 |
| Self-consistency (always) | {_fmt(cascade.self_consistency_accuracy)} | {always_cost:.2f} |
"""
    if selected is not None:
        text += (
            f"| **Uncertainty-gated cascade** | "
            f"**{_fmt(selected.test_accuracy)}** | "
            f"**{selected.mean_generations:.2f}** |\n"
        )

        gain = cascade.self_consistency_accuracy - cascade.greedy_accuracy
        recovered = selected.test_accuracy - cascade.greedy_accuracy
        share = recovered / gain if abs(gain) > 1e-9 else float("nan")
        text += (
            f"\nSampling buys {gain:+.3f} accuracy at {always_cost:.0f}× the "
            f"generation cost. The cascade recovers {recovered:+.3f} of that "
            f"({_pct(share)}) while escalating only "
            f"{_pct(selected.test_escalation_rate)} of questions, at "
            f"{_pct(selected.generation_cost_ratio)} of the always-sample cost.\n"
        )
    return text


def main() -> int:
    """Write the report.

    Returns:
        Process exit code.
    """
    args = parse_args()
    configure_logging()

    analyses = [analyse_task(task, args.results_dir) for task in args.tasks]

    parts = [header_section(), methods_section(), "\n## Results\n"]
    parts.extend(task_section(analysis) for analysis in analyses)
    parts.append(discussion_section(analyses))
    parts.append(reproduction_section())

    args.output.write_text("\n".join(parts), encoding="utf-8")
    logger.info("Wrote %s (%d chars)", args.output, len(args.output.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
