"""Build a printable study-guide PDF summarising methods and results.

Rendered with matplotlib's ``PdfPages`` rather than a LaTeX or HTML toolchain,
so it adds no dependency beyond what the project already pins and needs no
system packages.

The guide is a revision aid: the concepts and formulas on one side, and this
run's actual numbers on the other, so the two can be read together.

Usage::

    python scripts/make_study_guide.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from src.analysis import TaskAnalysis, analyse_task  # noqa: E402
from src.config import (  # noqa: E402
    FIGURES_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    TASKS,
)
from src.logging_utils import configure_logging  # noqa: E402
from src.plot_style import PALETTE, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY  # noqa: E402

logger = logging.getLogger("make_study_guide")

#: A4 portrait in inches.
_PAGE_SIZE: tuple[float, float] = (8.27, 11.69)

#: Left margin for body text, in axes coordinates.
_MARGIN: float = 0.07

#: Vertical cursor start, in axes coordinates.
_TOP: float = 0.95


class Page:
    """A single PDF page built by writing lines top-down."""

    def __init__(self, title: str) -> None:
        """Start a page with a title and rule.

        Args:
            title: Page heading.
        """
        self.figure = plt.figure(figsize=_PAGE_SIZE, facecolor=SURFACE)
        self.axis = self.figure.add_axes([0, 0, 1, 1])
        self.axis.set_axis_off()
        # Pin the data limits and disable autoscaling: every position below is
        # a page fraction, and a single autoscaling plot() call would otherwise
        # rescale the whole page out from under the text already placed.
        self.axis.set_xlim(0, 1)
        self.axis.set_ylim(0, 1)
        self.axis.set_autoscale_on(False)
        self.cursor = _TOP

        self._write(title, fontsize=17, weight="bold", color=TEXT_PRIMARY)
        self.cursor -= 0.018
        self.axis.plot(
            [_MARGIN, 1 - _MARGIN],
            [self.cursor, self.cursor],
            color=PALETTE[0],
            linewidth=2.0,
        )
        self.cursor -= 0.022

    def _write(
        self,
        text: str,
        fontsize: float,
        color: str,
        weight: str = "normal",
        family: str | None = None,
        indent: float = 0.0,
    ) -> None:
        """Place one line of text at the cursor without advancing it.

        Args:
            text: The line to draw.
            fontsize: Point size.
            color: Text colour.
            weight: Font weight.
            family: Optional font family, e.g. ``"monospace"``.
            indent: Extra left indent as a page fraction.
        """
        kwargs = {"family": family} if family else {}
        self.axis.text(
            _MARGIN + indent,
            self.cursor,
            text,
            fontsize=fontsize,
            fontweight=weight,
            color=color,
            va="top",
            **kwargs,
        )

    def heading(self, text: str) -> None:
        """Write a section heading.

        Args:
            text: Heading text.
        """
        self.cursor -= 0.010
        self._write(text, fontsize=12, weight="bold", color=PALETTE[0])
        self.cursor -= 0.024

    def body(self, text: str, width: int = 96) -> None:
        """Write a wrapped paragraph.

        Args:
            text: Paragraph text.
            width: Wrap width in characters.
        """
        for line in textwrap.wrap(text, width=width) or [""]:
            self._write(line, fontsize=9.5, color=TEXT_PRIMARY)
            self.cursor -= 0.0165
        self.cursor -= 0.008

    def mono(self, lines: Sequence[str]) -> None:
        """Write monospaced lines, used for tables and formulas.

        Args:
            lines: Lines to write verbatim.
        """
        for line in lines:
            self._write(
                line, fontsize=8.0, color=TEXT_SECONDARY, family="monospace"
            )
            self.cursor -= 0.0145
        self.cursor -= 0.008

    def bullets(self, items: Sequence[str], width: int = 92) -> None:
        """Write a bulleted list.

        Args:
            items: Bullet texts.
            width: Wrap width in characters.
        """
        for item in items:
            wrapped = textwrap.wrap(item, width=width)
            for index, line in enumerate(wrapped):
                prefix = "•  " if index == 0 else "   "
                self._write(
                    prefix + line,
                    fontsize=9.5,
                    color=TEXT_PRIMARY,
                    indent=0.01,
                )
                self.cursor -= 0.0165
            self.cursor -= 0.005
        self.cursor -= 0.006

    @property
    def space_remaining(self) -> float:
        """Vertical page fraction still free below the cursor."""
        return self.cursor - 0.05


def concepts_page() -> plt.Figure:
    """Build the concepts-and-formulas page.

    Returns:
        The page figure.
    """
    page = Page("Uncertainty Quantification — Concepts")

    page.heading("The problem")
    page.body(
        "Language models produce fluent, confident-sounding text whether or "
        "not they know the answer. Uncertainty quantification asks a narrower, "
        "answerable question: can we compute a number, at generation time, "
        "that ranks wrong answers above right ones?"
    )

    page.heading("Two granularities")
    page.mono(
        [
            "TOKEN-LEVEL    one greedy decode, per-step next-token distribution p_t",
            "  entropy      H(p_t)  = -SUM_v p_t(v) log p_t(v)            [nats]",
            "  max-softmax  max_v p_t(v)          margin  top1 - top2",
            "  log-perplexity  -(1/T) SUM_t log p_t(y_t)",
            "",
            "SENTENCE-LEVEL  N stochastic samples, compare their MEANINGS",
            "  semantic entropy   H over meaning-cluster probabilities",
            "  lexical entropy    H over exact-match strings   (the ablation)",
            "  self-consistency   share of samples equal to the modal answer",
        ]
    )

    page.heading("Semantic entropy — the key idea")
    page.body(
        "Sampling the same question ten times may give ten different strings "
        "that all mean the same thing. Counting strings overstates uncertainty. "
        "Kuhn et al. (2023) cluster samples by bidirectional NLI entailment: "
        "answers a and b share a meaning class when a entails b AND b entails "
        "a. Entropy is then taken over cluster probabilities, not string "
        "frequencies."
    )
    page.mono(
        [
            "  Discrete   SE = -SUM_c (n_c/N) log(n_c/N)        cluster counts",
            "  Weighted   SE = -SUM_c P(c) log P(c),  P(c) ∝ SUM_{i∈c} exp(logp_i)",
            "",
            "  Two traps:  premise must include the QUESTION",
            "              read the entailment index from config.id2label",
            "              (cross-encoders order [contradiction, entailment, neutral])",
        ]
    )

    page.heading("Evaluation metrics")
    page.mono(
        [
            "AUROC   P(uncertainty of a wrong answer > that of a right one).",
            "        Rank-based (Mann-Whitney U), ties get average ranks.",
            "        0.5 = useless.  Undefined if all-correct or all-wrong.",
            "",
            "ECE     SUM_b (n_b/N) |acc(b) - conf(b)|.  Needs a PROBABILITY, so",
            "        raw entropies are mapped through a dev-fitted calibrator.",
            "        Fixed-width bins go empty at small N -> also use equal-mass.",
            "",
            "AURC    mean risk over all coverage levels, sorting by confidence.",
            "        Lower is better. Directly models answer-or-abstain.",
        ]
    )

    page.heading("The leakage rule")
    page.body(
        "Thresholds, calibrators and operating points are fitted on the dev "
        "split and reported on the held-out test split. Fitting on the split "
        "you report makes every method look better calibrated and every "
        "abstention threshold look more transferable than it is."
    )

    return page.figure


def results_page(analysis: TaskAnalysis) -> plt.Figure:
    """Build the results-summary page for one task.

    One page per task rather than one shared page: two full result tables plus
    their application summaries overflow a single A4 sheet.

    Args:
        analysis: A completed task analysis.

    Returns:
        The page figure.
    """
    page = Page(f"This Run — {analysis.task}")

    page.heading("Signal ranking")
    page.mono(
        [
            f"greedy accuracy {analysis.greedy_accuracy:.3f}   "
            f"dev {analysis.num_dev} / test {analysis.num_test}",
            "",
            f"{'signal':<38} {'cost':<13} {'AUROC':>7}  {'95% CI':>16}",
            "-" * 80,
        ]
        + [
            f"{ev.display_name[:37]:<38} {ev.cost_tier:<13} "
            f"{ev.auroc.value:>7.3f}  "
            f"[{ev.auroc.lower:>5.3f}, {ev.auroc.upper:>5.3f}]"
            for ev in analysis.evaluations[:8]
            if np.isfinite(ev.auroc.value)
        ]
    )

    if analysis.selective:
        page.heading("Selective answering")
        page.mono(
            ["dev-fitted thresholds applied to held-out data:"]
            + [
                f"  target {p.target_accuracy:.0%} -> coverage "
                f"{p.test_coverage:>5.1%}, accuracy "
                f"{p.test_accuracy:.3f}" if np.isfinite(p.test_accuracy) else
                f"  target {p.target_accuracy:.0%} -> abstained on everything"
                for p in analysis.selective
            ]
        )

    if analysis.cascade is not None and analysis.cascade.selected is not None:
        selected = analysis.cascade.selected
        page.heading("Self-consistency cascade")
        page.mono(
            [
                f"  greedy            {analysis.cascade.greedy_accuracy:.3f}   1.00 gen",
                f"  always sample     {analysis.cascade.self_consistency_accuracy:.3f}  "
                f"{1 + analysis.cascade.num_samples:>5.2f} gen",
                f"  gated cascade     {selected.test_accuracy:.3f}  "
                f"{selected.mean_generations:>5.2f} gen  "
                f"({selected.test_escalation_rate:.0%} escalated)",
            ]
        )

    return page.figure


def figure_page(image_path: Path, caption: str) -> plt.Figure:
    """Wrap a saved PNG in a captioned page.

    Args:
        image_path: Path to the PNG.
        caption: Caption text.

    Returns:
        The page figure.
    """
    figure = plt.figure(figsize=_PAGE_SIZE, facecolor=SURFACE)
    axis = figure.add_axes([0.06, 0.12, 0.88, 0.78])
    axis.imshow(mpimg.imread(image_path))
    axis.set_axis_off()

    figure.text(
        _MARGIN,
        0.94,
        caption,
        fontsize=13,
        fontweight="bold",
        color=TEXT_PRIMARY,
        va="top",
    )
    figure.text(
        _MARGIN,
        0.08,
        image_path.name,
        fontsize=8,
        family="monospace",
        color=TEXT_SECONDARY,
    )
    return figure


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "study_guide.pdf"
    )
    return parser.parse_args()


def main() -> int:
    """Write the study-guide PDF.

    Returns:
        Process exit code.
    """
    args = parse_args()
    configure_logging()

    analyses = [analyse_task(task, args.results_dir) for task in TASKS]

    captions = {
        "auroc_by_signal": "Which signals predict an error?",
        "risk_coverage": "Risk falls as you abstain on the uncertain cases",
        "reliability_equal_mass": "Calibration, equal-mass bins",
        "cascade": "Spending compute only where the model is unsure",
        "cost_quality": "What better uncertainty actually costs",
        "span_comparison": "Whole generation vs final-answer span",
    }

    with PdfPages(args.output) as pdf:
        pdf.savefig(concepts_page())
        plt.close("all")
        for analysis in analyses:
            pdf.savefig(results_page(analysis))
            plt.close("all")

        for stem, caption in captions.items():
            for path in sorted(args.figures_dir.glob(f"{stem}_*.png")):
                task = path.stem.replace(f"{stem}_", "")
                pdf.savefig(figure_page(path, f"{caption}  —  {task}"))
                plt.close("all")

        pdf.infodict()["Title"] = "Uncertainty Quantification — Study Guide"

    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
