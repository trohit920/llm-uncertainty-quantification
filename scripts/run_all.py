"""Run the entire pipeline end to end, from generation to final artifacts.

This is the single entry point for reproducing the submission. It chains the
five stages in dependency order, stopping at the first failure so a broken
stage is never masked by a later one:

    generate -> figures + metrics -> report -> notebook -> study guide

Usage::

    python scripts/run_all.py                # full run (~40 min on an RTX 2070)
    python scripts/run_all.py --quick        # 15 examples per task (~6 min)
    python scripts/run_all.py --skip-generation   # rebuild artifacts only (~2 min)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging_utils import configure_logging  # noqa: E402

logger = logging.getLogger("run_all")

#: Directory holding the stage scripts.
SCRIPTS_DIR = Path(__file__).resolve().parent

#: Project root, used as the working directory for every stage.
PROJECT_ROOT = SCRIPTS_DIR.parent


@dataclass(frozen=True)
class Stage:
    """One pipeline stage.

    Attributes:
        name: Human-readable stage name for logging.
        script: Filename of the script under ``scripts/``.
        is_generation: Whether the stage performs model generation, and so is
            skipped by ``--skip-generation``.
    """

    name: str
    script: str
    is_generation: bool = False


#: Stages in dependency order. Each consumes what the previous one wrote.
STAGES: tuple[Stage, ...] = (
    Stage("Generate records", "run_experiment.py", is_generation=True),
    Stage("Figures and metrics", "make_figures.py"),
    Stage("Analysis report", "fill_report.py"),
    Stage("Study guide PDF", "make_study_guide.py"),
    Stage("Notebook", "build_notebook.py"),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Generate 15 examples per task instead of the full sweep.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Reuse existing records and rebuild artifacts only.",
    )
    return parser.parse_args()


def run_stage(stage: Stage, extra_args: list[str]) -> bool:
    """Execute one stage as a subprocess.

    Args:
        stage: The stage to run.
        extra_args: Additional command-line arguments for the stage.

    Returns:
        ``True`` when the stage exited successfully.
    """
    command = [sys.executable, str(SCRIPTS_DIR / stage.script), *extra_args]
    logger.info("START  %s  (%s)", stage.name, " ".join(command[1:]))

    started = time.perf_counter()
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    elapsed = time.perf_counter() - started

    if completed.returncode != 0:
        logger.error(
            "FAILED %s after %.1fs (exit code %d)",
            stage.name,
            elapsed,
            completed.returncode,
        )
        return False

    logger.info("OK     %s in %.1fs", stage.name, elapsed)
    return True


def main() -> int:
    """Run every stage in order, stopping at the first failure.

    Returns:
        Process exit code: 0 when every stage succeeded.
    """
    args = parse_args()
    configure_logging()

    stages = [
        stage
        for stage in STAGES
        if not (args.skip_generation and stage.is_generation)
    ]
    logger.info("Running %d stages", len(stages))

    started = time.perf_counter()
    for index, stage in enumerate(stages, start=1):
        extra_args = ["--quick"] if (args.quick and stage.is_generation) else []
        logger.info("--- stage %d/%d ---", index, len(stages))
        if not run_stage(stage, extra_args):
            logger.error("Pipeline aborted at stage %d: %s", index, stage.name)
            return 1

    minutes = (time.perf_counter() - started) / 60
    logger.info("Pipeline complete: %d stages in %.1f minutes", len(stages), minutes)
    logger.info("Artifacts: report.md, study_guide.pdf, figures/, notebooks/, results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
