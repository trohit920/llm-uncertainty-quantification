"""Run the generation pipeline over one or more tasks.

Typical use::

    # Fast validation over 15 examples per task, ~10 minutes
    python scripts/run_experiment.py --quick

    # Full run: 200 NQ-Open + 100 GSM8K
    python scripts/run_experiment.py

Records are streamed to ``results/records_{task}.jsonl`` as they are produced,
so an interrupted run leaves usable output behind.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    GENERATION_DTYPE,
    GENERATION_MODEL_ID,
    MAX_NEW_TOKENS,
    NUM_EXAMPLES,
    NUM_SAMPLES,
    QUICK_NUM_EXAMPLES,
    RESULTS_DIR,
    SEED,
    TASKS,
    ensure_output_dirs,
)
from src.experiment import run_task  # noqa: E402
from src.logging_utils import configure_logging  # noqa: E402
from src.model import UncertaintyAwareGenerator, seed_everything  # noqa: E402
from src.semantic_clustering import EntailmentModel  # noqa: E402

logger = logging.getLogger("run_experiment")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASKS),
        choices=list(TASKS),
        help="Tasks to run.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the small validation sweep instead of the full run.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Override the per-task example count.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_SAMPLES,
        help="Stochastic samples drawn per example.",
    )
    parser.add_argument(
        "--no-verbalized",
        action="store_true",
        help="Skip the verbalized-confidence follow-up pass.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for the JSONL record files.",
    )
    parser.add_argument(
        "--model",
        default=GENERATION_MODEL_ID,
        help="Override the generation model, e.g. a smaller one for smoke tests.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string. Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--dtype",
        default=GENERATION_DTYPE,
        help="Torch dtype for the generation model. Use float32 on CPU.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override the per-task generation cap.",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Global seed.")
    return parser.parse_args()


def resolve_example_count(args: argparse.Namespace, task: str) -> int:
    """Decide how many examples to run for a task.

    Args:
        args: Parsed arguments.
        task: Task name.

    Returns:
        The example count to use.
    """
    if args.num_examples is not None:
        return args.num_examples
    table = QUICK_NUM_EXAMPLES if args.quick else NUM_EXAMPLES
    return table[task]


def main() -> int:
    """Run the pipeline and report a summary.

    Returns:
        Process exit code.
    """
    args = parse_args()
    configure_logging()
    ensure_output_dirs()
    seed_everything(args.seed)

    logger.info(
        "Starting run | model=%s | tasks=%s | samples=%d | verbalized=%s",
        args.model,
        args.tasks,
        args.num_samples,
        not args.no_verbalized,
    )

    if args.max_new_tokens is not None:
        for task in args.tasks:
            MAX_NEW_TOKENS[task] = args.max_new_tokens
        logger.info("Overriding max_new_tokens to %d", args.max_new_tokens)

    generator = UncertaintyAwareGenerator(
        model_id=args.model, dtype=args.dtype, device=args.device
    )
    entailment_model = EntailmentModel(device=args.device)

    started = time.perf_counter()
    try:
        for task in args.tasks:
            run_task(
                task,
                resolve_example_count(args, task),
                generator,
                entailment_model,
                output_dir=args.output_dir,
                num_samples=args.num_samples,
                collect_verbalized=not args.no_verbalized,
                seed=args.seed,
            )
    finally:
        generator.close()

    logger.info(
        "All tasks complete in %.1f minutes", (time.perf_counter() - started) / 60
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
