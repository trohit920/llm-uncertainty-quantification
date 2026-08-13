"""Central configuration for the uncertainty-quantification experiments.

This module is the single source of truth for model identifiers, decoding
hyper-parameters, dataset sizes, evaluation settings and filesystem paths.
It is a leaf module: it imports nothing from the rest of the package, so every
other module may depend on it without creating a cycle.

All randomness in the project is seeded from :data:`SEED`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

#: Global seed applied to Python, NumPy and Torch RNGs.
SEED: Final[int] = 42

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

#: Instruction-tuned generation model. Small enough for 8 GB of VRAM in fp16.
GENERATION_MODEL_ID: Final[str] = "Qwen/Qwen2.5-1.5B-Instruct"

#: Cross-encoder NLI model used for bidirectional-entailment clustering.
NLI_MODEL_ID: Final[str] = "cross-encoder/nli-deberta-v3-small"

#: Torch dtype name for the generation model.
GENERATION_DTYPE: Final[str] = "float16"

#: Batch size for NLI premise/hypothesis pair scoring.
NLI_BATCH_SIZE: Final[int] = 32

# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------

#: Number of stochastic samples drawn per prompt for sentence-level methods.
NUM_SAMPLES: Final[int] = 10

#: Sampling temperature for the stochastic pass.
SAMPLING_TEMPERATURE: Final[float] = 1.0

#: Nucleus-sampling mass for the stochastic pass.
SAMPLING_TOP_P: Final[float] = 0.95

#: Maximum new tokens per task. Short-form QA needs far fewer than math CoT.
MAX_NEW_TOKENS: Final[dict[str, int]] = {
    "nq_open": 48,
    "gsm8k": 320,
}

#: Maximum new tokens for the verbalized-confidence follow-up question.
VERBALIZED_MAX_NEW_TOKENS: Final[int] = 12

# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

#: Hugging Face dataset identifiers, split names and streaming configuration.
NQ_OPEN_DATASET_ID: Final[str] = "google-research-datasets/nq_open"
NQ_OPEN_SPLIT: Final[str] = "validation"

GSM8K_DATASET_ID: Final[str] = "openai/gsm8k"
GSM8K_CONFIG: Final[str] = "main"
GSM8K_SPLIT: Final[str] = "test"

#: Number of examples drawn per task for the full run.
NUM_EXAMPLES: Final[dict[str, int]] = {
    "nq_open": 200,
    "gsm8k": 100,
}

#: Number of examples per task for the fast smoke-test run.
QUICK_NUM_EXAMPLES: Final[dict[str, int]] = {
    "nq_open": 15,
    "gsm8k": 15,
}

#: Canonical task names, in report order.
TASKS: Final[tuple[str, ...]] = ("nq_open", "gsm8k")

# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

#: Fraction of examples used to fit thresholds. The remainder is held out.
DEV_FRACTION: Final[float] = 0.5

#: Bin count for both fixed-width and equal-mass reliability diagrams.
NUM_CALIBRATION_BINS: Final[int] = 10

#: Bootstrap resamples used for confidence intervals on every headline metric.
NUM_BOOTSTRAP_RESAMPLES: Final[int] = 1000

#: Two-sided confidence level for bootstrap intervals.
BOOTSTRAP_CONFIDENCE_LEVEL: Final[float] = 0.95

#: Tolerance for comparing extracted GSM8K numeric answers.
NUMERIC_ANSWER_TOLERANCE: Final[float] = 1e-4

#: Target accuracy levels for the selective-answering operating points.
#: These must be reachable given the task's base accuracy: a target below the
#: base rate is trivially met at full coverage, while one far above it forces
#: the system to abstain on everything and reports nothing useful. Closed-book
#: NQ-Open runs near 14% accuracy, so it needs a different ladder from GSM8K.
SELECTIVE_TARGET_ACCURACIES_BY_TASK: Final[dict[str, tuple[float, ...]]] = {
    "nq_open": (0.25, 0.40, 0.50, 0.60),
    "gsm8k": (0.60, 0.70, 0.80, 0.90),
}

#: Fallback ladder for tasks with no explicit entry.
SELECTIVE_TARGET_ACCURACIES: Final[tuple[float, ...]] = (0.60, 0.70, 0.80, 0.90)


def selective_targets_for_task(task: str) -> tuple[float, ...]:
    """Return the accuracy ladder appropriate to a task's base rate.

    Args:
        task: Task name.

    Returns:
        Target accuracies for the selective-answering operating points.
    """
    return SELECTIVE_TARGET_ACCURACIES_BY_TASK.get(
        task, SELECTIVE_TARGET_ACCURACIES
    )

#: Coverage quantiles used to tag predictions High / Medium / Low confidence.
CONFIDENCE_TAG_QUANTILES: Final[tuple[float, float]] = (0.33, 0.66)

#: Labels for the three confidence tiers, most confident first.
CONFIDENCE_TAG_NAMES: Final[tuple[str, str, str]] = ("High", "Medium", "Low")

# --------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------

#: Root of the ``track_2`` project directory.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
FIGURES_DIR: Final[Path] = PROJECT_ROOT / "figures"
NOTEBOOK_DIR: Final[Path] = PROJECT_ROOT / "notebooks"

#: Raw per-example generation records, one JSON object per line.
RECORDS_FILENAME: Final[str] = "records_{task}.jsonl"

#: Aggregated metric tables written by the evaluation stage.
METRICS_FILENAME: Final[str] = "metrics_{task}.json"


def ensure_output_dirs() -> None:
    """Create the results, figures and notebook directories if absent."""
    for directory in (RESULTS_DIR, FIGURES_DIR, NOTEBOOK_DIR):
        directory.mkdir(parents=True, exist_ok=True)
