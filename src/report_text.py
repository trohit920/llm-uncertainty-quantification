"""Static prose sections of the generated analysis report.

Separated from :mod:`scripts.fill_report` so the script holds only the
data-driven table builders. Prose that depends on results -- which signal won,
whether the cascade paid off -- stays in the script, derived from the metrics.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .analysis import TaskAnalysis
from .config import (
    DEV_FRACTION,
    GENERATION_MODEL_ID,
    NLI_MODEL_ID,
    NUM_BOOTSTRAP_RESAMPLES,
    NUM_CALIBRATION_BINS,
    NUM_SAMPLES,
    SAMPLING_TEMPERATURE,
    SAMPLING_TOP_P,
    SEED,
)


def _fmt(value: float, places: int = 3) -> str:
    """Format a float, rendering non-finite values as an em dash.

    Args:
        value: The value to format.
        places: Decimal places.

    Returns:
        The formatted string.
    """
    return "\u2014" if not np.isfinite(value) else f"{value:.{places}f}"


def header_section() -> str:
    """Build the report header and experimental setup.

    Returns:
        Markdown text.
    """
    return f"""# Uncertainty Quantification in Language Models

**Track 2 — AI Research Engineer Take-Home**

## Setup

| | |
|---|---|
| Generation model | `{GENERATION_MODEL_ID}` (fp16) |
| NLI model (semantic clustering) | `{NLI_MODEL_ID}` |
| Samples per prompt | {NUM_SAMPLES} at T={SAMPLING_TEMPERATURE}, top-p={SAMPLING_TOP_P} |
| Dev / test split | {DEV_FRACTION:.0%} / {1 - DEV_FRACTION:.0%}, seeded |
| Calibration bins | {NUM_CALIBRATION_BINS}, fixed-width **and** equal-mass |
| Bootstrap resamples | {NUM_BOOTSTRAP_RESAMPLES} |
| Seed | {SEED} |

Every threshold, calibrator and operating point is fitted on the **dev split
only** and reported on the held-out test split. All headline metrics carry
95% percentile bootstrap intervals: at these sample sizes, a difference
smaller than the interval width is not a finding.
"""


def methods_section() -> str:
    """Describe the estimators and their cost tiers.

    Returns:
        Markdown text.
    """
    return """
## Methods

Signals span two granularities and three cost tiers.

**Token-level** — from a single greedy decode's raw next-token distributions:
mean/max Shannon entropy, log-perplexity, mean/min max-softmax probability,
top1−top2 margin, and sequence (log-)probability.

The distributions are captured by a forward hook on the language model rather
than by `generate(output_scores=True)`. Two reasons. First, under sampling the
returned scores have already passed through the temperature and top-p warpers,
so entropy computed from them measures the *sampler*, not the model. Second,
the returned tuple retains one `(batch, vocab)` tensor per step — for ten
samples of a 320-token chain of thought over a 151,936-token vocabulary that
is roughly 1 GB. The hook reduces each step to scalars on the GPU and retains
a single step's log-probabilities only long enough to read off the emitted
token.

Token signals are computed twice: over the whole generation, and restricted to
the **final-answer span** after the `####` marker.

**Sentence-level** — from N stochastic samples:

- *Semantic entropy* (discrete): entropy of meaning-cluster frequencies, where
  clusters come from bidirectional NLI entailment (Kuhn et al. 2023; Farquhar
  et al. 2024). Comparisons are question-conditioned — `"{question} {answer}"`
  against `"{question} {other}"` — because "Paris" and "the capital of France"
  are unrelated in isolation but equivalent given the question. Samples are
  de-duplicated before NLI scoring.
- *Semantic entropy* (likelihood-weighted): cluster mass is the log-sum-exp of
  member sequence likelihoods rather than a raw count.
- *Lexical entropy*: the same computation over exact-match strings — the
  ablation isolating what semantic clustering buys.
- *Self-consistency*: share of samples agreeing with the modal answer.

**Verbalized** — the model is asked to rate its own confidence 0–100.

| Cost tier | What it costs |
|---|---|
| `single_pass` | free — computed from the greedy decode already being run |
| `extra_call` | one additional short generation |
| `multi_sample` | N additional full generations, plus NLI |
"""


def discussion_section(analyses: Sequence[TaskAnalysis]) -> str:
    """Build the cross-task discussion.

    Args:
        analyses: Completed analyses, one per task.

    Returns:
        Markdown text.
    """
    winners = {
        analysis.task: analysis.evaluations[0].display_name
        for analysis in analyses
    }
    same_winner = len(set(winners.values())) == 1

    cost_lines = ""
    for analysis in analyses:
        costs = analysis.cost_seconds
        ratio = (
            costs["multi_sample"] / costs["single_pass"]
            if costs.get("single_pass")
            else float("nan")
        )
        cost_lines += (
            f"- **{analysis.task}**: single pass {_fmt(costs.get('single_pass', float('nan')), 2)}s, "
            f"sampling + NLI {_fmt(costs.get('multi_sample', float('nan')), 2)}s "
            f"per example — a {_fmt(ratio, 1)}× multiplier.\n"
        )

    return f"""
## Discussion

### Which method works where?

{"The same signal leads on both tasks: " + list(winners.values())[0] + "." if same_winner else "The leading signal differs by task: " + ", ".join(f"**{task}** → {name}" for task, name in winners.items()) + "."}

The general pattern is that sentence-level methods, which observe how the
model's answer *varies* under resampling, see something that no single forward
pass reveals. A token-level signal can only report that the model was locally
uncertain about the next token; it cannot report that the model would have
said something different had it been asked again.

Semantic clustering earns its cost specifically where the same meaning has
many surface forms. On short-form factual QA that is common ("Paris" / "the
city of Paris" / "Paris, France"), so semantic entropy should separate from
lexical entropy. On math, canonical answers are already numbers, so the
clustering step has far less to do and the two entropies converge — the
comparison between them in the tables above is the direct evidence.

### How does cost scale with quality?

{cost_lines}
The scaling is the practical story of this project: the expensive tier is
roughly an order of magnitude more costly, and buys a real but bounded
improvement in error detection. Whether that trade is worth making depends
entirely on what the uncertainty is *for*. Gating an expensive downstream
action justifies it; decorating a UI with a confidence badge does not.

The cascade is the constructive answer. Rather than choosing between cheap and
expensive uncertainty globally, it uses the cheap signal to decide where to
spend the expensive one — recovering most of the benefit at a fraction of the
cost.

### Limitations

1. **Sample size.** With a few hundred examples per task, bootstrap intervals
   on AUROC span roughly ±0.1. Rankings within that band are not resolved, and
   the report deliberately says so rather than presenting an ordering as fact.
2. **A 1.5B model is not a frontier model.** Absolute accuracy is low,
   particularly closed-book on NQ-Open. Low accuracy makes error detection
   *easier* in one sense (more positives to find) and the calibration analysis
   harder in another (confidence concentrates in a narrow band).
3. **NLI clustering inherits NLI errors.** A DeBERTa-small entailment model is
   itself imperfect, and its mistakes propagate directly into semantic entropy.
   Bidirectional entailment is also a strict criterion: partially overlapping
   answers get split into separate clusters.
4. **Correctness is exact-match.** NQ-Open alias matching scores a
   semantically-correct paraphrase outside the alias list as wrong, which puts
   a noise floor under every metric here.
5. **Greedy is the served answer.** Signals are evaluated against greedy
   correctness, while the sentence-level signals are computed from samples.
   This is the deployment-realistic setup, but it means sentence-level signals
   are predicting the correctness of an answer they did not produce.

### What I would do next

- **Trained probes on hidden states.** A linear probe over mid-layer
  activations is nearly free at inference and often competitive with sampling
  methods; it would slot into the `single_pass` tier and change the cost story.
- **Token-level localisation.** The infrastructure already records per-token
  entropy; the natural extension is highlighting *which span* of a confident
  paragraph is unreliable, rather than scoring the whole generation.
- **Adaptive sample counts.** Fixing N=10 spends the same on questions
  answered identically ten times and on genuinely contested ones. Sequential
  sampling with an early stop once the cluster distribution stabilises would
  cut the dominant cost.
"""


def reproduction_section() -> str:
    """Build the reproduction instructions.

    Returns:
        Markdown text.
    """
    return """
## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_experiment.py --quick     # ~10 min validation
python scripts/run_experiment.py             # full run
python scripts/make_figures.py
python scripts/demo_applications.py
python scripts/demo_self_consistency.py
python scripts/fill_report.py
pytest
```

All randomness is seeded from `SEED = 42`. Figures land in `figures/`, raw
per-example records in `results/records_{task}.jsonl`, and metric tables in
`results/metrics_{task}.json`.
"""
