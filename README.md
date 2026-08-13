Uncertainty Quantification in Language Models

Token-level and sentence-level uncertainty estimation for an open-source LLM,
evaluated across two task types, with three practical applications of
uncertainty-aware generation.

Everything runs locally on a single 8 GB consumer GPU.

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/run_all.py            # everything, end to end (~40 min)
```

That single command chains all five stages in dependency order and stops at the
first failure:

    generate -> figures + metrics -> report -> study guide -> notebook

Variants:

```bash
python scripts/run_all.py --quick             # 15 examples/task, ~6 min (sanity check)
python scripts/run_all.py --skip-generation   # rebuild artifacts from saved records, ~1 min
./verify_submission.sh                        # 25-point check: env, tests, artifacts, hygiene
```

Individual stages, if you want to run them piecemeal:

```bash
python scripts/run_experiment.py            # generation only -> results/records_*.jsonl
python scripts/make_figures.py              # figures + metrics JSON
python scripts/fill_report.py               # writes report.md from the metrics
python scripts/make_study_guide.py          # printable study_guide.pdf
python scripts/build_notebook.py            # builds and executes the notebook
python scripts/demo_applications.py         # selective answering + confidence tags
python scripts/demo_self_consistency.py     # cascade vs greedy vs always-sample
pytest                                      # 198 unit tests, no GPU or network needed
```

The first run downloads ~3.5 GB of model weights to your Hugging Face cache.

### Requirements

- **Python 3.12**, Linux or Windows
- **CUDA 12.6-capable GPU with ≥6 GB VRAM** (developed on an RTX 2070 Max-Q,
  8 GB, compute capability 7.5). CPU-only works but is impractically slow for
  the full run.
- ~13 GB free disk (venv ≈ 6 GB, model weights ≈ 3.5 GB, caches)

`requirements.txt` pins the CUDA build of torch via an extra index. On a
CPU-only machine, drop the `+cu126` local version specifier from the torch pin.

---

## What this does

Language models produce fluent, confident text whether or not they know the
answer. This project computes, at generation time, a number that ranks wrong
answers above right ones — and then uses that number to make decisions.

### Signals

| Granularity | Signal | Cost tier |
|---|---|---|
| Token | mean / max entropy, log-perplexity, mean / min max-softmax, top1−top2 margin, sequence (log-)probability | `single_pass` |
| Token | the same nine, restricted to the final-answer span | `single_pass` |
| Sentence | semantic entropy (discrete), semantic entropy (likelihood-weighted), lexical entropy, self-consistency, semantic self-consistency | `multi_sample` |
| Sentence | verbalized confidence ("rate yourself 0–100") | `extra_call` |

### Two implementation details that matter

**Raw logits, not `generate(output_scores=True)`.** Under sampling, the scores
that `generate` returns have already passed through the temperature and top-p
warpers — entropy computed from them measures the *sampler*, not the model. A
forward hook on the language model sees the logits as produced. It also avoids
retaining one `(batch, vocab)` tensor per step, which for ten samples of a
320-token chain of thought over Qwen's 151,936-token vocabulary is ~1 GB. The
hook reduces each step to scalars on the GPU and holds a single step's
log-probabilities only long enough to read off the emitted token.

**`####` is also a markdown heading.** Qwen2.5 opens chain-of-thought sections
with `#### Step 1`. Taking the first occurrence to locate the "final answer
span" anchors it to the top of the reasoning. Span detection selects the *last*
marker occurrence followed by a digit.

### Evaluation

All metrics are hand-implemented in `src/calibration.py` and unit-tested;
AUROC is cross-checked against `sklearn.roc_auc_score`, including on tied
scores.

- **Error-detection AUROC** — rank-based, incorrect answers as the positive
  class. Returns `nan` rather than a misleading 0.5 when a split is all-correct
  or all-wrong.
- **ECE + reliability diagrams** — with **both** fixed-width and equal-mass
  bins. At a few hundred examples, fixed-width bins go empty and the resulting
  number is unstable.
- **Risk-coverage / AURC** — error rate among the most confident fraction.
- **Bootstrap CIs** on every headline metric. Differences smaller than the
  interval width are not findings, and the generated report says so.

**No leakage.** Thresholds, logistic calibrators and operating points are
fitted on the dev split and reported on the held-out test split. Since a raw
entropy is not a probability, ECE requires mapping the signal to P(correct);
fitting that mapping on the split you report would flatter every method.

### Applications

1. **Selective answering** — answer or abstain, with thresholds fitted on dev
   to hit a target accuracy, reported on test with coverage operating points.
2. **Confidence tags** — High / Medium / Low tiers from dev quantiles, with the
   High-minus-Low accuracy gap as the test of whether the tags mean anything.
3. **Self-consistency decoding** — majority vote over samples for math, plus an
   **uncertainty-gated cascade** that escalates to sampling only when the
   greedy pass looks unreliable, recovering most of the accuracy gain at a
   fraction of the compute.

---

## Layout

```
track_2/
├── src/
│   ├── config.py              # all constants, paths, SEED
│   ├── logging_utils.py       # structured logging
│   ├── model.py               # generator + raw-logits forward hook
│   ├── token_metrics.py       # per-step reduction, sequence aggregation
│   ├── sentence_metrics.py    # semantic / lexical entropy, self-consistency
│   ├── semantic_clustering.py # bidirectional-entailment NLI clustering
│   ├── signals.py             # signal registry, orientation, cost tiers
│   ├── datasets_loader.py     # NQ-Open + GSM8K → one Example schema
│   ├── correctness.py         # alias matching, numeric extraction
│   ├── calibration.py         # AUROC, ECE, risk-coverage, bootstrap
│   ├── evaluate.py            # per-signal evaluation, dev-fitted calibration
│   ├── applications.py        # selective answering, confidence tags
│   ├── self_consistency.py    # majority vote, uncertainty-gated cascade
│   ├── experiment.py          # generation driver → records JSONL
│   ├── analysis.py            # composes the above into a per-task bundle
│   └── plots.py               # all figures
├── scripts/                   # entry points (see Quick start)
├── tests/                     # 198 unit tests, no GPU required
├── results/                   # records_{task}.jsonl, metrics_{task}.json
├── figures/                   # generated PNGs
├── notebooks/                 # generated, executed notebook
├── report.md                  # generated analysis report
└── study_guide.pdf            # generated printable summary
```

Dependencies point downward only: `config` and `logging_utils` are leaves,
`analysis` sits on top. Every module is single-responsibility and under 500
lines, with full type hints and docstrings.

## Reproducibility

All randomness derives from `SEED = 42` in `src/config.py` — dataset shuffling,
the dev/test split, sampling, and the bootstrap. Per-example sampling seeds are
offset deterministically from the base seed so samples differ across examples
while the run as a whole is reproducible.

`report.md`, the notebook and `study_guide.pdf` are all **generated** from
`results/metrics_{task}.json`, so no number in the write-up can drift from the
data it came from. The report's narrative claims are derived too — which signal
leads, whether answer-span restriction helped, whether the cascade paid off are
computed, not asserted.

## Testing

```bash
pytest              # 198 tests, ~30 s, no GPU or network needed
pytest -v           # per-test detail
```

Tests cover the pure logic end to end: correctness scoring, entropy and
clustering maths, the hand-implemented metrics (against scikit-learn where a
reference exists), signal orientation, answer-span detection against real
markdown-collision cases, the applications, and a full synthetic-record
pipeline run that plants a known signal-to-correctness relationship and checks
the pipeline recovers it.

## References

- Kuhn, Gal & Farquhar (2023). *Semantic Uncertainty: Linguistic Invariances
  for Uncertainty Estimation in Natural Language Generation.* ICLR.
- Farquhar, Kossen, Kuhn & Gal (2024). *Detecting hallucinations in large
  language models using semantic entropy.* Nature 630.
- Wang et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in
  Language Models.* ICLR.
- Guo et al. (2017). *On Calibration of Modern Neural Networks.* ICML.
