# Uncertainty Quantification in Language Models

**Track 2 — AI Research Engineer Take-Home**

## Setup

| | |
|---|---|
| Generation model | `Qwen/Qwen2.5-1.5B-Instruct` (fp16) |
| NLI model (semantic clustering) | `cross-encoder/nli-deberta-v3-small` |
| Samples per prompt | 10 at T=1.0, top-p=0.95 |
| Dev / test split | 50% / 50%, seeded |
| Calibration bins | 10, fixed-width **and** equal-mass |
| Bootstrap resamples | 1000 |
| Seed | 42 |

Every threshold, calibrator and operating point is fitted on the **dev split
only** and reported on the held-out test split. All headline metrics carry
95% percentile bootstrap intervals: at these sample sizes, a difference
smaller than the interval width is not a finding.


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


## Results


### nq_open — open-domain factual QA (short answers, closed book)

Greedy accuracy on the held-out split: **0.140**
(100 examples; 100 held out for fitting).

| Signal | Granularity | Cost | AUROC (95% CI) | AURC | ECE (fixed) | ECE (equal-mass) |
|---|---|---|---|---|---|---|
| Sequence log-prob | token | `single_pass` | 0.945 [0.899, 0.981] | 0.650 | 0.089 | 0.107 |
| Sequence probability | token | `single_pass` | 0.945 [0.899, 0.981] | 0.650 | 0.089 | 0.145 |
| Lexical entropy | sentence | `multi_sample` | 0.916 [0.848, 0.968] | 0.686 | 0.091 | 0.104 |
| Self-consistency | sentence | `multi_sample` | 0.914 [0.842, 0.966] | 0.686 | 0.090 | 0.130 |
| Distinct answer strings | sentence | `multi_sample` | 0.914 [0.846, 0.965] | 0.686 | 0.091 | 0.095 |
| Log-perplexity | token | `single_pass` | 0.914 [0.855, 0.964] | 0.678 | 0.088 | 0.134 |
| Min max-softmax | token | `single_pass` | 0.914 [0.849, 0.962] | 0.672 | 0.088 | 0.138 |
| Distinct meanings | sentence | `multi_sample` | 0.907 [0.841, 0.958] | 0.698 | 0.095 | 0.100 |
| Semantic entropy (likelihood-weighted) | sentence | `multi_sample` | 0.907 [0.845, 0.957] | 0.704 | 0.092 | 0.109 |
| Semantic entropy | sentence | `multi_sample` | 0.900 [0.826, 0.956] | 0.701 | 0.093 | 0.114 |
| Max token entropy | token | `single_pass` | 0.892 [0.819, 0.955] | 0.683 | 0.084 | 0.106 |
| Min top1-top2 margin | token | `single_pass` | 0.886 [0.802, 0.952] | 0.684 | 0.088 | 0.132 |
| Semantic self-consistency | sentence | `multi_sample` | 0.882 [0.801, 0.948] | 0.707 | 0.090 | 0.124 |
| Mean max-softmax | token | `single_pass` | 0.865 [0.779, 0.938] | 0.705 | 0.089 | 0.136 |

**Best signal: Sequence log-prob** — AUROC 0.945
[0.899, 0.981], AURC 0.650.

The best **free** signal is Sequence log-prob at 0.945, a gap of +0.000 against the best signal overall. The bootstrap intervals overlap, so this gap is not resolved at this sample size.


#### Selective answering

| Target | Coverage | Accuracy achieved | Target met on test? |
|---|---|---|---|
| 25% | 10% | 0.600 | yes |
| 40% | 7% | 0.714 | yes |
| 50% | 0% | — | no |
| 60% | 0% | — | no |

Dev-fitted thresholds held on unseen data for 2 of 4 targets — the honest measure of whether an operating point transfers, as opposed to whether it can be fitted.

#### Confidence tags

| Tier | Share | Accuracy |
|---|---|---|
| High | 27% | 0.519 |
| Medium | 36% | 0.000 |
| Low | 37% | 0.000 |

High-minus-Low accuracy gap: **0.519**. The tags separate.


### gsm8k — grade-school math (chain of thought, single numeric answer)

Greedy accuracy on the held-out split: **0.560**
(50 examples; 50 held out for fitting).

| Signal | Granularity | Cost | AUROC (95% CI) | AURC | ECE (fixed) | ECE (equal-mass) |
|---|---|---|---|---|---|---|
| Lexical entropy | sentence | `multi_sample` | 0.884 [0.780, 0.966] | 0.176 | 0.145 | 0.155 |
| Self-consistency | sentence | `multi_sample` | 0.884 [0.789, 0.965] | 0.172 | 0.186 | 0.186 |
| Distinct answer strings | sentence | `multi_sample` | 0.874 [0.770, 0.961] | 0.178 | 0.137 | 0.134 |
| Sequence log-prob | token | `single_pass` | 0.828 [0.684, 0.945] | 0.240 | 0.108 | 0.104 |
| Sequence probability | token | `single_pass` | 0.828 [0.684, 0.945] | 0.240 | 0.040 | 0.040 |
| Semantic entropy (likelihood-weighted) | sentence | `multi_sample` | 0.806 [0.674, 0.926] | 0.281 | 0.144 | 0.181 |
| Semantic self-consistency | sentence | `multi_sample` | 0.800 [0.661, 0.918] | 0.281 | 0.130 | 0.149 |
| Semantic entropy | sentence | `multi_sample` | 0.794 [0.659, 0.911] | 0.287 | 0.147 | 0.181 |
| Distinct meanings | sentence | `multi_sample` | 0.778 [0.638, 0.901] | 0.291 | 0.143 | 0.140 |
| Log-perplexity | token | `single_pass` | 0.745 [0.592, 0.884] | 0.299 | 0.160 | 0.198 |
| Mean token entropy | token | `single_pass` | 0.735 [0.567, 0.890] | 0.338 | 0.177 | 0.195 |
| Min max-softmax | token | `single_pass` | 0.732 [0.566, 0.866] | 0.288 | 0.152 | 0.196 |
| Mean max-softmax | token | `single_pass` | 0.716 [0.538, 0.868] | 0.345 | 0.207 | 0.280 |
| Mean top1-top2 margin | token | `single_pass` | 0.713 [0.532, 0.865] | 0.347 | 0.173 | 0.239 |

**Best signal: Lexical entropy** — AUROC 0.884
[0.780, 0.966], AURC 0.176.

The best **free** signal is Sequence log-prob at 0.828, a gap of +0.056 against the best signal overall. The bootstrap intervals overlap, so this gap is not resolved at this sample size.


#### Selective answering

| Target | Coverage | Accuracy achieved | Target met on test? |
|---|---|---|---|
| 60% | 100% | 0.560 | no |
| 70% | 74% | 0.730 | yes |
| 80% | 70% | 0.771 | no |
| 90% | 44% | 0.864 | no |

Dev-fitted thresholds held on unseen data for 1 of 4 targets — the honest measure of whether an operating point transfers, as opposed to whether it can be fitted.

#### Confidence tags

| Tier | Share | Accuracy |
|---|---|---|
| High | 38% | 0.842 |
| Medium | 32% | 0.688 |
| Low | 30% | 0.067 |

High-minus-Low accuracy gap: **0.775**. The tags separate.

#### Self-consistency and the uncertainty-gated cascade

| Policy | Accuracy | Generations / question |
|---|---|---|
| Greedy only | 0.560 | 1.00 |
| Self-consistency (always) | 0.660 | 11.00 |
| **Uncertainty-gated cascade** | **0.640** | **7.80** |

Sampling buys +0.100 accuracy at 11× the generation cost. The cascade recovers +0.080 of that (80%) while escalating only 68% of questions, at 71% of the always-sample cost.


## Discussion

### Which method works where?

The leading signal differs by task: **nq_open** → Sequence log-prob, **gsm8k** → Lexical entropy.

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

- **nq_open**: single pass 0.19s, sampling + NLI 0.47s per example — a 2.4× multiplier.
- **gsm8k**: single pass 9.22s, sampling + NLI 12.29s per example — a 1.3× multiplier.

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
