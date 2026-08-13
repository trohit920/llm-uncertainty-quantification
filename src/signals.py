"""The signal registry: every uncertainty estimator, with its orientation.

Signals arrive from two places with opposite conventions -- entropies grow
with uncertainty, probabilities shrink with it -- and every downstream metric
(AUROC, risk-coverage, selective answering) assumes a single orientation.
This module is the one place that knows which is which.

:func:`to_uncertainty` converts any signal to an uncertainty orientation
(higher means less trustworthy) by negating the confidence-oriented ones. It
is deliberately a negation rather than a reciprocal or ``1 - x``: AUROC and
risk-coverage depend only on rank order, and negation is the unique
monotone-decreasing map that leaves the resulting analysis interpretable.

Each signal also carries a **cost tier**, which is what makes the
cost-versus-quality discussion in the report quantitative rather than
hand-waved:

* ``single_pass`` -- free, computed from the greedy decode already being run.
* ``extra_call`` -- one additional short generation.
* ``multi_sample`` -- N additional full generations, plus NLI for the
  semantic variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

import numpy as np

from .sentence_metrics import SentenceUncertainty
from .token_metrics import TokenUncertainty

#: Prefix marking token signals restricted to the final-answer span.
ANSWER_SPAN_PREFIX: Final[str] = "answer_"


@dataclass(frozen=True)
class SignalSpec:
    """Metadata describing one uncertainty signal.

    Attributes:
        name: Column name used throughout results and figures.
        higher_means_more_uncertain: Orientation of the raw value.
        granularity: ``"token"`` or ``"sentence"``.
        cost_tier: ``"single_pass"``, ``"extra_call"`` or ``"multi_sample"``.
        display_name: Human-readable label for tables and plots.
    """

    name: str
    higher_means_more_uncertain: bool
    granularity: str
    cost_tier: str
    display_name: str


def _spec(
    name: str,
    uncertain: bool,
    granularity: str,
    cost: str,
    display: str,
) -> SignalSpec:
    """Construct a :class:`SignalSpec` with positional brevity."""
    return SignalSpec(name, uncertain, granularity, cost, display)


#: Token-level signals computed from the greedy decode's raw distributions.
_TOKEN_SPECS: Final[tuple[SignalSpec, ...]] = (
    _spec("mean_entropy", True, "token", "single_pass", "Mean token entropy"),
    _spec("max_entropy", True, "token", "single_pass", "Max token entropy"),
    _spec("log_perplexity", True, "token", "single_pass", "Log-perplexity"),
    _spec("mean_max_prob", False, "token", "single_pass", "Mean max-softmax"),
    _spec("min_max_prob", False, "token", "single_pass", "Min max-softmax"),
    _spec("mean_margin", False, "token", "single_pass", "Mean top1-top2 margin"),
    _spec("min_margin", False, "token", "single_pass", "Min top1-top2 margin"),
    _spec("seq_log_prob", False, "token", "single_pass", "Sequence log-prob"),
    _spec("seq_prob", False, "token", "single_pass", "Sequence probability"),
)

#: Sentence-level signals computed from the sampled generations.
_SENTENCE_SPECS: Final[tuple[SignalSpec, ...]] = (
    _spec(
        "semantic_entropy", True, "sentence", "multi_sample", "Semantic entropy"
    ),
    _spec(
        "weighted_semantic_entropy",
        True,
        "sentence",
        "multi_sample",
        "Semantic entropy (likelihood-weighted)",
    ),
    _spec("lexical_entropy", True, "sentence", "multi_sample", "Lexical entropy"),
    _spec(
        "self_consistency", False, "sentence", "multi_sample", "Self-consistency"
    ),
    _spec(
        "semantic_self_consistency",
        False,
        "sentence",
        "multi_sample",
        "Semantic self-consistency",
    ),
    _spec(
        "num_semantic_clusters",
        True,
        "sentence",
        "multi_sample",
        "Distinct meanings",
    ),
    _spec(
        "num_distinct_answers",
        True,
        "sentence",
        "multi_sample",
        "Distinct answer strings",
    ),
)

#: Signals obtained by asking the model to rate its own confidence.
_VERBALIZED_SPECS: Final[tuple[SignalSpec, ...]] = (
    _spec(
        "verbalized_confidence",
        False,
        "sentence",
        "extra_call",
        "Verbalized confidence",
    ),
)


def _answer_span_specs() -> tuple[SignalSpec, ...]:
    """Mirror the token specs onto the answer-span namespace."""
    return tuple(
        SignalSpec(
            name=f"{ANSWER_SPAN_PREFIX}{spec.name}",
            higher_means_more_uncertain=spec.higher_means_more_uncertain,
            granularity="token",
            cost_tier="single_pass",
            display_name=f"{spec.display_name} (answer span)",
        )
        for spec in _TOKEN_SPECS
    )


#: Every signal, keyed by name.
SIGNAL_REGISTRY: Final[dict[str, SignalSpec]] = {
    spec.name: spec
    for spec in (
        *_TOKEN_SPECS,
        *_answer_span_specs(),
        *_SENTENCE_SPECS,
        *_VERBALIZED_SPECS,
    )
}

#: Signal names in a stable reporting order.
SIGNAL_NAMES: Final[tuple[str, ...]] = tuple(SIGNAL_REGISTRY)


def get_spec(name: str) -> SignalSpec:
    """Look up a signal specification by name.

    Args:
        name: Signal name.

    Returns:
        The matching specification.

    Raises:
        KeyError: If the signal is not registered.
    """
    if name not in SIGNAL_REGISTRY:
        raise KeyError(
            f"Unknown signal {name!r}. Known: {sorted(SIGNAL_REGISTRY)}"
        )
    return SIGNAL_REGISTRY[name]


def to_uncertainty(name: str, values: np.ndarray | list[float]) -> np.ndarray:
    """Reorient a signal so that larger values always mean less trustworthy.

    Args:
        name: Signal name, used to look up its orientation.
        values: Raw signal values.

    Returns:
        Values negated when the signal is confidence-oriented, unchanged
        otherwise.

    Raises:
        KeyError: If the signal is not registered.
    """
    array = np.asarray(values, dtype=np.float64)
    return array if get_spec(name).higher_means_more_uncertain else -array


def signals_by_cost_tier(tier: str) -> tuple[str, ...]:
    """List signal names belonging to a cost tier.

    Args:
        tier: ``"single_pass"``, ``"extra_call"`` or ``"multi_sample"``.

    Returns:
        Matching signal names, in registry order.
    """
    return tuple(
        name for name, spec in SIGNAL_REGISTRY.items() if spec.cost_tier == tier
    )


def signals_by_granularity(granularity: str) -> tuple[str, ...]:
    """List signal names at a granularity.

    Args:
        granularity: ``"token"`` or ``"sentence"``.

    Returns:
        Matching signal names, in registry order.
    """
    return tuple(
        name
        for name, spec in SIGNAL_REGISTRY.items()
        if spec.granularity == granularity
    )


def build_signal_row(
    token_full: TokenUncertainty,
    token_answer_span: TokenUncertainty | None,
    sentence: SentenceUncertainty,
    verbalized_confidence: float | None,
) -> dict[str, float]:
    """Assemble the flat signal mapping for one example.

    Args:
        token_full: Token signals over the whole greedy generation.
        token_answer_span: Token signals restricted to the final-answer span,
            or ``None`` when the task has no distinguishable answer span.
        sentence: Sentence-level signals from the sampled generations.
        verbalized_confidence: Self-reported confidence in [0, 1], or ``None``
            when not collected.

    Returns:
        A mapping from signal name to value. Signals that could not be
        computed are present with a ``nan`` value, so every row has the same
        keys and downstream tables stay rectangular.
    """
    row: dict[str, float] = {}
    row.update(token_full.as_signal_dict())

    if token_answer_span is not None:
        row.update(token_answer_span.as_signal_dict(prefix=ANSWER_SPAN_PREFIX))
    else:
        row.update(
            {
                f"{ANSWER_SPAN_PREFIX}{spec.name}": float("nan")
                for spec in _TOKEN_SPECS
            }
        )

    row.update(sentence.as_signal_dict())
    row["verbalized_confidence"] = (
        float("nan")
        if verbalized_confidence is None
        else float(verbalized_confidence)
    )

    return row


def available_signals(rows: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """Filter the registry down to signals that are usable in a result set.

    A signal is usable when it has at least two distinct finite values: a
    constant or all-``nan`` column cannot rank anything, and feeding one to
    AUROC produces a meaningless 0.5 rather than an honest omission.

    Args:
        rows: Mapping from signal name to its column of values.

    Returns:
        Usable signal names, in registry order.
    """
    usable: list[str] = []
    for name in SIGNAL_NAMES:
        if name not in rows:
            continue
        column = np.asarray(rows[name], dtype=np.float64)
        finite = column[np.isfinite(column)]
        if finite.size >= 2 and np.unique(finite).size >= 2:
            usable.append(name)
    return tuple(usable)
