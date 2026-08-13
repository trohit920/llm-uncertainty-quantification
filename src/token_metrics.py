"""Token-level uncertainty signals derived from per-step next-token distributions.

The design constraint that shapes this module: a vocabulary-sized array must
never be accumulated across decoding steps. Qwen2.5 has a 151,936-token
vocabulary, so retaining the full distribution for a 320-step chain of thought
would cost ~97 MB per sequence. Instead :func:`step_statistics` reduces each
step's distribution to a handful of scalars *on the GPU*, and only those
scalars ever leave the device.

The module separates two concerns:

* :func:`step_statistics` -- the per-step reduction, which needs Torch.
* :func:`summarize_token_uncertainty` -- pure NumPy aggregation of those
  per-step scalars into the signals the evaluation consumes.

Aggregation supports restricting to an *answer span*. This matters for math:
entropy averaged over a 250-token chain of thought is dominated by ordinary
prose, whereas entropy over the ``#### <n>`` tokens speaks directly to the
answer the model committed to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray

#: Guards log(0) when a probability underflows to zero in fp16.
_LOG_EPSILON: Final[float] = 1e-12

#: Number of leading candidates needed to compute a top1-vs-top2 margin.
_TOP_K_FOR_MARGIN: Final[int] = 2

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class StepStatistics:
    """Scalars retained for a single decoding step, one entry per sequence.

    Attributes:
        entropy: Shannon entropy of the next-token distribution, in nats.
        max_probability: Probability mass on the most likely token.
        margin: Gap between the top-1 and top-2 probabilities.
        chosen_log_probability: Log-probability the model assigned to the
            token that was actually emitted. Populated once the emitted token
            is known, which is one step later during incremental decoding.
    """

    entropy: FloatArray
    max_probability: FloatArray
    margin: FloatArray
    chosen_log_probability: FloatArray


@dataclass(frozen=True)
class TokenUncertainty:
    """Aggregated token-level signals for one generated sequence.

    All fields are uncertainty-oriented or probability-oriented as named;
    :func:`as_signal_dict` documents the direction of each.
    """

    mean_entropy: float
    max_entropy: float
    log_perplexity: float
    mean_max_probability: float
    min_max_probability: float
    mean_margin: float
    min_margin: float
    sequence_log_probability: float
    sequence_probability: float
    num_tokens: int

    def as_signal_dict(self, prefix: str = "") -> dict[str, float]:
        """Flatten to a ``{signal_name: value}`` mapping.

        Args:
            prefix: Optional namespace prepended to every key, used to
                distinguish full-sequence signals from answer-span signals.

        Returns:
            A mapping from signal name to scalar value.
        """
        return {
            f"{prefix}mean_entropy": self.mean_entropy,
            f"{prefix}max_entropy": self.max_entropy,
            f"{prefix}log_perplexity": self.log_perplexity,
            f"{prefix}mean_max_prob": self.mean_max_probability,
            f"{prefix}min_max_prob": self.min_max_probability,
            f"{prefix}mean_margin": self.mean_margin,
            f"{prefix}min_margin": self.min_margin,
            f"{prefix}seq_log_prob": self.sequence_log_probability,
            f"{prefix}seq_prob": self.sequence_probability,
        }


@torch.no_grad()
def step_statistics(logits: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Reduce one step's raw logits to per-sequence scalars, on device.

    Args:
        logits: Raw (unwarped) next-token logits of shape
            ``(batch, vocab_size)``.

    Returns:
        A tuple ``(entropy, max_probability, margin, log_probabilities)``.
        The first three have shape ``(batch,)``; ``log_probabilities`` retains
        shape ``(batch, vocab_size)`` and is intended to be consumed
        immediately -- by the caller's chosen-token gather -- and then
        discarded, never accumulated.

    Raises:
        ValueError: If ``logits`` is not two-dimensional.
    """
    if logits.dim() != 2:
        raise ValueError(f"Expected (batch, vocab) logits, got {tuple(logits.shape)}")

    # float32 for numerical stability: fp16 softmax over a 152k vocabulary
    # loses meaningful precision in the tail, which distorts entropy.
    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    probabilities = log_probabilities.exp()

    entropy = -(probabilities * log_probabilities).sum(dim=-1)

    top_values, _ = probabilities.topk(_TOP_K_FOR_MARGIN, dim=-1)
    max_probability = top_values[:, 0]
    margin = top_values[:, 0] - top_values[:, 1]

    return entropy, max_probability, margin, log_probabilities


def summarize_token_uncertainty(
    entropy: FloatArray,
    max_probability: FloatArray,
    margin: FloatArray,
    chosen_log_probability: FloatArray,
) -> TokenUncertainty:
    """Aggregate per-step scalars into sequence-level token signals.

    Args:
        entropy: Per-step Shannon entropy, in nats.
        max_probability: Per-step top-1 probability.
        margin: Per-step top1-minus-top2 probability gap.
        chosen_log_probability: Per-step log-probability of the emitted token.

    Returns:
        The aggregated signals. When the sequence is empty every field is
        ``nan`` except ``num_tokens``, which is 0.

    Raises:
        ValueError: If the four arrays do not share a length.
    """
    lengths = {
        len(entropy),
        len(max_probability),
        len(margin),
        len(chosen_log_probability),
    }
    if len(lengths) != 1:
        raise ValueError(f"Per-step arrays disagree in length: {sorted(lengths)}")

    num_tokens = len(entropy)
    if num_tokens == 0:
        nan = float("nan")
        return TokenUncertainty(nan, nan, nan, nan, nan, nan, nan, nan, nan, 0)

    sequence_log_probability = float(np.sum(chosen_log_probability))
    # Log-perplexity is the length-normalised negative log-likelihood, which
    # keeps long chains of thought comparable with short answers.
    log_perplexity = -sequence_log_probability / num_tokens

    return TokenUncertainty(
        mean_entropy=float(np.mean(entropy)),
        max_entropy=float(np.max(entropy)),
        log_perplexity=log_perplexity,
        mean_max_probability=float(np.mean(max_probability)),
        min_max_probability=float(np.min(max_probability)),
        mean_margin=float(np.mean(margin)),
        min_margin=float(np.min(margin)),
        sequence_log_probability=sequence_log_probability,
        sequence_probability=float(np.exp(sequence_log_probability)),
        num_tokens=num_tokens,
    )


def slice_to_span(
    values: FloatArray, span: tuple[int, int] | None
) -> FloatArray:
    """Restrict a per-step array to a half-open token span.

    Args:
        values: Per-step scalars for the full generation.
        span: ``(start, end)`` token offsets, or ``None`` for the full array.

    Returns:
        The restricted view, or the full array when ``span`` is ``None`` or
        selects nothing.
    """
    if span is None:
        return values
    start, end = span
    start = max(0, start)
    end = min(len(values), end)
    if start >= end:
        return values[:0]
    return values[start:end]


def summarize_span(
    entropy: FloatArray,
    max_probability: FloatArray,
    margin: FloatArray,
    chosen_log_probability: FloatArray,
    span: tuple[int, int] | None,
) -> TokenUncertainty:
    """Aggregate token signals restricted to an answer span.

    Args:
        entropy: Per-step Shannon entropy for the full generation.
        max_probability: Per-step top-1 probability.
        margin: Per-step top1-minus-top2 gap.
        chosen_log_probability: Per-step log-probability of emitted tokens.
        span: ``(start, end)`` token offsets of the answer, or ``None``.

    Returns:
        Aggregated signals over the span.
    """
    return summarize_token_uncertainty(
        slice_to_span(entropy, span),
        slice_to_span(max_probability, span),
        slice_to_span(margin, span),
        slice_to_span(chosen_log_probability, span),
    )


def normalized_entropy(entropy_nats: float, vocab_size: int) -> float:
    """Scale an entropy in nats to [0, 1] by the uniform-distribution maximum.

    Args:
        entropy_nats: Shannon entropy in nats.
        vocab_size: Size of the token vocabulary.

    Returns:
        Entropy as a fraction of ``log(vocab_size)``.
    """
    if vocab_size <= 1:
        return 0.0
    return entropy_nats / float(np.log(vocab_size) + _LOG_EPSILON)
