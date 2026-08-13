"""Unit tests for token-level uncertainty signals.

The per-step reduction is exercised on CPU tensors with hand-constructed
distributions, so no GPU or model download is required.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.token_metrics import (
    normalized_entropy,
    slice_to_span,
    step_statistics,
    summarize_span,
    summarize_token_uncertainty,
)


class TestStepStatistics:
    """Per-step reduction of raw logits."""

    def test_uniform_distribution_has_maximum_entropy(self) -> None:
        vocab_size = 8
        logits = torch.zeros(1, vocab_size)
        entropy, max_probability, margin, log_probs = step_statistics(logits)

        assert entropy.item() == pytest.approx(math.log(vocab_size))
        assert max_probability.item() == pytest.approx(1 / vocab_size)
        assert margin.item() == pytest.approx(0.0, abs=1e-6)
        assert log_probs.shape == (1, vocab_size)

    def test_peaked_distribution_has_near_zero_entropy(self) -> None:
        logits = torch.tensor([[100.0, 0.0, 0.0, 0.0]])
        entropy, max_probability, margin, _ = step_statistics(logits)

        assert entropy.item() == pytest.approx(0.0, abs=1e-6)
        assert max_probability.item() == pytest.approx(1.0)
        assert margin.item() == pytest.approx(1.0)

    def test_known_two_way_split(self) -> None:
        # Softmax over [log 0.75, log 0.25] recovers those probabilities.
        logits = torch.tensor([[math.log(0.75), math.log(0.25)]])
        entropy, max_probability, margin, _ = step_statistics(logits)

        expected_entropy = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
        assert entropy.item() == pytest.approx(expected_entropy, abs=1e-5)
        assert max_probability.item() == pytest.approx(0.75, abs=1e-5)
        assert margin.item() == pytest.approx(0.5, abs=1e-5)

    def test_batch_is_handled_independently(self) -> None:
        logits = torch.tensor([[0.0, 0.0], [100.0, 0.0]])
        entropy, _, _, _ = step_statistics(logits)

        assert entropy.shape == (2,)
        assert entropy[0].item() == pytest.approx(math.log(2))
        assert entropy[1].item() == pytest.approx(0.0, abs=1e-6)

    def test_log_probs_are_normalised(self) -> None:
        logits = torch.randn(3, 16)
        _, _, _, log_probs = step_statistics(logits)
        totals = log_probs.exp().sum(dim=-1)
        assert torch.allclose(totals, torch.ones(3), atol=1e-5)

    def test_fp16_input_is_upcast(self) -> None:
        # fp16 softmax over a wide vocabulary loses tail precision; the
        # implementation must promote to fp32 before reducing.
        logits = torch.zeros(1, 4096, dtype=torch.float16)
        entropy, _, _, _ = step_statistics(logits)
        assert entropy.item() == pytest.approx(math.log(4096), rel=1e-4)

    def test_wrong_rank_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected"):
            step_statistics(torch.zeros(2, 3, 4))


class TestSummarizeTokenUncertainty:
    """Aggregation of per-step scalars."""

    def test_known_aggregates(self) -> None:
        entropy = np.array([0.0, 1.0, 2.0])
        max_probability = np.array([1.0, 0.6, 0.3])
        margin = np.array([0.9, 0.2, 0.1])
        chosen_log_probability = np.array([-0.1, -0.2, -0.3])

        summary = summarize_token_uncertainty(
            entropy, max_probability, margin, chosen_log_probability
        )

        assert summary.mean_entropy == pytest.approx(1.0)
        assert summary.max_entropy == pytest.approx(2.0)
        assert summary.mean_max_probability == pytest.approx(0.6333333)
        assert summary.min_max_probability == pytest.approx(0.3)
        assert summary.mean_margin == pytest.approx(0.4)
        assert summary.min_margin == pytest.approx(0.1)
        assert summary.sequence_log_probability == pytest.approx(-0.6)
        assert summary.log_perplexity == pytest.approx(0.2)
        assert summary.sequence_probability == pytest.approx(math.exp(-0.6))
        assert summary.num_tokens == 3

    def test_log_perplexity_is_length_normalised(self) -> None:
        # Doubling the sequence at the same per-token likelihood must leave
        # log-perplexity unchanged, unlike the raw sequence log-probability.
        short = summarize_token_uncertainty(
            np.zeros(2), np.ones(2), np.ones(2), np.full(2, -0.5)
        )
        long = summarize_token_uncertainty(
            np.zeros(8), np.ones(8), np.ones(8), np.full(8, -0.5)
        )
        assert short.log_perplexity == pytest.approx(long.log_perplexity)
        assert short.sequence_log_probability != long.sequence_log_probability

    def test_empty_sequence_is_nan(self) -> None:
        empty = np.array([])
        summary = summarize_token_uncertainty(empty, empty, empty, empty)
        assert summary.num_tokens == 0
        assert math.isnan(summary.mean_entropy)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="disagree in length"):
            summarize_token_uncertainty(
                np.zeros(3), np.zeros(2), np.zeros(3), np.zeros(3)
            )

    def test_signal_dict_prefix(self) -> None:
        summary = summarize_token_uncertainty(
            np.zeros(1), np.ones(1), np.ones(1), np.full(1, -0.1)
        )
        signals = summary.as_signal_dict(prefix="answer_")
        assert "answer_mean_entropy" in signals
        assert "mean_entropy" not in signals


class TestSliceToSpan:
    """Answer-span restriction."""

    def test_none_returns_full_array(self) -> None:
        values = np.arange(5.0)
        assert np.array_equal(slice_to_span(values, None), values)

    def test_span_selects_subrange(self) -> None:
        values = np.arange(5.0)
        assert np.array_equal(slice_to_span(values, (1, 3)), np.array([1.0, 2.0]))

    def test_span_is_clamped_to_bounds(self) -> None:
        values = np.arange(5.0)
        assert np.array_equal(slice_to_span(values, (3, 99)), np.array([3.0, 4.0]))

    def test_empty_span_returns_empty(self) -> None:
        values = np.arange(5.0)
        assert slice_to_span(values, (3, 3)).size == 0


class TestSummarizeSpan:
    """Span-restricted aggregation."""

    def test_span_isolates_the_answer_tokens(self) -> None:
        # Low entropy across a long "chain of thought", high entropy on the
        # final answer tokens: the span summary must surface the latter.
        entropy = np.array([0.1] * 8 + [3.0, 3.0])
        max_probability = np.array([0.95] * 8 + [0.3, 0.3])
        margin = np.array([0.9] * 8 + [0.05, 0.05])
        chosen = np.array([-0.05] * 8 + [-1.5, -1.5])

        full = summarize_span(entropy, max_probability, margin, chosen, None)
        answer = summarize_span(entropy, max_probability, margin, chosen, (8, 10))

        assert full.mean_entropy == pytest.approx(0.68)
        assert answer.mean_entropy == pytest.approx(3.0)
        assert answer.mean_entropy > full.mean_entropy
        assert answer.num_tokens == 2


class TestNormalizedEntropy:
    """Entropy scaled by the uniform maximum."""

    def test_uniform_entropy_normalises_to_one(self) -> None:
        assert normalized_entropy(math.log(100), 100) == pytest.approx(1.0, rel=1e-6)

    def test_zero_entropy_normalises_to_zero(self) -> None:
        assert normalized_entropy(0.0, 100) == pytest.approx(0.0)

    def test_degenerate_vocabulary_is_zero(self) -> None:
        assert normalized_entropy(1.0, 1) == 0.0
