"""Generation model wrapper that captures raw per-token distributions.

The central problem this module solves: obtaining the model's *unwarped*
next-token distribution at every decoding step, without ever accumulating a
vocabulary-sized array.

``generate(output_scores=True)`` is unsuitable on both counts. Under sampling
the returned scores have already passed through the temperature and top-p
logits warpers, so entropy computed from them measures the sampler rather than
the model; and the returned tuple retains one ``(batch, vocab)`` tensor per
step, which for ten samples of a 320-token chain of thought is roughly 1 GB.

Instead :class:`RawLogitsRecorder` attaches a forward hook to the language
model. The hook sees the logits as the model produced them -- before any
logits processor runs, so ordering inside the processor list is irrelevant --
reduces them to a few scalars on the GPU, and retains a single step's
log-probability tensor only long enough to read off the token that was
actually emitted on the following step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from .config import (
    GENERATION_DTYPE,
    GENERATION_MODEL_ID,
    SAMPLING_TEMPERATURE,
    SAMPLING_TOP_P,
    SEED,
)
from .token_metrics import step_statistics

logger = logging.getLogger(__name__)

#: Token width of an incremental decoding step, as opposed to prefill.
_DECODE_STEP_WIDTH: Final[int] = 1

FloatArray = NDArray[np.float64]


@dataclass
class GenerationRecord:
    """One generated sequence together with its per-step token statistics.

    Attributes:
        text: The decoded generation, excluding the prompt.
        token_ids: Generated token identifiers, excluding the prompt.
        entropy: Per-step Shannon entropy of the raw distribution, in nats.
        max_probability: Per-step top-1 probability.
        margin: Per-step top1-minus-top2 probability gap.
        chosen_log_probability: Per-step log-probability of the emitted token.
    """

    text: str
    token_ids: list[int]
    entropy: FloatArray
    max_probability: FloatArray
    margin: FloatArray
    chosen_log_probability: FloatArray

    @property
    def num_tokens(self) -> int:
        """Number of generated tokens."""
        return len(self.token_ids)

    @property
    def mean_log_probability(self) -> float:
        """Length-normalised sequence log-likelihood, in nats."""
        if self.chosen_log_probability.size == 0:
            return float("nan")
        return float(np.mean(self.chosen_log_probability))


class RawLogitsRecorder:
    """Forward hook reducing each step's raw logits to per-sequence scalars."""

    def __init__(self) -> None:
        """Initialise empty per-step buffers."""
        self.entropy: list[torch.Tensor] = []
        self.max_probability: list[torch.Tensor] = []
        self.margin: list[torch.Tensor] = []
        self.chosen_log_probability: list[torch.Tensor] = []
        self._pending_log_probs: torch.Tensor | None = None

    def reset(self) -> None:
        """Clear all buffers ahead of a new generation call."""
        self.entropy.clear()
        self.max_probability.clear()
        self.margin.clear()
        self.chosen_log_probability.clear()
        self._pending_log_probs = None

    def __call__(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        """Record statistics for the step whose logits are in ``output``.

        Args:
            module: The hooked module. Unused.
            args: Positional forward arguments.
            kwargs: Keyword forward arguments, read for ``input_ids``.
            output: The model output carrying a ``logits`` attribute.
        """
        del module, args  # Required by the hook signature.

        logits = getattr(output, "logits", None)
        if logits is None:
            return

        # Resolve the pending step first: at a decode step the freshly
        # supplied input token is the one sampled from the previous
        # distribution.
        input_ids = kwargs.get("input_ids")
        if self._pending_log_probs is not None and input_ids is not None:
            if input_ids.shape[-1] == _DECODE_STEP_WIDTH:
                chosen = input_ids[:, -1]
                self.chosen_log_probability.append(
                    self._pending_log_probs.gather(
                        1, chosen.unsqueeze(-1)
                    ).squeeze(-1)
                )
            self._pending_log_probs = None

        # The distribution for the next token always sits at the last position:
        # at prefill that is the end of the prompt, at decode the single step.
        entropy, max_probability, margin, log_probs = step_statistics(
            logits[:, -1, :]
        )
        self.entropy.append(entropy)
        self.max_probability.append(max_probability)
        self.margin.append(margin)
        self._pending_log_probs = log_probs

    def finalize(self, final_tokens: torch.Tensor) -> None:
        """Resolve the last pending step using the final emitted tokens.

        Args:
            final_tokens: Token identifiers emitted at the last step, shaped
                ``(batch,)``.
        """
        if self._pending_log_probs is None:
            return
        self.chosen_log_probability.append(
            self._pending_log_probs.gather(
                1, final_tokens.unsqueeze(-1).to(self._pending_log_probs.device)
            ).squeeze(-1)
        )
        self._pending_log_probs = None

    def stacked(self) -> dict[str, FloatArray]:
        """Stack per-step buffers into ``(num_steps, batch)`` NumPy arrays.

        Returns:
            A mapping with keys ``entropy``, ``max_probability``, ``margin``
            and ``chosen_log_probability``. Trailing steps for which no
            emitted token was observed are dropped so all arrays align.
        """
        num_steps = min(
            len(self.entropy),
            len(self.chosen_log_probability),
        )
        stack = lambda buffer: (  # noqa: E731 - local alias for brevity
            torch.stack(buffer[:num_steps]).float().cpu().numpy().astype(np.float64)
            if num_steps
            else np.zeros((0, 1), dtype=np.float64)
        )
        return {
            "entropy": stack(self.entropy),
            "max_probability": stack(self.max_probability),
            "margin": stack(self.margin),
            "chosen_log_probability": stack(self.chosen_log_probability),
        }


class UncertaintyAwareGenerator:
    """Loads the generation model and produces records with token statistics."""

    def __init__(
        self,
        model_id: str = GENERATION_MODEL_ID,
        dtype: str = GENERATION_DTYPE,
        device: str | None = None,
    ) -> None:
        """Load the tokenizer and model, and attach the logits hook.

        Args:
            model_id: Hugging Face identifier of the causal LM.
            dtype: Torch dtype name used for the weights.
            device: Torch device string; defaults to CUDA when available.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id

        logger.info("Loading %s (%s) on %s", model_id, dtype, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding keeps the final position aligned across a batch, which
        # is what the recorder reads.
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=getattr(torch, dtype)
        )
        self.model.to(self.device)
        self.model.eval()

        self.recorder = RawLogitsRecorder()
        self._hook_handle = self.model.register_forward_hook(
            self.recorder, with_kwargs=True
        )
        logger.info(
            "Model loaded: vocab=%d, hook attached", self.model.config.vocab_size
        )

    def close(self) -> None:
        """Detach the forward hook."""
        self._hook_handle.remove()

    def build_prompt(self, messages: Sequence[dict[str, str]]) -> str:
        """Render chat messages through the model's chat template.

        Args:
            messages: Chat-format messages.

        Returns:
            The rendered prompt string, ending at the assistant turn.
        """
        return self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def generate(
        self,
        messages: Sequence[dict[str, str]],
        max_new_tokens: int,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        temperature: float = SAMPLING_TEMPERATURE,
        top_p: float = SAMPLING_TOP_P,
        seed: int | None = None,
    ) -> list[GenerationRecord]:
        """Generate one or more continuations, capturing token statistics.

        All ``num_return_sequences`` samples are produced in a single batched
        call, which is what keeps the sampling pass affordable on 8 GB.

        Args:
            messages: Chat-format prompt messages.
            max_new_tokens: Generation cap.
            num_return_sequences: Number of sequences to return.
            do_sample: Whether to sample; ``False`` selects greedy decoding.
            temperature: Sampling temperature, ignored when greedy.
            top_p: Nucleus mass, ignored when greedy.
            seed: Optional per-call seed, for reproducible sampling.

        Returns:
            One :class:`GenerationRecord` per returned sequence.
        """
        if seed is not None:
            torch.manual_seed(seed)

        prompt = self.build_prompt(messages)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = int(encoded["input_ids"].shape[1])

        self.recorder.reset()
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": num_return_sequences,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            generation_kwargs.update(temperature=temperature, top_p=top_p)

        sequences = self.model.generate(**encoded, **generation_kwargs)
        generated = sequences[:, prompt_length:]
        if generated.shape[1] > 0:
            self.recorder.finalize(generated[:, -1])

        return self._build_records(generated)

    def _build_records(self, generated: torch.Tensor) -> list[GenerationRecord]:
        """Assemble per-sequence records, trimming padding after EOS.

        Args:
            generated: Generated token identifiers, shaped
                ``(batch, num_steps)``.

        Returns:
            One record per row of ``generated``.
        """
        stacked = self.recorder.stacked()
        num_steps = stacked["entropy"].shape[0]
        records: list[GenerationRecord] = []

        for row in range(generated.shape[0]):
            token_ids = generated[row].tolist()
            length = min(self._effective_length(token_ids), num_steps)
            kept_ids = token_ids[:length]

            records.append(
                GenerationRecord(
                    text=self.tokenizer.decode(kept_ids, skip_special_tokens=True),
                    token_ids=kept_ids,
                    entropy=stacked["entropy"][:length, row],
                    max_probability=stacked["max_probability"][:length, row],
                    margin=stacked["margin"][:length, row],
                    chosen_log_probability=stacked["chosen_log_probability"][
                        :length, row
                    ],
                )
            )

        return records

    def _effective_length(self, token_ids: Sequence[int]) -> int:
        """Length of a generation up to and including its first EOS token.

        Args:
            token_ids: Generated token identifiers for one sequence.

        Returns:
            The number of tokens that are genuine output rather than padding.
        """
        terminators = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
        terminators.discard(None)
        for index, token_id in enumerate(token_ids):
            if token_id in terminators:
                return index + 1
        return len(token_ids)

    def find_answer_span(
        self, token_ids: Sequence[int], marker: str
    ) -> tuple[int, int] | None:
        """Locate the token span following the final-answer marker.

        Args:
            token_ids: Generated token identifiers.
            marker: The textual marker preceding the answer.

        Returns:
            A ``(start, end)`` half-open span, or ``None`` when no usable
            marker occurrence exists.
        """
        return find_answer_span(
            token_ids,
            marker,
            lambda ids: self.tokenizer.decode(list(ids), skip_special_tokens=True),
        )


def find_answer_span(
    token_ids: Sequence[int],
    marker: str,
    decode: Callable[[Sequence[int]], str],
) -> tuple[int, int] | None:
    """Locate the token span holding a generation's final answer.

    Restricting token-level signals to this span matters for chain-of-thought
    tasks: entropy averaged over 250 tokens of prose says little about the
    answer, whereas entropy over the ``#### <n>`` tokens speaks to it directly.

    Two subtleties, both observed in real Qwen output:

    * ``####`` is also a markdown H4 heading, and instruction-tuned models
      routinely open sections with ``#### Step 1``. Taking the *first*
      occurrence would anchor the span to the start of the chain of thought.
    * A trailing heading with no number after it is not a final answer, so
      only occurrences followed by a digit are eligible.

    Args:
        token_ids: Generated token identifiers.
        marker: The textual marker preceding the answer.
        decode: Callable turning token identifiers into text.

    Returns:
        A ``(start, end)`` half-open span, or ``None`` when no eligible marker
        occurrence exists.
    """
    full_text = decode(token_ids)
    positions = [
        index
        for index in range(len(full_text))
        if full_text.startswith(marker, index)
    ]
    if not positions:
        return None

    eligible = [
        occurrence
        for occurrence, position in enumerate(positions)
        if any(char.isdigit() for char in full_text[position + len(marker) :])
    ]
    if not eligible:
        return None

    required_occurrences = eligible[-1] + 1
    for index in range(1, len(token_ids) + 1):
        if decode(token_ids[:index]).count(marker) >= required_occurrences:
            return (index, len(token_ids))
    return None


def seed_everything(seed: int = SEED) -> None:
    """Seed Python, NumPy and Torch RNGs.

    Args:
        seed: The seed value applied to every generator.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
