"""Dataset loading and prompt construction for the two task types.

Both tasks are normalised onto a single :class:`Example` schema so every
downstream stage -- generation, scoring, evaluation -- is task-agnostic and
dispatches only where the task genuinely differs.

* **NQ-Open** (``google-research-datasets/nq_open``, validation split) is
  streamed, so nothing beyond the sampled examples is written to disk.
* **GSM8K** (``openai/gsm8k``, main config, test split) is streamed likewise.

Sampling is a seeded shuffle followed by a take, which is deterministic for a
given seed and count.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Final, Iterator, Sequence

from .config import (
    GSM8K_CONFIG,
    GSM8K_DATASET_ID,
    GSM8K_SPLIT,
    NQ_OPEN_DATASET_ID,
    NQ_OPEN_SPLIT,
    SEED,
)

logger = logging.getLogger(__name__)

#: Shuffle buffer for streamed datasets. Large enough to mix, small enough to
#: keep memory flat.
_SHUFFLE_BUFFER_SIZE: Final[int] = 2_000

#: Instruction steering NQ-Open toward a short, extractable answer.
NQ_SYSTEM_PROMPT: Final[str] = (
    "You are a helpful assistant answering trivia questions. "
    "Reply with the short answer only -- a name, date, or phrase. "
    "Do not explain, and do not write a full sentence."
)

#: Instruction steering GSM8K toward a parseable final answer.
GSM8K_SYSTEM_PROMPT: Final[str] = (
    "You are a careful mathematician. Solve the problem step by step, "
    "then state the final numeric answer on its own last line in the exact "
    "format '#### <number>'."
)

#: Follow-up question used for the verbalized-confidence baseline.
VERBALIZED_CONFIDENCE_PROMPT: Final[str] = (
    "How confident are you that the answer above is correct? "
    "Reply with a single integer from 0 to 100 and nothing else."
)

#: Per-task system prompts.
SYSTEM_PROMPTS: Final[dict[str, str]] = {
    "nq_open": NQ_SYSTEM_PROMPT,
    "gsm8k": GSM8K_SYSTEM_PROMPT,
}


@dataclass(frozen=True)
class Example:
    """One evaluation item, normalised across tasks.

    Attributes:
        example_id: Stable identifier, unique within a task.
        task: Either ``"nq_open"`` or ``"gsm8k"``.
        question: The question text presented to the model.
        gold_answers: Accepted answer aliases for NQ-Open; a single-element
            tuple holding the full gold solution for GSM8K.
        metadata: Any extra task-specific fields, retained for the report.
    """

    example_id: str
    task: str
    question: str
    gold_answers: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the example."""
        return {
            "example_id": self.example_id,
            "task": self.task,
            "question": self.question,
            "gold_answers": list(self.gold_answers),
            "metadata": self.metadata,
        }


def build_chat_messages(task: str, question: str) -> list[dict[str, str]]:
    """Build the chat-template message list for a question.

    Args:
        task: Either ``"nq_open"`` or ``"gsm8k"``.
        question: The question text.

    Returns:
        Messages in Hugging Face chat format.

    Raises:
        ValueError: If ``task`` is not a recognised task name.
    """
    if task not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown task: {task!r}")
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[task]},
        {"role": "user", "content": question},
    ]


def build_verbalized_confidence_messages(
    task: str, question: str, answer: str
) -> list[dict[str, str]]:
    """Build the follow-up messages that elicit a verbalized confidence.

    Args:
        task: Either ``"nq_open"`` or ``"gsm8k"``.
        question: The original question text.
        answer: The model's own previous answer.

    Returns:
        Messages continuing the conversation with a confidence request.
    """
    messages = build_chat_messages(task, question)
    messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": VERBALIZED_CONFIDENCE_PROMPT})
    return messages


def _stream_dataset(
    dataset_id: str,
    split: str,
    num_examples: int,
    seed: int,
    config_name: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream, seed-shuffle and take rows from a Hugging Face dataset.

    Args:
        dataset_id: Hub identifier.
        split: Split name.
        num_examples: Number of rows to take after shuffling.
        seed: Shuffle seed.
        config_name: Optional dataset configuration name.

    Yields:
        Raw dataset rows as dictionaries.
    """
    from datasets import load_dataset

    logger.info(
        "Streaming %s (config=%s, split=%s), taking %d rows",
        dataset_id,
        config_name,
        split,
        num_examples,
    )
    dataset = load_dataset(
        dataset_id, config_name, split=split, streaming=True
    )
    shuffled = dataset.shuffle(seed=seed, buffer_size=_SHUFFLE_BUFFER_SIZE)
    yield from shuffled.take(num_examples)


def load_nq_open(num_examples: int, seed: int = SEED) -> list[Example]:
    """Load NQ-Open validation examples.

    Args:
        num_examples: Number of examples to sample.
        seed: Shuffle seed.

    Returns:
        Normalised examples. Rows whose alias list is empty are skipped,
        since they cannot be scored.
    """
    examples: list[Example] = []
    for index, row in enumerate(
        _stream_dataset(NQ_OPEN_DATASET_ID, NQ_OPEN_SPLIT, num_examples, seed)
    ):
        aliases = tuple(row.get("answer") or ())
        if not aliases:
            logger.warning("Skipping NQ-Open row %d with no gold answers", index)
            continue
        examples.append(
            Example(
                example_id=f"nq_open-{index:05d}",
                task="nq_open",
                question=str(row["question"]).strip(),
                gold_answers=aliases,
                metadata={"num_aliases": len(aliases)},
            )
        )
    logger.info("Loaded %d NQ-Open examples", len(examples))
    return examples


def load_gsm8k(num_examples: int, seed: int = SEED) -> list[Example]:
    """Load GSM8K test examples.

    Args:
        num_examples: Number of examples to sample.
        seed: Shuffle seed.

    Returns:
        Normalised examples, each carrying the full gold solution -- including
        its ``#### <n>`` suffix -- as the single gold answer.
    """
    examples: list[Example] = []
    for index, row in enumerate(
        _stream_dataset(
            GSM8K_DATASET_ID,
            GSM8K_SPLIT,
            num_examples,
            seed,
            config_name=GSM8K_CONFIG,
        )
    ):
        solution = str(row["answer"])
        examples.append(
            Example(
                example_id=f"gsm8k-{index:05d}",
                task="gsm8k",
                question=str(row["question"]).strip(),
                gold_answers=(solution,),
                metadata={"solution_steps": solution.count("\n") + 1},
            )
        )
    logger.info("Loaded %d GSM8K examples", len(examples))
    return examples


#: Dispatch table from task name to loader.
_LOADERS: Final[dict[str, Any]] = {
    "nq_open": load_nq_open,
    "gsm8k": load_gsm8k,
}


def load_task(task: str, num_examples: int, seed: int = SEED) -> list[Example]:
    """Load examples for a task by name.

    Args:
        task: Either ``"nq_open"`` or ``"gsm8k"``.
        num_examples: Number of examples to sample.
        seed: Shuffle seed.

    Returns:
        Normalised examples.

    Raises:
        ValueError: If ``task`` is not a recognised task name.
    """
    if task not in _LOADERS:
        raise ValueError(f"Unknown task: {task!r}. Known: {sorted(_LOADERS)}")
    return _LOADERS[task](num_examples, seed)


def split_dev_test(
    examples: Sequence[Example], dev_fraction: float, seed: int = SEED
) -> tuple[list[Example], list[Example]]:
    """Partition examples into a threshold-fitting dev set and a test set.

    Every threshold, calibrator and operating point in this project is fitted
    on the dev split only; the test split is touched once, at reporting time.

    Args:
        examples: Examples to split.
        dev_fraction: Fraction assigned to dev, in (0, 1).
        seed: Seed for the permutation.

    Returns:
        A ``(dev, test)`` pair.

    Raises:
        ValueError: If ``dev_fraction`` is not strictly between 0 and 1.
    """
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError(f"dev_fraction must be in (0, 1), got {dev_fraction}")

    import numpy as np

    indices = np.random.default_rng(seed).permutation(len(examples))
    split_point = int(round(len(examples) * dev_fraction))
    dev_indices = set(indices[:split_point].tolist())

    dev = [ex for i, ex in enumerate(examples) if i in dev_indices]
    test = [ex for i, ex in enumerate(examples) if i not in dev_indices]
    return dev, test
