"""The generation driver: turns a task into a file of per-example records.

For each example the pipeline runs three passes and records the wall-clock
cost of each, which is what makes the cost-versus-quality analysis concrete:

1. **Greedy pass** -- one deterministic decode. Yields every token-level
   signal plus the answer that the system would actually serve.
2. **Sampling pass** -- ``NUM_SAMPLES`` stochastic decodes issued as a single
   batched call. Yields the sentence-level signals and the majority-vote
   answer used by self-consistency decoding.
3. **Verbalized pass** -- one short decode asking the model to rate its own
   confidence.

Records are streamed to JSONL as they are produced, so a run that dies
halfway still leaves usable data behind.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator, Sequence

import numpy as np

from .config import (
    MAX_NEW_TOKENS,
    NUM_SAMPLES,
    RECORDS_FILENAME,
    RESULTS_DIR,
    SAMPLING_TEMPERATURE,
    SAMPLING_TOP_P,
    SEED,
    VERBALIZED_MAX_NEW_TOKENS,
)
from .correctness import (
    FINAL_ANSWER_MARKER,
    canonical_answer,
    majority_answer,
    score_generation,
)
from .datasets_loader import (
    Example,
    build_chat_messages,
    build_verbalized_confidence_messages,
    load_task,
)
from .model import GenerationRecord, UncertaintyAwareGenerator
from .semantic_clustering import EntailmentModel, cluster_unique_answers
from .sentence_metrics import summarize_sentence_uncertainty
from .signals import build_signal_row
from .token_metrics import summarize_span, summarize_token_uncertainty

logger = logging.getLogger(__name__)

#: Tasks whose generations contain a distinguishable final-answer span.
_TASKS_WITH_ANSWER_SPAN: Final[frozenset[str]] = frozenset({"gsm8k"})

#: Matches the integer in a verbalized-confidence reply.
_CONFIDENCE_RE: Final[re.Pattern[str]] = re.compile(r"\d{1,3}")

#: Upper bound of the verbalized confidence scale.
_CONFIDENCE_SCALE: Final[float] = 100.0


@dataclass(frozen=True)
class StageTimings:
    """Wall-clock seconds spent in each generation pass.

    Attributes:
        greedy_seconds: Time for the single deterministic decode.
        sampling_seconds: Time for the batched stochastic decodes.
        clustering_seconds: Time spent in NLI entailment scoring.
        verbalized_seconds: Time for the confidence follow-up decode.
    """

    greedy_seconds: float
    sampling_seconds: float
    clustering_seconds: float
    verbalized_seconds: float

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-serialisable view of the timings."""
        return {
            "greedy_seconds": self.greedy_seconds,
            "sampling_seconds": self.sampling_seconds,
            "clustering_seconds": self.clustering_seconds,
            "verbalized_seconds": self.verbalized_seconds,
        }


def parse_verbalized_confidence(text: str) -> float | None:
    """Extract a self-reported confidence and rescale it to [0, 1].

    Args:
        text: The model's reply to the confidence question.

    Returns:
        The confidence in [0, 1], or ``None`` when no integer in range was
        produced. Values above 100 are rejected rather than clipped, since
        they signal the model ignored the format instruction.
    """
    match = _CONFIDENCE_RE.search(text)
    if match is None:
        return None
    value = float(match.group())
    if value > _CONFIDENCE_SCALE:
        return None
    return value / _CONFIDENCE_SCALE


def _answer_span(
    generator: UncertaintyAwareGenerator, task: str, record: GenerationRecord
) -> tuple[int, int] | None:
    """Locate the final-answer token span, when the task has one.

    Args:
        generator: The generator, used for its tokenizer.
        task: Task name.
        record: The greedy generation record.

    Returns:
        A ``(start, end)`` span, or ``None`` when the task has no answer
        marker or the model omitted it.
    """
    if task not in _TASKS_WITH_ANSWER_SPAN:
        return None
    return generator.find_answer_span(record.token_ids, FINAL_ANSWER_MARKER)


def _sentence_signals(
    task: str,
    question: str,
    samples: Sequence[GenerationRecord],
    entailment_model: EntailmentModel,
) -> tuple[Any, list[str], float]:
    """Cluster sampled answers and summarise the sentence-level signals.

    Args:
        task: Task name, selecting the canonicalisation rule.
        question: The question, used to condition the NLI comparisons.
        samples: Sampled generation records.
        entailment_model: The NLI oracle.

    Returns:
        A ``(summary, canonical_answers, clustering_seconds)`` tuple.
    """
    canonical = [canonical_answer(task, sample.text) for sample in samples]
    log_probabilities = [sample.mean_log_probability for sample in samples]

    started = time.perf_counter()
    oracle = entailment_model.make_bidirectional_oracle(question)
    assignment = cluster_unique_answers(canonical, oracle)
    clustering_seconds = time.perf_counter() - started

    summary = summarize_sentence_uncertainty(
        answers=canonical,
        labels=assignment.labels,
        sequence_log_probabilities=log_probabilities,
    )
    return summary, canonical, clustering_seconds


def process_example(
    example: Example,
    generator: UncertaintyAwareGenerator,
    entailment_model: EntailmentModel,
    num_samples: int = NUM_SAMPLES,
    collect_verbalized: bool = True,
    seed: int = SEED,
) -> dict[str, Any]:
    """Run all generation passes for one example and assemble its record.

    Args:
        example: The example to process.
        generator: The loaded generation model.
        entailment_model: The loaded NLI model.
        num_samples: Number of stochastic samples to draw.
        collect_verbalized: Whether to run the confidence follow-up pass.
        seed: Base seed; offset per example so samples differ across items
            while the run as a whole stays reproducible.

    Returns:
        A JSON-serialisable record with generations, signals and timings.
    """
    task = example.task
    messages = build_chat_messages(task, example.question)
    max_new_tokens = MAX_NEW_TOKENS[task]
    example_seed = seed + abs(hash(example.example_id)) % 10_000

    started = time.perf_counter()
    greedy = generator.generate(messages, max_new_tokens, do_sample=False)[0]
    greedy_seconds = time.perf_counter() - started

    started = time.perf_counter()
    samples = generator.generate(
        messages,
        max_new_tokens,
        num_return_sequences=num_samples,
        do_sample=True,
        temperature=SAMPLING_TEMPERATURE,
        top_p=SAMPLING_TOP_P,
        seed=example_seed,
    )
    sampling_seconds = time.perf_counter() - started

    verbalized_confidence: float | None = None
    verbalized_seconds = 0.0
    if collect_verbalized:
        started = time.perf_counter()
        follow_up = generator.generate(
            build_verbalized_confidence_messages(
                task, example.question, greedy.text
            ),
            VERBALIZED_MAX_NEW_TOKENS,
            do_sample=False,
        )[0]
        verbalized_seconds = time.perf_counter() - started
        verbalized_confidence = parse_verbalized_confidence(follow_up.text)

    token_full = summarize_token_uncertainty(
        greedy.entropy,
        greedy.max_probability,
        greedy.margin,
        greedy.chosen_log_probability,
    )
    span = _answer_span(generator, task, greedy)
    token_span = (
        summarize_span(
            greedy.entropy,
            greedy.max_probability,
            greedy.margin,
            greedy.chosen_log_probability,
            span,
        )
        if span is not None
        else None
    )

    sentence, canonical, clustering_seconds = _sentence_signals(
        task, example.question, samples, entailment_model
    )

    greedy_correct = score_generation(task, greedy.text, example.gold_answers)
    majority = majority_answer(canonical)
    majority_correct = _score_canonical(task, majority, example)

    return {
        **example.as_dict(),
        "greedy_text": greedy.text,
        "greedy_num_tokens": greedy.num_tokens,
        "greedy_correct": greedy_correct,
        "greedy_canonical": canonical_answer(task, greedy.text),
        "answer_span": list(span) if span else None,
        "sample_texts": [sample.text for sample in samples],
        "sample_canonical": canonical,
        "majority_answer": majority,
        "majority_correct": majority_correct,
        "sample_correct": [
            _score_canonical(task, answer, example) for answer in canonical
        ],
        "signals": build_signal_row(
            token_full, token_span, sentence, verbalized_confidence
        ),
        "timings": StageTimings(
            greedy_seconds, sampling_seconds, clustering_seconds, verbalized_seconds
        ).as_dict(),
    }


def _score_canonical(task: str, answer: str, example: Example) -> bool:
    """Score a canonical answer string against the gold answers.

    Args:
        task: Task name.
        answer: Canonical answer string.
        example: The example holding the gold answers.

    Returns:
        ``True`` when the answer is correct. An empty answer is always wrong.
    """
    if not answer:
        return False
    if task == "gsm8k":
        # Canonical GSM8K answers are bare numbers; re-attach the marker so
        # the shared scorer sees the format it expects.
        return score_generation(
            task, f"{FINAL_ANSWER_MARKER} {answer}", example.gold_answers
        )
    return score_generation(task, answer, example.gold_answers)


def run_task(
    task: str,
    num_examples: int,
    generator: UncertaintyAwareGenerator,
    entailment_model: EntailmentModel,
    output_dir: Path = RESULTS_DIR,
    num_samples: int = NUM_SAMPLES,
    collect_verbalized: bool = True,
    seed: int = SEED,
) -> Path:
    """Run the full pipeline over one task and write its records.

    Args:
        task: Task name.
        num_examples: Number of examples to process.
        generator: The loaded generation model.
        entailment_model: The loaded NLI model.
        output_dir: Directory the JSONL record file is written to.
        num_samples: Number of stochastic samples per example.
        collect_verbalized: Whether to run the confidence follow-up pass.
        seed: Base seed.

    Returns:
        Path to the written JSONL file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / RECORDS_FILENAME.format(task=task)

    examples = load_task(task, num_examples, seed)
    logger.info("Running %s over %d examples -> %s", task, len(examples), output_path)

    started = time.perf_counter()
    num_correct = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(examples, start=1):
            record = process_example(
                example,
                generator,
                entailment_model,
                num_samples=num_samples,
                collect_verbalized=collect_verbalized,
                seed=seed,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

            num_correct += int(record["greedy_correct"])
            if index % 10 == 0 or index == len(examples):
                elapsed = time.perf_counter() - started
                logger.info(
                    "[%s] %d/%d  greedy acc %.3f  %.1fs elapsed  %.1fs/example",
                    task,
                    index,
                    len(examples),
                    num_correct / index,
                    elapsed,
                    elapsed / index,
                )

    logger.info(
        "Finished %s: %d records, greedy accuracy %.3f, %.1f minutes",
        task,
        len(examples),
        num_correct / max(len(examples), 1),
        (time.perf_counter() - started) / 60,
    )
    return output_path


def load_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL record file.

    Args:
        path: Path to the file.

    Returns:
        The parsed records.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No records at {path}. Run scripts/run_experiment.py first."
        )
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def records_for_task(task: str, results_dir: Path = RESULTS_DIR) -> list[dict[str, Any]]:
    """Load the records written for a task.

    Args:
        task: Task name.
        results_dir: Directory holding the record files.

    Returns:
        The parsed records.
    """
    return load_records(results_dir / RECORDS_FILENAME.format(task=task))


def signal_columns(
    records: Sequence[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Transpose records into a mapping from signal name to column.

    Args:
        records: Parsed records.

    Returns:
        A mapping from signal name to a float array over the records.
    """
    if not records:
        return {}
    names = sorted({name for record in records for name in record["signals"]})
    return {
        name: np.asarray(
            [record["signals"].get(name, float("nan")) for record in records],
            dtype=np.float64,
        )
        for name in names
    }


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream records from a JSONL file one at a time.

    Args:
        path: Path to the file.

    Yields:
        Parsed records.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
