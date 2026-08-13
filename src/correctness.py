"""Answer normalisation and correctness scoring for each task.

Two task families need different notions of "correct":

* **NQ-Open** ships a list of acceptable answer aliases. We score an exact
  match after SQuAD-style normalisation (lowercase, strip punctuation and
  articles, collapse whitespace) against *any* alias.
* **GSM8K** has a single numeric answer marked by a ``#### <n>`` suffix. We
  extract the final number from the model's chain of thought -- preferring the
  ``####`` marker when present -- and compare within a small tolerance.

Both scorers are pure functions over strings so they can be unit-tested
without loading a model.
"""

from __future__ import annotations

import re
import string
from typing import Final, Iterable, Sequence

from .config import NUMERIC_ANSWER_TOLERANCE

# --------------------------------------------------------------------------
# Shared text normalisation
# --------------------------------------------------------------------------

#: Articles removed before exact-match comparison, per the SQuAD convention.
_ARTICLES: Final[frozenset[str]] = frozenset({"a", "an", "the"})

_PUNCTUATION_TABLE: Final[dict[int, None]] = str.maketrans(
    "", "", string.punctuation
)

_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """Normalise free-form text for exact-match comparison.

    Lowercases, removes punctuation, drops leading articles and collapses
    runs of whitespace.

    Args:
        text: Raw answer string.

    Returns:
        The normalised string, possibly empty.
    """
    lowered = text.lower()
    depunctuated = lowered.translate(_PUNCTUATION_TABLE)
    tokens = [t for t in depunctuated.split() if t not in _ARTICLES]
    return _WHITESPACE_RE.sub(" ", " ".join(tokens)).strip()


# --------------------------------------------------------------------------
# NQ-Open: alias exact match
# --------------------------------------------------------------------------

#: Conversational prefixes a chat model tends to emit before the actual answer.
_ANSWER_PREFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(the\s+answer\s+is|answer\s*:|a\s*:)\s*", re.IGNORECASE
)


def extract_short_answer(generation: str) -> str:
    """Reduce a chat-model generation to its short-answer span.

    Strips a leading "The answer is"-style prefix and keeps only the first
    line, which is where an instruction-tuned model puts a short answer.

    Args:
        generation: Raw decoded model output.

    Returns:
        The candidate short answer, stripped of surrounding whitespace.
    """
    first_line = generation.strip().split("\n")[0]
    without_prefix = _ANSWER_PREFIX_RE.sub("", first_line)
    return without_prefix.strip().rstrip(".").strip()


def score_nq_open(generation: str, gold_aliases: Sequence[str]) -> bool:
    """Score an NQ-Open generation against its accepted aliases.

    Args:
        generation: Raw decoded model output.
        gold_aliases: Accepted answer strings from the dataset.

    Returns:
        ``True`` when the normalised prediction equals any normalised alias.
    """
    prediction = normalize_answer(extract_short_answer(generation))
    if not prediction:
        return False
    return any(prediction == normalize_answer(alias) for alias in gold_aliases)


# --------------------------------------------------------------------------
# GSM8K: numeric match
# --------------------------------------------------------------------------

#: Marker separating the chain of thought from the final answer in GSM8K.
FINAL_ANSWER_MARKER: Final[str] = "####"

#: Matches integers and decimals, with optional sign and thousands separators.
_NUMBER_RE: Final[re.Pattern[str]] = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _parse_number(raw: str) -> float | None:
    """Parse a number that may carry thousands separators.

    Args:
        raw: Numeric substring such as ``"1,234.5"`` or ``"-7"``.

    Returns:
        The parsed float, or ``None`` if it is not parseable.
    """
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def extract_final_number(text: str) -> float | None:
    """Extract the final numeric answer from a chain of thought.

    Prefers the substring after the last ``####`` marker; otherwise falls back
    to the last number appearing anywhere in the text.

    Args:
        text: Model generation or gold solution.

    Returns:
        The extracted number, or ``None`` when the text contains no number.
    """
    search_region = text
    if FINAL_ANSWER_MARKER in text:
        search_region = text.rsplit(FINAL_ANSWER_MARKER, 1)[1]
        matches = _NUMBER_RE.findall(search_region)
        if matches:
            return _parse_number(matches[0])
        # A dangling marker with no number after it: fall back to full text.
        search_region = text

    matches = _NUMBER_RE.findall(search_region)
    if not matches:
        return None
    return _parse_number(matches[-1])


def score_gsm8k(
    generation: str,
    gold_answer: str,
    tolerance: float = NUMERIC_ANSWER_TOLERANCE,
) -> bool:
    """Score a GSM8K generation against the gold solution.

    Args:
        generation: Raw decoded model output.
        gold_answer: Gold solution text, including its ``#### <n>`` suffix.
        tolerance: Absolute tolerance for the numeric comparison.

    Returns:
        ``True`` when both numbers parse and agree within ``tolerance``.
    """
    predicted = extract_final_number(generation)
    expected = extract_final_number(gold_answer)
    if predicted is None or expected is None:
        return False
    return abs(predicted - expected) <= tolerance


# --------------------------------------------------------------------------
# Task dispatch
# --------------------------------------------------------------------------


def score_generation(
    task: str, generation: str, gold_answers: Sequence[str]
) -> bool:
    """Score a generation using the scorer appropriate to ``task``.

    Args:
        task: Either ``"nq_open"`` or ``"gsm8k"``.
        generation: Raw decoded model output.
        gold_answers: Alias list for NQ-Open, or a single-element sequence
            holding the gold solution for GSM8K.

    Returns:
        ``True`` when the generation is judged correct.

    Raises:
        ValueError: If ``task`` is not a recognised task name.
    """
    if task == "nq_open":
        return score_nq_open(generation, gold_answers)
    if task == "gsm8k":
        return score_gsm8k(generation, gold_answers[0])
    raise ValueError(f"Unknown task: {task!r}")


def canonical_answer(task: str, generation: str) -> str:
    """Reduce a generation to the canonical form used for clustering.

    Sentence-level methods compare *answers*, not full chains of thought; for
    math this means the final number, and for QA the normalised short answer.

    Args:
        task: Either ``"nq_open"`` or ``"gsm8k"``.
        generation: Raw decoded model output.

    Returns:
        A canonical string. Empty when nothing could be extracted.

    Raises:
        ValueError: If ``task`` is not a recognised task name.
    """
    if task == "nq_open":
        return normalize_answer(extract_short_answer(generation))
    if task == "gsm8k":
        number = extract_final_number(generation)
        return "" if number is None else _format_number(number)
    raise ValueError(f"Unknown task: {task!r}")


def _format_number(value: float) -> str:
    """Render a number canonically so ``18`` and ``18.0`` compare equal.

    Args:
        value: Parsed numeric answer.

    Returns:
        An integer-formatted string when the value is integral, else a
        fixed-precision decimal string.
    """
    if abs(value - round(value)) <= NUMERIC_ANSWER_TOLERANCE:
        return str(int(round(value)))
    return f"{value:.6g}"


def majority_answer(answers: Iterable[str]) -> str:
    """Return the most frequent non-empty answer, breaking ties by first seen.

    Args:
        answers: Canonical answer strings.

    Returns:
        The modal answer, or ``""`` when every input is empty.
    """
    counts: dict[str, int] = {}
    for answer in answers:
        if answer:
            counts[answer] = counts.get(answer, 0) + 1
    if not counts:
        return ""
    return max(counts, key=lambda key: counts[key])
