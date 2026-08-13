"""Bidirectional-entailment clustering of sampled answers.

Implements the semantic-equivalence step of Kuhn et al. (2023), "Semantic
Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural
Language Generation". Two answers belong to the same meaning class when each
entails the other under an NLI model.

Two design points worth stating explicitly:

* **The question is part of the premise.** Kuhn et al. compare
  ``"{question} {answer_a}"`` against ``"{question} {answer_b}"`` rather than
  the bare answers. "Paris" and "the capital of France" are unrelated in
  isolation but equivalent once the question is supplied.
* **The entailment label index is read from the model config.** Cross-encoder
  NLI checkpoints commonly order their labels
  ``[contradiction, entailment, neutral]``, which is not the ordering people
  assume. Hard-coding index 2 -- as is common -- silently scores *neutral* as
  entailment.

The clustering algorithm itself takes an entailment callable, so it can be
unit-tested against a synthetic oracle without loading a transformer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Final, Sequence

import torch

from .config import NLI_BATCH_SIZE, NLI_MODEL_ID

logger = logging.getLogger(__name__)

#: Label name that marks entailment in a cross-encoder NLI config.
_ENTAILMENT_LABEL: Final[str] = "entailment"

#: Maximum token length for an NLI premise/hypothesis pair.
_NLI_MAX_LENGTH: Final[int] = 256

#: Signature of a batched bidirectional-entailment oracle.
EntailmentFn = Callable[[Sequence[tuple[str, str]]], list[bool]]


@dataclass(frozen=True)
class ClusterAssignment:
    """Result of clustering a list of answers by meaning.

    Attributes:
        labels: Cluster index for each input answer, parallel to the input.
        num_clusters: Total number of distinct meaning classes found.
    """

    labels: list[int]
    num_clusters: int


class EntailmentModel:
    """Batched cross-encoder NLI scorer used as the equivalence oracle."""

    def __init__(
        self,
        model_id: str = NLI_MODEL_ID,
        device: str | None = None,
        batch_size: int = NLI_BATCH_SIZE,
    ) -> None:
        """Load the NLI cross-encoder and resolve its entailment label index.

        Args:
            model_id: Hugging Face identifier of the NLI checkpoint.
            device: Torch device string; defaults to CUDA when available.
            batch_size: Number of pairs scored per forward pass.

        Raises:
            RuntimeError: If the checkpoint config exposes no entailment label.
        """
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        logger.info("Loading NLI model %s on %s", model_id, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        self.entailment_index = self._resolve_entailment_index()
        logger.info(
            "NLI labels %s -> entailment index %d",
            self.model.config.id2label,
            self.entailment_index,
        )

    def _resolve_entailment_index(self) -> int:
        """Find the logit index corresponding to the entailment class.

        Returns:
            The integer index of the entailment label.

        Raises:
            RuntimeError: If no label matches ``entailment``.
        """
        id2label: dict[int, str] = self.model.config.id2label
        for index, label in id2label.items():
            if label.lower() == _ENTAILMENT_LABEL:
                return int(index)
        raise RuntimeError(
            f"No 'entailment' label in NLI config: {id2label}. "
            "Cannot score semantic equivalence."
        )

    @torch.no_grad()
    def predict_entailment(self, pairs: Sequence[tuple[str, str]]) -> list[bool]:
        """Score premise/hypothesis pairs for entailment.

        Args:
            pairs: ``(premise, hypothesis)`` tuples.

        Returns:
            One boolean per pair: ``True`` when the arg-max label is
            entailment.
        """
        if not pairs:
            return []

        predictions: list[bool] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            encoded = self.tokenizer(
                [premise for premise, _ in batch],
                [hypothesis for _, hypothesis in batch],
                padding=True,
                truncation=True,
                max_length=_NLI_MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**encoded).logits
            predicted_index = logits.argmax(dim=-1)
            predictions.extend(
                (predicted_index == self.entailment_index).tolist()
            )

        return predictions

    def make_bidirectional_oracle(self, question: str) -> EntailmentFn:
        """Build a question-conditioned bidirectional-entailment oracle.

        Args:
            question: The prompt both answers respond to.

        Returns:
            A callable mapping ``(answer_a, answer_b)`` pairs to booleans that
            are ``True`` only when entailment holds in both directions.
        """

        def oracle(pairs: Sequence[tuple[str, str]]) -> list[bool]:
            if not pairs:
                return []
            forward = [
                (f"{question} {left}", f"{question} {right}")
                for left, right in pairs
            ]
            backward = [
                (f"{question} {right}", f"{question} {left}")
                for left, right in pairs
            ]
            # One batched call over both directions keeps GPU utilisation high.
            results = self.predict_entailment([*forward, *backward])
            half = len(pairs)
            return [
                results[index] and results[index + half] for index in range(half)
            ]

        return oracle


def cluster_by_entailment(
    answers: Sequence[str], oracle: EntailmentFn
) -> ClusterAssignment:
    """Greedily group answers into meaning classes.

    Each answer is compared against one representative per existing cluster --
    the first member, following Kuhn et al. -- and joins the first cluster it
    is mutually entailed with, otherwise seeding a new one. Comparisons for a
    given answer are issued as a single batch.

    Args:
        answers: Answer strings to cluster. Duplicates are permitted but
            callers should de-duplicate first to save NLI calls.
        oracle: Batched bidirectional-entailment predicate.

    Returns:
        The cluster assignment, parallel to ``answers``.
    """
    if not answers:
        return ClusterAssignment(labels=[], num_clusters=0)

    labels: list[int] = [0]
    representatives: list[str] = [answers[0]]

    for answer in answers[1:]:
        pairs = [(representative, answer) for representative in representatives]
        matches = oracle(pairs)

        assigned: int | None = None
        for cluster_index, is_match in enumerate(matches):
            if is_match:
                assigned = cluster_index
                break

        if assigned is None:
            assigned = len(representatives)
            representatives.append(answer)

        labels.append(assigned)

    return ClusterAssignment(labels=labels, num_clusters=len(representatives))


def cluster_unique_answers(
    answers: Sequence[str], oracle: EntailmentFn
) -> ClusterAssignment:
    """Cluster after de-duplicating, then expand back to the full list.

    Identical strings are trivially equivalent, so comparing them wastes NLI
    calls -- with 10 samples of a short factual answer, de-duplication often
    cuts the pair count by an order of magnitude.

    Args:
        answers: Answer strings, possibly containing duplicates.
        oracle: Batched bidirectional-entailment predicate.

    Returns:
        Cluster assignment parallel to the original ``answers``.
    """
    if not answers:
        return ClusterAssignment(labels=[], num_clusters=0)

    unique: list[str] = []
    index_of: dict[str, int] = {}
    for answer in answers:
        if answer not in index_of:
            index_of[answer] = len(unique)
            unique.append(answer)

    unique_assignment = cluster_by_entailment(unique, oracle)
    labels = [unique_assignment.labels[index_of[answer]] for answer in answers]
    return ClusterAssignment(
        labels=labels, num_clusters=unique_assignment.num_clusters
    )
