"""An LLM relevance judge, and the machinery to check it against human assessors.

"LLM-as-judge" is how RAG systems are usually evaluated now, and the judge's verdict is
usually reported as if it were ground truth. It is not: it is another measurement, made by
a model, with an error rate nobody has quoted. This module exists to quote it.

The judge is deliberately ordinary — an instruction-tuned seq2seq model asked whether a
document is relevant to a query. What is not ordinary is that its answers are scored
against the human relevance assessments that ship with the benchmark, using chance-
corrected agreement from :mod:`groundwork.eval.agreement`.

**Where this can honestly be done, and where it cannot.** Kappa needs human labels on both
sides, including negative ones. Of the four datasets here, only two carry explicit
non-relevant judgements:

===========  =============  =========================================================
Dataset      Judged pairs   Non-relevant judgements
===========  =============  =========================================================
SciFact      339            none - every judged pair is relevant
NFCorpus     12,334         none - grades are 1 and 2 only
SciDocs      29,928         25,000
TREC-COVID   66,336         41,661
===========  =============  =========================================================

On SciFact and NFCorpus a judge could only be scored against the *assumption* that
unjudged documents are non-relevant. That assumption is sound for computing nDCG, where it
is applied identically to every system being compared. It is not sound as ground truth for
an assessor, because an unjudged document is one no human ever looked at, and a judge
calling it relevant may be right. Calibrating against it would measure the judge's
agreement with a convention rather than with a person, so those two datasets are excluded
and :func:`judgeable_pairs` refuses them rather than quietly returning a pool of one class.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from groundwork.eval.agreement import AgreementResult, cohens_kappa

logger = logging.getLogger(__name__)

DEFAULT_JUDGE = "google/flan-t5-base"

JUDGE_PROMPT = (
    "Is the document relevant to the query? Answer yes or no.\n\n"
    "Query: {query}\n\nDocument: {document}\n\nAnswer:"
)

# Generated text is matched against these before anything else. A judge that answers
# something unparseable is counted, not silently coerced to a class - coercing to "no"
# would inflate agreement on a pool that is mostly non-relevant.
AFFIRMATIVE = ("yes", "true", "relevant")
NEGATIVE = ("no", "false", "not relevant", "irrelevant")


def judgeable_pairs(
    qrels: Mapping[str, Mapping[str, int]],
    run: Mapping[str, Mapping[str, float]],
    depth: int = 10,
) -> list[tuple[str, str, int]]:
    """Retrieved query-document pairs that a human actually labelled.

    Restricted to the top ``depth`` of the run, because those are the documents a RAG
    system would put in front of a model, and unjudged pairs are dropped because there is
    no human label to agree or disagree with.

    Args:
        qrels: ``{query_id: {doc_id: grade}}``.
        run: ``{query_id: {doc_id: score}}``.
        depth: How far down each ranking to go.

    Returns:
        ``[(query_id, doc_id, human_grade), ...]``.

    Raises:
        ValueError: If the qrels carry no non-relevant judgements. Agreement over a pool
            in which every human label is "relevant" is not agreement, it is a measure of
            how often the judge says yes.
    """
    grades = {grade for relevance in qrels.values() for grade in relevance.values()}
    if not any(grade <= 0 for grade in grades):
        raise ValueError(
            "these qrels contain no non-relevant judgements, so a judge cannot be "
            "calibrated against them: every human label is 'relevant', and agreement "
            "would only measure how often the judge says yes. Use a dataset whose qrels "
            "include explicit zeros, such as SciDocs or TREC-COVID."
        )

    pairs: list[tuple[str, str, int]] = []
    for query_id, scores in run.items():
        relevance = qrels.get(query_id, {})
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:depth]
        for doc_id, _ in ranked:
            if doc_id in relevance:
                pairs.append((query_id, doc_id, int(relevance[doc_id])))
    return pairs


def parse_verdict(text: str) -> int | None:
    """Map generated text to 1, 0, or None when it says neither.

    Args:
        text: The model's raw output.

    Returns:
        1 for an affirmative answer, 0 for a negative one, None if unparseable.
    """
    lowered = text.strip().lower()
    if not lowered:
        return None
    # Negatives are checked first: "not relevant" contains "relevant".
    for token in NEGATIVE:
        if lowered.startswith(token):
            return 0
    for token in AFFIRMATIVE:
        if lowered.startswith(token):
            return 1
    return None


class RelevanceJudge:
    """Ask a language model whether a document is relevant to a query.

    Args:
        model_name: A seq2seq model on the Hugging Face hub.
        max_document_chars: Truncate documents to this many characters before prompting,
            so one long document cannot push the query out of the model's context.
        batch_size: Pairs per forward pass.
        seed: Seed, recorded with results. Decoding is greedy, so this affects nothing
            unless the model is changed to sample; it is recorded anyway.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_JUDGE,
        max_document_chars: int = 1200,
        batch_size: int = 16,
        seed: int = 0,
    ) -> None:
        self.model_name = model_name
        self.max_document_chars = max_document_chars
        self.batch_size = batch_size
        self.seed = seed
        self._model = None
        self._tokenizer = None
        self._device: str | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(device).eval()
        self._device = device
        logger.info("Loaded judge %s onto %s", self.model_name, device)

    def judge(
        self,
        pairs: Sequence[tuple[str, str]],
        show_progress: bool = True,
    ) -> list[int | None]:
        """Judge each (query text, document text) pair.

        Args:
            pairs: ``[(query_text, document_text), ...]``.
            show_progress: Display a progress bar.

        Returns:
            One verdict per pair: 1, 0, or None where the model said neither.
        """
        import torch
        from tqdm.auto import tqdm

        self._load()
        torch.manual_seed(self.seed)

        verdicts: list[int | None] = []
        batches = range(0, len(pairs), self.batch_size)
        for start in tqdm(batches, desc="Judging", unit="batch", disable=not show_progress):
            chunk = pairs[start : start + self.batch_size]
            prompts = [
                JUDGE_PROMPT.format(query=query, document=document[: self.max_document_chars])
                for query, document in chunk
            ]
            encoded = self._tokenizer(
                prompts, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self._device)
            with torch.no_grad():
                outputs = self._model.generate(**encoded, max_new_tokens=4, do_sample=False)
            for text in self._tokenizer.batch_decode(outputs, skip_special_tokens=True):
                verdicts.append(parse_verdict(text))
        return verdicts

    def describe(self) -> dict[str, object]:
        """Settings, for recording alongside results."""
        return {
            "method": "llm-relevance-judge",
            "model": self.model_name,
            "prompt": JUDGE_PROMPT,
            "max_document_chars": self.max_document_chars,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "device": self._device,
        }


def calibrate(
    human_grades: Sequence[int],
    verdicts: Sequence[int | None],
) -> tuple[AgreementResult, int]:
    """Score a judge against human assessors, dropping pairs it could not answer.

    Human grades are collapsed to binary — any grade above zero is relevant — because the
    judge is asked a yes/no question. Graded agreement would need a graded prompt.

    Args:
        human_grades: Human relevance grades, aligned with ``verdicts``.
        verdicts: Judge verdicts, with None where the model said neither.

    Returns:
        ``(agreement, num_unparseable)``.

    Raises:
        ValueError: If every verdict was unparseable, or the inputs differ in length.
    """
    if len(human_grades) != len(verdicts):
        raise ValueError(f"{len(human_grades)} human labels against {len(verdicts)} verdicts")

    human: list[int] = []
    judge: list[int] = []
    unparseable = 0
    for grade, verdict in zip(human_grades, verdicts, strict=True):
        if verdict is None:
            unparseable += 1
            continue
        human.append(1 if grade > 0 else 0)
        judge.append(verdict)

    if not human:
        raise ValueError("the judge produced no parseable verdicts")
    return cohens_kappa(human, judge), unparseable
