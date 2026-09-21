"""Answer a question from retrieved documents, with citations.

The generation half of roadmap item 13. It is deliberately small, and the reason is worth
stating plainly rather than hiding in a docstring nobody reads:

**This benchmark cannot measure whether the answers are any good.** BEIR ships queries,
documents and relevance judgements. It does not ship reference answers. So there is no
held-out target to score a generated answer against, and any quality number produced here
would come from a model judging another model — which experiment 16 exists to show is not
free. Rather than invent a metric, this module measures the one thing about a cited answer
that *can* be checked mechanically and matters most in practice: **whether the citations
resolve to documents that were actually retrieved.**

That is the failure mode that makes RAG dangerous rather than merely wrong. An answer with
a fabricated citation looks more trustworthy than an uncited one and is less so. It is
also the failure a benchmark with no reference answers can still catch, because the check
is against the retrieved set rather than against a ground truth.

What is *not* claimed here: that the answers are accurate, complete, well written, or that
a cited document actually supports the sentence citing it. The last of those is answerable
in principle with an entailment model, and would need calibrating against human labels
first — the same discipline experiment 16 applies to relevance judging.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_GENERATOR = "google/flan-t5-base"

ANSWER_PROMPT = (
    "Answer the question using only the numbered sources below. "
    "Cite every claim with the source number in square brackets, like [1].\n\n"
    "{sources}\n\nQuestion: {query}\n\nAnswer:"
)

CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class CitedAnswer:
    """A generated answer and what its citations point at."""

    query: str
    answer: str
    doc_ids: list[str]
    cited_positions: list[int]
    cited_doc_ids: list[str]
    invalid_citations: list[int]
    uncited: bool

    def describe(self) -> dict[str, object]:
        """A JSON-safe record."""
        return {
            "query": self.query,
            "answer": self.answer,
            "retrieved_doc_ids": self.doc_ids,
            "cited_positions": self.cited_positions,
            "cited_doc_ids": self.cited_doc_ids,
            "invalid_citations": self.invalid_citations,
            "uncited": self.uncited,
            "citation_precision": (
                None
                if not self.cited_positions
                else 1.0 - len(self.invalid_citations) / len(self.cited_positions)
            ),
        }


def format_sources(
    doc_ids: Sequence[str],
    corpus: Mapping[str, Mapping[str, str]],
    max_chars: int = 700,
) -> str:
    """Number the retrieved documents for the prompt, starting at 1.

    Numbering is one-based because that is how citations are written and read, and an
    off-by-one here would silently attribute every claim to the wrong document.
    """
    blocks = []
    for position, doc_id in enumerate(doc_ids, start=1):
        fields = corpus.get(doc_id, {})
        title = fields.get("title", "").strip()
        text = fields.get("text", "").strip()[:max_chars]
        blocks.append(f"[{position}] {title}\n{text}".strip())
    return "\n\n".join(blocks)


def parse_citations(answer: str, num_sources: int) -> tuple[list[int], list[int]]:
    """Find citation markers, split into ones that resolve and ones that do not.

    Args:
        answer: Generated text.
        num_sources: How many documents were supplied.

    Returns:
        ``(valid_positions, invalid_positions)``, both in order of first appearance and
        without duplicates. A position is valid when it is between 1 and ``num_sources``.
    """
    seen: list[int] = []
    for match in CITATION.finditer(answer):
        position = int(match.group(1))
        if position not in seen:
            seen.append(position)
    valid = [p for p in seen if 1 <= p <= num_sources]
    invalid = [p for p in seen if not (1 <= p <= num_sources)]
    return valid, invalid


def answer_from_documents(
    query: str,
    doc_ids: Sequence[str],
    corpus: Mapping[str, Mapping[str, str]],
    model_name: str = DEFAULT_GENERATOR,
    max_new_tokens: int = 160,
    seed: int = 0,
) -> CitedAnswer:
    """Generate an answer to ``query`` from the given documents.

    Args:
        query: The question.
        doc_ids: Retrieved document ids, in rank order.
        corpus: ``{doc_id: {"title": ..., "text": ...}}``.
        model_name: A seq2seq model on the Hugging Face hub.
        max_new_tokens: Cap on answer length.
        seed: Recorded; decoding is greedy so it changes nothing unless that changes.

    Returns:
        A :class:`CitedAnswer`.

    Raises:
        ValueError: If no documents are supplied. An "answer" generated from nothing is
            the model's prior with a citation format bolted on, which is exactly the
            output this module exists to make visible.
    """
    if not doc_ids:
        raise ValueError("no documents to answer from")

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
    torch.manual_seed(seed)

    prompt = ANSWER_PROMPT.format(sources=format_sources(doc_ids, corpus), query=query)
    encoded = tokenizer(prompt, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
    answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    valid, invalid = parse_citations(answer, len(doc_ids))
    return CitedAnswer(
        query=query,
        answer=answer,
        doc_ids=list(doc_ids),
        cited_positions=valid + invalid,
        cited_doc_ids=[doc_ids[p - 1] for p in valid],
        invalid_citations=invalid,
        uncited=not valid and not invalid,
    )
