"""Expand queries or documents with generated text before retrieval.

Two techniques with one mechanism between them. Both try to close the vocabulary gap
between a short query and the documents that answer it, and both do it by adding words.

- **Query expansion** (HyDE, Query2Doc): a language model writes the document the query is
  looking for, and that text is appended to the query. The generated text does not have to
  be *true* — it has to contain the vocabulary a relevant document would contain.
- **Document expansion** (doc2query): a model predicts queries each document would answer,
  and appends them to the document before indexing. The cost moves from search time to
  index time, which is the practical reason to prefer it.

Neither is new in kind. RM3, already measured in this repository, closes the same gap by
reading the corpus instead of a model's parameters. That is why experiments 13 and 14 are
framed as comparisons against RM3 rather than against BM25 alone: a method that beats the
baseline but not the older method solving the same problem has not earned its dependencies.

**Generation is separated from retrieval on purpose.** ``generate_expansions`` produces
text and nothing else; ``expand_queries`` and ``expand_documents`` apply it. Generation is
the slow, non-deterministic, GPU-shaped part, so it is run once, committed, and re-used.
Re-scoring an expanded run must not require re-generating the expansions, or the results
stop being reproducible from the repository.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Query2Doc concatenates the query repeated ``n`` times with the generated passage, so the
# original terms are not swamped by generated ones. The repetition count is the method's
# only real parameter and is swept on a training split like any other.
DEFAULT_QUERY_WEIGHT = 5

DEFAULT_QUERY_EXPANDER = "google/flan-t5-base"
DEFAULT_DOCUMENT_EXPANDER = "doc2query/all-t5-base-msmarco"

HYDE_PROMPT = (
    "Write a short scientific abstract that would answer this question.\n\n"
    "Question: {text}\n\nAbstract:"
)


def expand_queries(
    queries: Mapping[str, str],
    expansions: Mapping[str, str],
    weight: int = DEFAULT_QUERY_WEIGHT,
) -> dict[str, str]:
    """Append each query's generated passage, repeating the query ``weight`` times.

    Args:
        queries: ``{query_id: text}``.
        expansions: ``{query_id: generated_passage}``. Queries missing from this mapping
            are passed through unchanged.
        weight: How many times to repeat the original query. ``0`` discards the original
            and retrieves on the generated passage alone, which is HyDE in its pure form;
            ``1`` is a plain concatenation.

    Returns:
        ``{query_id: expanded_text}`` over exactly the keys of ``queries``.

    Raises:
        ValueError: If ``weight`` is negative, which would silently drop the query.
    """
    if weight < 0:
        raise ValueError(f"weight must be non-negative, got {weight}")

    expanded: dict[str, str] = {}
    for query_id, text in queries.items():
        passage = expansions.get(query_id, "").strip()
        if not passage:
            expanded[query_id] = text
            continue
        repeated = " ".join([text] * weight)
        expanded[query_id] = f"{repeated} {passage}".strip()
    return expanded


def expand_documents(
    corpus: Mapping[str, Mapping[str, str]],
    expansions: Mapping[str, Sequence[str]],
    field: str = "text",
) -> dict[str, dict[str, str]]:
    """Append generated queries to each document's ``field``.

    The expansion goes into the body rather than the title, because under multi-field
    indexing the title is a short high-weight field and padding it with generated text
    would change what a title match means. Documents missing from ``expansions`` are
    passed through unchanged.

    Args:
        corpus: ``{doc_id: {"title": ..., "text": ...}}``.
        expansions: ``{doc_id: [generated_query, ...]}``.
        field: Which field to append to.

    Returns:
        A new corpus; the input is not modified.
    """
    grown: dict[str, dict[str, str]] = {}
    for doc_id, fields in corpus.items():
        document = dict(fields)
        generated = [q.strip() for q in expansions.get(doc_id, ()) if q and q.strip()]
        if generated:
            document[field] = f"{document.get(field, '')} {' '.join(generated)}".strip()
        grown[doc_id] = document
    return grown


def _load(model_name: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, str]:
    """Load a seq2seq model onto the best available device."""
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
    logger.info("Loaded %s onto %s", model_name, device)
    return model, tokenizer, device


def generate_expansions(
    texts: Mapping[str, str],
    model_name: str,
    prompt: str | None = None,
    num_return_sequences: int = 1,
    max_new_tokens: int = 128,
    batch_size: int = 16,
    seed: int = 0,
    do_sample: bool = False,
    show_progress: bool = True,
) -> dict[str, list[str]]:
    """Generate text for each input, returning ``{id: [generation, ...]}``.

    Args:
        texts: ``{id: input_text}``.
        model_name: A seq2seq model on the Hugging Face hub.
        prompt: Format string containing ``{text}``, or None to pass the text through.
            doc2query models take raw document text; instruction-tuned models need a
            prompt.
        num_return_sequences: Generations per input. Above 1 requires ``do_sample``,
            since greedy decoding would return the same string repeatedly.
        max_new_tokens: Cap on generated length.
        batch_size: Inputs per forward pass.
        seed: Seed for sampling, recorded so the run reproduces.
        do_sample: Sample instead of decoding greedily.
        show_progress: Display a progress bar.

    Returns:
        ``{id: [generation, ...]}`` in the order the ids were given.

    Raises:
        ValueError: If several generations are requested without sampling, which would
            silently produce duplicates.
    """
    if num_return_sequences > 1 and not do_sample:
        raise ValueError(
            "num_return_sequences > 1 with greedy decoding returns the same string "
            "repeatedly; pass do_sample=True"
        )

    import torch
    from tqdm.auto import tqdm

    model, tokenizer, device = _load(model_name)
    torch.manual_seed(seed)

    ids = list(texts)
    results: dict[str, list[str]] = {}
    batches = range(0, len(ids), batch_size)
    for start in tqdm(batches, desc="Generating", unit="batch", disable=not show_progress):
        chunk = ids[start : start + batch_size]
        inputs = [prompt.format(text=texts[i]) if prompt else texts[i] for i in chunk]
        encoded = tokenizer(
            inputs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_k=10 if do_sample else None,
                num_return_sequences=num_return_sequences,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for offset, doc_id in enumerate(chunk):
            window = decoded[offset * num_return_sequences : (offset + 1) * num_return_sequences]
            results[doc_id] = [text.strip() for text in window]
    return results


def describe_expansion(
    kind: str,
    model_name: str,
    num_return_sequences: int,
    seed: int,
    do_sample: bool,
    weight: int | None = None,
) -> dict[str, object]:
    """Settings, for recording alongside results."""
    return {
        "method": f"{kind}-expansion",
        "model": model_name,
        "generations_per_input": num_return_sequences,
        "sampling": do_sample,
        "seed": seed,
        "query_weight": weight,
    }
