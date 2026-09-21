"""Learned sparse retrieval (SPLADE).

The third kind of retriever in this repository, and the one that makes the other two
comparable. BM25 weights the terms a document contains. A dense bi-encoder replaces terms
with a single vector. SPLADE does neither: it learns a weight for every term in the BERT
vocabulary, keeps the ones that survive a ReLU, and scores with a dot product over an
inverted index exactly as BM25 does.

**Why this belongs here specifically.** Experiments 13 and 14 measured two ways of
expanding text with a language model — writing a pseudo-document for the query, and
writing queries for the document — and neither moved retrieval. SPLADE expands too: a
document acquires weight on terms it does not contain. The difference is that its
expansion is trained end to end against relevance rather than generated to read
plausibly. So it is the control those two experiments were missing, in the same way RM3
was the control for HyDE.

**The representation**, from the SPLADE paper, is the whole model::

    w = max over sequence positions of log(1 + ReLU(logits)) * attention_mask

The MLM head produces a distribution over the vocabulary at every position; the log
saturates large weights, the ReLU is what makes the vector sparse, and the max over
positions is what lets one mention anywhere in the document carry a term. Scoring is a
plain dot product. There is no length normalisation and no ``k1``/``b``: the weights are
trained, so the saturation BM25 gets from a formula, SPLADE is supposed to have learnt.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# The checkpoint the published BEIR figures for SPLADE++ use.
DEFAULT_SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"


class SpladeRetriever:
    """Retrieve with learned sparse representations.

    Args:
        model_name: A masked-language-model checkpoint with an MLM head.
        max_length: Token cap per document. Documents longer than this are truncated,
            and the rate is measured rather than assumed, as in :class:`DenseRetriever`.
        batch_size: Documents per forward pass. On unified-memory hardware this is the
            main lever on memory; see ``docs/decisions.md``.
        top_terms: Keep at most this many terms per document, largest weight first, or
            None to keep every non-zero. Kept because it is the knob every SPLADE
            deployment turns, and recorded with the run because it changes results.
        cache_dir: Where to cache encoded representations, or None to disable.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SPLADE_MODEL,
        max_length: int = 256,
        batch_size: int = 16,
        top_terms: int | None = None,
        cache_dir: str | Path | None = "data/splade",
    ) -> None:
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if top_terms is not None and top_terms <= 0:
            raise ValueError(f"top_terms must be positive or None, got {top_terms}")

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.top_terms = top_terms
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.doc_ids: list[str] = []
        self.truncation_rate: float | None = None
        self.mean_terms_per_document: float | None = None
        self._postings: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._model = None
        self._tokenizer = None
        self._device: str | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(device).eval()
        self._device = device
        logger.info("Loaded %s onto %s", self.model_name, device)

    def _encode(self, texts: list[str], show_progress: bool) -> list[tuple[np.ndarray, np.ndarray]]:
        """Encode texts to ``[(term_ids, weights), ...]``, sorted by term id.

        Sorted because the postings lists are built by concatenation and a stable term
        order makes the index reproducible.
        """
        import torch
        from tqdm.auto import tqdm

        self._load()
        encoded: list[tuple[np.ndarray, np.ndarray]] = []
        truncated = 0
        batches = range(0, len(texts), self.batch_size)
        for start in tqdm(batches, desc="Encoding", unit="batch", disable=not show_progress):
            chunk = texts[start : start + self.batch_size]
            inputs = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            lengths = self._tokenizer(chunk, truncation=False, padding=False)["input_ids"]
            truncated += sum(1 for ids in lengths if len(ids) > self.max_length)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits
                # The SPLADE representation. The mask keeps padding positions from
                # contributing a term the document does not have.
                weights = torch.log1p(torch.relu(logits))
                weights = weights * inputs["attention_mask"].unsqueeze(-1)
                pooled = weights.max(dim=1).values
            for row in pooled.cpu().numpy():
                term_ids = np.flatnonzero(row)
                values = row[term_ids]
                if self.top_terms is not None and len(term_ids) > self.top_terms:
                    keep = np.argpartition(-values, self.top_terms - 1)[: self.top_terms]
                    keep.sort()
                    term_ids, values = term_ids[keep], values[keep]
                encoded.append((term_ids.astype(np.int32), values.astype(np.float32)))
        if texts:
            self.truncation_rate = truncated / len(texts)
        return encoded

    def index(self, corpus: Mapping[str, Mapping[str, str]], show_progress: bool = True) -> None:
        """Encode the corpus and build the inverted index.

        Args:
            corpus: ``{doc_id: {"title": ..., "text": ...}}``. Title and body are joined,
                because SPLADE has no notion of fields — unlike BM25, where indexing them
                separately is the whole of experiment 10.
        """
        self.doc_ids = list(corpus)
        texts = [
            f"{corpus[d].get('title', '')} {corpus[d].get('text', '')}".strip()
            for d in self.doc_ids
        ]
        encoded = self._encode(texts, show_progress=show_progress)

        by_term: dict[int, list[tuple[int, float]]] = {}
        total_terms = 0
        for doc_index, (term_ids, values) in enumerate(encoded):
            total_terms += len(term_ids)
            for term_id, value in zip(term_ids.tolist(), values.tolist(), strict=True):
                by_term.setdefault(term_id, []).append((doc_index, value))

        self._postings = {}
        for term_id, postings in by_term.items():
            indices = np.fromiter((d for d, _ in postings), dtype=np.int32, count=len(postings))
            values = np.fromiter((v for _, v in postings), dtype=np.float32, count=len(postings))
            self._postings[term_id] = (indices, values)
        self.mean_terms_per_document = total_terms / len(encoded) if encoded else None
        logger.info(
            "Indexed %d documents, %.1f terms each on average",
            len(self.doc_ids),
            self.mean_terms_per_document or 0.0,
        )

    def score_query(self, term_ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Dot product of one sparse query against every document.

        Args:
            term_ids: Query term ids.
            weights: Their weights, aligned with ``term_ids``.

        Returns:
            One score per document, in ``doc_ids`` order.
        """
        scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        for term_id, weight in zip(term_ids.tolist(), weights.tolist(), strict=True):
            posting = self._postings.get(term_id)
            if posting is None:
                continue
            indices, values = posting
            scores[indices] += weight * values
        return scores

    def rank_scores(self, scores: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Top ``top_k`` documents, ties broken by document id.

        Zero-scoring documents are dropped rather than ranked arbitrarily, matching
        :class:`BM25Retriever`; see ``docs/decisions.md``.
        """
        nonzero = np.flatnonzero(scores > 0.0)
        if nonzero.size == 0:
            return []
        if nonzero.size > top_k:
            partition = np.argpartition(-scores[nonzero], top_k - 1)[:top_k]
            nonzero = nonzero[partition]
        return sorted(
            ((self.doc_ids[i], float(scores[i])) for i in nonzero),
            key=lambda item: (-item[1], item[0]),
        )

    def retrieve(
        self,
        queries: Mapping[str, str],
        top_k: int = 100,
        show_progress: bool = True,
    ) -> dict[str, dict[str, float]]:
        """Score documents for every query.

        Args:
            queries: ``{query_id: query_text}``.
            top_k: Documents per query.
            show_progress: Display an encoding progress bar.

        Returns:
            ``{query_id: {doc_id: score}}``.
        """
        if not self.doc_ids:
            raise RuntimeError("index() must be called before retrieve()")
        query_ids = list(queries)
        # Queries are short; the document truncation rate must not be overwritten by them.
        document_truncation = self.truncation_rate
        encoded = self._encode([queries[q] for q in query_ids], show_progress=show_progress)
        self.truncation_rate = document_truncation
        return {
            query_id: dict(self.rank_scores(self.score_query(*encoded[row]), top_k))
            for row, query_id in enumerate(query_ids)
        }

    def describe(self) -> dict[str, object]:
        """Settings, for recording alongside results."""
        return {
            "method": "splade",
            "model": self.model_name,
            "similarity": "dot-product",
            "pooling": "max",
            "activation": "log1p-relu",
            "max_length": self.max_length,
            "truncation_rate": self.truncation_rate,
            "top_terms": self.top_terms,
            "mean_terms_per_document": self.mean_terms_per_document,
            "batch_size": self.batch_size,
            "device": self._device,
        }
