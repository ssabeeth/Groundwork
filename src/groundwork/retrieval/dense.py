"""Dense retrieval with a sentence-transformer bi-encoder.

The first method here that does not match tokens at all. BM25 scores a document by the
query terms it literally contains; a bi-encoder maps both into a vector space and scores
by cosine similarity, so it can connect "myocardial infarction" to "heart attack" and
can equally well connect a gene symbol to a different gene symbol that keeps similar
company. Which of those two behaviours dominates is the question this project exists to
answer, and it is why the comparison is run per query rather than as a single headline.

**Three things are recorded with every run, because each silently changes the result.**

*The model.* A bi-encoder's score is a property of its training data as much as of the
query. Several popular encoders are fine-tuned on data overlapping BEIR, and BEIR is
meant to be a zero-shot benchmark — so a model that has seen the test set is not
measuring what the leaderboard claims. The model id goes in `describe()` so the claim
can be checked rather than assumed.

*The truncation.* Encoders have a maximum sequence length, commonly 256 or 512 word
pieces. Scientific abstracts frequently exceed it, and the excess is silently discarded.
"Dense retrieval underperforms on long documents" and "half the document was never
encoded" are different findings, so this class measures the truncation rate rather than
leaving it to be assumed — and ``chunk_words`` offers the alternative: split a document
into overlapping windows, embed each, and score the document by its best-matching window.
That is the standard MaxP arrangement, and it trades more encoding for keeping the text
the model would otherwise never see.

*The pooling and normalisation.* Mean pooling with L2 normalisation, so the inner
product is cosine similarity. Stated because it is a choice, not a law.

**Embeddings are cached** to disk, keyed by model, truncation length and a digest of the
corpus. Encoding is by far the slowest thing in this repo, and a sweep that re-encoded
per cell would be unusable.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _require_sentence_transformers():  # noqa: ANN202 - third-party class, not ours to name
    """Import sentence-transformers, or explain how to install it.

    Kept lazy and behind an extra for the same reason stemming is: the core package
    depends on numpy and tqdm, and a torch install is a reason people do not run
    portfolio code.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Dense retrieval requires sentence-transformers. "
            "Install with: pip install -e '.[dense]'"
        ) from exc
    return SentenceTransformer


def corpus_digest(corpus: Mapping[str, Mapping[str, str]]) -> str:
    """Stable digest of a corpus, for keying an embedding cache.

    Hashes ids and text in iteration order. Two corpora with the same documents in a
    different order digest differently, which is deliberate: the cache stores a matrix
    whose rows are positional, so reusing it across orderings would silently misalign
    every document.

    Args:
        corpus: ``{doc_id: {"title": ..., "text": ...}}``.

    Returns:
        A hex digest.
    """
    hasher = hashlib.sha256()
    for doc_id, fields in corpus.items():
        hasher.update(doc_id.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update((fields.get("title", "") or "").encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update((fields.get("text", "") or "").encode("utf-8"))
        hasher.update(b"\x01")
    return hasher.hexdigest()[:16]


class DenseRetriever:
    """Bi-encoder retrieval over cosine similarity.

    Args:
        model_name: Sentence-transformers model id, recorded with every result.
        batch_size: Encoding batch size. Affects speed only.
        max_seq_length: Word-piece cap. None keeps the model's own default.
        chunk_words: Split documents into windows of this many whitespace-separated
            words and score by the best-matching window. None encodes each document
            once, truncating whatever exceeds the model's limit.
        chunk_overlap: Words shared between consecutive windows, so a passage straddling
            a boundary still appears whole in one of them. Must be less than
            ``chunk_words``.
        cache_dir: Where to cache embeddings. None disables caching.
        device: Torch device string, or None to let the library choose.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 64,
        max_seq_length: int | None = None,
        chunk_words: int | None = None,
        chunk_overlap: int = 0,
        cache_dir: Path | str | None = "data/embeddings",
        device: str | None = None,
    ) -> None:
        if chunk_words is not None:
            if chunk_words <= 0:
                raise ValueError("chunk_words must be positive")
            if not 0 <= chunk_overlap < chunk_words:
                raise ValueError("chunk_overlap must be in [0, chunk_words)")

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.chunk_words = chunk_words
        self.chunk_overlap = chunk_overlap
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.device = device

        self._model = None
        self.doc_ids: list[str] = []
        self.embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.truncation_rate: float | None = None
        self.effective_max_seq_length: int | None = None
        # Index of each document's first embedding row. With chunking off this is simply
        # arange(len(doc_ids)); with it on, documents own a contiguous run of rows and
        # this is what collapses chunk scores back to document scores.
        self._chunk_starts: np.ndarray = np.zeros(0, dtype=np.int64)
        self.num_chunks: int = 0

    @property
    def model(self):  # noqa: ANN201 - third-party class, not ours to name
        """The encoder, loaded on first use."""
        if self._model is None:
            SentenceTransformer = _require_sentence_transformers()
            self._model = SentenceTransformer(self.model_name, device=self.device)
            if self.max_seq_length is not None:
                self._model.max_seq_length = self.max_seq_length
            self.effective_max_seq_length = int(self._model.max_seq_length)
        return self._model

    def _encode(self, texts: list[str], show_progress: bool) -> np.ndarray:
        """Encode to L2-normalised float32, so an inner product is cosine similarity."""
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def measure_truncation(self, texts: list[str]) -> float:
        """Fraction of ``texts`` the encoder will cut off.

        Args:
            texts: Document or query texts as they will be encoded.

        Returns:
            The fraction exceeding the model's maximum sequence length, in [0, 1].
        """
        tokenizer = self.model.tokenizer
        limit = int(self.model.max_seq_length)
        over = 0
        for text in texts:
            # add_special_tokens matches what encoding actually does, so the count is
            # the one that decides whether this document loses its tail.
            length = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
            if length > limit:
                over += 1
        return over / len(texts) if texts else 0.0

    def split_into_chunks(self, text: str) -> list[str]:
        """Split ``text`` into overlapping word windows.

        Always returns at least one chunk, including for empty text, so every document
        owns a row in the embedding matrix and the document boundaries stay aligned.

        Args:
            text: Document text.

        Returns:
            The windows, in order.
        """
        if self.chunk_words is None:
            return [text]

        words = text.split()
        if len(words) <= self.chunk_words:
            return [" ".join(words)]

        stride = self.chunk_words - self.chunk_overlap
        chunks = [
            " ".join(words[start : start + self.chunk_words])
            for start in range(0, len(words), stride)
            if start == 0 or start < len(words)
        ]
        # The final stride can produce a window entirely inside its predecessor when the
        # overlap is large; those add cost and no information.
        return [chunk for chunk in chunks if chunk] or [""]

    def _document_scores(self, chunk_scores: np.ndarray) -> np.ndarray:
        """Collapse per-chunk similarities to one score per document, by maximum.

        A document matches a query if *any* part of it does, which is what MaxP encodes.
        Averaging instead would penalise a long document for the passages that happen to
        be about something else.

        Args:
            chunk_scores: Scores over embedding rows, either 1-D or (queries, chunks).

        Returns:
            Scores over documents, same leading shape.
        """
        if self.chunk_words is None:
            return chunk_scores
        axis = chunk_scores.ndim - 1
        return np.maximum.reduceat(chunk_scores, self._chunk_starts, axis=axis)

    def _cache_path(self, corpus: Mapping[str, Mapping[str, str]]) -> Path | None:
        if self.cache_dir is None:
            return None
        model_slug = self.model_name.replace("/", "__")
        length = self.max_seq_length if self.max_seq_length is not None else "default"
        chunking = (
            "nochunk"
            if self.chunk_words is None
            else f"chunk{self.chunk_words}o{self.chunk_overlap}"
        )
        return (
            self.cache_dir / f"{model_slug}__len{length}__{chunking}__{corpus_digest(corpus)}.npz"
        )

    def index(
        self,
        corpus: Mapping[str, Mapping[str, str]],
        show_progress: bool = True,
        measure_truncation: bool = True,
    ) -> None:
        """Encode the corpus, reusing a cached matrix when one matches.

        Args:
            corpus: ``{doc_id: {"title": ..., "text": ...}}``.
            show_progress: Display an encoding progress bar.
            measure_truncation: Also compute what fraction of documents get cut off.
                Costs a full tokenisation pass, which is worth it once per corpus.
        """
        if not corpus:
            raise ValueError("corpus is empty")

        self.doc_ids = list(corpus)
        texts = [
            f"{(corpus[d].get('title', '') or '')} {(corpus[d].get('text', '') or '')}".strip()
            for d in self.doc_ids
        ]

        # One contiguous run of chunks per document, so scores collapse by reduceat.
        chunks: list[str] = []
        starts: list[int] = []
        for text in texts:
            starts.append(len(chunks))
            chunks.extend(self.split_into_chunks(text))
        self._chunk_starts = np.array(starts, dtype=np.int64)
        self.num_chunks = len(chunks)

        cache_path = self._cache_path(corpus)
        if cache_path is not None and cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as cached:
                self.embeddings = cached["embeddings"]
                stored = cached["truncation_rate"]
                self.truncation_rate = float(stored[0]) if stored.size else None
                self.effective_max_seq_length = int(cached["max_seq_length"].item())
            logger.info("Loaded %d cached rows from %s", self.embeddings.shape[0], cache_path)
            if self.embeddings.shape[0] != self.num_chunks:
                raise ValueError(
                    f"cached embeddings have {self.embeddings.shape[0]} rows but this corpus "
                    f"produces {self.num_chunks}; delete {cache_path} and re-index"
                )
            return

        if measure_truncation:
            # Measured on the text as it will actually be encoded: with chunking on, the
            # question is whether a *chunk* overflows, not whether the document would.
            self.truncation_rate = self.measure_truncation(chunks)
            logger.info("Truncation rate: %.1f%% of encoded units", 100 * self.truncation_rate)

        self.embeddings = self._encode(chunks, show_progress=show_progress)
        self.effective_max_seq_length = int(self.model.max_seq_length)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                embeddings=self.embeddings,
                truncation_rate=np.array(
                    [] if self.truncation_rate is None else [self.truncation_rate]
                ),
                max_seq_length=np.array(self.effective_max_seq_length),
            )
            logger.info("Cached %d embeddings to %s", self.num_chunks, cache_path)

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Return the ``top_k`` nearest documents to ``query`` by cosine similarity.

        Unlike BM25, every document has a score, and a score near zero means "unrelated"
        rather than "shares no term". Nothing is dropped, so the ranking is always
        ``top_k`` long.

        Args:
            query: Query text.
            top_k: Documents to return.

        Returns:
            ``[(doc_id, score), ...]``, highest first, ties broken by doc id ascending.
        """
        if not self.doc_ids:
            raise RuntimeError("index() must be called before search()")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        vector = self._encode([query], show_progress=False)[0]
        scores = self._document_scores(self.embeddings @ vector)

        k = min(top_k, scores.shape[0])
        candidates = np.argpartition(-scores, k - 1)[:k]
        results = [(self.doc_ids[i], float(scores[i])) for i in candidates]
        results.sort(key=lambda item: (-item[1], item[0]))
        return results

    def retrieve(
        self,
        queries: Mapping[str, str],
        top_k: int = 100,
        show_progress: bool = True,
    ) -> dict[str, dict[str, float]]:
        """Score documents for every query, encoding the queries in one batch.

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
        vectors = self._encode([queries[q] for q in query_ids], show_progress=show_progress)
        similarities = self._document_scores(vectors @ self.embeddings.T)

        k = min(top_k, len(self.doc_ids))
        run: dict[str, dict[str, float]] = {}
        for row, query_id in enumerate(query_ids):
            scores = similarities[row]
            candidates = np.argpartition(-scores, k - 1)[:k]
            ranked = sorted(
                ((self.doc_ids[i], float(scores[i])) for i in candidates),
                key=lambda item: (-item[1], item[0]),
            )
            run[query_id] = dict(ranked)
        return run

    def describe(self) -> dict[str, object]:
        """Settings, for recording alongside results."""
        return {
            "method": "dense",
            "model": self.model_name,
            "similarity": "cosine",
            "pooling": "mean",
            "normalized": True,
            "max_seq_length": self.effective_max_seq_length,
            "truncation_rate": self.truncation_rate,
            "chunk_words": self.chunk_words,
            "chunk_overlap": self.chunk_overlap if self.chunk_words else None,
            "num_chunks": self.num_chunks,
            "batch_size": self.batch_size,
            # Recorded because the timings beside it are meaningless without it: the same
            # encode runs several times faster on "mps" than on "cpu", and the library
            # chooses silently. A run that does not say which it used cannot be compared
            # against one that does.
            "device": self._device_description(),
        }

    def _device_description(self) -> str | None:
        """The device the model loaded onto, None before it loads, "unknown" if it
        will not say.

        Deliberately total. ``describe()`` is called on the way to writing a results
        file, so a version of this that could raise would turn a missing attribute into
        a lost run.
        """
        if self._model is None:
            return None
        device = getattr(self._model, "device", None)
        return "unknown" if device is None else str(device)
