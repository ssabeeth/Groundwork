"""Late-interaction retrieval (ColBERT).

Every dense method measured elsewhere in this repository compresses a document into one
vector. ColBERT keeps one vector per token and scores a pair by **MaxSim**: for each query
token, the best match against any document token, summed over query tokens.

.. math::

    S(q, d) = \\sum_{i \\in q} \\max_{j \\in d} \\; \\hat{q}_i \\cdot \\hat{d}_j

Nothing is pooled, so nothing is averaged away. That is the interesting property here:
experiment 9 found that letting a bi-encoder see the whole document by chunking made
results slightly *worse*, which says the tails of these documents carry no signal a
single vector can use. Late interaction is the method most likely to use it, because a
single decisive term can win its own query token without having to survive a mean.

**Settings come from the checkpoint, not from memory.** ``colbert-ir/colbertv2.0`` ships an
``artifact.metadata`` recording dimension 128, query length 32, document length 180, cosine
similarity and punctuation masking, and those are read rather than guessed. Getting one of
them wrong produces a model that runs and retrieves badly, which is indistinguishable from
a negative result.

**Three details that are easy to miss and change the numbers.**

- The projection to 128 dimensions lives *outside* the BERT model in the published
  checkpoint. Loading with ``AutoModel`` silently drops it and leaves 768-dimensional
  unprojected vectors, so it is loaded explicitly and its absence is an error.
- Queries are padded to their full length with ``[MASK]`` rather than ignored. This is
  query augmentation, and it is a deliberate part of the method: the mask positions learn
  to stand in for terms the query did not state.
- Punctuation is masked out of *document* representations, so full stops cannot win a
  MaxSim.

**What this implementation is not.** Published ColBERT numbers come with an engineered
serving stack — centroid candidate generation, residual compression, PLAID. This scores
exhaustively against every document, which is the quality ceiling of that stack and none
of its efficiency. Timings here are not comparable with anything and none are claimed.
"""

from __future__ import annotations

import logging
import string
from collections.abc import Mapping

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_COLBERT_MODEL = "colbert-ir/colbertv2.0"

# From the checkpoint's own artifact.metadata.
DEFAULT_DIM = 128
DEFAULT_QUERY_LENGTH = 32
DEFAULT_DOC_LENGTH = 180

# ColBERT marks which side of the pair it is encoding by inserting an unused vocabulary
# token straight after [CLS]. In bert-base-uncased these are [unused0] and [unused1].
QUERY_MARKER = "[unused0]"
DOC_MARKER = "[unused1]"


class ColbertRetriever:
    """Retrieve by late interaction over token-level embeddings.

    Args:
        model_name: A ColBERT checkpoint.
        dim: Projection dimension. Must match the checkpoint.
        query_length: Query tokens, padded with ``[MASK]`` to exactly this many.
        doc_length: Token cap per document.
        batch_size: Documents per forward pass.
        mask_punctuation: Drop punctuation tokens from document representations.
        score_block: Documents scored per block, which bounds peak memory during
            retrieval without changing any score.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_COLBERT_MODEL,
        dim: int = DEFAULT_DIM,
        query_length: int = DEFAULT_QUERY_LENGTH,
        doc_length: int = DEFAULT_DOC_LENGTH,
        batch_size: int = 16,
        mask_punctuation: bool = True,
        score_block: int = 8192,
    ) -> None:
        for name, value in (
            ("dim", dim),
            ("query_length", query_length),
            ("doc_length", doc_length),
            ("batch_size", batch_size),
            ("score_block", score_block),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        self.model_name = model_name
        self.dim = dim
        self.query_length = query_length
        self.doc_length = doc_length
        self.batch_size = batch_size
        self.mask_punctuation = mask_punctuation
        self.score_block = score_block

        self.doc_ids: list[str] = []
        self.truncation_rate: float | None = None
        self.mean_tokens_per_document: float | None = None
        # All document token vectors end to end, with the start offset of each document.
        # One array rather than a list of arrays so MaxSim is a single matmul.
        self._embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._offsets: np.ndarray = np.zeros(0, dtype=np.int64)
        self._model = None
        self._tokenizer = None
        self._projection = None
        self._device: str | None = None
        self._skiplist: set[int] = set()

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name).to(device).eval()
        self._projection = self._load_projection(device)

        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        if self.mask_punctuation:
            self._skiplist = {
                tokenizer.convert_tokens_to_ids(symbol)
                for symbol in string.punctuation
                if tokenizer.convert_tokens_to_ids(symbol) != tokenizer.unk_token_id
            }
        logger.info("Loaded %s onto %s", self.model_name, device)

    def _load_projection(self, device: str):  # noqa: ANN202 - a torch tensor, not ours to name
        """Load the linear projection that ``AutoModel`` leaves behind.

        Raises:
            RuntimeError: If the checkpoint has no projection weight. Continuing without
                it would produce unprojected 768-dimensional vectors that still retrieve,
                just badly — a silent wrong answer rather than a failure.
        """
        import torch
        from huggingface_hub import hf_hub_download

        errors = []
        for filename, loader in (
            ("model.safetensors", "safetensors"),
            ("pytorch_model.bin", "torch"),
        ):
            try:
                path = hf_hub_download(self.model_name, filename)
            except Exception as exc:  # noqa: BLE001 - any download failure, try the next
                errors.append(f"{filename}: {exc}")
                continue
            if loader == "safetensors":
                from safetensors.torch import load_file

                state = load_file(path)
            else:
                state = torch.load(path, map_location="cpu", weights_only=True)
            for key in ("linear.weight", "bert.linear.weight", "colbert.linear.weight"):
                if key in state:
                    weight = state[key].to(torch.float32)
                    if weight.shape[0] != self.dim:
                        raise RuntimeError(
                            f"{self.model_name} projects to {weight.shape[0]} dimensions "
                            f"but dim={self.dim} was requested"
                        )
                    return weight.to(device)
            errors.append(f"{filename}: no linear weight among {sorted(state)[:8]}")
        raise RuntimeError(
            f"could not find ColBERT's projection weight in {self.model_name}. Without it "
            f"the model returns unprojected vectors and retrieves badly rather than "
            f"failing. Tried: {'; '.join(errors)}"
        )

    def _encode(self, texts: list[str], is_query: bool, show_progress: bool) -> list[np.ndarray]:
        """Encode texts to per-token unit vectors."""
        import torch
        from tqdm.auto import tqdm

        self._load()
        marker = QUERY_MARKER if is_query else DOC_MARKER
        limit = self.query_length if is_query else self.doc_length
        marked = [f"{marker} {text}" for text in texts]

        out: list[np.ndarray] = []
        truncated = 0
        batches = range(0, len(marked), self.batch_size)
        label = "Encoding queries" if is_query else "Encoding documents"
        for start in tqdm(batches, desc=label, unit="batch", disable=not show_progress):
            chunk = marked[start : start + self.batch_size]
            inputs = self._tokenizer(
                chunk,
                # Queries are padded to the full length on purpose: the [MASK] positions
                # are query augmentation, not filler.
                padding="max_length" if is_query else True,
                truncation=True,
                max_length=limit,
                return_tensors="pt",
            )
            lengths = self._tokenizer(chunk, truncation=False, padding=False)["input_ids"]
            truncated += sum(1 for ids in lengths if len(ids) > limit)
            ids = inputs["input_ids"]
            if is_query:
                # Padding becomes [MASK], and attends, which is what query augmentation is.
                ids = ids.masked_fill(
                    ids == self._tokenizer.pad_token_id, self._tokenizer.mask_token_id
                )
                inputs["input_ids"] = ids
                inputs["attention_mask"] = torch.ones_like(ids)
            device_inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                hidden = self._model(**device_inputs).last_hidden_state
                projected = hidden @ self._projection.T
                projected = torch.nn.functional.normalize(projected, p=2, dim=-1)

            keep = inputs["attention_mask"].bool()
            if not is_query and self._skiplist:
                punctuation = torch.zeros_like(ids, dtype=torch.bool)
                for token_id in self._skiplist:
                    punctuation |= ids == token_id
                keep &= ~punctuation
            for row in range(projected.shape[0]):
                selected = projected[row][keep[row].to(self._device)]
                out.append(selected.cpu().numpy().astype(np.float32))
        if texts and not is_query:
            self.truncation_rate = truncated / len(texts)
        return out

    def index(self, corpus: Mapping[str, Mapping[str, str]], show_progress: bool = True) -> None:
        """Encode the corpus to token vectors and lay them out for MaxSim.

        Args:
            corpus: ``{doc_id: {"title": ..., "text": ...}}``. Title and body are joined;
                ColBERT has no notion of fields.
        """
        self.doc_ids = list(corpus)
        texts = [
            f"{corpus[d].get('title', '')} {corpus[d].get('text', '')}".strip()
            for d in self.doc_ids
        ]
        per_document = self._encode(texts, is_query=False, show_progress=show_progress)

        counts = np.fromiter(
            (len(v) for v in per_document), dtype=np.int64, count=len(per_document)
        )
        if np.any(counts == 0):
            empty = int(np.count_nonzero(counts == 0))
            raise RuntimeError(
                f"{empty} documents encoded to zero tokens, which would make their MaxSim "
                "undefined. This usually means punctuation masking removed everything."
            )
        self._offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        self._embeddings = np.concatenate(per_document, axis=0)
        self.mean_tokens_per_document = float(counts.mean())
        logger.info(
            "Indexed %d documents, %d token vectors, %.1f MB",
            len(self.doc_ids),
            self._embeddings.shape[0],
            self._embeddings.nbytes / 1024**2,
        )

    def _maxsim(self, query: np.ndarray) -> np.ndarray:
        """MaxSim of one query against every document.

        Args:
            query: ``(num_query_tokens, dim)`` unit vectors.

        Returns:
            One score per document, in ``doc_ids`` order.
        """
        scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        # Blocked over documents so peak memory is bounded by score_block rather than by
        # the corpus. Scores are identical either way; only the allocation changes.
        for start in range(0, len(self.doc_ids), self.score_block):
            stop = min(start + self.score_block, len(self.doc_ids))
            first = int(self._offsets[start])
            last = (
                int(self._offsets[stop]) if stop < len(self.doc_ids) else self._embeddings.shape[0]
            )
            similarity = query @ self._embeddings[first:last].T
            boundaries = (self._offsets[start:stop] - first).astype(np.int64)
            # Per-document maximum over that document's token columns, then the sum over
            # query tokens, which is MaxSim written out.
            per_document = np.maximum.reduceat(similarity, boundaries, axis=1)
            scores[start:stop] = per_document.sum(axis=0)
        return scores

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
            show_progress: Display a progress bar.

        Returns:
            ``{query_id: {doc_id: score}}``.
        """
        if not self.doc_ids:
            raise RuntimeError("index() must be called before retrieve()")
        query_ids = list(queries)
        encoded = self._encode(
            [queries[q] for q in query_ids], is_query=True, show_progress=show_progress
        )

        k = min(top_k, len(self.doc_ids))
        run: dict[str, dict[str, float]] = {}
        for row, query_id in enumerate(query_ids):
            scores = self._maxsim(encoded[row])
            candidates = np.argpartition(-scores, k - 1)[:k]
            run[query_id] = dict(
                sorted(
                    ((self.doc_ids[i], float(scores[i])) for i in candidates),
                    key=lambda item: (-item[1], item[0]),
                )
            )
        return run

    def describe(self) -> dict[str, object]:
        """Settings, for recording alongside results."""
        return {
            "method": "colbert",
            "model": self.model_name,
            "similarity": "maxsim-cosine",
            "dim": self.dim,
            "query_length": self.query_length,
            "doc_length": self.doc_length,
            "mask_punctuation": self.mask_punctuation,
            "truncation_rate": self.truncation_rate,
            "mean_tokens_per_document": self.mean_tokens_per_document,
            "batch_size": self.batch_size,
            "exhaustive": True,
            "device": self._device,
        }
