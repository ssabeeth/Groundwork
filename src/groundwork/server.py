"""An MCP server exposing this project's retrieval over scientific literature.

Roadmap item 14. The point is not to ship a search box — it is to make the measured
results usable, and in particular to make the *uncertainty* usable. Every tool here
reports what it retrieved and how it retrieved it, and the one finding this project spent
four experiments on is exposed directly: ``compare_methods`` runs BM25 and a dense encoder
side by side on one query so the caller can see where they disagree.

**What the tools deliberately do not do.** There is no single "best" method exposed as a
default that hides the choice. Experiments 11 and 12 established that which method wins
varies by query, that the obvious predictor cannot tell you which in advance, and that
fusing beats choosing. So ``search`` defaults to fusion, ``compare_methods`` shows the
disagreement, and nothing here claims to route intelligently, because nothing measured
here can.

Install with the ``mcp`` extra and run::

    pip install -e ".[dense,mcp]"
    python -m groundwork.server --dataset scifact

The first call loads a dataset and builds an index, which takes seconds for SciFact and
minutes for TREC-COVID. Indexes are cached per dataset for the life of the process.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Any

from groundwork.data import REFERENCE_NDCG_10, load_beir_dataset
from groundwork.retrieval import (
    LUCENE_ENGLISH_STOPWORDS,
    MultiFieldBM25Retriever,
    Tokenizer,
    reciprocal_rank_fusion,
)

logger = logging.getLogger(__name__)

# The fusion constant. Swept on train for every dataset with a train split in this
# project, and it landed between 1 and 10 every time - far below the conventional 60.
DEFAULT_RRF_K = 5.0
DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass
class DatasetIndex:
    """One dataset, indexed lazily and kept for the life of the process."""

    name: str
    data_dir: str = "data"
    split: str = "test"
    dense_model: str = DEFAULT_DENSE_MODEL
    _dataset: Any = None
    _bm25: Any = None
    _dense: Any = None
    # None means "not loaded yet" and would otherwise be indistinguishable from "cannot
    # be loaded", so the unavailable case gets its own flag. Without it the property
    # retries the import on every call and appends its note again each time.
    _dense_unavailable: bool = False
    _notes: list[str] = field(default_factory=list)

    @property
    def dataset(self) -> Any:
        if self._dataset is None:
            logger.info("Loading %s (%s)", self.name, self.split)
            self._dataset = load_beir_dataset(self.name, split=self.split, data_dir=self.data_dir)
        return self._dataset

    @property
    def bm25(self) -> MultiFieldBM25Retriever:
        if self._bm25 is None:
            tokenizer = Tokenizer(stopwords=LUCENE_ENGLISH_STOPWORDS, stem=True)
            retriever = MultiFieldBM25Retriever(tokenizer=tokenizer)
            retriever.index(self.dataset.corpus, show_progress=False)
            self._bm25 = retriever
        return self._bm25

    @property
    def dense(self) -> Any:
        """The dense retriever, or None if the optional extra is not installed.

        Returning None rather than raising is deliberate: BM25 needs only numpy, so a
        caller without torch should still get lexical search rather than a server that
        will not start.
        """
        if self._dense_unavailable:
            return None
        if self._dense is None:
            try:
                from groundwork.retrieval import DenseRetriever
            except ImportError:  # pragma: no cover - depends on the install
                self._dense_unavailable = True
                self._notes.append("dense retrieval unavailable: install the 'dense' extra")
                return None
            retriever = DenseRetriever(model_name=self.dense_model)
            retriever.index(self.dataset.corpus, show_progress=False)
            self._dense = retriever
        return self._dense


def _documents(index: DatasetIndex, scores: dict[str, float], top_k: int) -> list[dict[str, Any]]:
    """Turn a scored run into readable results."""
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    corpus = index.dataset.corpus
    return [
        {
            "rank": rank,
            "doc_id": doc_id,
            "score": round(float(score), 4),
            "title": corpus[doc_id].get("title", ""),
            "text": corpus[doc_id].get("text", "")[:600],
        }
        for rank, (doc_id, score) in enumerate(ranked, start=1)
    ]


def search(
    index: DatasetIndex, query: str, top_k: int = 10, method: str = "hybrid"
) -> dict[str, Any]:
    """Retrieve documents for a query.

    Args:
        index: The dataset to search.
        query: Free text.
        top_k: How many documents to return.
        method: ``"bm25"``, ``"dense"`` or ``"hybrid"``.

    Returns:
        The documents, and the settings that produced them.
    """
    runs: list[dict[str, dict[str, float]]] = []
    used: list[str] = []

    if method in {"bm25", "hybrid"}:
        runs.append(index.bm25.retrieve({"q": query}, top_k=100, show_progress=False))
        used.append("bm25")
    if method in {"dense", "hybrid"}:
        dense = index.dense
        if dense is not None:
            runs.append(dense.retrieve({"q": query}, top_k=100, show_progress=False))
            used.append("dense")

    if not runs:
        raise ValueError(f"no retriever available for method {method!r}")

    if len(runs) == 1:
        scores = runs[0]["q"]
    else:
        scores = reciprocal_rank_fusion(runs, k=DEFAULT_RRF_K, top_k=100)["q"]

    return {
        "query": query,
        "dataset": index.name,
        "method": "+".join(used),
        "rrf_k": DEFAULT_RRF_K if len(runs) > 1 else None,
        "results": _documents(index, scores, top_k),
        "notes": list(index._notes),
    }


def compare_methods(index: DatasetIndex, query: str, top_k: int = 5) -> dict[str, Any]:
    """Run BM25 and the dense encoder separately and show where they disagree.

    This is the project's central question made usable. Experiments 8, 9, 11 and 12
    established that which method wins varies by query, that term rarity does not reliably
    predict which, and that no router built on it beats simply fusing the two. So rather
    than pretending to choose, this returns both rankings and their overlap.
    """
    lexical = index.bm25.retrieve({"q": query}, top_k=100, show_progress=False)["q"]
    dense = index.dense
    semantic = dense.retrieve({"q": query}, top_k=100, show_progress=False)["q"] if dense else {}

    lexical_top = [d["doc_id"] for d in _documents(index, lexical, top_k)]
    semantic_top = [d["doc_id"] for d in _documents(index, semantic, top_k)] if semantic else []
    overlap = sorted(set(lexical_top) & set(semantic_top))

    statistics = (
        index.bm25.query_statistics(query) if hasattr(index.bm25, "query_statistics") else {}
    )

    return {
        "query": query,
        "dataset": index.name,
        "bm25": _documents(index, lexical, top_k),
        "dense": _documents(index, semantic, top_k) if semantic else [],
        "overlap_at_k": len(overlap),
        "overlapping_doc_ids": overlap,
        "query_statistics": statistics,
        "interpretation": (
            "Overlap is the honest signal here. This project measured that BM25's "
            "per-query advantage correlates with term rarity on two of three datasets "
            "and not at all on the third, and that a router using that signal loses to "
            "fusing both systems. Treat a low overlap as a reason to use both, not as a "
            "reason to pick one."
        ),
        "notes": list(index._notes),
    }


def describe_benchmark(index: DatasetIndex) -> dict[str, Any]:
    """What this dataset is, and how this implementation scores on it."""
    dataset = index.dataset
    levels = sorted({level for r in dataset.qrels.values() for level in r.values()})
    return {
        "dataset": index.name,
        "split": index.split,
        "num_documents": len(dataset.corpus),
        "num_judged_queries": len(dataset.queries),
        "relevance_levels": levels,
        "published_bm25_ndcg_at_10": REFERENCE_NDCG_10.get(index.name),
        "retriever": index.bm25.describe(),
        "caveat": (
            "Scores in results/ are measured on the test split with these settings. "
            "Numbers reported by this server for ad-hoc queries are not benchmark "
            "results: there are no relevance judgements for a query you invented."
        ),
    }


def build_server(index: DatasetIndex) -> Any:
    """Construct the MCP server. Imported lazily so the module loads without the extra."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("groundwork")

    @server.tool()
    def groundwork_search(query: str, top_k: int = 10, method: str = "hybrid") -> dict:
        """Search scientific literature. method is bm25, dense or hybrid."""
        return search(index, query, top_k=top_k, method=method)

    @server.tool()
    def groundwork_compare_methods(query: str, top_k: int = 5) -> dict:
        """Run lexical and semantic retrieval side by side and report their overlap."""
        return compare_methods(index, query, top_k=top_k)

    @server.tool()
    def groundwork_describe_benchmark() -> dict:
        """Describe the loaded dataset and the retrieval settings in use."""
        return describe_benchmark(index)

    return server


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    index = DatasetIndex(
        name=args.dataset,
        data_dir=args.data_dir,
        split=args.split,
        dense_model=args.dense_model,
    )
    build_server(index).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
