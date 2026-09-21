"""End-to-end test on a synthetic dataset.

CI cannot download BEIR, so the full path - index, retrieve, score - is exercised on a
small corpus built here. It catches the wiring failures that unit tests miss: a run
shaped wrongly for the scorer, query ids that do not line up with qrels, a retriever
that silently returns nothing.
"""

import pytest

from groundwork.eval import evaluate_run
from groundwork.retrieval import BM25Retriever

CORPUS = {
    f"doc{i}": {"title": title, "text": text}
    for i, (title, text) in enumerate(
        [
            ("Photosynthesis in C4 plants", "Carbon fixation pathways in maize and sugarcane."),
            ("CRISPR off-target effects", "Guide RNA specificity in Cas9 genome editing."),
            ("Antarctic ice sheet mass", "Satellite gravimetry of ice loss since 1990."),
            ("Gut microbiome diversity", "Bacterial community composition and host diet."),
            ("Neutrino oscillation", "Flavour change observed in solar neutrino flux."),
            ("Maize drought tolerance", "Stomatal conductance under water stress in maize."),
            ("Cas9 delivery methods", "Lipid nanoparticle delivery of genome editing tools."),
            ("Sea level rise projections", "Contributions from ice sheets and thermal expansion."),
        ],
        start=1,
    )
}

QUERIES = {
    "q1": "carbon fixation in maize",
    "q2": "Cas9 guide RNA specificity",
    "q3": "ice sheet loss and sea level",
}

QRELS = {
    "q1": {"doc1": 1, "doc6": 1},
    "q2": {"doc2": 1, "doc7": 1},
    "q3": {"doc3": 1, "doc8": 1},
}


@pytest.fixture(scope="module")
def run():
    retriever = BM25Retriever()
    retriever.index(CORPUS, show_progress=False)
    return retriever.retrieve(QUERIES, top_k=10, show_progress=False)


def test_every_query_retrieves_something(run):
    assert set(run) == set(QUERIES)
    assert all(scores for scores in run.values())


def test_relevant_documents_rank_above_irrelevant_ones(run):
    for query_id, relevance in QRELS.items():
        ranked = sorted(run[query_id].items(), key=lambda kv: (-kv[1], kv[0]))
        top_doc = ranked[0][0]
        assert top_doc in relevance, f"{query_id} ranked {top_doc} first"


def test_metrics_are_in_range_and_beat_chance(run):
    metrics = evaluate_run(run, QRELS, k_values=(10,))
    assert 0.0 <= metrics["ndcg@10"] <= 1.0
    assert 0.0 <= metrics["recall@10"] <= 1.0
    assert metrics["num_queries"] == 3.0
    # With lexically obvious queries over eight documents, BM25 should be strong.
    assert metrics["ndcg@10"] > 0.5


def test_scores_are_unchanged_by_corpus_ordering(run):
    shuffled = dict(reversed(list(CORPUS.items())))
    retriever = BM25Retriever()
    retriever.index(shuffled, show_progress=False)
    other = retriever.retrieve(QUERIES, top_k=10, show_progress=False)
    for query_id in QUERIES:
        assert run[query_id] == pytest.approx(other[query_id])
