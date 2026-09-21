"""Cross-encoder reranking tests, offline with a stub model.

As with the dense tests, what is worth testing here is the reordering contract, not
whether somebody else's transformer works. The stub scores a pair by a rule stated in
the test, so expected orderings are derived rather than observed.
"""

import pytest

from groundwork.retrieval.rerank import CrossEncoderReranker

CORPUS = {
    "d1": {"title": "alpha", "text": "one"},
    "d2": {"title": "beta", "text": "two"},
    "d3": {"title": "gamma", "text": "three"},
    "d4": {"title": "delta", "text": "four"},
}
QUERIES = {"q1": "beta"}


class StubCrossEncoder:
    """Scores 1.0 when the query string appears in the document, else -1.0."""

    def __init__(self):
        self.seen = []

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        self.seen = list(pairs)
        return [1.0 if query in document else -1.0 for query, document in pairs]


def build(depth=100):
    reranker = CrossEncoderReranker(depth=depth)
    reranker._model = StubCrossEncoder()
    return reranker


class TestReordering:
    def test_a_relevant_document_is_promoted_from_the_bottom(self):
        # First stage ranks d2 last; the cross-encoder is the only thing that likes it.
        run = {"q1": {"d1": 9.0, "d3": 8.0, "d4": 7.0, "d2": 1.0}}
        reranked = build().rerank(run, QUERIES, CORPUS, show_progress=False)
        assert list(reranked["q1"])[0] == "d2"

    def test_the_same_documents_come_back(self):
        run = {"q1": {"d1": 9.0, "d2": 8.0, "d3": 7.0, "d4": 6.0}}
        reranked = build().rerank(run, QUERIES, CORPUS, show_progress=False)
        assert set(reranked["q1"]) == {"d1", "d2", "d3", "d4"}

    def test_ties_are_broken_by_document_id(self):
        # Every document scores -1.0 against a query none of them contain.
        run = {"q2": {"d3": 3.0, "d1": 2.0, "d4": 1.0}}
        reranked = build().rerank(run, {"q2": "zeta"}, CORPUS, show_progress=False)
        assert list(reranked["q2"]) == ["d1", "d3", "d4"]

    def test_every_query_is_handled(self):
        run = {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}}
        reranked = build().rerank(run, {"q1": "beta", "q2": "beta"}, CORPUS, show_progress=False)
        assert set(reranked) == {"q1", "q2"}


class TestDepth:
    def test_only_the_top_depth_candidates_are_scored(self):
        run = {"q1": {"d1": 9.0, "d2": 8.0, "d3": 7.0, "d4": 6.0}}
        reranker = build(depth=2)
        reranker.rerank(run, QUERIES, CORPUS, show_progress=False)
        assert len(reranker._model.seen) == 2

    def test_the_untouched_tail_sorts_below_every_reranked_document(self):
        # d4 is outside depth=2. Even though the reranked block contains documents the
        # model scored -1.0, the tail must still rank beneath them.
        run = {"q1": {"d1": 9.0, "d3": 8.0, "d4": 7.0, "d2": 1.0}}
        reranked = build(depth=2).rerank(run, QUERIES, CORPUS, show_progress=False)
        order = sorted(reranked["q1"].items(), key=lambda kv: (-kv[1], kv[0]))
        positions = {doc: i for i, (doc, _) in enumerate(order)}
        assert positions["d1"] < positions["d4"]
        assert positions["d3"] < positions["d4"]

    def test_the_tail_keeps_its_original_order(self):
        run = {"q1": {"d1": 9.0, "d2": 8.0, "d3": 7.0, "d4": 6.0}}
        reranked = build(depth=1).rerank(run, QUERIES, CORPUS, show_progress=False)
        order = [doc for doc, _ in sorted(reranked["q1"].items(), key=lambda kv: (-kv[1], kv[0]))]
        assert order[1:] == ["d2", "d3", "d4"]

    def test_depth_beyond_the_candidate_count_is_harmless(self):
        run = {"q1": {"d1": 1.0}}
        reranked = build(depth=500).rerank(run, QUERIES, CORPUS, show_progress=False)
        assert set(reranked["q1"]) == {"d1"}


class TestValidation:
    def test_non_positive_depth_raises(self):
        with pytest.raises(ValueError, match="depth must be positive"):
            CrossEncoderReranker(depth=0)

    def test_an_empty_run_returns_empty_results(self):
        assert build().rerank({"q1": {}}, QUERIES, CORPUS, show_progress=False) == {"q1": {}}


class TestDescribe:
    def test_records_model_and_depth(self):
        described = CrossEncoderReranker(depth=25).describe()
        assert described["method"] == "cross-encoder-rerank"
        assert described["depth"] == 25
        assert "model" in described
