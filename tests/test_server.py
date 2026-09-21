"""The MCP server's tools.

The MCP transport is not exercised — that would test the `mcp` package. What is tested is
the layer underneath it, against a tiny synthetic corpus, because these are the functions
that decide what a caller is told about retrieval quality.

The important tests are the ones about degradation and honesty: the server must work
without the dense extra installed rather than refusing to start, and it must not present
ad-hoc query results as benchmark scores.
"""

import pytest

from groundwork.server import (
    DatasetIndex,
    compare_methods,
    describe_benchmark,
    search,
)

CORPUS = {
    "d1": {"title": "CRISPR off-target effects", "text": "Guide RNA specificity in Cas9 editing."},
    "d2": {"title": "Maize drought tolerance", "text": "Stomatal conductance under water stress."},
    "d3": {
        "title": "Ice sheet mass balance",
        "text": "Satellite gravimetry of Antarctic ice loss.",
    },
}
QUERIES = {"q1": "Cas9 guide RNA specificity"}
QRELS = {"q1": {"d1": 1, "d2": 0}}


class FakeDataset:
    corpus = CORPUS
    queries = QUERIES
    qrels = QRELS


@pytest.fixture
def index():
    """An index over the synthetic corpus, with no dense model available."""
    built = DatasetIndex(name="fake")
    built._dataset = FakeDataset()
    # Mark dense as unavailable the way a missing extra would.
    built._dense_unavailable = True
    built._notes = ["dense retrieval unavailable: install the 'dense' extra"]
    return built


class TestSearch:
    def test_bm25_search_returns_ranked_documents(self, index):
        result = search(index, "Cas9 guide RNA", top_k=3, method="bm25")
        assert result["results"][0]["doc_id"] == "d1"
        assert result["results"][0]["rank"] == 1

    def test_results_carry_the_text_a_caller_needs_to_judge_them(self, index):
        result = search(index, "Cas9 guide RNA", top_k=1, method="bm25")
        top = result["results"][0]
        assert top["title"] == "CRISPR off-target effects"
        assert "Guide RNA" in top["text"]

    def test_the_method_actually_used_is_reported(self, index):
        """A caller must be able to tell that hybrid silently degraded to lexical only,
        because the two have different measured behaviour."""
        result = search(index, "Cas9", top_k=1, method="hybrid")
        assert result["method"] == "bm25"

    def test_hybrid_degrades_to_bm25_when_dense_is_unavailable(self, index):
        """BM25 needs only numpy. A caller without torch should get lexical search, not a
        server that refuses to start."""
        result = search(index, "Cas9", top_k=1, method="hybrid")
        assert result["results"]
        assert any("dense retrieval unavailable" in note for note in result["notes"])

    def test_dense_only_with_no_dense_model_fails_loudly(self, index):
        """Asking specifically for dense and silently getting lexical would be a lie."""
        with pytest.raises(ValueError, match="no retriever available"):
            search(index, "Cas9", method="dense")

    def test_top_k_limits_the_results(self, index):
        assert len(search(index, "ice maize Cas9", top_k=2, method="bm25")["results"]) == 2


class TestCompareMethods:
    def test_both_rankings_are_returned(self, index):
        result = compare_methods(index, "Cas9 guide RNA", top_k=2)
        assert result["bm25"]
        assert result["dense"] == []  # no dense model in this fixture

    def test_overlap_is_reported(self, index):
        result = compare_methods(index, "Cas9 guide RNA", top_k=2)
        assert result["overlap_at_k"] == 0
        assert result["overlapping_doc_ids"] == []

    def test_the_interpretation_does_not_tell_the_caller_to_pick_one(self, index):
        """Experiments 11 and 12 measured that routing loses to fusion. The server must
        not undo that finding in its own prose."""
        interpretation = compare_methods(index, "Cas9", top_k=1)["interpretation"]
        assert "use both" in interpretation.lower()


class TestDescribeBenchmark:
    def test_reports_the_corpus_and_the_settings(self, index):
        described = describe_benchmark(index)
        assert described["num_documents"] == 3
        assert described["retriever"]["variant"] == "lucene-multifield"

    def test_states_that_ad_hoc_queries_are_not_benchmark_results(self, index):
        """The failure this guards against is a caller quoting a server response as a
        measured score. There are no relevance judgements for an invented query."""
        assert "not benchmark" in describe_benchmark(index)["caveat"]

    def test_reports_the_published_baseline_for_a_known_dataset(self):
        known = DatasetIndex(name="scifact")
        known._dataset = FakeDataset()
        assert describe_benchmark(known)["published_bm25_ndcg_at_10"] == 0.665

    def test_reports_none_for_a_dataset_with_no_published_baseline(self, index):
        assert describe_benchmark(index)["published_bm25_ndcg_at_10"] is None
