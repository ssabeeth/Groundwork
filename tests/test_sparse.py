"""Learned sparse retrieval.

The model is not loaded here — CI runs offline — so what is tested is the retrieval
arithmetic around it, with an index built by hand. That is the part where a bug produces
a plausible ranking rather than an error.

Expected values below are worked out by hand and the arithmetic is shown. Do not replace
them with numbers copied from a failing test.
"""

import numpy as np
import pytest

from groundwork.retrieval.sparse import SpladeRetriever


def built() -> SpladeRetriever:
    """A retriever with a three-document index installed directly.

    Postings, as ``term id -> (document indices, weights)``::

        term 5 -> doc 0 at 0.5, doc 2 at 2.0
        term 9 -> doc 1 at 1.0
    """
    retriever = SpladeRetriever(cache_dir=None)
    retriever.doc_ids = ["a", "b", "c"]
    retriever._postings = {
        5: (np.array([0, 2], dtype=np.int32), np.array([0.5, 2.0], dtype=np.float32)),
        9: (np.array([1], dtype=np.int32), np.array([1.0], dtype=np.float32)),
    }
    return retriever


class TestConstructorGuards:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"max_length": 0}, "max_length"),
            ({"batch_size": -1}, "batch_size"),
            ({"top_terms": 0}, "top_terms"),
        ],
    )
    def test_nonsense_settings_fail_loudly(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            SpladeRetriever(**kwargs)

    def test_top_terms_may_be_none_meaning_keep_everything(self):
        assert SpladeRetriever(top_terms=None).top_terms is None


class TestScoreQuery:
    def test_scores_are_the_sparse_dot_product(self):
        # Query: term 5 at weight 2.0, term 9 at weight 3.0.
        #   doc a = 2.0 * 0.5 = 1.0
        #   doc b = 3.0 * 1.0 = 3.0
        #   doc c = 2.0 * 2.0 = 4.0
        scores = built().score_query(
            np.array([5, 9], dtype=np.int32), np.array([2.0, 3.0], dtype=np.float32)
        )
        assert scores.tolist() == pytest.approx([1.0, 3.0, 4.0])

    def test_a_query_term_absent_from_the_index_contributes_nothing(self):
        """Not an error: SPLADE queries carry terms no document happens to have."""
        scores = built().score_query(
            np.array([5, 4242], dtype=np.int32), np.array([2.0, 9.0], dtype=np.float32)
        )
        assert scores.tolist() == pytest.approx([1.0, 0.0, 4.0])

    def test_a_query_with_no_terms_scores_everything_zero(self):
        scores = built().score_query(np.array([], dtype=np.int32), np.array([], dtype=np.float32))
        assert scores.tolist() == [0.0, 0.0, 0.0]

    def test_weights_multiply_rather_than_replace(self):
        """Doubling a query weight doubles that term's contribution, which is what makes
        the score a dot product rather than a count of matches."""
        one = built().score_query(np.array([5], np.int32), np.array([1.0], np.float32))
        two = built().score_query(np.array([5], np.int32), np.array([2.0], np.float32))
        assert two.tolist() == pytest.approx((one * 2).tolist())


class TestRankScores:
    def test_documents_come_back_in_descending_score(self):
        ranked = built().rank_scores(np.array([1.0, 3.0, 4.0], dtype=np.float32), top_k=3)
        assert [doc for doc, _ in ranked] == ["c", "b", "a"]

    def test_zero_scoring_documents_are_dropped_not_ranked(self):
        """Matches BM25: a document sharing no term is not "rank 3", it is unretrieved."""
        ranked = built().rank_scores(np.array([0.0, 3.0, 0.0], dtype=np.float32), top_k=3)
        assert [doc for doc, _ in ranked] == ["b"]

    def test_ties_are_broken_by_document_id(self):
        ranked = built().rank_scores(np.array([2.0, 2.0, 1.0], dtype=np.float32), top_k=3)
        assert [doc for doc, _ in ranked] == ["a", "b", "c"]

    def test_top_k_truncates_after_sorting_not_before(self):
        ranked = built().rank_scores(np.array([1.0, 3.0, 4.0], dtype=np.float32), top_k=2)
        assert [doc for doc, _ in ranked] == ["c", "b"]

    def test_an_all_zero_score_vector_retrieves_nothing(self):
        assert built().rank_scores(np.zeros(3, dtype=np.float32), top_k=3) == []


class TestRetrieveGuards:
    def test_retrieving_before_indexing_fails_loudly(self):
        with pytest.raises(RuntimeError, match="index"):
            SpladeRetriever().retrieve({"q": "anything"})


class TestDescribe:
    def test_the_settings_that_change_results_are_recorded(self):
        described = SpladeRetriever(max_length=128, top_terms=64).describe()
        assert described["method"] == "splade"
        assert described["max_length"] == 128
        assert described["top_terms"] == 64
        assert described["pooling"] == "max"
        assert described["similarity"] == "dot-product"

    def test_the_record_is_json_safe(self):
        import json

        json.dumps(SpladeRetriever().describe())
