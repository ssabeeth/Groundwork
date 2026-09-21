"""RM3 tests.

Two kinds of check, in the spirit of the rest of the suite.

The first is an identity: with `alpha=0` the interpolation puts no weight on the
relevance model, so RM3 must reduce to plain BM25 *exactly*. That one assertion
exercises the expansion, the weighted scorer and the ranking path in a single stroke,
and it cannot be satisfied by a subtly wrong relevance model.

The second is arithmetic: the relevance model is small enough on a toy corpus to work
out by hand, and the expected weights below are derived in the comments rather than
copied from what the code produced.
"""

import pytest

from groundwork.retrieval.bm25 import BM25Retriever, MultiFieldBM25Retriever
from groundwork.retrieval.rm3 import RM3Retriever
from groundwork.retrieval.tokenize import Tokenizer

PLAIN = Tokenizer(stopwords=None, stem=False)

CORPUS = {
    "d1": {"title": "", "text": "apple banana"},
    "d2": {"title": "", "text": "apple cherry cherry"},
    "d3": {"title": "", "text": "durian"},
}

BIGGER = {
    "d1": {"title": "", "text": "the quick brown fox"},
    "d2": {"title": "", "text": "the quick brown fox jumps over the lazy dog"},
    "d3": {"title": "", "text": "lazy dog"},
    "d4": {"title": "", "text": "quick foxes are brown and lazy"},
}


def build(**kwargs):
    retriever = RM3Retriever(tokenizer=PLAIN, **kwargs)
    retriever.index(BIGGER, show_progress=False)
    return retriever


class TestReducesToBM25:
    """alpha=0 means no weight on the relevance model, so nothing may change."""

    @pytest.mark.parametrize("query", ["quick fox", "lazy dog", "brown", "quick quick lazy"])
    def test_alpha_zero_reproduces_plain_bm25(self, query):
        bm25 = BM25Retriever(k1=0.9, b=0.4, tokenizer=PLAIN)
        bm25.index(BIGGER, show_progress=False)

        rm3 = build(alpha=0.0, fb_docs=10, fb_terms=10)

        expected = bm25.search(query, top_k=10)
        actual = rm3.search(query, top_k=10)
        assert [doc for doc, _ in actual] == [doc for doc, _ in expected]
        # BM25 scores a bag of terms; RM3 with alpha=0 scores the same terms weighted
        # by 1/|Q|, so scores are proportional rather than equal.
        scale = len(PLAIN(query))
        assert [score * scale for _, score in actual] == pytest.approx(
            [score for _, score in expected], rel=1e-5
        )

    def test_zero_feedback_documents_disables_expansion(self):
        rm3 = build(alpha=0.9, fb_docs=0, fb_terms=10)
        assert set(rm3.expanded_query("quick fox")) == {"quick", "fox"}

    def test_zero_feedback_terms_disables_expansion(self):
        rm3 = build(alpha=0.9, fb_docs=3, fb_terms=0)
        assert set(rm3.expanded_query("quick fox")) == {"quick", "fox"}


class TestRelevanceModelArithmetic:
    def test_single_feedback_document_is_its_own_term_distribution(self):
        # One feedback document, so w_d = 1. d2 is "apple cherry cherry": three tokens,
        # P(apple|R) = 1/3, P(cherry|R) = 2/3. Renormalising over both keeps that.
        rm3 = RM3Retriever(tokenizer=PLAIN, fb_docs=1, fb_terms=10, alpha=1.0)
        rm3.index(CORPUS, show_progress=False)
        model = rm3.relevance_model([("d2", 1.0)])
        assert model == pytest.approx({"apple": 1 / 3, "cherry": 2 / 3})

    def test_documents_are_weighted_by_normalised_score(self):
        # Feedback: d1 score 3, d3 score 1 -> weights 0.75 and 0.25.
        # d1 "apple banana" (2 tokens): apple 1/2, banana 1/2
        # d3 "durian"       (1 token):  durian 1
        # P(apple|R)  = 0.75 * 0.5 = 0.375
        # P(banana|R) = 0.75 * 0.5 = 0.375
        # P(durian|R) = 0.25 * 1.0 = 0.25
        # These already sum to 1, so renormalisation is a no-op.
        rm3 = RM3Retriever(tokenizer=PLAIN, fb_docs=2, fb_terms=10, alpha=1.0)
        rm3.index(CORPUS, show_progress=False)
        model = rm3.relevance_model([("d1", 3.0), ("d3", 1.0)])
        assert model == pytest.approx({"apple": 0.375, "banana": 0.375, "durian": 0.25})

    def test_only_the_top_terms_are_kept_and_renormalised(self):
        # Same feedback as above; keeping 2 of 3 terms drops durian (0.25) and leaves
        # apple and banana at 0.375 each, renormalised to 0.5 each.
        rm3 = RM3Retriever(tokenizer=PLAIN, fb_docs=2, fb_terms=2, alpha=1.0)
        rm3.index(CORPUS, show_progress=False)
        model = rm3.relevance_model([("d1", 3.0), ("d3", 1.0)])
        assert model == pytest.approx({"apple": 0.5, "banana": 0.5})

    def test_model_weights_sum_to_one(self):
        rm3 = build(fb_docs=3, fb_terms=5, alpha=0.5)
        model = rm3.relevance_model(rm3.bm25.search("quick fox", top_k=3))
        assert sum(model.values()) == pytest.approx(1.0)

    def test_no_feedback_gives_an_empty_model(self):
        rm3 = build(fb_docs=3, fb_terms=5, alpha=0.5)
        assert rm3.relevance_model([]) == {}

    def test_non_positive_scores_give_an_empty_model(self):
        rm3 = build(fb_docs=3, fb_terms=5, alpha=0.5)
        assert rm3.relevance_model([("d1", 0.0)]) == {}


class TestInterpolation:
    def test_interpolated_weights_sum_to_one(self):
        # (1-alpha) over a distribution summing to 1, plus alpha over another, must
        # itself sum to 1 whatever the overlap between the two term sets.
        rm3 = build(fb_docs=3, fb_terms=6, alpha=0.4)
        weights = rm3.expanded_query("quick fox")
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_expansion_adds_terms_the_query_did_not_have(self):
        rm3 = build(fb_docs=3, fb_terms=8, alpha=0.6)
        weights = rm3.expanded_query("quick")
        assert set(weights) > {"quick"}

    def test_alpha_one_discards_the_original_query_terms_weight(self):
        # With alpha=1 the original distribution contributes nothing, so the expanded
        # query must be exactly the relevance model.
        rm3 = build(fb_docs=1, fb_terms=3, alpha=1.0)
        model = rm3.relevance_model(rm3.bm25.search("dog", top_k=1))
        assert model  # the query must actually retrieve something for this to mean anything
        assert rm3.expanded_query("dog") == pytest.approx(model)

    def test_a_query_matching_nothing_falls_back_to_the_original_terms(self):
        # No first-pass results means no evidence to expand with. Falling back to the
        # unexpanded query is the only safe move: inventing an expansion from an empty
        # feedback set would be strictly worse than doing nothing.
        rm3 = build(fb_docs=5, fb_terms=5, alpha=1.0)
        assert rm3.expanded_query("aardvark") == pytest.approx({"aardvark": 1.0})
        assert rm3.search("aardvark", top_k=10) == []

    def test_empty_query_retrieves_nothing(self):
        rm3 = build(fb_docs=3, fb_terms=5, alpha=0.5)
        assert rm3.search("", top_k=10) == []


class TestParameterReuse:
    def test_with_parameters_shares_the_index(self):
        rm3 = build(fb_docs=3, fb_terms=5, alpha=0.5)
        clone = rm3.with_parameters(alpha=0.2)
        assert clone.bm25 is rm3.bm25
        assert clone.alpha == 0.2
        assert clone.fb_docs == 3
        assert clone.fb_terms == 5

    def test_clone_scores_identically_to_a_freshly_indexed_retriever(self):
        rm3 = build(fb_docs=3, fb_terms=5, alpha=0.5)
        clone = rm3.with_parameters(alpha=0.3, fb_terms=4)

        fresh = RM3Retriever(tokenizer=PLAIN, fb_docs=3, fb_terms=4, alpha=0.3)
        fresh.index(BIGGER, show_progress=False)

        assert dict(clone.search("quick fox", top_k=10)) == pytest.approx(
            dict(fresh.search("quick fox", top_k=10))
        )

    def test_requires_an_index(self):
        with pytest.raises(RuntimeError):
            RM3Retriever().with_parameters(alpha=0.5)


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs"),
        [{"alpha": -0.1}, {"alpha": 1.1}, {"fb_docs": -1}, {"fb_terms": -1}],
    )
    def test_invalid_parameters_raise(self, kwargs):
        with pytest.raises(ValueError):
            RM3Retriever(**kwargs)

    def test_searching_before_indexing_raises(self):
        with pytest.raises(RuntimeError):
            RM3Retriever().search("anything")


class TestDescribe:
    def test_records_every_parameter_that_moves_the_result(self):
        described = build(fb_docs=7, fb_terms=13, alpha=0.35).describe()
        assert described["method"] == "rm3"
        assert described["fb_docs"] == 7
        assert described["fb_terms"] == 13
        assert described["alpha"] == 0.35
        assert described["tokenizer"] == PLAIN.describe()


class TestFeedbackTokenCache:
    """Caching feedback tokens must change speed and nothing else."""

    def test_cached_and_uncached_searches_agree(self):
        warm = build(fb_docs=3, fb_terms=5, alpha=0.5)
        warm.search("quick fox", top_k=10)  # populates the cache
        assert warm._token_cache  # the search really did cache something

        cold = build(fb_docs=3, fb_terms=5, alpha=0.5)
        assert dict(warm.search("quick fox", top_k=10)) == pytest.approx(
            dict(cold.search("quick fox", top_k=10))
        )

    def test_cache_holds_only_documents_used_as_feedback(self):
        rm3 = build(fb_docs=1, fb_terms=5, alpha=0.5)
        rm3.search("dog", top_k=10)
        assert len(rm3._token_cache) == 1
        assert len(rm3._token_cache) < len(BIGGER)

    def test_reindexing_clears_the_cache(self):
        rm3 = build(fb_docs=3, fb_terms=5, alpha=0.5)
        rm3.search("quick fox", top_k=10)
        rm3.index(CORPUS, show_progress=False)
        assert rm3._token_cache == {}

    def test_clone_shares_the_cache(self):
        rm3 = build(fb_docs=3, fb_terms=5, alpha=0.5)
        rm3.search("quick fox", top_k=10)
        clone = rm3.with_parameters(alpha=0.2)
        assert clone._token_cache is rm3._token_cache


class TestMultiFieldRM3:
    """RM3 over a multi-field index must score the expansion the same way the first pass was."""

    FIELDED = {
        "d1": {"title": "quick fox", "text": "lazy dog sleeps through the afternoon"},
        "d2": {"title": "lazy dog", "text": ""},
        "d3": {"title": "", "text": "quick brown fox jumps far"},
    }

    def test_alpha_zero_reproduces_plain_multi_field_bm25(self):
        # The same identity that anchors the single-field case: with no weight on the
        # relevance model, RM3 must be its own underlying retriever.
        rm3 = RM3Retriever(tokenizer=PLAIN, alpha=0.0, fb_docs=5, fb_terms=5, multi_field=True)
        rm3.index(self.FIELDED, show_progress=False)

        multi = MultiFieldBM25Retriever(k1=0.9, b=0.4, tokenizer=PLAIN)
        multi.index(self.FIELDED, show_progress=False)

        expected = multi.search("quick fox", top_k=10)
        actual = rm3.search("quick fox", top_k=10)
        assert [doc for doc, _ in actual] == [doc for doc, _ in expected]
        scale = len(PLAIN("quick fox"))
        assert [score * scale for _, score in actual] == pytest.approx(
            [score for _, score in expected], rel=1e-5
        )

    def test_it_differs_from_single_field_rm3(self):
        # If these agreed the migration would be a no-op.
        single = RM3Retriever(tokenizer=PLAIN, alpha=0.5, fb_docs=2, fb_terms=5)
        single.index(self.FIELDED, show_progress=False)
        multi = RM3Retriever(tokenizer=PLAIN, alpha=0.5, fb_docs=2, fb_terms=5, multi_field=True)
        multi.index(self.FIELDED, show_progress=False)
        assert dict(single.search("quick fox", top_k=10)) != pytest.approx(
            dict(multi.search("quick fox", top_k=10))
        )

    def test_feedback_terms_come_from_the_whole_document(self):
        # The relevance model is about vocabulary, not field structure, so a term in the
        # body must be available to expand with even under a multi-field index.
        rm3 = RM3Retriever(tokenizer=PLAIN, alpha=1.0, fb_docs=1, fb_terms=10, multi_field=True)
        rm3.index(self.FIELDED, show_progress=False)
        model = rm3.relevance_model([("d1", 1.0)])
        assert "sleeps" in model

    def test_with_parameters_keeps_the_field_setting(self):
        rm3 = RM3Retriever(tokenizer=PLAIN, alpha=0.5, fb_docs=2, fb_terms=5, multi_field=True)
        rm3.index(self.FIELDED, show_progress=False)
        assert rm3.with_parameters(alpha=0.2).multi_field is True

    def test_describe_reports_the_multi_field_variant(self):
        rm3 = RM3Retriever(tokenizer=PLAIN, multi_field=True)
        rm3.index(self.FIELDED, show_progress=False)
        assert rm3.describe()["base"]["variant"] == "lucene-multifield"
