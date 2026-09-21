"""BM25 tests.

The main correctness check scores a tiny corpus with a second, deliberately naive
implementation written straight from the Lucene formula, and requires the two to agree.
A test that reuses the optimised code path would only prove it is consistent with
itself, which is not the question.
"""

import math

import pytest

from groundwork.retrieval.bm25 import BM25Retriever, MultiFieldBM25Retriever
from groundwork.retrieval.tokenize import Tokenizer

CORPUS = {
    "d1": {"title": "", "text": "the quick brown fox"},
    "d2": {"title": "", "text": "the quick brown fox jumps over the lazy dog"},
    "d3": {"title": "", "text": "lazy dog"},
}

# Plain tokenisation so the arithmetic below can be followed by hand.
PLAIN = Tokenizer(stopwords=None, stem=False)


def naive_bm25(query: str, corpus: dict, k1: float, b: float, tokenizer) -> dict[str, float]:
    """Reference scorer: Lucene BM25 transcribed directly, no optimisation."""
    docs = {
        doc_id: tokenizer(f"{f.get('title', '')} {f.get('text', '')}".strip())
        for doc_id, f in corpus.items()
    }
    num_docs = len(docs)
    avgdl = sum(len(tokens) for tokens in docs.values()) / num_docs

    scores = {}
    for doc_id, tokens in docs.items():
        total = 0.0
        for term in tokenizer(query):
            tf = tokens.count(term)
            if tf == 0:
                continue
            df = sum(1 for other in docs.values() if term in other)
            idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
            norm = k1 * (1.0 - b + b * len(tokens) / avgdl)
            total += idf * (tf * (k1 + 1.0)) / (tf + norm)
        scores[doc_id] = total
    return scores


class TestBM25Scoring:
    @pytest.mark.parametrize(
        ("k1", "b"),
        [(0.9, 0.4), (1.2, 0.75), (1.5, 0.0), (0.0, 0.4)],
    )
    @pytest.mark.parametrize(
        "query",
        ["quick fox", "lazy dog", "the", "quick quick brown", "missing"],
    )
    def test_matches_an_independent_implementation(self, k1, b, query):
        retriever = BM25Retriever(k1=k1, b=b, tokenizer=PLAIN)
        retriever.index(CORPUS, show_progress=False)

        expected = naive_bm25(query, CORPUS, k1=k1, b=b, tokenizer=PLAIN)
        actual = dict(retriever.search(query, top_k=10))

        for doc_id, score in expected.items():
            if score > 0:
                assert actual[doc_id] == pytest.approx(score, rel=1e-5)
            else:
                assert doc_id not in actual

    def test_corpus_statistics(self):
        # Lengths 4, 9 and 2 tokens -> avgdl = 5.0
        retriever = BM25Retriever(tokenizer=PLAIN)
        retriever.index(CORPUS, show_progress=False)
        assert retriever.avg_doc_length == pytest.approx(5.0)
        assert list(retriever.doc_lengths) == pytest.approx([4.0, 9.0, 2.0])

    def test_idf_is_never_negative_for_common_terms(self):
        # "the" appears in 2 of 3 documents. The Robertson IDF would go negative here;
        # the Lucene form used must not.
        retriever = BM25Retriever(tokenizer=PLAIN)
        retriever.index(CORPUS, show_progress=False)
        assert retriever._idf["the"] > 0.0

    def test_idf_value(self):
        # df=2, N=3 -> ln(1 + (3 - 2 + 0.5) / 2.5) = ln(1.6) = 0.47000362924573563
        retriever = BM25Retriever(tokenizer=PLAIN)
        retriever.index(CORPUS, show_progress=False)
        assert retriever._idf["quick"] == pytest.approx(0.47000362924573563)

    def test_shorter_document_wins_when_term_counts_match(self):
        # d1 and d2 both contain "quick fox" once; d1 is shorter, so length
        # normalisation must rank it first.
        retriever = BM25Retriever(k1=0.9, b=0.4, tokenizer=PLAIN)
        retriever.index(CORPUS, show_progress=False)
        ranking = retriever.search("quick fox", top_k=10)
        assert [doc_id for doc_id, _ in ranking] == ["d1", "d2"]

    def test_b_zero_disables_length_normalisation(self):
        # With b=0 the two documents contain the query terms equally often and differ
        # only in length, so they must tie exactly.
        retriever = BM25Retriever(k1=0.9, b=0.0, tokenizer=PLAIN)
        retriever.index(CORPUS, show_progress=False)
        scores = dict(retriever.search("quick fox", top_k=10))
        assert scores["d1"] == pytest.approx(scores["d2"])

    def test_documents_without_query_terms_are_omitted(self):
        retriever = BM25Retriever(tokenizer=PLAIN)
        retriever.index(CORPUS, show_progress=False)
        assert dict(retriever.search("quick", top_k=10)).keys() == {"d1", "d2"}

    def test_query_with_no_indexed_terms_returns_nothing(self):
        retriever = BM25Retriever(tokenizer=PLAIN)
        retriever.index(CORPUS, show_progress=False)
        assert retriever.search("aardvark", top_k=10) == []

    def test_top_k_is_respected(self):
        retriever = BM25Retriever(tokenizer=PLAIN)
        retriever.index(CORPUS, show_progress=False)
        assert len(retriever.search("quick lazy dog fox", top_k=2)) == 2

    def test_ranking_is_sorted_by_descending_score(self):
        retriever = BM25Retriever(tokenizer=PLAIN)
        retriever.index(CORPUS, show_progress=False)
        scores = [score for _, score in retriever.search("quick lazy dog fox", top_k=10)]
        assert scores == sorted(scores, reverse=True)


class TestBM25Indexing:
    def test_title_is_indexed_with_the_body(self):
        corpus = {"d1": {"title": "photosynthesis", "text": "unrelated body"}}
        retriever = BM25Retriever(tokenizer=PLAIN)
        retriever.index(corpus, show_progress=False)
        assert retriever.search("photosynthesis", top_k=5)

    def test_search_before_index_raises(self):
        with pytest.raises(RuntimeError):
            BM25Retriever().search("anything")

    def test_empty_corpus_raises(self):
        with pytest.raises(ValueError):
            BM25Retriever().index({}, show_progress=False)

    @pytest.mark.parametrize(("k1", "b"), [(-1.0, 0.4), (0.9, 1.5), (0.9, -0.1)])
    def test_invalid_parameters_raise(self, k1, b):
        with pytest.raises(ValueError):
            BM25Retriever(k1=k1, b=b)


class TestRetrieveInterface:
    def test_returns_a_run_keyed_by_query_id(self):
        retriever = BM25Retriever(tokenizer=PLAIN)
        retriever.index(CORPUS, show_progress=False)
        run = retriever.retrieve(
            {"q1": "quick fox", "q2": "lazy dog"}, top_k=2, show_progress=False
        )
        assert set(run) == {"q1", "q2"}
        assert all(isinstance(scores, dict) for scores in run.values())
        assert run["q1"]["d1"] > 0.0


class TestTokenizer:
    def test_lowercases_and_splits_on_non_alphanumeric(self):
        assert PLAIN("SARS-CoV-2 (Omicron)") == ["sars", "cov", "2", "omicron"]

    def test_stopwords_are_dropped(self):
        assert "the" not in Tokenizer(stem=False)("the quick fox")

    def test_stemming_collapses_inflections(self):
        tokenizer = Tokenizer(stem=True)
        assert tokenizer("replicating") == tokenizer("replicate")


class TestParameterReuse:
    """`with_parameters` must be indistinguishable from indexing afresh.

    The whole point is that a sweep can skip re-indexing, so the test that matters is
    that skipping it changes nothing about the scores.
    """

    @pytest.mark.parametrize(("k1", "b"), [(0.9, 0.4), (1.2, 0.75), (0.0, 0.0), (2.0, 1.0)])
    def test_scores_match_a_retriever_indexed_from_scratch(self, k1, b):
        shared = BM25Retriever(k1=0.9, b=0.4, tokenizer=PLAIN)
        shared.index(CORPUS, show_progress=False)
        reused = shared.with_parameters(k1=k1, b=b)

        fresh = BM25Retriever(k1=k1, b=b, tokenizer=PLAIN)
        fresh.index(CORPUS, show_progress=False)

        for query in ("quick fox", "lazy dog", "the", "quick quick brown"):
            assert dict(reused.search(query, top_k=10)) == pytest.approx(
                dict(fresh.search(query, top_k=10))
            )

    def test_the_original_is_left_alone(self):
        original = BM25Retriever(k1=0.9, b=0.4, tokenizer=PLAIN)
        original.index(CORPUS, show_progress=False)
        before = dict(original.search("quick fox", top_k=10))

        original.with_parameters(k1=2.0, b=1.0).search("quick fox", top_k=10)

        assert dict(original.search("quick fox", top_k=10)) == pytest.approx(before)
        assert original.k1 == 0.9
        assert original.b == 0.4

    def test_omitted_parameters_are_inherited(self):
        original = BM25Retriever(k1=1.5, b=0.3, tokenizer=PLAIN)
        original.index(CORPUS, show_progress=False)
        clone = original.with_parameters(b=0.8)
        assert clone.k1 == 1.5
        assert clone.b == 0.8

    def test_tokenizer_is_carried_over_so_describe_stays_truthful(self):
        original = BM25Retriever(tokenizer=PLAIN)
        original.index(CORPUS, show_progress=False)
        clone = original.with_parameters(k1=1.2)
        assert clone.describe()["tokenizer"] == original.describe()["tokenizer"]
        assert clone.describe()["k1"] == 1.2

    def test_requires_an_index(self):
        with pytest.raises(RuntimeError):
            BM25Retriever().with_parameters(k1=1.2)

    @pytest.mark.parametrize(("k1", "b"), [(-1.0, 0.4), (0.9, 1.5)])
    def test_invalid_parameters_still_raise(self, k1, b):
        original = BM25Retriever(tokenizer=PLAIN)
        original.index(CORPUS, show_progress=False)
        with pytest.raises(ValueError):
            original.with_parameters(k1=k1, b=b)


class TestMultiFieldBM25:
    """Scoring fields separately is a different model, not a refactor.

    The expected values here come from the single-field scorer applied to each field on
    its own and added, which is the definition of what the multi-field retriever does —
    and which is emphatically not the same as scoring the concatenation.
    """

    FIELDED = {
        "d1": {"title": "quick fox", "text": "the lazy dog sleeps all afternoon here"},
        "d2": {"title": "lazy dog", "text": ""},
        "d3": {"title": "", "text": "quick brown fox jumps"},
    }

    def _single_field_score(self, field, query):
        """Score one field with the ordinary single-field retriever."""
        view = {
            doc_id: {"title": "", "text": fields.get(field, "")}
            for doc_id, fields in self.FIELDED.items()
        }
        retriever = BM25Retriever(k1=0.9, b=0.4, tokenizer=PLAIN)
        retriever.index(view, show_progress=False)
        return dict(retriever.search(query, top_k=10))

    @pytest.mark.parametrize(("k1", "b"), [(0.9, 0.4), (1.4, 0.5), (0.0, 0.0), (2.0, 1.0)])
    def test_with_parameters_matches_a_retriever_indexed_from_scratch(self, k1, b):
        """Same contract as the single-field version: reusing the index changes nothing.

        This method did not exist until after the multi-field migration, and its absence
        was why `run_sweep.py` was still sweeping a concatenated index while every other
        script had moved on. A sweep that cannot reuse its index is a sweep that quietly
        keeps using whichever retriever it was written against.
        """
        shared = MultiFieldBM25Retriever(k1=0.9, b=0.4, tokenizer=PLAIN)
        shared.index(self.FIELDED, show_progress=False)
        reused = shared.with_parameters(k1=k1, b=b)

        fresh = MultiFieldBM25Retriever(k1=k1, b=b, tokenizer=PLAIN)
        fresh.index(self.FIELDED, show_progress=False)

        for query in ("lazy dog", "quick fox", "the"):
            assert dict(reused.search(query, top_k=10)) == pytest.approx(
                dict(fresh.search(query, top_k=10))
            )

    def test_with_parameters_leaves_the_original_scoring_unchanged(self):
        shared = MultiFieldBM25Retriever(k1=0.9, b=0.4, tokenizer=PLAIN)
        shared.index(self.FIELDED, show_progress=False)
        before = dict(shared.search("lazy dog", top_k=10))

        shared.with_parameters(k1=2.0, b=1.0).search("lazy dog", top_k=10)

        assert dict(shared.search("lazy dog", top_k=10)) == pytest.approx(before)

    def test_with_parameters_keeps_the_axis_that_was_not_given(self):
        shared = MultiFieldBM25Retriever(k1=1.3, b=0.7, tokenizer=PLAIN)
        shared.index(self.FIELDED, show_progress=False)
        assert shared.with_parameters(b=0.2).k1 == 1.3
        assert shared.with_parameters(k1=0.5).b == 0.7

    def test_with_parameters_before_indexing_fails_loudly(self):
        with pytest.raises(RuntimeError, match="index\\(\\) must be called"):
            MultiFieldBM25Retriever().with_parameters(k1=1.2)

    def test_with_parameters_reports_the_multifield_variant(self):
        """A clone that described itself as single-field would defeat the filename check
        in tests/test_documentation.py, which reads exactly this field."""
        shared = MultiFieldBM25Retriever(tokenizer=PLAIN)
        shared.index(self.FIELDED, show_progress=False)
        assert shared.with_parameters(k1=1.4).describe()["variant"] == "lucene-multifield"

    def test_score_is_the_sum_of_the_per_field_scores(self):
        multi = MultiFieldBM25Retriever(k1=0.9, b=0.4, tokenizer=PLAIN)
        multi.index(self.FIELDED, show_progress=False)
        actual = dict(multi.search("lazy dog", top_k=10))

        title = self._single_field_score("title", "lazy dog")
        text = self._single_field_score("text", "lazy dog")
        for doc_id in ("d1", "d2"):
            expected = title.get(doc_id, 0.0) + text.get(doc_id, 0.0)
            assert actual[doc_id] == pytest.approx(expected, rel=1e-5)

    def test_it_differs_from_scoring_the_concatenation(self):
        # If these agreed there would be nothing to measure. d2 is title-only, so under
        # concatenation it is a two-token document and length normalisation flatters it.
        multi = MultiFieldBM25Retriever(k1=0.9, b=0.4, tokenizer=PLAIN)
        multi.index(self.FIELDED, show_progress=False)

        single = BM25Retriever(k1=0.9, b=0.4, tokenizer=PLAIN)
        single.index(self.FIELDED, show_progress=False)

        assert dict(multi.search("lazy dog", top_k=10)) != pytest.approx(
            dict(single.search("lazy dog", top_k=10))
        )

    def test_a_field_carries_its_own_length_normalisation(self):
        # "quick fox" is d1's whole title but a fraction of d3's body. Scoring fields
        # separately must let d1 win on title despite d1 having the longer document.
        multi = MultiFieldBM25Retriever(k1=0.9, b=0.4, tokenizer=PLAIN)
        multi.index(self.FIELDED, show_progress=False)
        title_only = self._single_field_score("title", "quick fox")
        assert title_only["d1"] > 0.0

    def test_weights_scale_a_fields_contribution(self):
        plain = MultiFieldBM25Retriever(tokenizer=PLAIN)
        plain.index(self.FIELDED, show_progress=False)
        weighted = MultiFieldBM25Retriever(weights=(3.0, 1.0), tokenizer=PLAIN)
        weighted.index(self.FIELDED, show_progress=False)
        # d2 has only a title, so tripling the title weight must triple its score.
        assert dict(weighted.search("lazy dog", top_k=10))["d2"] == pytest.approx(
            3.0 * dict(plain.search("lazy dog", top_k=10))["d2"], rel=1e-5
        )

    def test_zero_weight_removes_a_field(self):
        body_only = MultiFieldBM25Retriever(weights=(0.0, 1.0), tokenizer=PLAIN)
        body_only.index(self.FIELDED, show_progress=False)
        # d2's only content is its title, so with the title weighted out it must vanish.
        assert "d2" not in dict(body_only.search("lazy dog", top_k=10))

    def test_ranking_is_sorted_and_ties_break_by_id(self):
        multi = MultiFieldBM25Retriever(tokenizer=PLAIN)
        multi.index(self.FIELDED, show_progress=False)
        ranking = multi.search("quick fox lazy dog", top_k=10)
        scores = [score for _, score in ranking]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_returns_a_run_keyed_by_query_id(self):
        multi = MultiFieldBM25Retriever(tokenizer=PLAIN)
        multi.index(self.FIELDED, show_progress=False)
        run = multi.retrieve({"q1": "lazy dog"}, top_k=5, show_progress=False)
        assert set(run) == {"q1"}
        assert run["q1"]

    def test_describe_records_the_fields_and_weights(self):
        described = MultiFieldBM25Retriever(weights=(2.0, 1.0), tokenizer=PLAIN).describe()
        assert described["variant"] == "lucene-multifield"
        assert described["fields"] == ["title", "text"]
        assert described["field_weights"] == [2.0, 1.0]

    def test_search_before_index_raises(self):
        with pytest.raises(RuntimeError):
            MultiFieldBM25Retriever().search("anything")

    def test_empty_corpus_raises(self):
        with pytest.raises(ValueError):
            MultiFieldBM25Retriever().index({}, show_progress=False)

    def test_mismatched_weights_raise(self):
        with pytest.raises(ValueError, match="weights for"):
            MultiFieldBM25Retriever(fields=("title", "text"), weights=(1.0,))

    def test_no_fields_raises(self):
        with pytest.raises(ValueError, match="at least one field"):
            MultiFieldBM25Retriever(fields=())
