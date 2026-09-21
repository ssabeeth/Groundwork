"""Dense retriever tests.

These run offline, with a stub encoder standing in for the real model. That is not a
compromise: the parts worth testing here are the ranking, the caching and the truncation
accounting, none of which are properties of any particular transformer. Downloading a
model to assert that it embeds text would test somebody else's software over the
network, which the project's testing rules rule out.

The stub encodes deterministically from the text itself, so expected rankings below are
worked out by hand from the vectors rather than read off the implementation.
"""

import numpy as np
import pytest

from groundwork.retrieval.dense import DenseRetriever, corpus_digest


class StubTokenizer:
    """Counts a "word piece" per whitespace token, so truncation is easy to reason about."""

    def encode(self, text, add_special_tokens=True, truncation=False):
        pieces = text.split()
        return pieces + (["[CLS]"] if add_special_tokens else [])


class StubModel:
    """Maps text to a unit vector over a fixed three-term vocabulary."""

    VOCAB = ("alpha", "beta", "gamma")

    def __init__(self, max_seq_length=8):
        self.max_seq_length = max_seq_length
        self.tokenizer = StubTokenizer()

    def encode(
        self,
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = np.array([float(lowered.count(term)) for term in self.VOCAB], dtype=np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
            vectors.append(vector)
        return np.stack(vectors)


CORPUS = {
    "d1": {"title": "alpha", "text": "alpha alpha"},
    "d2": {"title": "beta", "text": "beta"},
    "d3": {"title": "gamma", "text": "gamma gamma gamma"},
}


def build(**kwargs):
    retriever = DenseRetriever(cache_dir=None, **kwargs)
    retriever._model = StubModel()
    retriever.index(CORPUS, show_progress=False)
    return retriever


class TestRanking:
    def test_nearest_neighbour_by_cosine_wins(self):
        # d1 is pure alpha, so a pure-alpha query has cosine 1 with it and 0 with the rest.
        retriever = build()
        ranking = retriever.search("alpha", top_k=3)
        assert ranking[0][0] == "d1"
        assert ranking[0][1] == pytest.approx(1.0)

    def test_every_document_is_scored_unlike_bm25(self):
        # A dense score of zero means "unrelated", not "shares no term", so nothing is
        # dropped and the ranking is always top_k long.
        retriever = build()
        assert len(retriever.search("alpha", top_k=3)) == 3

    def test_top_k_is_respected_and_clamped_to_the_corpus(self):
        retriever = build()
        assert len(retriever.search("alpha", top_k=2)) == 2
        assert len(retriever.search("alpha", top_k=99)) == 3

    def test_ties_are_broken_by_document_id(self):
        # A query orthogonal to every document scores 0.0 everywhere, so the tie-break
        # is the only thing ordering the result.
        retriever = build()
        ranking = retriever.search("delta", top_k=3)
        assert [doc for doc, _ in ranking] == ["d1", "d2", "d3"]

    def test_ranking_is_sorted_by_descending_score(self):
        retriever = build()
        scores = [score for _, score in retriever.search("alpha beta", top_k=3)]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_matches_search_query_by_query(self):
        retriever = build()
        run = retriever.retrieve({"q1": "alpha", "q2": "gamma"}, top_k=3, show_progress=False)
        assert set(run) == {"q1", "q2"}
        for qid, text in (("q1", "alpha"), ("q2", "gamma")):
            assert run[qid] == pytest.approx(dict(retriever.search(text, top_k=3)))


class TestTruncationAccounting:
    def test_counts_documents_over_the_limit(self):
        # StubModel allows 8 pieces; the stub tokenizer counts whitespace tokens plus one.
        # "alpha " * 20 is 20 pieces + 1 -> over. "alpha" is 1 + 1 -> under.
        retriever = DenseRetriever(cache_dir=None)
        retriever._model = StubModel(max_seq_length=8)
        assert retriever.measure_truncation(["alpha", "alpha " * 20]) == pytest.approx(0.5)

    def test_no_truncation_reports_zero(self):
        retriever = DenseRetriever(cache_dir=None)
        retriever._model = StubModel(max_seq_length=8)
        assert retriever.measure_truncation(["alpha", "beta"]) == 0.0

    def test_empty_input_reports_zero(self):
        retriever = DenseRetriever(cache_dir=None)
        retriever._model = StubModel()
        assert retriever.measure_truncation([]) == 0.0

    def test_index_records_the_rate_for_describe(self):
        retriever = build()
        assert retriever.truncation_rate == 0.0
        assert retriever.describe()["truncation_rate"] == 0.0


class TestCorpusDigest:
    def test_same_corpus_digests_the_same(self):
        assert corpus_digest(CORPUS) == corpus_digest(dict(CORPUS))

    def test_changing_text_changes_the_digest(self):
        altered = {**CORPUS, "d2": {"title": "beta", "text": "beta beta"}}
        assert corpus_digest(altered) != corpus_digest(CORPUS)

    def test_reordering_changes_the_digest(self):
        # The cache stores a matrix whose rows are positional, so a reordered corpus
        # must not hit the same cache entry or every document would be misaligned.
        reordered = dict(reversed(list(CORPUS.items())))
        assert corpus_digest(reordered) != corpus_digest(CORPUS)

    def test_field_boundaries_cannot_be_forged(self):
        # Concatenation without separators would let title/text be shifted across the
        # boundary without changing the digest.
        a = {"d1": {"title": "ab", "text": "c"}}
        b = {"d1": {"title": "a", "text": "bc"}}
        assert corpus_digest(a) != corpus_digest(b)


class TestCaching:
    def test_a_cached_index_reproduces_the_ranking(self, tmp_path):
        first = DenseRetriever(cache_dir=tmp_path)
        first._model = StubModel()
        first.index(CORPUS, show_progress=False)
        expected = first.search("alpha", top_k=3)

        second = DenseRetriever(cache_dir=tmp_path)
        second._model = StubModel()
        second.index(CORPUS, show_progress=False)

        assert list(tmp_path.glob("*.npz"))
        assert second.search("alpha", top_k=3) == pytest.approx(expected)

    def test_cache_key_separates_models_and_lengths(self, tmp_path):
        a = DenseRetriever(cache_dir=tmp_path, model_name="model-a")
        b = DenseRetriever(cache_dir=tmp_path, model_name="model-b")
        c = DenseRetriever(cache_dir=tmp_path, model_name="model-a", max_seq_length=128)
        paths = {a._cache_path(CORPUS), b._cache_path(CORPUS), c._cache_path(CORPUS)}
        assert len(paths) == 3

    def test_caching_can_be_disabled(self):
        assert DenseRetriever(cache_dir=None)._cache_path(CORPUS) is None


class TestValidation:
    def test_empty_corpus_raises(self):
        retriever = DenseRetriever(cache_dir=None)
        retriever._model = StubModel()
        with pytest.raises(ValueError):
            retriever.index({}, show_progress=False)

    def test_search_before_index_raises(self):
        with pytest.raises(RuntimeError):
            DenseRetriever(cache_dir=None).search("anything")

    def test_retrieve_before_index_raises(self):
        with pytest.raises(RuntimeError):
            DenseRetriever(cache_dir=None).retrieve({"q1": "anything"})

    def test_non_positive_top_k_raises(self):
        with pytest.raises(ValueError):
            build().search("alpha", top_k=0)


class TestDescribe:
    def test_records_what_changes_the_result(self):
        described = build().describe()
        assert described["method"] == "dense"
        assert described["similarity"] == "cosine"
        assert described["normalized"] is True
        assert described["max_seq_length"] == 8
        assert "model" in described
