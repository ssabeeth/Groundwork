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


LONG_CORPUS = {
    # Twelve words each, so chunking at 4 gives exactly three windows with no overlap.
    "d1": {"title": "", "text": " ".join(["alpha"] * 12)},
    "d2": {"title": "", "text": " ".join(["beta"] * 4 + ["gamma"] * 4 + ["alpha"] * 4)},
    "d3": {"title": "", "text": "gamma"},
}


def build_chunked(**kwargs):
    retriever = DenseRetriever(cache_dir=None, **kwargs)
    retriever._model = StubModel()
    retriever.index(LONG_CORPUS, show_progress=False)
    return retriever


class TestChunkSplitting:
    def test_chunking_off_returns_the_text_unchanged(self):
        retriever = DenseRetriever(cache_dir=None)
        assert retriever.split_into_chunks("a b c d e") == ["a b c d e"]

    def test_text_shorter_than_a_chunk_is_one_chunk(self):
        retriever = DenseRetriever(cache_dir=None, chunk_words=10)
        assert retriever.split_into_chunks("a b c") == ["a b c"]

    def test_exact_multiple_splits_evenly(self):
        # 6 words, chunk 3, no overlap -> two windows.
        retriever = DenseRetriever(cache_dir=None, chunk_words=3)
        assert retriever.split_into_chunks("a b c d e f") == ["a b c", "d e f"]

    def test_overlap_shares_words_between_windows(self):
        # chunk 3, overlap 1 -> stride 2: [a b c], [c d e], [e]
        retriever = DenseRetriever(cache_dir=None, chunk_words=3, chunk_overlap=1)
        assert retriever.split_into_chunks("a b c d e") == ["a b c", "c d e", "e"]

    def test_empty_text_still_yields_one_chunk(self):
        # Every document must own at least one embedding row, or the document
        # boundaries used to collapse chunk scores would shift.
        retriever = DenseRetriever(cache_dir=None, chunk_words=4)
        assert retriever.split_into_chunks("") == [""]

    @pytest.mark.parametrize(
        "kwargs", [{"chunk_words": 0}, {"chunk_words": -1}, {"chunk_words": 3, "chunk_overlap": 3}]
    )
    def test_invalid_chunk_settings_raise(self, kwargs):
        with pytest.raises(ValueError):
            DenseRetriever(cache_dir=None, **kwargs)


class TestChunkedRetrieval:
    def test_one_embedding_row_per_chunk(self):
        # d1: 12 words / 4 = 3 chunks. d2: 3 chunks. d3: 1 chunk. Total 7.
        retriever = build_chunked(chunk_words=4)
        assert retriever.num_chunks == 7
        assert retriever.embeddings.shape[0] == 7
        assert list(retriever._chunk_starts) == [0, 3, 6]

    def test_a_document_is_scored_by_its_best_chunk(self):
        # d2's last chunk is pure alpha, so against an alpha query d2 must score 1.0 even
        # though two thirds of the document is about something else. Truncating at the
        # first 4 words would have scored it 0.
        retriever = build_chunked(chunk_words=4)
        scores = dict(retriever.search("alpha", top_k=3))
        assert scores["d2"] == pytest.approx(1.0)

    def test_chunking_recovers_a_match_that_truncation_loses(self):
        truncated = DenseRetriever(cache_dir=None, chunk_words=None)
        truncated._model = StubModel(max_seq_length=8)
        truncated.index(LONG_CORPUS, show_progress=False)

        chunked = build_chunked(chunk_words=4)

        # The stub encodes whole text, so truncation does not bite in the stub itself;
        # what this asserts is the ranking contract: with chunking, d2's alpha tail is
        # worth as much as d1's, so the two tie. Without it, d2 is diluted by its
        # other eight words and ranks below d1.
        assert chunked.search("alpha", top_k=3)[0][1] == pytest.approx(1.0)
        assert dict(truncated.search("alpha", top_k=3))["d2"] < 1.0

    def test_retrieve_and_search_agree_under_chunking(self):
        retriever = build_chunked(chunk_words=4)
        run = retriever.retrieve({"q1": "alpha", "q2": "gamma"}, top_k=3, show_progress=False)
        for qid, text in (("q1", "alpha"), ("q2", "gamma")):
            assert run[qid] == pytest.approx(dict(retriever.search(text, top_k=3)))

    def test_results_are_still_one_entry_per_document(self):
        retriever = build_chunked(chunk_words=4)
        ranking = retriever.search("alpha", top_k=10)
        assert len(ranking) == len(LONG_CORPUS)
        assert len({doc for doc, _ in ranking}) == len(LONG_CORPUS)

    def test_describe_records_the_chunk_settings(self):
        described = build_chunked(chunk_words=4, chunk_overlap=1).describe()
        assert described["chunk_words"] == 4
        assert described["chunk_overlap"] == 1
        assert described["num_chunks"] >= len(LONG_CORPUS)


class TestChunkCacheKeys:
    def test_chunked_and_unchunked_use_different_cache_entries(self, tmp_path):
        plain = DenseRetriever(cache_dir=tmp_path)
        chunked = DenseRetriever(cache_dir=tmp_path, chunk_words=4)
        assert plain._cache_path(LONG_CORPUS) != chunked._cache_path(LONG_CORPUS)

    def test_different_chunk_sizes_use_different_cache_entries(self, tmp_path):
        a = DenseRetriever(cache_dir=tmp_path, chunk_words=4)
        b = DenseRetriever(cache_dir=tmp_path, chunk_words=8)
        c = DenseRetriever(cache_dir=tmp_path, chunk_words=4, chunk_overlap=2)
        assert (
            len(
                {a._cache_path(LONG_CORPUS), b._cache_path(LONG_CORPUS), c._cache_path(LONG_CORPUS)}
            )
            == 3
        )
