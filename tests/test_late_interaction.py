"""Late-interaction retrieval.

The model is not loaded here — CI runs offline — so what is tested is MaxSim itself, with
token vectors chosen so the arithmetic can be done on paper.

MaxSim is::

    S(q, d) = sum over query tokens i of  max over document tokens j of  q_i . d_j

Every expected value below is worked out from that definition, in the comments. The
blocking in :meth:`_maxsim` exists to bound memory and must not change a score, so the
same fixtures are run at a block size that splits the corpus.
"""

import numpy as np
import pytest

from groundwork.retrieval.late_interaction import ColbertRetriever

# Two orthogonal unit vectors, so every dot product below is 0, 1 or -1.
RIGHT = [1.0, 0.0]
UP = [0.0, 1.0]
DOWN = [0.0, -1.0]

# Document a: tokens RIGHT, DOWN.  Document b: token UP.  Document c: tokens RIGHT, UP.
TOKENS = np.array([RIGHT, DOWN, UP, RIGHT, UP], dtype=np.float32)
OFFSETS = np.array([0, 2, 3], dtype=np.int64)
QUERY = np.array([RIGHT, UP], dtype=np.float32)

# Worked out from the definition:
#   query token RIGHT: doc a max(1, 0) = 1 | doc b max(0) = 0 | doc c max(1, 0) = 1
#   query token UP:    doc a max(0, -1) = 0 | doc b max(1) = 1 | doc c max(0, 1) = 1
#   sums:              a = 1 + 0 = 1        | b = 0 + 1 = 1    | c = 1 + 1 = 2
EXPECTED = [1.0, 1.0, 2.0]


def built(score_block: int = 8192) -> ColbertRetriever:
    retriever = ColbertRetriever(dim=2, score_block=score_block)
    retriever.doc_ids = ["a", "b", "c"]
    retriever._embeddings = TOKENS
    retriever._offsets = OFFSETS
    return retriever


class TestConstructorGuards:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"dim": 0}, "dim"),
            ({"query_length": 0}, "query_length"),
            ({"doc_length": -5}, "doc_length"),
            ({"batch_size": 0}, "batch_size"),
            ({"score_block": 0}, "score_block"),
        ],
    )
    def test_nonsense_settings_fail_loudly(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            ColbertRetriever(**kwargs)

    def test_the_defaults_are_the_checkpoints_own_settings(self):
        """These come from colbertv2.0's artifact.metadata. Changing one silently would
        produce a model that retrieves badly rather than failing."""
        retriever = ColbertRetriever()
        assert (retriever.dim, retriever.query_length, retriever.doc_length) == (128, 32, 180)
        assert retriever.mask_punctuation is True


class TestMaxSim:
    def test_maxsim_matches_the_definition(self):
        assert built()._maxsim(QUERY).tolist() == pytest.approx(EXPECTED)

    def test_blocking_does_not_change_any_score(self):
        """score_block bounds memory during retrieval. If it changed a score it would be
        a correctness knob wearing the clothes of a performance one."""
        for block in (1, 2, 3, 8192):
            assert built(score_block=block)._maxsim(QUERY).tolist() == pytest.approx(EXPECTED)

    def test_a_document_is_scored_by_its_best_token_not_its_average(self):
        """The whole point of late interaction. Document d has one perfectly matching
        token and one that actively disagrees; a mean would cancel them to zero."""
        retriever = ColbertRetriever(dim=2)
        retriever.doc_ids = ["d"]
        retriever._embeddings = np.array([RIGHT, DOWN], dtype=np.float32)
        retriever._offsets = np.array([0], dtype=np.int64)
        # One query token RIGHT: max(RIGHT.RIGHT, RIGHT.DOWN) = max(1, 0) = 1.
        assert retriever._maxsim(np.array([RIGHT], dtype=np.float32)).tolist() == pytest.approx(
            [1.0]
        )

    def test_each_query_token_contributes_independently(self):
        """Scores are a sum over query tokens, so repeating a query token doubles the
        score. Asserting this pins the sum, which is what separates MaxSim from a mean
        over query tokens."""
        one = built()._maxsim(np.array([RIGHT], dtype=np.float32))
        twice = built()._maxsim(np.array([RIGHT, RIGHT], dtype=np.float32))
        assert twice.tolist() == pytest.approx((one * 2).tolist())

    def test_a_query_matching_nothing_scores_at_most_zero(self):
        # DOWN against doc c's tokens RIGHT and UP: max(0, -1) = 0.
        scores = built()._maxsim(np.array([DOWN], dtype=np.float32))
        assert scores[2] == pytest.approx(0.0)

    def test_the_last_document_reaches_the_end_of_the_token_array(self):
        """Off-by-one insurance on the offsets. Document c owns the final two vectors, so
        dropping the tail would cost it the UP token and change its score."""
        assert built()._maxsim(np.array([UP], dtype=np.float32))[2] == pytest.approx(1.0)


class TestRetrieveGuards:
    def test_retrieving_before_indexing_fails_loudly(self):
        with pytest.raises(RuntimeError, match="index"):
            ColbertRetriever().retrieve({"q": "anything"})


class TestDescribe:
    def test_the_settings_that_change_results_are_recorded(self):
        described = ColbertRetriever(doc_length=220, mask_punctuation=False).describe()
        assert described["method"] == "colbert"
        assert described["doc_length"] == 220
        assert described["mask_punctuation"] is False
        assert described["similarity"] == "maxsim-cosine"

    def test_the_record_says_the_search_was_exhaustive(self):
        """Published ColBERT numbers come with an approximate serving stack. A record that
        did not say which was measured could not be compared with them."""
        assert ColbertRetriever().describe()["exhaustive"] is True

    def test_the_record_is_json_safe(self):
        import json

        json.dumps(ColbertRetriever().describe())
