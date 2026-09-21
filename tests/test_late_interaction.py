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

from groundwork.retrieval.late_interaction import ColbertRetriever, prepare_ids

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


class TestPrepareIds:
    """ColBERT's input conventions, which are where it goes silently wrong.

    This function exists as a pure array operation so these can be checked without torch
    or a downloaded checkpoint — CI has neither. The first implementation wrote the
    marker as text, which this tokenizer lowercases and splits into '[', 'unused', '##0',
    ']'. That produced a model that ran, retrieved, and missed its published nDCG@10 by
    0.0859 without erroring once.
    """

    # [CLS] placeholder tok1 tok2 [SEP] [PAD] [PAD], with 101/102/0/103 as BERT's ids.
    IDS = np.array([[101, 1012, 7592, 2088, 102, 0, 0]], dtype=np.int64)
    MASK = np.array([[1, 1, 1, 1, 1, 0, 0]], dtype=np.int64)
    MARKER, PAD, MASK_ID = 1, 0, 103

    def prepared(self, is_query, skiplist=frozenset()):
        return prepare_ids(
            self.IDS, self.MASK, self.MARKER, self.PAD, self.MASK_ID, is_query, skiplist
        )

    def test_the_marker_occupies_exactly_one_position(self):
        """The bug this function was extracted for. One token, at position 1, not four."""
        ids, _ = self.prepared(is_query=True)
        assert ids[0][1] == self.MARKER
        assert list(ids[0]).count(self.MARKER) == 1

    def test_the_marker_replaces_the_placeholder_and_keeps_cls_first(self):
        ids, _ = self.prepared(is_query=False)
        assert ids[0][0] == 101
        assert ids[0][2:5].tolist() == [7592, 2088, 102]

    def test_query_padding_becomes_mask_not_pad(self):
        ids, _ = self.prepared(is_query=True)
        assert ids[0][-2:].tolist() == [self.MASK_ID, self.MASK_ID]

    def test_a_query_keeps_every_position_including_the_augmented_ones(self):
        """Query augmentation is the method, not padding to be discarded: all positions
        contribute a vector."""
        _, keep = self.prepared(is_query=True)
        assert keep.all()
        assert keep.shape == self.IDS.shape

    def test_a_document_drops_its_padding(self):
        _, keep = self.prepared(is_query=False)
        assert keep[0].tolist() == [True, True, True, True, True, False, False]

    def test_a_document_drops_skiplisted_tokens(self):
        # 7592 stands in for a punctuation id here.
        _, keep = self.prepared(is_query=False, skiplist=frozenset({7592}))
        assert keep[0].tolist() == [True, True, False, True, True, False, False]

    def test_document_padding_is_not_turned_into_mask(self):
        """Only queries are augmented. Doing it to documents would add vectors for
        tokens the document does not contain."""
        ids, _ = self.prepared(is_query=False)
        assert ids[0][-2:].tolist() == [self.PAD, self.PAD]

    def test_the_input_array_is_not_modified(self):
        before = self.IDS.copy()
        self.prepared(is_query=True)
        assert (before == self.IDS).all()
