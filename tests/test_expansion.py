"""Query and document expansion.

Generation itself needs a model download, so it is not exercised here — CI runs offline.
What is tested is everything around it: how generated text is combined with the original,
and the failure modes that would quietly produce a wrong run rather than an error.

The combination rules are where the bugs hide. An expander that dropped the original query
terms, or that mutated the corpus it was handed, would still produce plausible-looking
numbers, and the only sign would be a result that did not reproduce.
"""

import pytest

from groundwork.retrieval import LUCENE_ENGLISH_STOPWORDS, Tokenizer
from groundwork.retrieval.expansion import (
    DEFAULT_QUERY_WEIGHT,
    added_terms,
    expand_documents,
    expand_queries,
    generate_expansions,
    summarise_expansions,
)

QUERIES = {"q1": "statin muscle pain", "q2": "vitamin d deficiency"}
CORPUS = {
    "d1": {"title": "Statins", "text": "Myopathy in patients taking statins."},
    "d2": {"title": "Vitamin D", "text": "Serum levels and supplementation."},
}


class TestQueryExpansion:
    def test_query_is_repeated_weight_times_before_the_passage(self):
        # weight=2 means the query appears twice, then the generated passage once.
        expanded = expand_queries(QUERIES, {"q1": "Statins cause myopathy."}, weight=2)
        assert expanded["q1"] == "statin muscle pain statin muscle pain Statins cause myopathy."

    def test_weight_zero_retrieves_on_the_generated_passage_alone(self):
        """HyDE in its pure form discards the query. It must be reachable, because it is
        the form the method was published in."""
        expanded = expand_queries(QUERIES, {"q1": "Statins cause myopathy."}, weight=0)
        assert expanded["q1"] == "Statins cause myopathy."

    def test_weight_one_is_plain_concatenation(self):
        expanded = expand_queries(QUERIES, {"q1": "Statins cause myopathy."}, weight=1)
        assert expanded["q1"] == "statin muscle pain Statins cause myopathy."

    def test_a_query_with_no_expansion_is_passed_through_unchanged(self):
        """A generation failure must degrade to the unexpanded query, not to an empty
        one, or a single bad generation silently scores zero for that query."""
        expanded = expand_queries(QUERIES, {"q1": "something"}, weight=2)
        assert expanded["q2"] == "vitamin d deficiency"

    def test_a_blank_expansion_is_treated_as_no_expansion(self):
        expanded = expand_queries(QUERIES, {"q1": "   "}, weight=3)
        assert expanded["q1"] == "statin muscle pain"

    def test_every_query_survives_expansion(self):
        """The scorer averages over the queries present, so losing one inflates the mean."""
        expanded = expand_queries(QUERIES, {"q1": "a passage"}, weight=DEFAULT_QUERY_WEIGHT)
        assert set(expanded) == set(QUERIES)

    def test_negative_weight_fails_loudly(self):
        with pytest.raises(ValueError, match="non-negative"):
            expand_queries(QUERIES, {}, weight=-1)

    def test_expansion_does_not_mutate_the_queries_it_was_given(self):
        original = dict(QUERIES)
        expand_queries(QUERIES, {"q1": "a passage"}, weight=2)
        assert original == QUERIES


class TestDocumentExpansion:
    def test_generated_queries_are_appended_to_the_body(self):
        grown = expand_documents(CORPUS, {"d1": ["does statin cause pain", "statin side effects"]})
        assert grown["d1"]["text"] == (
            "Myopathy in patients taking statins. does statin cause pain statin side effects"
        )

    def test_the_title_is_left_alone(self):
        """Under multi-field indexing the title is a short, high-weight field. Padding it
        with generated text would change what a title match means."""
        grown = expand_documents(CORPUS, {"d1": ["does statin cause pain"]})
        assert grown["d1"]["title"] == "Statins"

    def test_a_document_with_no_expansion_is_passed_through_unchanged(self):
        grown = expand_documents(CORPUS, {"d1": ["something"]})
        assert grown["d2"] == CORPUS["d2"]

    def test_blank_generations_are_dropped_rather_than_padding_the_document(self):
        """Empty strings would add whitespace and, worse, inflate nothing while looking
        like an expansion had been applied."""
        grown = expand_documents(CORPUS, {"d1": ["", "   ", "real query"]})
        assert grown["d1"]["text"] == "Myopathy in patients taking statins. real query"

    def test_expansion_does_not_mutate_the_corpus_it_was_given(self):
        """The same corpus object is reused to build an unexpanded baseline index in the
        same process, so mutating it would make the baseline silently wrong."""
        before = CORPUS["d1"]["text"]
        expand_documents(CORPUS, {"d1": ["a generated query"]})
        assert CORPUS["d1"]["text"] == before

    def test_every_document_survives_expansion(self):
        grown = expand_documents(CORPUS, {"d1": ["q"]})
        assert set(grown) == set(CORPUS)

    def test_a_chosen_field_other_than_text_can_be_expanded(self):
        grown = expand_documents(CORPUS, {"d1": ["extra"]}, field="title")
        assert grown["d1"]["title"] == "Statins extra"
        assert grown["d1"]["text"] == CORPUS["d1"]["text"]


class TestGenerationGuards:
    def test_several_greedy_generations_fail_loudly_instead_of_duplicating(self):
        """Greedy decoding returns the same string every time. Asking for five and
        receiving five copies would look like a working doc2query and expand nothing."""
        with pytest.raises(ValueError, match="do_sample=True"):
            generate_expansions({"d1": "text"}, "any-model", num_return_sequences=5)


# Tokenisation without stemming, so the arithmetic in these tests is readable: every
# expected term below is the literal lowercased word. Stemming gets its own test.
PLAIN = Tokenizer(stopwords=LUCENE_ENGLISH_STOPWORDS, stem=False)


class TestAddedTerms:
    """What an expansion contributes that the query did not already have.

    This is the measurement that separates a real expansion from a fluent restatement.
    Only terms absent from the original can change which documents match; repeating terms
    the query already has changes term frequency, not reachability.
    """

    def test_a_pure_restatement_adds_nothing(self):
        assert added_terms("vitamin d deficiency", "vitamin d deficiency", PLAIN) == set()

    def test_new_words_are_returned(self):
        # "in" is a Lucene stopword, so the additions are statins, cause, myopathy.
        assert added_terms("statin muscle pain", "statins cause myopathy in", PLAIN) == {
            "statins",
            "cause",
            "myopathy",
        }

    def test_terms_already_in_the_query_are_not_counted_however_often_repeated(self):
        assert added_terms("aspirin dose", "aspirin aspirin dose aspirin", PLAIN) == set()

    def test_stopwords_in_the_generated_text_are_not_additions(self):
        """A generated passage padded with function words adds no retrievable vocabulary."""
        assert added_terms("aspirin", "the aspirin and the dose", PLAIN) == {"dose"}

    def test_the_comparison_happens_after_stemming(self):
        """'cells' cannot reach a document 'cell' does not, so it is not an addition.

        The comparison has to happen in the index's term space or the statistic overstates
        what the expansion bought.
        """
        pytest.importorskip("snowballstemmer")
        stemming = Tokenizer(stopwords=LUCENE_ENGLISH_STOPWORDS, stem=True)
        assert added_terms("cell", "cells", stemming) == set()
        assert added_terms("cell", "cells", PLAIN) == {"cells"}

    def test_an_empty_expansion_adds_nothing(self):
        assert added_terms("aspirin", "", PLAIN) == set()


class TestSummariseExpansions:
    """Worked by hand against the four inputs below; see the arithmetic in comments."""

    ORIGINALS = {
        "q1": "vitamin d deficiency",
        "q2": "statin muscle pain",
        "q3": "aspirin",
        "q4": "cancer screening",
    }
    EXPANSIONS = {
        "q1": ["vitamin d deficiency"],              # restatement: 3 words, 0 new
        "q2": ["statins cause myopathy in adults"],  # 5 words, 4 new of 4 distinct
        "q3": [],                                    # nothing generated
        "q4": ["screening reduces mortality"],       # 3 words, 2 new of 3 distinct
    }

    def summary(self):
        return summarise_expansions(self.ORIGINALS, self.EXPANSIONS, PLAIN)

    def test_inputs_with_no_generated_text_are_counted_separately(self):
        """An id that generated nothing is not an expansion that added nothing; conflating
        them would hide a broken generation run inside a weak-expansion statistic."""
        summary = self.summary()
        assert summary.num_inputs == 4
        assert summary.num_empty == 1

    def test_restatements_are_counted(self):
        # q1 only: q3 generated nothing and is excluded rather than counted here.
        assert self.summary().num_adding_nothing == 1

    def test_generated_word_counts(self):
        # Word counts [3, 5, 3] -> sorted [3, 3, 5]; mean 11/3 = 3.6667.
        words = self.summary().generated_words
        assert words["median"] == 3.0
        assert words["mean"] == 3.6667
        assert words["min"] == 3.0
        assert words["max"] == 5.0

    def test_added_term_counts(self):
        # Added [0, 4, 2] -> sorted [0, 2, 4]; median 2, mean 6/3 = 2.0.
        added = self.summary().added_terms
        assert added["median"] == 2.0
        assert added["mean"] == 2.0

    def test_added_term_fraction_is_the_mean_of_per_input_fractions(self):
        # q1 0/3 = 0.0, q2 4/4 = 1.0, q4 2/3 = 0.6667; (0 + 1 + 2/3)/3 = 0.5556.
        assert self.summary().added_term_fraction == 0.5556

    def test_several_generations_for_one_input_are_pooled(self):
        """doc2query produces many short queries per document and all are appended, so the
        statistic must describe their union rather than the first one."""
        summary = summarise_expansions(
            {"d1": "cell growth"},
            {"d1": ["what regulates division", "how do tumours spread"]},
            PLAIN,
        )
        # Union of new terms: regulates, division, how, do, tumours, spread = 6
        # ("what" is not a Lucene stopword, so it counts too -> 7).
        assert summary.added_terms["median"] == 7.0
        assert summary.num_adding_nothing == 0

    def test_an_input_missing_from_the_expansions_counts_as_empty(self):
        summary = summarise_expansions({"q1": "aspirin"}, {}, PLAIN)
        assert summary.num_empty == 1
        assert summary.added_term_fraction == 0.0

    def test_the_record_is_json_safe(self):
        import json

        json.dumps(self.summary().describe())
