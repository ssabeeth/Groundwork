"""Query and document expansion.

Generation itself needs a model download, so it is not exercised here — CI runs offline.
What is tested is everything around it: how generated text is combined with the original,
and the failure modes that would quietly produce a wrong run rather than an error.

The combination rules are where the bugs hide. An expander that dropped the original query
terms, or that mutated the corpus it was handed, would still produce plausible-looking
numbers, and the only sign would be a result that did not reproduce.
"""

import pytest

from groundwork.retrieval.expansion import (
    DEFAULT_QUERY_WEIGHT,
    expand_documents,
    expand_queries,
    generate_expansions,
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
