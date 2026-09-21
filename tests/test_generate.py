"""Answer generation with citations.

The model is not loaded here. What is tested is the part that can be checked without a
ground-truth answer, which is also the part that matters most: whether a citation resolves
to a document that was actually retrieved.

BEIR ships no reference answers, so answer *quality* is not measurable on this benchmark
and nothing here pretends otherwise. A fabricated citation, though, is catchable against
the retrieved set alone — and an answer with a fabricated citation is more dangerous than
an uncited one, because it looks more trustworthy and is not.
"""

import pytest

from groundwork.generate import answer_from_documents, format_sources, parse_citations

CORPUS = {
    "d1": {"title": "Statins and myopathy", "text": "Muscle pain is a known adverse effect."},
    "d2": {"title": "Vitamin D", "text": "Serum levels vary with sun exposure."},
    "d3": {"title": "Unrelated", "text": "Antarctic ice mass balance."},
}


class TestFormatSources:
    def test_sources_are_numbered_from_one(self):
        """Citations are written and read one-based. An off-by-one here would attribute
        every claim to the wrong document while looking completely normal."""
        formatted = format_sources(["d1", "d2"], CORPUS)
        assert formatted.startswith("[1] Statins and myopathy")
        assert "[2] Vitamin D" in formatted

    def test_numbering_follows_rank_order_not_document_id(self):
        formatted = format_sources(["d2", "d1"], CORPUS)
        assert formatted.startswith("[2] Statins") is False
        assert formatted.startswith("[1] Vitamin D")

    def test_long_documents_are_truncated(self):
        corpus = {"d1": {"title": "T", "text": "x" * 5000}}
        assert len(format_sources(["d1"], corpus, max_chars=100)) < 200

    def test_a_missing_document_does_not_crash_the_prompt(self):
        """A retrieved id absent from the corpus is a wiring bug, but producing a broken
        prompt is better than producing no answer and losing the rest."""
        assert "[1]" in format_sources(["nonexistent"], CORPUS)


class TestParseCitations:
    def test_valid_citations_are_found_in_order(self):
        valid, invalid = parse_citations("Statins cause myopathy [1] and [2] agrees.", 3)
        assert valid == [1, 2]
        assert invalid == []

    def test_duplicate_citations_are_reported_once(self):
        valid, _ = parse_citations("As [1] says, and again [1].", 3)
        assert valid == [1]

    def test_a_citation_beyond_the_retrieved_set_is_invalid(self):
        """The failure this module exists to catch: the model invents a source number."""
        valid, invalid = parse_citations("This is established [7].", 3)
        assert valid == []
        assert invalid == [7]

    def test_zero_is_invalid_because_numbering_starts_at_one(self):
        _, invalid = parse_citations("See [0].", 3)
        assert invalid == [0]

    def test_valid_and_invalid_are_separated(self):
        valid, invalid = parse_citations("Both [1] and [9] say so.", 2)
        assert valid == [1]
        assert invalid == [9]

    def test_an_answer_with_no_citations_yields_nothing(self):
        valid, invalid = parse_citations("Statins cause muscle pain.", 3)
        assert valid == []
        assert invalid == []

    def test_bracketed_text_that_is_not_a_number_is_ignored(self):
        valid, invalid = parse_citations("See [source] and [1].", 3)
        assert valid == [1]
        assert invalid == []


class TestAnswerGuards:
    def test_answering_with_no_documents_fails_loudly(self):
        """An answer generated from no sources is the model's prior wearing a citation
        format. It must not be producible by accident."""
        with pytest.raises(ValueError, match="no documents"):
            answer_from_documents("does vitamin D help?", [], CORPUS)
