"""Gold-evidence alignment, and the leakage rule that keeps it honest.

Aligning human-marked evidence to spans is what lets the evaluation ask whether
compression removed the passage the answer needed — the question answer-F1 cannot answer,
and the one the restoration mechanism exists to handle.

The alignment is deliberately lenient in stages, so these tests pin two things: that each
stage does what it claims, and that leniency never silently invents a match.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.gold_evidence import (
    Alignment,
    GoldSpan,
    align,
    covered_characters,
    gold_character_total,
    recall,
)
from src.types import Span

DOC = (
    "Neural Machine Translation Survey\n\n"
    "We evaluate on the Europarl corpus and the MultiUN dataset. "
    "Our model uses a transformer encoder with eight layers BIBREF19. "
    "Results are shown in Table TABREF12 for all language pairs.\n\n"
    "Training took three days on four GPUs."
)


class TestExactAlignment:
    def test_verbatim_passage_is_located(self):
        result = align(["We evaluate on the Europarl corpus and the MultiUN dataset."], DOC)
        assert result.n_aligned == 1
        span = result.spans[0]
        assert span.method == "verbatim"
        assert DOC[span.start : span.end].startswith("We evaluate on the Europarl")

    def test_offsets_round_trip_through_the_document(self):
        """The whole point of a span: it must re-resolve to the text it came from."""
        passage = "Training took three days on four GPUs."
        result = align([passage], DOC)
        span = result.spans[0]
        assert DOC[span.start : span.end] == passage


class TestLenientAlignment:
    def test_whitespace_and_case_differences_still_align(self):
        result = align(["we  evaluate on the EUROPARL corpus"], DOC)
        assert result.n_aligned == 1
        assert result.spans[0].method == "normalised"

    def test_stripped_citation_markers_still_align(self):
        """QASPER records evidence with markers removed; the document keeps them."""
        result = align(["Results are shown in Table  for all language pairs."], DOC)
        assert result.n_aligned == 1
        assert result.spans[0].method == "markers_stripped"

    def test_marker_masking_preserves_offsets(self):
        """Markers are blanked, not deleted — otherwise every later offset shifts."""
        result = align(["Results are shown in Table  for all language pairs."], DOC)
        span = result.spans[0]
        assert "Results are shown in Table" in DOC[span.start : span.end]
        assert "language pairs" in DOC[span.start : span.end]

    def test_unrelated_text_is_not_matched(self):
        """Leniency must not manufacture alignments."""
        result = align(["The authors propose a novel reinforcement learning objective."], DOC)
        assert result.n_aligned == 0
        assert result.n_unmatched == 1


class TestTableAndFigureEvidence:
    def test_float_evidence_is_counted_separately_not_as_failure(self):
        """Table captions are absent from a text-only corpus. Scoring them as dropped
        evidence would blame the compressor for an ingestion limit."""
        result = align(["FLOAT SELECTED: Table 2: Main results."], DOC)
        assert result.n_table_or_figure == 1
        assert result.n_unmatched == 0
        assert result.n_aligned == 0

    def test_question_with_only_float_evidence_is_unusable(self):
        result = align(["FLOAT SELECTED: Table 1: Data."], DOC)
        assert not result.usable


class TestCoverage:
    def test_total_merges_overlapping_gold_passages(self):
        """Annotators often mark the same paragraph twice; recall must not exceed 1."""
        gold = [GoldSpan(0, 100, "verbatim"), GoldSpan(50, 150, "verbatim")]
        assert gold_character_total(gold) == 150

    def test_partial_overlap_is_counted_in_characters(self):
        gold = [GoldSpan(100, 200, "verbatim")]
        spans = [Span(doc_id="d", start=150, end=250)]
        assert covered_characters(gold, spans) == 50

    def test_disjoint_units_do_not_double_count(self):
        gold = [GoldSpan(0, 100, "verbatim")]
        spans = [Span(doc_id="d", start=0, end=60), Span(doc_id="d", start=40, end=80)]
        assert covered_characters(gold, spans) == 80

    def test_recall_ignores_units_from_other_documents(self):
        """Cross-document text cannot support a single-paper question, and counting it
        would let retrieval noise inflate evidence recall."""

        class FakeUnit:
            def __init__(self, span):
                self.span = span

        gold = [GoldSpan(0, 100, "verbatim")]
        units = [FakeUnit(Span(doc_id="other", start=0, end=100))]
        assert recall(gold, units, doc_id="target") == 0.0

    def test_recall_of_fully_retained_evidence_is_one(self):
        class FakeUnit:
            def __init__(self, span):
                self.span = span

        gold = [GoldSpan(10, 60, "verbatim")]
        units = [FakeUnit(Span(doc_id="paper", start=0, end=100))]
        assert recall(gold, units, doc_id="paper") == pytest.approx(1.0)

    def test_no_gold_evidence_scores_zero_not_an_error(self):
        assert recall([], [], doc_id="paper") == 0.0


class TestAlignmentReporting:
    def test_aligned_fraction_reports_honestly(self):
        result = Alignment(
            spans=[GoldSpan(0, 5, "verbatim")],
            n_evidence=4,
            n_table_or_figure=2,
            n_unmatched=1,
        )
        assert result.aligned_fraction == pytest.approx(0.25)


class TestLeakageRule:
    """Gold evidence is evaluation-only. If the pipeline could reach it, every result
    measured with it would be suspect — and the failure would be invisible."""

    def test_no_pipeline_module_imports_gold_evidence(self):
        src = Path(__file__).resolve().parent.parent / "src"
        offenders = [
            path.name
            for path in src.glob("*.py")
            if path.name != "gold_evidence.py"
            and "gold_evidence" in path.read_text(encoding="utf-8")
        ]
        assert offenders == [], (
            f"{offenders} import gold_evidence; evaluation-only data must never reach "
            "the pipeline. The oracle arm belongs in scripts/run_eval.py."
        )
