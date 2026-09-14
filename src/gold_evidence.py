"""Align QASPER's human-marked evidence to character spans in our documents.

WHY THIS EXISTS

Answer F1 tells you whether the final answer was right. It cannot tell you whether
compression removed the evidence the answer needed — a system can drop the crucial
passage and still produce a plausible answer, and a system can keep it and still answer
badly because the generator is weak. Those are different failures with different fixes,
and the project's central safety claim is about the first one.

QASPER annotators marked which passages support each answer. Aligned to spans, they give
the measurement the restoration mechanism has never actually been tested against:

    was the answer's evidence dropped, and if so, did restoration fire?

**LEAKAGE RULE.** Nothing in the pipeline may import this module. Gold evidence is
evaluation-only. The single deliberate exception is the oracle arm, which is built in
`scripts/run_eval.py` and reported as an upper bound, never as a system configuration.
`tests/test_gold_evidence.py` enforces the rule.

ALIGNMENT IS IMPERFECT AND SAYS SO

Measured over all 756 evidence strings in the 110-paper corpus:

    verbatim            78.8%
    whitespace/case      2.9%
    markers stripped     ~6%   (evidence has "Table ." where the document has "Table TABREF3")
    table_or_figure     12.3%  <- not a failure: genuinely absent from our corpus
    unmatched            ~1%

The 12.3% matters and must not be quietly dropped. QASPER evidence sometimes points at a
table or figure caption, and `fetch_qasper.paper_text()` ingests only title, abstract and
body paragraphs. Those questions are *unanswerable from the corpus we built*, which is a
property of our ingestion rather than of the compressor — and it inflates the false-abstain
rate through no fault of the pipeline. Reported separately for that reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Same markers fetch_qasper.py strips from answers. Substituted with equal-length runs of
# spaces rather than deleted, so every character index into the document survives the
# rewrite and a match found in the stripped text maps straight back to the original.
_MARKERS = re.compile(r"\b(?:BIBREF|TABREF|FIGREF|SECREF|FLOAT|UNKREF)\d*\b")

# QASPER marks table and figure evidence with this sentinel.
_FLOAT_PREFIX = "FLOAT SELECTED"


@dataclass(frozen=True)
class GoldSpan:
    """A human-marked supporting passage, as offsets into the document."""

    start: int
    end: int
    method: str  # verbatim | normalised | markers_stripped

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class Alignment:
    spans: list[GoldSpan] = field(default_factory=list)
    n_evidence: int = 0
    n_table_or_figure: int = 0
    n_unmatched: int = 0

    @property
    def n_aligned(self) -> int:
        return len(self.spans)

    @property
    def aligned_fraction(self) -> float:
        return self.n_aligned / self.n_evidence if self.n_evidence else 0.0

    @property
    def usable(self) -> bool:
        """Whether evidence-level metrics can be computed for this question at all.

        A question whose evidence is entirely table captions is not a compression failure
        and must be excluded from evidence recall rather than scored as zero.
        """
        return self.n_aligned > 0


def _mask_markers(text: str) -> str:
    """Blank citation markers while preserving every character offset."""
    return _MARKERS.sub(lambda m: " " * len(m.group()), text)


def _normalise(text: str) -> tuple[str, list[int]]:
    """Lowercase and collapse whitespace, keeping a map back to original offsets.

    `index[i]` is the original position of normalised character `i`, so a match at
    `[a, b)` in normalised space maps to `[index[a], index[b - 1] + 1)` in the document.
    """
    chars: list[str] = []
    index: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            chars.append(" ")
            index.append(i)
            prev_space = True
        else:
            chars.append(ch.lower())
            index.append(i)
            prev_space = False
    return "".join(chars), index


def align(evidence: list[str], document: str) -> Alignment:
    """Locate each evidence string in the document, by escalating leniency.

    Ordered strictly cheapest-and-most-exact first, so a passage that matches verbatim is
    never attributed to a fuzzier rule that happened to also match somewhere else.
    """
    result = Alignment()

    normalised_doc, doc_index = _normalise(document)
    masked_doc = _mask_markers(document)
    normalised_masked, masked_index = _normalise(masked_doc)

    for raw in evidence:
        if not raw or not raw.strip():
            continue
        result.n_evidence += 1
        needle = raw.strip()

        if needle.startswith(_FLOAT_PREFIX):
            result.n_table_or_figure += 1
            continue

        # 1. Exact.
        position = document.find(needle)
        if position >= 0:
            result.spans.append(
                GoldSpan(position, position + len(needle), "verbatim")
            )
            continue

        # 2. Whitespace and case.
        normalised_needle, _ = _normalise(needle)
        normalised_needle = normalised_needle.strip()
        span = _find_normalised(normalised_needle, normalised_doc, doc_index, "normalised")
        if span:
            result.spans.append(span)
            continue

        # 3. Citation markers, blanked on both sides. Catches evidence recorded as
        #    "shown in Table ." against a document reading "shown in Table TABREF19".
        masked_needle, _ = _normalise(_mask_markers(needle))
        masked_needle = masked_needle.strip()
        span = _find_normalised(
            masked_needle, normalised_masked, masked_index, "markers_stripped"
        )
        if span:
            result.spans.append(span)
            continue

        result.n_unmatched += 1

    return result


def _find_normalised(
    needle: str, haystack: str, index: list[int], method: str
) -> GoldSpan | None:
    if not needle:
        return None
    at = haystack.find(needle)
    if at < 0:
        return None
    end = at + len(needle) - 1
    if end >= len(index):
        return None
    return GoldSpan(index[at], index[end] + 1, method)


def covered_characters(gold: list[GoldSpan], spans) -> int:
    """Gold-evidence characters covered by `spans`, counted without double-counting.

    Overlap is measured in characters rather than in whole units because an evidence unit
    may cover part of a marked passage — a sentence chunk inside a marked paragraph is a
    partial hit, and rounding it to 0 or 1 would misreport recall in both directions.
    """
    intervals = sorted(
        (max(g.start, s.start), min(g.end, s.end))
        for g in gold
        for s in spans
        if min(g.end, s.end) > max(g.start, s.start)
    )
    return _merged_length(intervals)


def _merged_length(intervals: list[tuple[int, int]]) -> int:
    """Total length of a set of intervals, counting overlapped regions once."""
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return total + (current_end - current_start)


def gold_character_total(gold: list[GoldSpan]) -> int:
    """Total gold characters, merging overlapping marked passages.

    Annotators frequently mark the same paragraph for several answers to one question;
    without merging, a unit covering it would score recall above 1.0.
    """
    return _merged_length([(g.start, g.end) for g in gold])


def recall(gold: list[GoldSpan], units, doc_id: str) -> float:
    """Fraction of marked evidence characters present in `units`.

    Units from other documents are ignored: in a single-paper benchmark they cannot
    contribute support, and counting them would let cross-document noise inflate recall.
    """
    total = gold_character_total(gold)
    if total == 0:
        return 0.0
    spans = [u.span for u in units if u.span.doc_id == doc_id]
    return covered_characters(gold, spans) / total
