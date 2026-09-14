"""Pipeline stage 6.1 — evidence restoration.

Reverses compression when the compressed evidence turned out insufficient. Because
compression is extractive (units are dropped, never rewritten) restoration is exact
rather than approximate.

Two recovery routes exist, and they must agree:

* `restore` reinstates the retained `dropped` units — the fast path used at runtime.
* `restore_from_corpus` ignores those retained copies entirely and re-reads each unit's
  text from its `Span`. This is the route that proves reversibility is a real property
  of the span index rather than a side effect of keeping a backup in memory.

tests/test_compression.py asserts both produce identical evidence.
"""

from __future__ import annotations

from .types import ClaimEvidence, CompressedEvidence, Corpus, EvidenceUnit


def _ordered(units: list[EvidenceUnit]) -> list[EvidenceUnit]:
    """Stable ordering by descending retrieval score, then by span position.

    Restoration must be deterministic: the same compressed input always yields the same
    restored evidence, regardless of the order units happened to be dropped in.
    """
    return sorted(
        units,
        key=lambda u: (-u.rank_score, u.span.doc_id, u.span.start),
    )


def restore(compressed: CompressedEvidence) -> ClaimEvidence:
    """Reinstate dropped evidence, returning the claim's full evidence set."""
    units = _ordered(list(compressed.kept) + list(compressed.dropped))
    return ClaimEvidence(
        claim=compressed.claim,
        units=units,
        token_count=sum(u.token_count for u in units),
        uncertainty=compressed.uncertainty,
    )


def restore_from_corpus(
    compressed: CompressedEvidence, corpus: Corpus
) -> ClaimEvidence:
    """Rebuild evidence text from spans alone, ignoring any retained copies.

    If this diverges from `restore`, the span index is corrupt and the reversibility
    guarantee no longer holds — which is why the round-trip test checks it.
    """
    rebuilt: list[EvidenceUnit] = []
    for unit in list(compressed.kept) + list(compressed.dropped):
        rebuilt.append(
            EvidenceUnit(
                span=unit.span,
                text=unit.span.resolve(corpus),
                retrieval_score=unit.retrieval_score,
                uncertainty=unit.uncertainty,
                token_count=unit.token_count,
            )
        )

    units = _ordered(rebuilt)
    return ClaimEvidence(
        claim=compressed.claim,
        units=units,
        token_count=sum(u.token_count for u in units),
        uncertainty=compressed.uncertainty,
    )


def restore_partial(compressed: CompressedEvidence, n: int) -> ClaimEvidence:
    """Reinstate only the `n` best-ranked dropped units.

    Full restoration is all-or-nothing: the moment anything looks wrong the claim gives
    back its entire saving, including the units that were correctly dropped. Measured on
    QASPER, restoration fires on a quarter to a half of questions, so that is a large
    share of the budget surrendered on suspicion.

    Restoring a few units at a time lets the check run again against a cheaper
    intermediate, and it only pays for the full set when the cheaper one still fails.
    Ordering is by retrieval score, so "next best" means the strongest evidence
    compression chose to discard.
    """
    if n <= 0:
        return ClaimEvidence(
            claim=compressed.claim,
            units=list(compressed.kept),
            token_count=compressed.compressed_token_count,
            uncertainty=compressed.uncertainty,
        )

    extra = _ordered(list(compressed.dropped))[:n]
    units = _ordered(list(compressed.kept) + extra)
    return ClaimEvidence(
        claim=compressed.claim,
        units=units,
        token_count=sum(u.token_count for u in units),
        uncertainty=compressed.uncertainty,
    )


def restore_neighbours(compressed: CompressedEvidence) -> ClaimEvidence:
    """Reinstate dropped units that sit immediately beside a kept one in the source text.

    Sentence chunking is what makes reversibility exact, but it splits a claim from its
    context: "The function has two tunable parameters." and the sentence that names them
    become separate units, and compression can keep one without the other. A chunk's
    neighbours are the cheapest possible guess at its missing context — no model call, no
    scoring, just adjacency in the document.

    Cheaper than full restoration and more targeted than restoring the next-best by score,
    because "next best" is a *relevance* judgement and this is a *cohesion* one. The two
    are different repairs and the neighbour is often the one actually needed.
    """
    kept = list(compressed.kept)
    if not kept or not compressed.dropped:
        return restore_partial(compressed, 0)

    # Adjacency is measured in the source document, not in rank order.
    ends = {(u.span.doc_id, u.span.end) for u in kept}
    starts = {(u.span.doc_id, u.span.start) for u in kept}

    neighbours = [
        u
        for u in compressed.dropped
        # A dropped unit that ends where a kept one begins, or begins where one ends,
        # allowing a small gap for the whitespace between sentences.
        if any(
            u.span.doc_id == doc_id and abs(u.span.end - start) <= 2
            for doc_id, start in starts
        )
        or any(
            u.span.doc_id == doc_id and abs(u.span.start - end) <= 2
            for doc_id, end in ends
        )
    ]

    units = _ordered(kept + neighbours)
    return ClaimEvidence(
        claim=compressed.claim,
        units=units,
        token_count=sum(u.token_count for u in units),
        uncertainty=compressed.uncertainty,
    )


def restoration_ladder(compressed: CompressedEvidence, step: int) -> list[int]:
    """How many units to reinstate at each attempt, ending at everything dropped.

    The last rung is always full restoration, so graded restoration can never recover
    *less* than the all-at-once policy it replaces — it only reaches the same place more
    cheaply when an earlier rung suffices.
    """
    total = len(compressed.dropped)
    if total == 0:
        return []
    step = max(1, step)
    rungs = list(range(step, total, step))
    rungs.append(total)
    return rungs


def restore_all(compressed: list[CompressedEvidence]) -> list[ClaimEvidence]:
    return [restore(c) for c in compressed]
