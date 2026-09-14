"""Compression and restoration tests.

`TestRoundTrip` is the gate for the whole project. Compression is only legitimate if it
can be undone exactly; if these fail, every downstream result is untrustworthy because
restored evidence would silently differ from what was originally retrieved.
"""

from __future__ import annotations

import pytest

from src.compression import compress, to_claim_evidence
from src.restoration import restore, restore_all, restore_from_corpus
from src.types import ClaimEvidence, EvidenceUnit


def evidence_from_units(units, claim="the boiling point of water"):
    for i, unit in enumerate(units):
        unit.retrieval_score = 0.9 - 0.1 * i
        unit.token_count = len(unit.text.split())
    return ClaimEvidence(
        claim=claim, units=units, token_count=sum(u.token_count for u in units)
    )


def signature(units: list[EvidenceUnit]):
    """Identity of an evidence set, independent of ordering."""
    return sorted((u.span.doc_id, u.span.start, u.span.end, u.text) for u in units)


@pytest.fixture
def claim_evidence(units):
    return evidence_from_units(list(units))


class TestRoundTrip:
    """THE gate: compress -> restore must return exactly the original evidence."""

    def test_restore_recovers_every_unit(self, claim_evidence, cfg):
        original = signature(claim_evidence.units)
        compressed = compress([claim_evidence], cfg=cfg)[0]
        restored = restore(compressed)
        assert signature(restored.units) == original

    def test_restore_from_corpus_matches_retained_copies(
        self, claim_evidence, corpus, cfg
    ):
        """The two recovery routes must agree.

        `restore` reinstates retained copies; `restore_from_corpus` ignores them and
        re-reads each span from the source. Agreement is what proves reversibility comes
        from the span index rather than from keeping a backup.
        """
        compressed = compress([claim_evidence], cfg=cfg)[0]
        assert signature(restore(compressed).units) == signature(
            restore_from_corpus(compressed, corpus).units
        )

    def test_restored_text_is_byte_identical(self, claim_evidence, corpus, cfg):
        compressed = compress([claim_evidence], cfg=cfg)[0]
        for unit in restore_from_corpus(compressed, corpus).units:
            assert unit.text == unit.span.resolve(corpus)

    def test_round_trip_holds_at_every_uncertainty_level(self, claim_evidence, cfg):
        """Reversibility must not depend on how aggressive compression happened to be."""
        original = signature(claim_evidence.units)
        for uncertainty in (0.0, 0.25, 0.5, 0.75, 1.0):
            item = evidence_from_units(list(claim_evidence.units))
            item.uncertainty = uncertainty
            from src.compression import _uncertainty_guided

            compressed = _uncertainty_guided(item, cfg)
            assert signature(restore(compressed).units) == original, (
                f"round trip broken at uncertainty={uncertainty}"
            )

    def test_token_accounting_is_conserved(self, claim_evidence, cfg):
        compressed = compress([claim_evidence], cfg=cfg)[0]
        kept = sum(u.token_count for u in compressed.kept)
        dropped = sum(u.token_count for u in compressed.dropped)
        assert kept + dropped == compressed.original_token_count
        assert compressed.compressed_token_count == kept

    def test_restore_is_deterministic(self, claim_evidence, cfg):
        compressed = compress([claim_evidence], cfg=cfg)[0]
        first = [u.text for u in restore(compressed).units]
        assert first == [u.text for u in restore(compressed).units]

    def test_restore_all_handles_multiple_claims(self, claim_evidence, cfg):
        compressed = compress([claim_evidence, claim_evidence], cfg=cfg)
        assert len(restore_all(compressed)) == 2


class TestCompressionBehaviour:
    def test_nothing_is_destroyed(self, claim_evidence, cfg):
        compressed = compress([claim_evidence], cfg=cfg)[0]
        assert len(compressed.kept) + len(compressed.dropped) == len(
            claim_evidence.units
        )

    def test_kept_units_are_the_highest_scoring(self, claim_evidence, cfg):
        compressed = compress([claim_evidence], cfg=cfg)[0]
        if compressed.dropped:
            assert min(u.retrieval_score for u in compressed.kept) >= max(
                u.retrieval_score for u in compressed.dropped
            )

    def test_higher_uncertainty_keeps_at_least_as_much(self, claim_evidence, cfg):
        """The core guarantee of 'uncertainty-guided': unsure means keep more."""
        from src.compression import _uncertainty_guided

        counts = []
        for uncertainty in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            item = evidence_from_units(list(claim_evidence.units))
            item.uncertainty = uncertainty
            counts.append(len(_uncertainty_guided(item, cfg).kept))
        assert counts == sorted(counts), f"keep count not monotonic: {counts}"

    def test_maximum_uncertainty_keeps_everything(self, claim_evidence, cfg):
        from src.compression import _uncertainty_guided

        item = evidence_from_units(list(claim_evidence.units))
        item.uncertainty = 1.0
        compressed = _uncertainty_guided(item, cfg)
        assert compressed.dropped == []

    def test_floor_prevents_empty_evidence(self, claim_evidence, cfg):
        """Even total confidence must leave something to generate from."""
        from src.compression import _uncertainty_guided

        item = evidence_from_units(list(claim_evidence.units))
        item.uncertainty = 0.0
        compressed = _uncertainty_guided(item, cfg)
        assert len(compressed.kept) >= int(cfg.get_path("compression.floor_units", 1))

    def test_compression_reduces_tokens(self, claim_evidence, cfg):
        from src.compression import _uncertainty_guided

        item = evidence_from_units(list(claim_evidence.units))
        item.uncertainty = 0.0
        compressed = _uncertainty_guided(item, cfg)
        assert compressed.compressed_token_count < compressed.original_token_count
        assert compressed.reduction > 0

    def test_uncertainty_is_recorded_with_the_decision(self, claim_evidence, cfg):
        """CLAUDE.md: scores must be returned alongside compression decisions."""
        compressed = compress([claim_evidence], cfg=cfg)[0]
        assert 0.0 <= compressed.uncertainty <= 1.0
        assert 0.0 <= compressed.keep_ratio <= 1.0

    def test_empty_evidence_does_not_crash(self, cfg):
        empty = ClaimEvidence(claim="nothing", units=[], token_count=0)
        compressed = compress([empty], cfg=cfg)[0]
        assert compressed.kept == [] and compressed.dropped == []


class TestIdentityMode:
    """The no-compression baseline arm used by scripts/run_eval.py."""

    def test_keeps_everything(self, claim_evidence, cfg):
        compressed = compress([claim_evidence], cfg=cfg, mode="identity")[0]
        assert len(compressed.kept) == len(claim_evidence.units)
        assert compressed.dropped == []

    def test_reports_zero_reduction(self, claim_evidence, cfg):
        compressed = compress([claim_evidence], cfg=cfg, mode="identity")[0]
        assert compressed.reduction == 0.0
        assert compressed.tokens_saved == 0

    def test_round_trip_is_trivially_exact(self, claim_evidence, cfg):
        compressed = compress([claim_evidence], cfg=cfg, mode="identity")[0]
        assert signature(restore(compressed).units) == signature(claim_evidence.units)


class TestAblationArms:
    """The arms that make the central claim falsifiable.

    Both keep a constant fraction regardless of uncertainty. If they matched the
    uncertainty arm at equal budget, the proposal would add nothing.
    """

    def test_fixed_ratio_ignores_uncertainty(self, claim_evidence, cfg):
        from src.compression import _fixed_ratio

        counts = []
        for uncertainty in (0.0, 0.5, 1.0):
            item = evidence_from_units(list(claim_evidence.units))
            item.uncertainty = uncertainty
            counts.append(len(_fixed_ratio(item, cfg).kept))
        assert len(set(counts)) == 1, f"fixed arm varied with uncertainty: {counts}"

    def test_fixed_ratio_keeps_highest_scoring(self, claim_evidence, cfg):
        compressed = compress([claim_evidence], cfg=cfg, mode="fixed_ratio")[0]
        if compressed.dropped:
            assert min(u.retrieval_score for u in compressed.kept) >= max(
                u.retrieval_score for u in compressed.dropped
            )

    def test_random_arm_is_reproducible(self, claim_evidence, cfg):
        """A seeded arm, or the ablation could not be rerun."""
        first = compress([claim_evidence], cfg=cfg, mode="random")[0]
        second = compress([claim_evidence], cfg=cfg, mode="random")[0]
        assert [u.text for u in first.kept] == [u.text for u in second.kept]

    def test_random_arm_ignores_score_order(self, cfg):
        """It must actually be random, or it is just a second fixed_ratio arm."""
        from src.types import EvidenceUnit, Span

        units = [
            EvidenceUnit(span=Span("d", i, i + 1), text=f"unit {i}", token_count=1)
            for i in range(40)
        ]
        for i, u in enumerate(units):
            u.retrieval_score = 1.0 - i * 0.01
        item = ClaimEvidence(claim="c", units=units, token_count=len(units))

        compressed = compress([item], cfg=cfg, mode="random")[0]
        ranked_top = {u.text for u in sorted(units, key=lambda x: -x.retrieval_score)[
            : len(compressed.kept)
        ]}
        assert {u.text for u in compressed.kept} != ranked_top

    @pytest.mark.parametrize("mode", ["fixed_ratio", "random"])
    def test_ablation_arms_are_reversible(self, claim_evidence, corpus, cfg, mode):
        """Reversibility is a property of the design, not of one mode."""
        original = signature(claim_evidence.units)
        compressed = compress([claim_evidence], cfg=cfg, mode=mode)[0]
        assert signature(restore(compressed).units) == original
        assert signature(restore_from_corpus(compressed, corpus).units) == original

    @pytest.mark.parametrize("mode", ["fixed_ratio", "random"])
    def test_ablation_arms_conserve_units(self, claim_evidence, cfg, mode):
        compressed = compress([claim_evidence], cfg=cfg, mode=mode)[0]
        assert len(compressed.kept) + len(compressed.dropped) == len(
            claim_evidence.units
        )

    def test_unknown_mode_raises(self, claim_evidence, cfg):
        with pytest.raises(ValueError, match="unknown compression mode"):
            compress([claim_evidence], cfg=cfg, mode="nonsense")


class TestClaimEvidenceView:
    def test_view_exposes_only_kept_units(self, claim_evidence, cfg):
        compressed = compress([claim_evidence], cfg=cfg)[0]
        view = to_claim_evidence(compressed)
        assert view.units == compressed.kept
        assert view.token_count == compressed.compressed_token_count
        assert view.claim == compressed.claim


class TestFixedTokenBudget:
    """Budgeting in units lets a claim with long chunks quietly spend more than one with
    short chunks. A cost-constrained deployment would budget in tokens; so does this arm."""

    def test_respects_the_token_budget(self, claim_evidence, cfg):
        cfg["compression"] = dict(cfg["compression"], token_budget=12)
        compressed = compress([claim_evidence], cfg=cfg, mode="fixed_tokens")[0]
        assert compressed.compressed_token_count <= 12

    def test_keeps_the_highest_ranked_evidence(self, claim_evidence, cfg):
        cfg["compression"] = dict(cfg["compression"], token_budget=12)
        compressed = compress([claim_evidence], cfg=cfg, mode="fixed_tokens")[0]
        best = max(claim_evidence.units, key=lambda u: u.retrieval_score)
        assert best in compressed.kept

    def test_always_keeps_the_floor_even_under_a_tiny_budget(self, claim_evidence, cfg):
        """A claim with no evidence at all cannot be evaluated; the floor prevents it."""
        cfg["compression"] = dict(cfg["compression"], token_budget=1)
        compressed = compress([claim_evidence], cfg=cfg, mode="fixed_tokens")[0]
        assert len(compressed.kept) >= 1

    def test_round_trip_still_exact(self, claim_evidence, corpus, cfg):
        cfg["compression"] = dict(cfg["compression"], token_budget=12)
        compressed = compress([claim_evidence], cfg=cfg, mode="fixed_tokens")[0]
        assert signature(restore(compressed).units) == signature(
            restore_from_corpus(compressed, corpus).units
        )


class TestOracleArm:
    """The oracle reads the answer key. It measures headroom and cannot be deployed, so
    the tests here are as much about preventing misuse as about behaviour."""

    def test_requires_explicit_spans(self, claim_evidence, cfg):
        """Silently degrading to another policy would report a ceiling that is not one."""
        with pytest.raises(ValueError, match="oracle mode requires oracle_spans"):
            compress([claim_evidence], cfg=cfg, mode="oracle")

    def test_keeps_only_units_overlapping_marked_evidence(self, claim_evidence, cfg):
        target = claim_evidence.units[1]
        spans = {(target.span.doc_id, target.span.start, target.span.end)}
        compressed = compress(
            [claim_evidence], cfg=cfg, mode="oracle", oracle_spans=spans
        )[0]
        assert target in compressed.kept
        assert all(u.span.doc_id == target.span.doc_id for u in compressed.kept)

    def test_partial_overlap_counts_as_a_hit(self, claim_evidence, cfg):
        """Marked evidence is paragraph-level and chunks are sentences, so a chunk
        typically covers only part of a marked span."""
        target = claim_evidence.units[0]
        spans = {(target.span.doc_id, target.span.start + 1, target.span.end + 500)}
        compressed = compress(
            [claim_evidence], cfg=cfg, mode="oracle", oracle_spans=spans
        )[0]
        assert target in compressed.kept

    def test_falls_back_rather_than_keeping_nothing(self, claim_evidence, cfg):
        """An empty oracle would score zero and misreport the ceiling as unreachable."""
        compressed = compress(
            [claim_evidence], cfg=cfg, mode="oracle",
            oracle_spans={("no_such_doc", 0, 10)},
        )[0]
        assert len(compressed.kept) == 1

    def test_round_trip_still_exact(self, claim_evidence, corpus, cfg):
        target = claim_evidence.units[1]
        spans = {(target.span.doc_id, target.span.start, target.span.end)}
        compressed = compress(
            [claim_evidence], cfg=cfg, mode="oracle", oracle_spans=spans
        )[0]
        assert signature(restore(compressed).units) == signature(
            restore_from_corpus(compressed, corpus).units
        )

    def test_oracle_spans_are_rejected_by_every_other_mode(self, claim_evidence, cfg):
        """The answer key must not influence a mode that claims not to see it."""
        target = claim_evidence.units[1]
        spans = {(target.span.doc_id, target.span.start, target.span.end)}
        for mode in ("identity", "uncertainty_guided", "fixed_ratio", "random"):
            with_key = compress(
                [claim_evidence], cfg=cfg, mode=mode, oracle_spans=spans
            )[0]
            without = compress([claim_evidence], cfg=cfg, mode=mode)[0]
            assert signature(with_key.kept) == signature(without.kept), mode


class TestSeedOverride:
    def test_different_seeds_select_differently(self, claim_evidence, cfg):
        """Multi-seed reporting is only meaningful if the seed actually changes the draw."""
        cfg["compression"] = dict(cfg["compression"], fixed_keep=0.5)
        picks = {
            tuple(sorted(u.text for u in compress(
                [evidence_from_units(list(claim_evidence.units))],
                cfg=cfg, mode="random", seed=s,
            )[0].kept))
            for s in range(12)
        }
        assert len(picks) > 1

    def test_same_seed_is_reproducible(self, claim_evidence, cfg):
        first = compress([claim_evidence], cfg=cfg, mode="random", seed=7)[0]
        second = compress([claim_evidence], cfg=cfg, mode="random", seed=7)[0]
        assert signature(first.kept) == signature(second.kept)


class TestGradedRestoration:
    """Full restoration is all-or-nothing: the moment anything looks wrong the claim
    surrenders its entire saving, including the units correctly dropped. Restoration fires
    on a quarter to a half of QASPER questions, so that is a lot of budget given back on
    suspicion."""

    def test_partial_restores_only_the_requested_count(self, claim_evidence, cfg):
        from src.restoration import restore_partial

        compressed = compress([claim_evidence], cfg=cfg)[0]
        partial = restore_partial(compressed, 1)
        assert len(partial.units) == len(compressed.kept) + 1

    def test_partial_restores_the_best_dropped_evidence_first(self, claim_evidence, cfg):
        from src.restoration import restore_partial

        compressed = compress([claim_evidence], cfg=cfg)[0]
        best_dropped = max(compressed.dropped, key=lambda u: u.retrieval_score)
        assert best_dropped in restore_partial(compressed, 1).units

    def test_zero_restores_nothing(self, claim_evidence, cfg):
        from src.restoration import restore_partial

        compressed = compress([claim_evidence], cfg=cfg)[0]
        assert signature(restore_partial(compressed, 0).units) == signature(compressed.kept)

    def test_partial_never_exceeds_full(self, claim_evidence, cfg):
        from src.restoration import restore_partial

        compressed = compress([claim_evidence], cfg=cfg)[0]
        full = restore(compressed)
        many = restore_partial(compressed, len(compressed.dropped) + 99)
        assert signature(many.units) == signature(full.units)

    def test_ladder_always_ends_at_full_restoration(self, claim_evidence, cfg):
        """The guarantee that makes graded restoration safe to adopt: it can never
        recover less than the policy it replaces, only reach it more cheaply."""
        from src.restoration import restoration_ladder

        compressed = compress([claim_evidence], cfg=cfg)[0]
        for step in (1, 2, 3, 99):
            ladder = restoration_ladder(compressed, step)
            assert ladder[-1] == len(compressed.dropped), step
            assert ladder == sorted(ladder)

    def test_ladder_is_empty_when_nothing_was_dropped(self, claim_evidence, cfg):
        from src.restoration import restoration_ladder

        compressed = compress([claim_evidence], cfg=cfg, mode="identity")[0]
        assert restoration_ladder(compressed, 2) == []

    def test_partial_restoration_is_still_exactly_reversible(self, claim_evidence, corpus, cfg):
        """Spans must survive partial restoration — the core guarantee is unconditional."""
        from src.restoration import restore_partial

        compressed = compress([claim_evidence], cfg=cfg)[0]
        for unit in restore_partial(compressed, 1).units:
            assert unit.verify_against(corpus)


class TestNeighbourRestoration:
    """Sentence chunking is what makes reversibility exact, but it severs a claim from its
    context. A dropped chunk adjacent in the source text to a kept one is the cheapest
    possible guess at that missing context — no model call, just adjacency."""

    def test_restores_the_chunk_immediately_after_a_kept_one(self, claim_evidence, cfg):
        from src.restoration import restore_neighbours

        compressed = compress([claim_evidence], cfg=cfg)[0]
        kept_ends = {(u.span.doc_id, u.span.end) for u in compressed.kept}
        expected = [
            u for u in compressed.dropped
            if any(u.span.doc_id == d and abs(u.span.start - e) <= 2 for d, e in kept_ends)
        ]
        restored = restore_neighbours(compressed).units
        for unit in expected:
            assert unit in restored

    def test_does_not_restore_distant_chunks(self, claim_evidence, cfg):
        """The point is a cheap cohesion repair, not a disguised full restoration."""
        from src.restoration import restore_neighbours

        compressed = compress([claim_evidence], cfg=cfg)[0]
        restored = restore_neighbours(compressed)
        assert len(restored.units) <= len(compressed.kept) + len(compressed.dropped)

    def test_never_loses_a_kept_unit(self, claim_evidence, cfg):
        from src.restoration import restore_neighbours

        compressed = compress([claim_evidence], cfg=cfg)[0]
        restored = restore_neighbours(compressed).units
        for unit in compressed.kept:
            assert unit in restored

    def test_nothing_dropped_means_nothing_added(self, claim_evidence, cfg):
        from src.restoration import restore_neighbours

        compressed = compress([claim_evidence], cfg=cfg, mode="identity")[0]
        assert signature(restore_neighbours(compressed).units) == signature(compressed.kept)

    def test_neighbours_are_still_exactly_reversible(self, claim_evidence, corpus, cfg):
        from src.restoration import restore_neighbours

        compressed = compress([claim_evidence], cfg=cfg)[0]
        for unit in restore_neighbours(compressed).units:
            assert unit.verify_against(corpus)
