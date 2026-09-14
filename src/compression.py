"""Pipeline stage 4 — uncertainty-guided reversible evidence compression.

WHAT IS ACTUALLY COMPRESSED (stated explicitly, per CLAUDE.md):
the saving is in tokens *sent to the generator*, not in memory. Dropped units are
retained in `CompressedEvidence.dropped` so restoration is immediate, and each one keeps
its `Span`, so the original text remains recoverable from the corpus even if those
retained copies were discarded. Both recovery routes are asserted equivalent by the
round-trip test in tests/test_compression.py.

Compression is extractive — whole evidence units are kept or dropped, never rewritten.
That is what makes restoration exact rather than approximate. Abstractive compression
would reach higher ratios but could not be inverted from its output alone, which would
break the core guarantee of the project.

Four modes, three of which exist to make the central claim falsifiable:

  identity           — no-op. The no-compression baseline.
  uncertainty_guided — keep budget scales with claim uncertainty. The proposal.
  fixed_ratio        — keeps a constant fraction, top-ranked first. Ablates the
                       *adaptivity* while holding the budget and the ranking fixed.
  random             — keeps a constant fraction chosen at random. Ablates the
                       ranking too, so it separates "score order matters" from
                       "uncertainty-driven budgeting matters".

Comparing uncertainty_guided against identity only shows that compression saves
tokens, which is trivially true of any method that discards evidence. The fixed_ratio
arm is what tests whether *uncertainty* is doing the work, and it is only meaningful
when both arms spend the same budget — scripts/run_eval.py calibrates it to do so.
"""

from __future__ import annotations

import random

from .config import Config, default_config
from .types import ClaimEvidence, CompressedEvidence
from .uncertainty import estimator_from_config

MODES = ("identity", "uncertainty_guided", "fixed_ratio", "fixed_tokens", "random", "oracle")

# Modes that cannot be run as a system, only measured. `oracle` reads human-marked answer
# evidence, so it is an upper bound on what any selection policy could achieve, not a
# configuration anyone could deploy. Named here so the distinction is enforced rather than
# remembered.
MEASUREMENT_ONLY_MODES = ("oracle",)


def _identity(item: ClaimEvidence) -> CompressedEvidence:
    """Keep everything. The baseline arm."""
    return CompressedEvidence(
        claim=item.claim,
        kept=list(item.units),
        dropped=[],
        uncertainty=item.uncertainty,
        keep_ratio=1.0,
        original_token_count=item.token_count,
        compressed_token_count=item.token_count,
    )


def _take(
    item: ClaimEvidence, ordered_units: list, keep_ratio: float, cfg: Config
) -> CompressedEvidence:
    """Keep the first `keep_ratio` fraction of an already-ordered unit list.

    Shared by every compressing mode so that the arms differ only in how they order
    units and how they choose the ratio — never in the accounting.
    """
    floor_units = int(cfg.get_path("compression.floor_units", 1))
    n_keep = max(floor_units, round(keep_ratio * len(ordered_units)))
    n_keep = min(n_keep, len(ordered_units))

    kept, dropped = ordered_units[:n_keep], ordered_units[n_keep:]
    return CompressedEvidence(
        claim=item.claim,
        kept=kept,
        dropped=dropped,
        uncertainty=item.uncertainty,
        keep_ratio=keep_ratio,
        original_token_count=item.token_count,
        compressed_token_count=sum(u.token_count for u in kept),
    )


def _by_score(item: ClaimEvidence) -> list:
    """Rank order. Uses the reranked score when a second stage has run — otherwise a
    reranker would reorder the candidates and compression would immediately undo it."""
    return sorted(item.units, key=lambda u: u.rank_score, reverse=True)


def _uncertainty_guided(item: ClaimEvidence, cfg: Config) -> CompressedEvidence:
    """Drop the weakest evidence, keeping more of it when uncertainty is high.

        keep_ratio = min_keep + u * (max_keep - min_keep)

    Low uncertainty means retrieval was decisive — one clear winner well above the rest —
    so most units are redundant and can go. High uncertainty means the ranking is
    unstable, so more is retained to avoid discarding the unit that actually mattered.
    """
    min_keep = float(cfg.require("compression.min_keep"))
    max_keep = float(cfg.require("compression.max_keep"))

    units = _by_score(item)
    if not units:
        return _identity(item)

    keep_ratio = min_keep + item.uncertainty * (max_keep - min_keep)
    return _take(item, units, keep_ratio, cfg)


def _fixed_ratio(item: ClaimEvidence, cfg: Config) -> CompressedEvidence:
    """Keep a constant top fraction, ignoring uncertainty. Ablates adaptivity only."""
    units = _by_score(item)
    if not units:
        return _identity(item)
    return _take(item, units, float(cfg.get_path("compression.fixed_keep", 0.6)), cfg)


def _fixed_tokens(item: ClaimEvidence, cfg: Config) -> CompressedEvidence:
    """Keep top-ranked units until a token budget is spent. Ablates the *unit* of budget.

    Every other arm budgets in units, which silently lets a claim whose chunks happen to
    be long spend more than one whose chunks are short. Budgeting in tokens is what a cost
    -constrained deployment would actually do, and it is the honest comparison once
    reduction is reported end-to-end in tokens rather than in evidence units.
    """
    budget = int(cfg.get_path("compression.token_budget", 0))
    units = _by_score(item)
    if not units or budget <= 0:
        return _identity(item)

    floor_units = int(cfg.get_path("compression.floor_units", 1))
    kept: list = []
    spent = 0
    for unit in units:
        if kept and spent + unit.token_count > budget:
            continue
        kept.append(unit)
        spent += unit.token_count

    if len(kept) < floor_units:
        kept = units[:floor_units]
    keep_set = {id(u) for u in kept}
    dropped = [u for u in units if id(u) not in keep_set]

    return CompressedEvidence(
        claim=item.claim,
        kept=kept,
        dropped=dropped,
        uncertainty=item.uncertainty,
        keep_ratio=len(kept) / len(units),
        original_token_count=item.token_count,
        compressed_token_count=sum(u.token_count for u in kept),
    )


def _oracle(item: ClaimEvidence, keep_spans: set) -> CompressedEvidence:
    """Keep exactly the units overlapping human-marked answer evidence.

    NOT A SYSTEM. This reads the answer key, so it cannot be deployed and must never be
    reported as a configuration. It exists to answer one question the ablation cannot:
    *how much of the achievable saving does ranked selection actually capture?*

    Without it, "53% reduction at equal quality" has no scale. If the oracle reaches 70%,
    ranked selection is leaving a third on the table; if it reaches 55%, ranked selection
    is close to optimal and further work on the selector is wasted.
    """
    units = list(item.units)
    if not units:
        return _identity(item)

    kept = [
        u
        for u in units
        if any(
            u.span.doc_id == doc_id and u.span.start < end and start < u.span.end
            for doc_id, start, end in keep_spans
        )
    ]
    # An oracle that keeps nothing would score zero and misrepresent the ceiling as
    # unreachable; fall back to the top-ranked unit, matching every other arm's floor.
    if not kept:
        kept = _by_score(item)[:1]

    keep_set = {id(u) for u in kept}
    dropped = [u for u in units if id(u) not in keep_set]
    return CompressedEvidence(
        claim=item.claim,
        kept=kept,
        dropped=dropped,
        uncertainty=item.uncertainty,
        keep_ratio=len(kept) / len(units),
        original_token_count=item.token_count,
        compressed_token_count=sum(u.token_count for u in kept),
    )


def _random(item: ClaimEvidence, cfg: Config, rng: random.Random) -> CompressedEvidence:
    """Keep a constant fraction chosen at random. Ablates the ranking as well.

    If this matches the ranked arms, retrieval order is carrying no information and the
    saving is coming from prompt length alone.
    """
    units = list(item.units)
    if not units:
        return _identity(item)
    rng.shuffle(units)
    return _take(item, units, float(cfg.get_path("compression.fixed_keep", 0.6)), cfg)


def compress(
    evidence: list[ClaimEvidence],
    cfg: Config | None = None,
    mode: str | None = None,
    oracle_spans: set | None = None,
    seed: int | None = None,
) -> list[CompressedEvidence]:
    """Compress each claim's evidence set.

    Uncertainty is estimated here and stored on the result, so every compression
    decision is returned alongside the score that drove it (CLAUDE.md reproducibility
    requirement) rather than being recomputed or lost.
    """
    cfg = cfg or default_config()
    mode = mode or "uncertainty_guided"
    if mode not in MODES:
        raise ValueError(f"unknown compression mode {mode!r}; available: {list(MODES)}")
    if mode == "oracle" and oracle_spans is None:
        # Failing loudly beats silently degrading to some other policy and reporting the
        # result as a ceiling it is not.
        raise ValueError(
            "oracle mode requires oracle_spans (human-marked evidence). It is a "
            "measurement upper bound supplied by the eval harness, not a configuration."
        )

    estimator = estimator_from_config(cfg)
    # Seeded per call, so the random arm is reproducible across runs and independent of
    # how many queries preceded it. `seed` overrides the config for multi-seed sweeps,
    # where a single draw would be mistaken for the behaviour of random selection.
    rng = random.Random(
        int(cfg.get_path("compression.random_seed", 0)) if seed is None else seed
    )

    out: list[CompressedEvidence] = []
    for item in evidence:
        # Uncertainty is estimated in every mode, even those that ignore it, so all arms
        # carry the same scores in their traces and stay comparable when analysed.
        item.uncertainty = estimator.estimate(item.units)

        if mode == "identity":
            out.append(_identity(item))
        elif mode == "fixed_ratio":
            out.append(_fixed_ratio(item, cfg))
        elif mode == "fixed_tokens":
            out.append(_fixed_tokens(item, cfg))
        elif mode == "random":
            out.append(_random(item, cfg, rng))
        elif mode == "oracle":
            out.append(_oracle(item, oracle_spans or set()))
        else:
            out.append(_uncertainty_guided(item, cfg))
    return out


def to_claim_evidence(compressed: CompressedEvidence) -> ClaimEvidence:
    """View a compressed set as a ClaimEvidence, for prompt assembly and evaluation."""
    return ClaimEvidence(
        claim=compressed.claim,
        units=list(compressed.kept),
        token_count=compressed.compressed_token_count,
        uncertainty=compressed.uncertainty,
    )
