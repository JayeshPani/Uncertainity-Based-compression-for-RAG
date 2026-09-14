"""Pipeline stage 1b — rerank retrieved candidates before compression decides what stays.

WHY THIS EXISTS

Compression keeps the top-ranked units, and until now "top-ranked" meant *bi-encoder
cosine similarity*. That is a first-stage recall signal: it is computed without the query
and the passage ever meeting, which is exactly what makes it cheap enough to run over a
whole corpus, and exactly what makes it a blunt instrument for deciding which of eight
candidates carries the answer.

The cost is measurable. Retrieval surfaces 44-55% of the human-marked answer evidence, but
only ~20% survives compression. Most of that loss is not compression being too aggressive;
it is compression being aggressive in the wrong order.

A cross-encoder reads the query and passage together and is far better at that ordering.
The same `cross-encoder/ms-marco-MiniLM-L-6-v2` already loaded for support scoring is
reused, so this adds no new model to the dependency set — it is the second stage the
retrieval design always implied and never had.

WHAT IT DOES NOT FIX

Reranking cannot recover evidence retrieval never returned. The ~45% of marked evidence
that never enters the candidate set is a first-stage recall problem, addressed by raising
`retrieval.top_k` (which gives the reranker more to work with), not by reordering.
"""

from __future__ import annotations

import logging

from .config import Config, default_config
from .types import EvidenceUnit

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reordering of retrieved candidates.

    Deliberately a thin wrapper over the existing scorer rather than a new model class:
    the relevance scorer already answers "how well does this passage address this query?",
    which is precisely the reranking question.
    """

    def __init__(self, cfg: Config | None = None, scorer=None) -> None:
        self.cfg = cfg or default_config()
        self._scorer = scorer

    @property
    def scorer(self):
        if self._scorer is None:
            from .evaluation import _SCORERS
            from .config import resolve_device

            name = self.cfg.get_path("rerank.scorer", "relevance")
            settings = self.cfg.get_path(f"evaluation.scorers.{name}") or {}
            self._scorer = _SCORERS[name](
                model_id=self.cfg.get_path("rerank.model") or settings["model"],
                threshold=float(settings.get("threshold", 0.5)),
                device=resolve_device(self.cfg.get_path("models.device", "auto")),
            )
        return self._scorer

    def rerank(
        self, query: str, units: list[EvidenceUnit], contextualize_units: bool = True
    ) -> list[EvidenceUnit]:
        """Return `units` reordered by cross-encoder relevance to `query`.

        **The bi-encoder score is preserved.** `retrieval_score` continues to hold the
        cosine similarity, because the uncertainty estimator is calibrated against its
        scale — overwriting it would silently redefine `u_claim` and change the
        compression budget as a side effect of a reranking change. The new score lands in
        `rerank_score`, and ordering alone is what compression consumes.
        """
        if len(units) < 2:
            return units

        from .evidence_mapping import contextualize

        premises = [contextualize(u, contextualize_units) for u in units]
        scores = self.scorer.score(premises, query)
        if len(scores) != len(units):  # pragma: no cover - defensive
            logger.warning("reranker returned %d scores for %d units", len(scores), len(units))
            return units

        for unit, score in zip(units, scores):
            unit.rerank_score = score

        # Sort by the new score, tie-broken by the original ranking so the result is
        # deterministic and degrades to retrieval order when the reranker is indifferent.
        return sorted(
            units,
            key=lambda u: (-(u.rerank_score or 0.0), -u.retrieval_score, u.span.start),
        )


def rerank_evidence(
    evidence: list, cfg: Config | None = None, reranker: Reranker | None = None
) -> list:
    """Rerank each claim's candidates in place, returning the same ClaimEvidence list."""
    cfg = cfg or default_config()
    if not bool(cfg.get_path("rerank.enabled", False)):
        return evidence

    reranker = reranker or Reranker(cfg=cfg)
    contextualize_units = bool(cfg.get_path("evaluation.contextualize", True))
    for item in evidence:
        item.units = reranker.rerank(
            item.claim, item.units, contextualize_units=contextualize_units
        )
    return evidence
