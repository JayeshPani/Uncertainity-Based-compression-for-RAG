"""Pipeline stages 2-3 — map retrieved evidence to claims, producing the initial set.

The output of this module is the "Initial Evidence Set" from the README: full,
uncompressed, high-token evidence per claim. The token count recorded here is the
baseline every later compression ratio is measured against, so it is computed before
any compression runs.
"""

from __future__ import annotations

from .config import Config, default_config
from .retrieval import Retriever
from .tokens import TokenCounter, shared_counter
from .types import ClaimEvidence, EvidenceUnit


def document_title(doc_id: str) -> str:
    """Human-readable title for a document id ("dense_retrieval" -> "dense retrieval").

    Used to prefix evidence when scoring support. Isolated sentence chunks are often
    anaphoric — "The function has two tunable parameters" never names BM25 — and without
    the title the NLI evaluator cannot resolve the subject. Measured effect on the
    seeded KB: P(entailment) 0.004 -> 0.997 for a known-relevant chunk.
    """
    return doc_id.replace("_", " ").strip()


def contextualize(unit: EvidenceUnit, enabled: bool = True) -> str:
    """Evidence text prefixed with its document title, for support scoring.

    WHETHER THIS HELPS DEPENDS ENTIRELY ON THE CORPUS, and getting it wrong is
    catastrophic in either direction. Measured mean best-support over the same claims:

        corpus                         with title    without title
        hand-built (24 docs)               0.997            0.004
        QASPER (110 real papers)           0.004            0.176

    On the hand-built corpus, doc ids are clean single concepts ("bm25", "calibration")
    that genuinely resolve the anaphora in a chunk like "The function has two tunable
    parameters". On real papers the id is a 60-character slug of the paper title, which
    is near-identical across every chunk of that paper and simply drowns short passages
    in noise.

    The +0.99 result was therefore an artefact of a corpus we wrote ourselves. Keep this
    switchable, and set it per corpus rather than assuming.
    """
    if not enabled:
        return unit.text
    return f"{document_title(unit.span.doc_id)}. {unit.text}"


def map_evidence(
    claims: list[str],
    retriever: Retriever,
    cfg: Config | None = None,
    counter: TokenCounter | None = None,
    top_k: int | None = None,
    restrict_to: str | None = None,
) -> list[ClaimEvidence]:
    """Retrieve and attach evidence for each claim, claim-wise.

    Every claim receives its own retrieval pass, so evidence is genuinely per-claim
    rather than a shared pool sliced up afterwards.
    """
    cfg = cfg or default_config()
    counter = counter or shared_counter(cfg)

    mapped: list[ClaimEvidence] = []
    for claim in claims:
        units = retriever.retrieve(claim, top_k=top_k, restrict_to=restrict_to)
        for unit in units:
            unit.token_count = counter.count(unit.text)

        mapped.append(
            ClaimEvidence(
                claim=claim,
                units=units,
                token_count=sum(u.token_count for u in units),
            )
        )
    return mapped


def total_baseline_tokens(evidence: list[ClaimEvidence]) -> int:
    """Uncompressed token cost across all claims — the denominator for reduction."""
    return sum(e.token_count for e in evidence)
