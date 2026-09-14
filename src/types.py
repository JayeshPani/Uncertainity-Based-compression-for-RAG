"""Core data model for the uncertainty-guided reversible compression pipeline.

The central design decision lives here: evidence is never a free-floating string. Every
piece of evidence is a `Span` — (doc_id, start, end) character offsets into a source
document. Compression drops spans from the *prompt*; it never destroys them. Restoration
re-resolves dropped spans from the corpus.

That makes reversibility a verifiable property of the span index rather than an artifact
of "we happened to keep a backup" — see `Span.resolve` and the round-trip test in
tests/test_compression.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Document:
    """A source document in the knowledge base."""

    doc_id: str
    text: str
    meta: dict = field(default_factory=dict)


# The corpus is the authority for span resolution: doc_id -> Document.
Corpus = dict[str, Document]


@dataclass(frozen=True)
class Span:
    """Character offsets into a source document.

    This is the reversibility handle. As long as a Span is retained, the exact original
    text can always be recovered from the corpus, regardless of what compression did.
    """

    doc_id: str
    start: int
    end: int

    def resolve(self, corpus: Corpus) -> str:
        """Recover the exact original text this span points at."""
        doc = corpus.get(self.doc_id)
        if doc is None:
            raise KeyError(f"span references unknown document {self.doc_id!r}")
        if self.start < 0 or self.end > len(doc.text) or self.start >= self.end:
            raise ValueError(
                f"span [{self.start}:{self.end}] out of bounds for document "
                f"{self.doc_id!r} of length {len(doc.text)}"
            )
        return doc.text[self.start : self.end]


@dataclass
class EvidenceUnit:
    """One retrievable, droppable piece of evidence.

    `text` is denormalised from `span` for convenience and prompt assembly. The two must
    always agree — `verify_against` asserts it, and the chunker invariant test in
    tests/test_kb.py checks it holds for every chunk in the corpus.
    """

    span: Span
    text: str
    # First-stage bi-encoder cosine. Kept separate from `rerank_score` because the
    # uncertainty estimator is calibrated against this scale — overwriting it would
    # silently redefine the compression budget as a side effect of a reranking change.
    retrieval_score: float = 0.0
    # Second-stage cross-encoder score, when reranking is enabled. None means not reranked.
    rerank_score: float | None = None
    uncertainty: float = 0.0
    token_count: int = 0

    def verify_against(self, corpus: Corpus) -> bool:
        """True if this unit's cached text still matches what its span resolves to."""
        return self.span.resolve(corpus) == self.text

    @property
    def rank_score(self) -> float:
        """The score every stage orders by — reranked if available, retrieval otherwise.

        Compression and restoration must agree on this, or reranking would be silently
        undone: compression sorting by `retrieval_score` after a reranker had reordered
        the candidates would discard exactly the work the reranker did.
        """
        return self.retrieval_score if self.rerank_score is None else self.rerank_score


@dataclass
class ClaimEvidence:
    """The full, uncompressed evidence set for a single claim (pipeline stage 3).

    `token_count` recorded here is the baseline every compression ratio is measured
    against, so it is computed before any compression runs.
    """

    claim: str
    units: list[EvidenceUnit] = field(default_factory=list)
    token_count: int = 0
    uncertainty: float = 0.0


@dataclass
class CompressedEvidence:
    """Result of uncertainty-guided compression for one claim.

    IMPORTANT — what is actually compressed: the token saving is in what gets *sent to
    the LLM*, not in memory. Dropped units are retained in `dropped` so restoration is
    cheap, and their spans mean the original text is recoverable from the corpus even if
    the retained copies were discarded. Both recovery paths are asserted equivalent by
    the round-trip test.
    """

    claim: str
    kept: list[EvidenceUnit] = field(default_factory=list)
    dropped: list[EvidenceUnit] = field(default_factory=list)
    uncertainty: float = 0.0
    keep_ratio: float = 1.0
    original_token_count: int = 0
    compressed_token_count: int = 0

    @property
    def tokens_saved(self) -> int:
        return self.original_token_count - self.compressed_token_count

    @property
    def reduction(self) -> float:
        """Fraction of baseline tokens removed. 0.0 when nothing was compressed."""
        if self.original_token_count == 0:
            return 0.0
        return self.tokens_saved / self.original_token_count


# Which branch of the pipeline produced the final output.
Branch = Literal[
    "direct",              # compressed evidence was sufficient straight away
    "restored",            # insufficient -> restored -> then sufficient
    "abstain_clarify",     # still insufficient; question looks ambiguous
    "abstain_retrieve",    # still insufficient; KB looks to lack the evidence
]


@dataclass
class ClaimVerdict:
    """Output of claim support evaluation (stages 5 and 6.2)."""

    claim: str
    is_sufficient: bool
    support_score: float
    best_unit_index: int = -1


@dataclass
class Answer:
    """Final pipeline output — either a generated answer or a deliberate abstention."""

    query: str
    text: str
    branch: Branch
    abstained: bool = False
    claims: list[str] = field(default_factory=list)
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
