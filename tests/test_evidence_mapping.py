"""Evidence mapping tests — run against a stub retriever so no model weights are needed."""

from __future__ import annotations

from src.evidence_mapping import (
    contextualize,
    document_title,
    map_evidence,
    total_baseline_tokens,
)
from src.types import EvidenceUnit


class StubRetriever:
    """Returns fixture chunks with descending scores, no embedding model involved."""

    def __init__(self, units, cfg=None):
        self.units = units
        self.cfg = cfg
        self.calls: list[str] = []

    def retrieve(self, query, top_k=None, restrict_to=None):
        self.calls.append(query)
        k = top_k or 4
        return [
            EvidenceUnit(
                span=u.span, text=u.text, retrieval_score=1.0 - 0.1 * i
            )
            for i, u in enumerate(self.units[:k])
        ]


class StubCounter:
    """Word-count stand-in, so token expectations in tests are obvious by inspection."""

    is_exact = False

    def count(self, text):
        return len(text.split())


class TestDocumentTitle:
    def test_underscores_become_spaces(self):
        assert document_title("dense_retrieval") == "dense retrieval"

    def test_plain_id_unchanged(self):
        assert document_title("bm25") == "bm25"

    def test_contextualize_prefixes_title(self):
        unit = EvidenceUnit(
            span=type("S", (), {"doc_id": "dense_retrieval"})(), text="It encodes text."
        )
        assert contextualize(unit) == "dense retrieval. It encodes text."


class TestMapEvidence:
    def test_every_claim_gets_evidence(self, units, cfg):
        claims = ["boiling point of water", "the Pacific ocean"]
        mapped = map_evidence(claims, StubRetriever(units), cfg=cfg, counter=StubCounter())
        assert len(mapped) == len(claims)
        assert all(m.units for m in mapped), "no claim may end up with empty evidence"

    def test_claims_are_preserved_in_order(self, units, cfg):
        claims = ["alpha", "beta", "gamma"]
        mapped = map_evidence(claims, StubRetriever(units), cfg=cfg, counter=StubCounter())
        assert [m.claim for m in mapped] == claims

    def test_retrieval_is_per_claim(self, units, cfg):
        """Each claim must get its own retrieval pass, not a slice of a shared pool."""
        retriever = StubRetriever(units)
        claims = ["alpha", "beta", "gamma"]
        map_evidence(claims, retriever, cfg=cfg, counter=StubCounter())
        assert retriever.calls == claims

    def test_baseline_tokens_recorded(self, units, cfg):
        mapped = map_evidence(["alpha"], StubRetriever(units), cfg=cfg, counter=StubCounter())
        claim = mapped[0]
        assert claim.token_count > 0
        assert claim.token_count == sum(u.token_count for u in claim.units)

    def test_units_carry_retrieval_scores(self, units, cfg):
        mapped = map_evidence(["alpha"], StubRetriever(units), cfg=cfg, counter=StubCounter())
        scores = [u.retrieval_score for u in mapped[0].units]
        assert scores == sorted(scores, reverse=True)

    def test_total_baseline_tokens_sums_claims(self, units, cfg):
        mapped = map_evidence(
            ["alpha", "beta"], StubRetriever(units), cfg=cfg, counter=StubCounter()
        )
        assert total_baseline_tokens(mapped) == sum(m.token_count for m in mapped)

    def test_spans_survive_mapping(self, units, cfg, corpus):
        """Mapping must not break the span invariant retrieval depends on."""
        mapped = map_evidence(["alpha"], StubRetriever(units), cfg=cfg, counter=StubCounter())
        for unit in mapped[0].units:
            assert unit.verify_against(corpus)
