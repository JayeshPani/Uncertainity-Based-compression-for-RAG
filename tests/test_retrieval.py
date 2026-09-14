"""Retrieval tests, including single-document restriction.

`restrict_to` exists because of a real measurement error. QASPER questions are asked
about one specific paper and are not self-identifying — "which datasets did they
experiment with?" means nothing without knowing which paper "they" is. Run against a
110-paper corpus, retrieval returned confident evidence from the wrong papers, and the
whole evaluation reported ~1% answer F1 and 83% false abstention *even with no
compression at all*. The bug was in the harness, not the pipeline, and these tests pin
the fix so it cannot regress silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.retrieval import Retriever
from src.types import Document


@pytest.fixture
def multi_doc_corpus():
    return {
        "paper_a": Document(
            doc_id="paper_a",
            text=(
                "We evaluate on the Europarl corpus. "
                "Our model uses a transformer encoder with eight layers. "
                "Training took three days on four GPUs."
            ),
        ),
        "paper_b": Document(
            doc_id="paper_b",
            text=(
                "We evaluate on the SQuAD dataset. "
                "Our model uses a recurrent encoder with two layers. "
                "Training took six hours on one GPU."
            ),
        ),
    }


class FakeEmbedder:
    """Deterministic bag-of-words vectors — keeps the test offline and exact."""

    model_id = "fake"

    def __init__(self, vocab):
        self.vocab = vocab

    def encode(self, texts, batch_size=32):
        out = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for i, text in enumerate(texts):
            words = set(text.lower().replace(".", "").split())
            for j, term in enumerate(self.vocab):
                out[i, j] = 1.0 if term in words else 0.0
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


@pytest.fixture
def retriever(multi_doc_corpus, cfg):
    r = Retriever(corpus=multi_doc_corpus, cfg=cfg)
    r.embedder = FakeEmbedder(
        ["europarl", "squad", "transformer", "recurrent", "gpus", "gpu", "dataset", "corpus"]
    )
    r._matrix = r.embedder.encode([u.text for u in r.units])
    return r


class TestRestrictTo:
    def test_unrestricted_search_spans_the_corpus(self, retriever):
        docs = {u.span.doc_id for u in retriever.retrieve("encoder layers", top_k=6)}
        assert docs == {"paper_a", "paper_b"}

    def test_restriction_confines_results_to_one_paper(self, retriever):
        results = retriever.retrieve("encoder layers", top_k=6, restrict_to="paper_b")
        assert results
        assert {u.span.doc_id for u in results} == {"paper_b"}

    def test_restriction_returns_that_paper_s_own_answer(self, retriever):
        """The point of the fix: the right paper's evidence, not a confident wrong one."""
        results = retriever.retrieve("which dataset", top_k=3, restrict_to="paper_a")
        assert any("Europarl" in u.text for u in results)
        assert not any("SQuAD" in u.text for u in results)

    def test_unknown_document_returns_nothing(self, retriever):
        assert retriever.retrieve("anything", restrict_to="paper_zzz") == []

    def test_spans_still_resolve_after_restriction(self, retriever, multi_doc_corpus):
        """Index remapping must not corrupt the span invariant."""
        for unit in retriever.retrieve("encoder", top_k=3, restrict_to="paper_b"):
            assert unit.verify_against(multi_doc_corpus)

    def test_scores_match_the_unrestricted_run(self, retriever):
        """Restriction filters candidates; it must not change how they are scored."""
        restricted = retriever.retrieve("encoder layers", top_k=6, restrict_to="paper_a")
        everything = retriever.retrieve("encoder layers", top_k=99)
        expected = {u.text: u.retrieval_score for u in everything}
        for unit in restricted:
            assert unit.retrieval_score == pytest.approx(expected[unit.text])

    def test_results_stay_sorted(self, retriever):
        scores = [
            u.retrieval_score
            for u in retriever.retrieve("encoder", top_k=5, restrict_to="paper_a")
        ]
        assert scores == sorted(scores, reverse=True)
