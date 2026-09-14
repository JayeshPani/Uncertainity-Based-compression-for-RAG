"""Pipeline stage 1 — retrieve candidate evidence from the knowledge base.

Deliberately dependency-light: embeddings come from `transformers.AutoModel` with mean
pooling (no sentence-transformers), and search is exact brute-force cosine over numpy (no
FAISS). For a KB of this size exact search is fast, always returns the true nearest
neighbours, and removes an approximate-index tuning burden that would otherwise be a
silent source of recall loss.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .config import REPO_ROOT, Config, default_config, resolve_device
from .kb import chunk_corpus, corpus_fingerprint, load_corpus
from .types import Corpus, EvidenceUnit

logger = logging.getLogger(__name__)


def _mean_pool(last_hidden_state, attention_mask):
    """Mean-pool token embeddings, ignoring padding positions."""
    import torch

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class Embedder:
    """Mean-pooled transformer encoder producing L2-normalised vectors.

    Vectors are unit-normalised at creation, so cosine similarity is a plain dot product
    and scores land in [-1, 1] — which the uncertainty estimator relies on.
    """

    def __init__(self, model_id: str | None = None, cfg: Config | None = None) -> None:
        cfg = cfg or default_config()
        self.model_id = model_id or cfg.require("models.embedding")
        self.device = resolve_device(cfg.get_path("models.device", "auto"))
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            logger.info("loading embedding model %s on %s", self.model_id, self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id).to(self.device)
            self._model.eval()
            self._torch = torch
        return self._model, self._tokenizer

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to a (n, dim) float32 array of unit vectors."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        model, tokenizer = self._load()
        torch = self._torch
        out: list[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                hidden = model(**encoded).last_hidden_state
                pooled = _mean_pool(hidden, encoded["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out.append(pooled.cpu().numpy().astype(np.float32))

        return np.vstack(out)


class Retriever:
    """Brute-force cosine retriever over chunked KB evidence."""

    def __init__(self, corpus: Corpus | None = None, cfg: Config | None = None) -> None:
        self.cfg = cfg or default_config()
        self.corpus = corpus if corpus is not None else load_corpus(cfg=self.cfg)
        self.units: list[EvidenceUnit] = chunk_corpus(self.corpus, cfg=self.cfg)
        self.embedder = Embedder(cfg=self.cfg)
        self._matrix: np.ndarray | None = None

    def _cache_path(self) -> Path:
        cache_dir = Path(self.cfg.get_path("retrieval.cache_dir", ".cache/embeddings"))
        if not cache_dir.is_absolute():
            cache_dir = REPO_ROOT / cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Keyed by model AND corpus content: changing either misses rather than
        # returning a stale vector for text that no longer exists.
        model_key = self.embedder.model_id.replace("/", "__")
        return cache_dir / f"{model_key}__{corpus_fingerprint(self.corpus)}.npy"

    def build(self, use_cache: bool = True) -> np.ndarray:
        """Embed every chunk, loading from disk cache when valid."""
        if self._matrix is not None:
            return self._matrix

        cache_path = self._cache_path()
        if use_cache and cache_path.exists():
            matrix = np.load(cache_path)
            if matrix.shape[0] == len(self.units):
                logger.info("loaded %d cached embeddings", matrix.shape[0])
                self._matrix = matrix
                return matrix
            logger.warning("cache size mismatch, re-embedding")

        logger.info("embedding %d chunks", len(self.units))
        matrix = self.embedder.encode([u.text for u in self.units])
        if use_cache:
            np.save(cache_path, matrix)
        self._matrix = matrix
        return matrix

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        restrict_to: str | None = None,
    ) -> list[EvidenceUnit]:
        """Return the top-k evidence units for a query, scored by cosine similarity.

        `restrict_to` limits candidates to a single document. This is needed for
        single-document benchmarks such as QASPER, where questions are asked *about a
        given paper* and are not self-identifying — "which datasets did they experiment
        with?" has no answer without knowing which paper "they" refers to. Searching the
        whole corpus for such a question retrieves confident, well-scoring evidence from
        entirely the wrong paper.

        Fresh EvidenceUnit copies are returned so per-query scores never mutate the
        retriever's shared chunk list.
        """
        k = top_k if top_k is not None else int(self.cfg.require("retrieval.top_k"))
        matrix = self.build()
        if matrix.shape[0] == 0:
            return []

        allowed = None
        if restrict_to is not None:
            allowed = np.array(
                [i for i, u in enumerate(self.units) if u.span.doc_id == restrict_to]
            )
            if allowed.size == 0:
                return []
            matrix = matrix[allowed]

        query_vec = self.embedder.encode([query])[0]
        # Unit vectors, so the dot product is cosine similarity.
        # errstate: numpy 2.x on Apple Accelerate leaks spurious divide/overflow/invalid
        # flags out of its vectorised matmul kernel. Verified harmless here — inputs
        # contain no NaN or inf, all rows are unit-norm, and results match a float64
        # computation to ~1e-8. Suppressed narrowly so real numerical faults elsewhere
        # still surface.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scores = matrix @ query_vec

        k = min(k, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        results: list[EvidenceUnit] = []
        for idx in top_idx:
            # Map back to corpus positions when the candidate set was narrowed.
            unit = self.units[allowed[idx] if allowed is not None else idx]
            results.append(
                EvidenceUnit(
                    span=unit.span,
                    text=unit.text,
                    retrieval_score=float(scores[idx]),
                )
            )
        return results
