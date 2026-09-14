"""Pipeline stages 5 and 6.2 — claim support evaluation.

Called twice per claim: once on compressed evidence, and again on restored evidence when
the first check fails. It is therefore stateless — the same evaluator instance answers
both questions and nothing carries over between calls.

Support is scored by a cross-encoder returning a continuous score, not by asking a
generative model "is this supported, yes or no?". A 1.5B generator gives a poorly
calibrated binary; a scalar can be thresholded, logged, tuned, and ablated.

WHICH SCORER, AND WHY IT DEPENDS ON WHERE CLAIMS COME FROM
---------------------------------------------------------
NLI entailment is the textbook choice for claim verification, and it is the wrong
default *here*. Claims in this pipeline are decomposed from the user's question, so they
are topics ("dense retrieval differ from lexical retrieval"), not assertions. Entailment
asks "does the evidence make this statement true", and a single retrieved sentence
genuinely does not entail a topic phrase. Measured with DeBERTa-v3-large-mnli:

    claim                                        P(entailment)
    "dense retrieval differ from lexical..."         0.009
    "dense retrieval"                                0.877
    "hallucination in language models"               0.014
    "hallucination"                                  0.502

Only bare head nouns scored well, and adding any modifier collapsed the score — the
model was answering its own question correctly, just not ours. Across the eval set that
produced a 14.3% false-abstain rate on questions the KB plainly answers.

A relevance cross-encoder asks the question we actually mean — "does this evidence
address this claim?" — and separates cleanly on the same set: lowest answerable claim
0.205, highest unanswerable 0.001.

Both scorers ship. `nli` becomes the right choice if claims are ever decomposed from a
draft answer instead (see README, Fall extension), because those claims *are*
assertions. Keeping both behind one interface makes that switch a config change and
gives a ready-made ablation.

Evidence is scored *contextualised* — prefixed with its document title. Sentence chunks
are frequently anaphoric ("The function has two tunable parameters" never names BM25);
without the prefix, support collapses (measured 0.004 -> 0.997 for the same chunk).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from .config import Config, default_config, resolve_device
from .evidence_mapping import contextualize
from .types import ClaimVerdict, EvidenceUnit

logger = logging.getLogger(__name__)


class SupportScorer(ABC):
    """Scores how strongly each premise supports a claim, on a [0, 1] scale."""

    name: str = "base"

    def __init__(self, model_id: str, threshold: float, device: str) -> None:
        self.model_id = model_id
        self.threshold = threshold
        self.device = device
        self._model = None
        self._tokenizer = None
        # Cost instrumentation. Compression is not free: it buys generator tokens by
        # spending scorer forward passes, and a token saving reported without that
        # exchange rate is only half a cost claim. `relative` restoration in particular
        # doubles these.
        self.calls = 0
        self.premises_scored = 0
        # Scoring cache, keyed by (claim, premise). The ablation runs the same claim
        # against overlapping evidence sets many times over — every arm scores the same
        # question, the relative policy scores the compressed and full sets, and graded
        # restoration scores each rung — so the same (claim, premise) pair recurs
        # constantly. The cache is per-scorer, so a change of model or device gets a fresh
        # one by construction rather than by remembering to invalidate.
        self._cache: dict[tuple[str, str], float] = {}
        self.cache_hits = 0

    def _load(self):
        if self._model is None:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            logger.info("loading %s scorer %s on %s", self.name, self.model_id, self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id
            ).to(self.device)
            self._model.eval()
            self._torch = torch
            self._post_load()
        return self._model, self._tokenizer

    def _post_load(self) -> None:
        """Hook for subclasses needing to inspect the loaded config."""

    def _encode(self, premises: list[str], claim: str):
        model, tokenizer = self._load()
        self.calls += 1
        self.premises_scored += len(premises)
        return model, tokenizer, tokenizer(
            premises,
            [claim] * len(premises),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

    def score(self, premises: list[str], claim: str) -> list[float]:
        """Score each premise against `claim`, computing only what is not already known.

        Caching happens here rather than in each subclass so both scorers get it, and so
        `_score_uncached` stays a pure "run the model on these" function that is easy to
        reason about and to test.
        """
        if not premises:
            return []

        missing = [p for p in dict.fromkeys(premises) if (claim, p) not in self._cache]
        self.cache_hits += len(premises) - len(missing)
        if missing:
            for premise, score in zip(missing, self._score_uncached(missing, claim)):
                self._cache[(claim, premise)] = score
        return [self._cache[(claim, p)] for p in premises]

    @abstractmethod
    def _score_uncached(self, premises: list[str], claim: str) -> list[float]:
        ...


class RelevanceScorer(SupportScorer):
    """Query-passage relevance cross-encoder. Single logit, squashed with a sigmoid."""

    name = "relevance"

    def _score_uncached(self, premises: list[str], claim: str) -> list[float]:
        model, _, encoded = self._encode(premises, claim)
        torch = self._torch
        with torch.no_grad():
            logits = model(**encoded).logits
        return torch.sigmoid(logits[:, 0]).cpu().tolist()


class NliScorer(SupportScorer):
    """Three-way NLI model. Support is P(entailment)."""

    name = "nli"

    def _post_load(self) -> None:
        # Resolve the entailment column from the model's own label map rather than
        # assuming an index — MNLI checkpoints order their labels differently.
        id2label = self._model.config.id2label
        self._entailment_index = None
        for idx, label in id2label.items():
            if str(label).upper().startswith("ENTAIL"):
                self._entailment_index = int(idx)
                break
        if self._entailment_index is None:
            raise ValueError(f"no ENTAILMENT label in {self.model_id}: {id2label}")

    def _score_uncached(self, premises: list[str], claim: str) -> list[float]:
        model, _, encoded = self._encode(premises, claim)
        torch = self._torch
        with torch.no_grad():
            probs = torch.softmax(model(**encoded).logits, dim=-1)
        return probs[:, self._entailment_index].cpu().tolist()


_SCORERS: dict[str, type[SupportScorer]] = {
    RelevanceScorer.name: RelevanceScorer,
    NliScorer.name: NliScorer,
}


def scorer_from_config(cfg: Config | None = None) -> SupportScorer:
    """Build the scorer named by `evaluation.scorer`, with its own threshold."""
    cfg = cfg or default_config()
    name = cfg.get_path("evaluation.scorer", RelevanceScorer.name)
    if name not in _SCORERS:
        raise ValueError(f"unknown support scorer {name!r}; available: {sorted(_SCORERS)}")

    settings = cfg.get_path(f"evaluation.scorers.{name}") or {}
    if "model" not in settings or "threshold" not in settings:
        raise KeyError(f"evaluation.scorers.{name} needs both 'model' and 'threshold'")

    return _SCORERS[name](
        model_id=settings["model"],
        threshold=float(settings["threshold"]),
        device=resolve_device(cfg.get_path("models.device", "auto")),
    )


class SupportEvaluator:
    """Decides whether evidence supports a claim. Stateless across calls."""

    def __init__(self, cfg: Config | None = None, scorer: SupportScorer | None = None) -> None:
        self.cfg = cfg or default_config()
        self.scorer = scorer or scorer_from_config(self.cfg)
        self.aggregate = self.cfg.get_path("evaluation.aggregate", "max")
        self.contextualize = bool(self.cfg.get_path("evaluation.contextualize", True))

    @property
    def threshold(self) -> float:
        return self.scorer.threshold

    def score_units(self, claim: str, units: list[EvidenceUnit]) -> list[float]:
        """Score each unit independently.

        Units are scored one at a time rather than concatenated: measured on the seeded
        KB, concatenating a claim's units dropped its score from 0.997 to 0.510 because
        irrelevant neighbours dilute the signal.
        """
        if not units:
            return []
        premises = [contextualize(u, self.contextualize) for u in units]
        return self.scorer.score(premises, claim)

    def evaluate(self, claim: str, units: list[EvidenceUnit]) -> ClaimVerdict:
        scores = self.score_units(claim, units)
        if not scores:
            return ClaimVerdict(
                claim=claim, is_sufficient=False, support_score=0.0, best_unit_index=-1
            )

        best = max(range(len(scores)), key=scores.__getitem__)
        if self.aggregate == "mean":
            support = sum(scores) / len(scores)
        else:  # "max" — a claim needs one convincing piece of evidence, not consensus
            support = scores[best]

        return ClaimVerdict(
            claim=claim,
            is_sufficient=support >= self.scorer.threshold,
            support_score=float(support),
            best_unit_index=best,
        )
