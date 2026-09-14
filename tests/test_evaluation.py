"""Claim support evaluation tests.

Split in two:

* `TestScorerSelection` is pure config wiring and runs everywhere.
* The rest run the real cross-encoder — the behaviour worth testing (relevant evidence
  scores high, irrelevant scores low) only exists in the weights. Marked `integration`
  and skipped when the model is unavailable.

Run just these:      pytest -m integration
Skip them entirely:  pytest -m "not integration"
"""

from __future__ import annotations

import pytest

from src.config import load_config
from src.evaluation import (
    NliScorer,
    RelevanceScorer,
    SupportEvaluator,
    scorer_from_config,
)
from src.types import ClaimVerdict, EvidenceUnit, Span

BM25_TEXT = (
    "The function has two tunable parameters, k1 and b, which control saturation "
    "and length normalisation respectively."
)
OCEAN_TEXT = "The Pacific is the largest ocean on Earth."
BM25_CLAIM = "the two tunable parameters of BM25"


def unit(doc_id: str, text: str) -> EvidenceUnit:
    return EvidenceUnit(span=Span(doc_id, 0, len(text)), text=text, retrieval_score=0.8)


class StubScorer:
    """Returns queued scores, so aggregation can be tested without model weights."""

    threshold = 0.15

    def __init__(self, scores):
        self.scores = scores

    def score(self, premises, claim):
        return self.scores[: len(premises)]


class TestScorerSelection:
    def test_default_scorer_is_relevance(self, cfg):
        """Claims come from the query, so relevance is correct; NLI would misfire."""
        assert cfg.require("evaluation.scorer") == "relevance"
        assert isinstance(scorer_from_config(cfg), RelevanceScorer)

    def test_nli_scorer_is_selectable(self, cfg):
        swapped = type(cfg)(dict(cfg))
        swapped["evaluation"] = dict(cfg["evaluation"], scorer="nli")
        assert isinstance(scorer_from_config(swapped), NliScorer)

    def test_unknown_scorer_raises(self, cfg):
        broken = type(cfg)(dict(cfg))
        broken["evaluation"] = dict(cfg["evaluation"], scorer="nope")
        with pytest.raises(ValueError, match="unknown support scorer"):
            scorer_from_config(broken)

    def test_each_scorer_has_its_own_threshold(self, cfg):
        """A shared threshold would silently misbehave — the scales differ."""
        scorers = cfg.require("evaluation.scorers")
        assert scorers["relevance"]["threshold"] != scorers["nli"]["threshold"]
        for settings in scorers.values():
            assert 0.0 <= float(settings["threshold"]) <= 1.0

    def test_missing_scorer_settings_raise(self, cfg):
        broken = type(cfg)(dict(cfg))
        broken["evaluation"] = dict(cfg["evaluation"], scorer="relevance", scorers={})
        with pytest.raises(KeyError):
            scorer_from_config(broken)


class TestAggregationLogic:
    """Aggregation is arithmetic, so it is tested without loading a model."""

    def test_max_aggregate_takes_the_best_unit(self, cfg):
        evaluator = SupportEvaluator(cfg=cfg, scorer=StubScorer([0.02, 0.91, 0.10]))
        verdict = evaluator.evaluate("c", [unit("a", "x")] * 3)
        assert verdict.support_score == pytest.approx(0.91)
        assert verdict.best_unit_index == 1
        assert verdict.is_sufficient

    def test_mean_aggregate_is_selectable(self, cfg):
        tweaked = type(cfg)(dict(cfg))
        tweaked["evaluation"] = dict(cfg["evaluation"], aggregate="mean")
        evaluator = SupportEvaluator(cfg=tweaked, scorer=StubScorer([0.0, 1.0]))
        assert evaluator.evaluate("c", [unit("a", "x")] * 2).support_score == pytest.approx(0.5)

    def test_threshold_decides_sufficiency(self, cfg):
        evaluator = SupportEvaluator(cfg=cfg, scorer=StubScorer([0.14]))
        assert not evaluator.evaluate("c", [unit("a", "x")]).is_sufficient
        evaluator = SupportEvaluator(cfg=cfg, scorer=StubScorer([0.16]))
        assert evaluator.evaluate("c", [unit("a", "x")]).is_sufficient

    def test_empty_evidence_is_insufficient(self, cfg):
        verdict = SupportEvaluator(cfg=cfg, scorer=StubScorer([])).evaluate("c", [])
        assert isinstance(verdict, ClaimVerdict)
        assert not verdict.is_sufficient
        assert verdict.support_score == 0.0
        assert verdict.best_unit_index == -1


@pytest.fixture(scope="module")
def evaluator():
    """Loaded once for the whole module — the weights are expensive."""
    ev = SupportEvaluator(cfg=load_config())
    try:
        ev.scorer._load()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"support model unavailable: {exc}")
    return ev


@pytest.mark.integration
class TestRealScorer:
    def test_relevant_evidence_is_supported(self, evaluator):
        verdict = evaluator.evaluate(BM25_CLAIM, [unit("bm25", BM25_TEXT)])
        assert verdict.is_sufficient
        assert verdict.support_score > 0.5

    def test_irrelevant_evidence_is_not_supported(self, evaluator):
        verdict = evaluator.evaluate(BM25_CLAIM, [unit("oceans", OCEAN_TEXT)])
        assert not verdict.is_sufficient

    def test_separation_makes_the_threshold_meaningful(self, evaluator):
        good = evaluator.evaluate(BM25_CLAIM, [unit("bm25", BM25_TEXT)]).support_score
        bad = evaluator.evaluate(BM25_CLAIM, [unit("oceans", OCEAN_TEXT)]).support_score
        assert good - bad > 0.5

    def test_title_prefix_rescues_anaphoric_evidence(self, evaluator):
        """The chunk says "The function has..." and never names BM25.

        Without the document title prefixed, the scorer cannot resolve the subject.
        """
        with_title = evaluator.score_units(BM25_CLAIM, [unit("bm25", BM25_TEXT)])[0]
        without_title = evaluator.score_units(BM25_CLAIM, [unit("", BM25_TEXT)])[0]
        assert with_title > without_title

    def test_verb_fragment_claims_still_score(self, evaluator):
        """Regression: these are exactly the claims NLI entailment scored near zero.

        Query decomposition produces ungrammatical fragments like this, and an
        evaluator that cannot handle them shows up as false abstentions on questions
        the knowledge base plainly answers.
        """
        for claim in (
            "dense retrieval differ from lexical retrieval",
            "chunking need character offsets",
            "causes hallucination in language models",
        ):
            evidence = [unit("dense_retrieval", claim.replace("differ", "differs"))]
            assert evaluator.score_units(claim, evidence)[0] > 0.15, claim

    def test_max_aggregate_ignores_irrelevant_neighbours(self, evaluator):
        units = [unit("oceans", OCEAN_TEXT), unit("bm25", BM25_TEXT)]
        verdict = evaluator.evaluate(BM25_CLAIM, units)
        assert verdict.is_sufficient
        assert verdict.best_unit_index == 1

    def test_evaluator_is_stateless_across_calls(self, evaluator):
        """It is invoked twice per claim (stage 5 and 6.2) — no state may carry over."""
        first = evaluator.evaluate(BM25_CLAIM, [unit("bm25", BM25_TEXT)])
        evaluator.evaluate("something unrelated", [unit("oceans", OCEAN_TEXT)])
        second = evaluator.evaluate(BM25_CLAIM, [unit("bm25", BM25_TEXT)])
        assert first.support_score == pytest.approx(second.support_score)


class TestScoringCache:
    """The ablation scores the same (claim, premise) pair many times: every arm sees the
    same question, the relative policy scores compressed and full sets, and graded
    restoration scores each rung. Caching is the difference between a sweep taking minutes
    and taking an hour."""

    class CountingScorer(RelevanceScorer):
        def __init__(self):
            super().__init__(model_id="fake", threshold=0.5, device="cpu")
            self.uncached_calls = 0

        def _score_uncached(self, premises, claim):
            self.uncached_calls += 1
            return [0.5 + 0.01 * len(p) for p in premises]

    def test_repeated_scoring_hits_the_cache(self):
        scorer = self.CountingScorer()
        first = scorer.score(["alpha", "beta"], "claim")
        second = scorer.score(["alpha", "beta"], "claim")
        assert first == second
        assert scorer.uncached_calls == 1
        assert scorer.cache_hits == 2

    def test_only_the_missing_premises_are_computed(self):
        scorer = self.CountingScorer()
        scorer.score(["alpha"], "claim")
        scorer.score(["alpha", "beta"], "claim")
        assert scorer.uncached_calls == 2
        assert scorer.cache_hits == 1

    def test_a_different_claim_is_a_different_key(self):
        """Caching on the premise alone would return one claim's score for another."""
        scorer = self.CountingScorer()
        scorer.score(["alpha"], "claim one")
        scorer.score(["alpha"], "claim two")
        assert scorer.uncached_calls == 2
        assert scorer.cache_hits == 0

    def test_duplicate_premises_in_one_call_are_scored_once(self):
        scorer = self.CountingScorer()
        scores = scorer.score(["alpha", "alpha", "beta"], "claim")
        assert len(scores) == 3
        assert scores[0] == scores[1]

    def test_results_are_returned_in_input_order(self):
        """Deduplication must not permute the output — callers index into it by position,
        and `best_unit_index` would otherwise point at the wrong evidence."""
        scorer = self.CountingScorer()
        premises = ["mid", "a much longer premise", "short"]
        scores = scorer.score(premises, "claim")
        assert scores == [0.5 + 0.01 * len(p) for p in premises]

    def test_order_is_preserved_when_answers_come_from_the_cache(self):
        scorer = self.CountingScorer()
        scorer.score(["beta"], "claim")  # warm one entry only
        premises = ["alpha", "beta", "gamma-long"]
        assert scorer.score(premises, "claim") == [
            0.5 + 0.01 * len(p) for p in premises
        ]

    def test_empty_input_short_circuits(self):
        scorer = self.CountingScorer()
        assert scorer.score([], "claim") == []
        assert scorer.uncached_calls == 0
