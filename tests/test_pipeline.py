"""Pipeline orchestration tests.

Run against stubbed retrieval, evaluation, and generation so all three branches
(direct / restored / abstain) can be forced deterministically without model weights.
Branch selection is control flow, and control flow is what these tests pin down.
"""

from __future__ import annotations

import pytest

from src.pipeline import Pipeline
from src.types import ClaimVerdict, EvidenceUnit


class StubRetriever:
    def __init__(self, units, score_fn=None):
        self.units = units
        self.score_fn = score_fn or (lambda i: 0.9 - 0.1 * i)
        self.returned: list[EvidenceUnit] = []

    def retrieve(self, query, top_k=None, restrict_to=None):
        k = top_k or 6
        out = [
            EvidenceUnit(span=u.span, text=u.text, retrieval_score=self.score_fn(i))
            for i, u in enumerate(self.units[:k])
        ]
        self.returned.extend(out)
        return out


class ScriptedEvaluator:
    """Returns support scores from a queue, so branches can be forced exactly."""

    def __init__(self, scores, threshold=0.55):
        self.scores = list(scores)
        self.threshold = threshold
        self.calls = 0

    def evaluate(self, claim, units):
        score = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        return ClaimVerdict(
            claim=claim,
            is_sufficient=score >= self.threshold,
            support_score=score,
            best_unit_index=0,
        )


class StubGenerator:
    def __init__(self):
        self.prompts = []

    def complete(self, prompt, max_new_tokens=None):
        self.prompts.append(prompt)
        return "stub answer"


class StubCounter:
    is_exact = True

    def count(self, text):
        return len(text.split())


def build(cfg, units, scores, score_fn=None):
    scorer = cfg.require("evaluation.scorer")
    threshold = float(cfg.require(f"evaluation.scorers.{scorer}.threshold"))
    evaluator = ScriptedEvaluator(scores, threshold)
    generator = StubGenerator()
    pipeline = Pipeline(
        cfg=cfg,
        retriever=StubRetriever(units, score_fn=score_fn),
        evaluator=evaluator,
        generator=generator,
        counter=StubCounter(),
    )
    return pipeline, evaluator, generator


class TestDirectBranch:
    def test_sufficient_evidence_answers_without_restoring(self, cfg, units):
        pipeline, _, generator = build(cfg, units, scores=[0.95])
        answer, trace = pipeline.run("What is the boiling point of water?")
        assert answer.branch == "direct"
        assert not answer.abstained
        assert not trace.restoration_triggered
        assert generator.prompts, "generator should have been called"

    def test_direct_branch_saves_tokens(self, cfg, units):
        pipeline, _, _ = build(cfg, units, scores=[0.95])
        _, trace = pipeline.run("What is the boiling point of water?")
        assert trace.final_tokens < trace.baseline_tokens
        assert trace.reduction > 0


class TestRestorationBranch:
    def test_insufficient_then_sufficient_restores(self, cfg, units):
        """First check fails, restoration runs, second check passes."""
        pipeline, evaluator, _ = build(cfg, units, scores=[0.10, 0.95])
        answer, trace = pipeline.run("What is the boiling point of water?")
        assert answer.branch == "restored"
        assert not answer.abstained
        assert trace.restoration_triggered
        assert evaluator.calls == 2, "evaluator must be called again after restoring"

    def test_restored_claim_is_charged_full_token_cost(self, cfg, units):
        """A restored claim gives its savings back — the trace must not pretend otherwise."""
        pipeline, _, _ = build(cfg, units, scores=[0.10, 0.95])
        _, trace = pipeline.run("What is the boiling point of water?")
        assert trace.final_tokens == trace.baseline_tokens
        assert trace.reduction == 0.0

    def test_restored_evidence_reaches_the_generator(self, cfg, units):
        """Every retrieved unit — including the dropped ones — must be back in the prompt."""
        pipeline, _, generator = build(cfg, units, scores=[0.10, 0.95])
        pipeline.run("What is the boiling point of water?")
        prompt = generator.prompts[0]
        for unit in pipeline.retriever.returned:
            assert unit.text in prompt, "restored prompt should contain all evidence"

    def test_dropped_evidence_is_absent_before_restoration(self, cfg, units):
        """Guards the test above: it only proves something if compression dropped units."""
        pipeline, _, generator = build(cfg, units, scores=[0.95])
        pipeline.run("What is the boiling point of water?")
        prompt = generator.prompts[0]
        missing = [u for u in pipeline.retriever.returned if u.text not in prompt]
        assert missing, "compression should have dropped at least one unit"

    def test_trace_records_both_evaluations(self, cfg, units):
        pipeline, _, _ = build(cfg, units, scores=[0.10, 0.95])
        _, trace = pipeline.run("What is the boiling point of water?")
        claim = trace.claims[0]
        assert claim.support_score == pytest.approx(0.10)
        assert claim.restored_support_score == pytest.approx(0.95)
        assert claim.restored_sufficient is True


class TestAbstainBranch:
    def test_still_insufficient_after_restoration_abstains(self, cfg, units):
        pipeline, _, generator = build(cfg, units, scores=[0.10, 0.12])
        answer, trace = pipeline.run("What is the boiling point of water?")
        assert answer.abstained
        assert answer.branch.startswith("abstain")
        assert trace.abstained
        assert generator.prompts == [], "must not generate an answer when abstaining"

    def test_abstain_names_the_unsupported_claims(self, cfg, units):
        pipeline, _, _ = build(cfg, units, scores=[0.10, 0.12])
        answer, _ = pipeline.run("What is the boiling point of water?")
        assert answer.unsupported_claims
        assert answer.unsupported_claims[0] in answer.text

    def test_low_retrieval_scores_recommend_retrieving_more(self, cfg, units):
        """Nothing matched well -> the KB is the problem, not the question."""
        pipeline, _, _ = build(cfg, units, scores=[0.05, 0.05], score_fn=lambda i: 0.05)
        answer, _ = pipeline.run("What is the airspeed of an unladen swallow?")
        assert answer.branch == "abstain_retrieve"
        assert "knowledge base" in answer.text.lower()

    def test_abstain_text_is_a_refusal_not_an_answer(self, cfg, units):
        pipeline, _, _ = build(cfg, units, scores=[0.10, 0.12])
        answer, _ = pipeline.run("What is the boiling point of water?")
        assert "cannot answer" in answer.text.lower()


class TestTracing:
    def test_trace_covers_every_claim(self, cfg, units):
        pipeline, _, _ = build(cfg, units, scores=[0.95])
        answer, trace = pipeline.run("What is chunking and also why do offsets matter?")
        assert len(trace.claims) == len(answer.claims) == 2

    def test_trace_records_uncertainty_and_keep_ratio(self, cfg, units):
        pipeline, _, _ = build(cfg, units, scores=[0.95])
        _, trace = pipeline.run("What is the boiling point of water?")
        claim = trace.claims[0]
        assert 0.0 <= claim.uncertainty <= 1.0
        assert 0.0 <= claim.keep_ratio <= 1.0
        assert claim.n_kept + claim.n_dropped == claim.n_candidates

    def test_trace_serialises(self, cfg, units):
        pipeline, _, _ = build(cfg, units, scores=[0.95])
        _, trace = pipeline.run("What is the boiling point of water?")
        payload = trace.to_dict()
        assert "reduction" in payload and "final_tokens" in payload
        assert trace.to_json().startswith("{")

    def test_summary_renders(self, cfg, units):
        pipeline, _, _ = build(cfg, units, scores=[0.95])
        _, trace = pipeline.run("What is the boiling point of water?")
        assert "tokens:" in trace.summary()


class TestBaselineMode:
    def test_identity_mode_keeps_all_evidence(self, cfg, units):
        pipeline, _, _ = build(cfg, units, scores=[0.95])
        _, trace = pipeline.run(
            "What is the boiling point of water?", compression_mode="identity"
        )
        assert trace.reduction == 0.0
        assert trace.claims[0].n_dropped == 0


def with_policy(cfg, policy, retain_fraction=0.9):
    cfg["restoration"] = dict(
        cfg.get("restoration", {}), policy=policy, retain_fraction=retain_fraction
    )
    return cfg


class TestRestorationPolicy:
    """Restoration and abstention were decided by one threshold. They answer different
    questions, and the shared signal answered the restoration one badly: measured on
    QASPER it fired on 0 of 6 cases where compression removed every trace of the marked
    answer evidence."""

    def test_unknown_policy_fails_loudly(self, cfg, units):
        pipeline, _, _ = build(with_policy(cfg, "nonsense"), units, scores=[0.10, 0.95])
        with pytest.raises(ValueError, match="unknown restoration.policy"):
            pipeline.run("What is the boiling point of water?")

    def test_identity_never_restores_under_either_policy(self, cfg, units):
        """Nothing was dropped, so there is nothing to put back — and the relative test
        must not be allowed to fire on a set it never compressed."""
        for policy in ("absolute", "relative"):
            pipeline, _, _ = build(
                with_policy(cfg, policy), units, scores=[0.01, 0.99, 0.99]
            )
            _, trace = pipeline.run(
                "What is the boiling point of water?", compression_mode="identity"
            )
            assert not trace.restoration_triggered, policy

    def test_relative_restores_when_support_falls_against_the_full_set(self, cfg, units):
        # compressed 0.50, full 1.00 -> 0.50 < 0.9 * 1.00, so restore; then 0.95 passes.
        pipeline, _, _ = build(
            with_policy(cfg, "relative"), units, scores=[0.50, 1.00, 0.95]
        )
        _, trace = pipeline.run("What is the boiling point of water?")
        assert trace.restoration_triggered

    def test_relative_leaves_a_near_lossless_compression_alone(self, cfg, units):
        # compressed 0.98 against full 1.00 -> above the 0.9 retention bar.
        pipeline, _, _ = build(
            with_policy(cfg, "relative"), units, scores=[0.98, 1.00]
        )
        _, trace = pipeline.run("What is the boiling point of water?")
        assert not trace.restoration_triggered

    def test_relative_catches_the_drop_absolute_misses(self, cfg, units):
        """The failure this policy exists for.

        Compressed support is 0.60 — far above the abstain threshold, so `absolute` sees
        a healthy claim and never restores. But the full evidence scored 1.00, meaning
        compression removed 40% of the support. `relative` sees that; `absolute` cannot.
        """
        absolute, _, _ = build(with_policy(cfg, "absolute"), units, scores=[0.60])
        _, absolute_trace = absolute.run("What is the boiling point of water?")
        assert not absolute_trace.restoration_triggered

        relative, _, _ = build(
            with_policy(cfg, "relative"), units, scores=[0.60, 1.00, 0.99]
        )
        _, relative_trace = relative.run("What is the boiling point of water?")
        assert relative_trace.restoration_triggered

    def test_relative_falls_back_when_the_full_set_supports_nothing(self, cfg, units):
        """A zero-support claim would make the ratio meaningless; restoring on it would
        pay full token cost to recover evidence that never helped."""
        pipeline, _, _ = build(
            with_policy(cfg, "relative"), units, scores=[0.0, 0.0, 0.0]
        )
        answer, trace = pipeline.run("What is the boiling point of water?")
        assert answer.abstained

    def test_relative_costs_an_extra_scoring_pass(self, cfg, units):
        """Stated so the cost is visible: the policy is not free."""
        absolute, abs_eval, _ = build(with_policy(cfg, "absolute"), units, scores=[0.95])
        absolute.run("What is the boiling point of water?")

        relative, rel_eval, _ = build(
            with_policy(cfg, "relative"), units, scores=[0.95, 1.00]
        )
        relative.run("What is the boiling point of water?")
        assert rel_eval.calls > abs_eval.calls


class TestAbstainGate:
    """Which signal decides a refusal. Measured on a real uploaded paper, the sentence
    containing the answer scored 0.00085 on the support scorer while still being the
    top-ranked unit retrieved — the ranking is informative, the absolute scale is not.
    Refusing a question whose answer sits at rank 1 is the worst failure the reading room
    can produce."""

    def test_support_gate_refuses_on_low_support(self, cfg, units):
        pipeline, _, _ = build(cfg, units, scores=[0.001, 0.001])
        answer, _ = pipeline.run("What is the boiling point of water?")
        assert answer.abstained

    def test_retrieval_gate_answers_despite_low_support(self, cfg, units):
        """The fix: strong retrieval overrides an uninformative support score."""
        cfg["abstain"] = dict(cfg["abstain"], gate="retrieval", low_retrieval_score=0.25)
        pipeline, _, _ = build(cfg, units, scores=[0.001, 0.001])
        answer, _ = pipeline.run("What is the boiling point of water?")
        assert not answer.abstained

    def test_retrieval_gate_still_refuses_when_nothing_is_close(self, cfg, units):
        """It must remain able to say no, or the honesty claim is gone entirely."""
        cfg["abstain"] = dict(cfg["abstain"], gate="retrieval", low_retrieval_score=0.25)
        pipeline, _, _ = build(cfg, units, scores=[0.99], score_fn=lambda i: 0.05)
        answer, _ = pipeline.run("What is the boiling point of water?")
        assert answer.abstained

    def test_default_gate_is_unchanged(self, cfg, units):
        """Archived experiments must keep the behaviour they were measured with."""
        assert cfg.get_path("abstain.gate", "support") == "support"
