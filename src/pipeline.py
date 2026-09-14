"""End-to-end orchestration, including the restoration loop and the abstain branch.

    query
      -> decompose into claims                     (claims.py)
      -> retrieve + map evidence claim-wise        (evidence_mapping.py)
      -> uncertainty-guided compression            (compression.py)
      -> claim support evaluation                  (evaluation.py)
           sufficient   -> generate                (generation.py)
           insufficient -> restore                 (restoration.py)
                           -> re-evaluate          (evaluation.py, same instance)
                                sufficient   -> generate
                                insufficient -> abstain / clarify / retrieve more

Restoration is applied per claim, not globally: a claim whose compressed evidence was
already sufficient keeps its savings, and only the claims that actually failed pay the
cost of their full evidence set back.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .claims import decompose
from .compression import compress, to_claim_evidence
from .config import Config, default_config
from .evaluation import SupportEvaluator
from .evidence_mapping import map_evidence
from .generation import Generator, abstain, build_prompt, generate_answer
from .restoration import (
    restore,
    restore_neighbours,
    restore_partial,
    restoration_ladder,
)
from .rerank import Reranker, rerank_evidence
from .retrieval import Retriever
from .tokens import TokenCounter, shared_counter
from .trace import ClaimTrace, EvidenceRecord, PipelineTrace
from .types import Answer, Branch, ClaimEvidence, EvidenceUnit

logger = logging.getLogger(__name__)


def _record(units: list[EvidenceUnit]) -> list[EvidenceRecord]:
    return [
        EvidenceRecord(
            doc_id=u.span.doc_id,
            text=u.text,
            score=round(u.retrieval_score, 4),
            start=u.span.start,
            end=u.span.end,
        )
        for u in units
    ]


class Pipeline:
    """Holds the loaded models so they are paid for once, not per query."""

    def __init__(
        self,
        cfg: Config | None = None,
        retriever: Retriever | None = None,
        evaluator: SupportEvaluator | None = None,
        generator: Generator | None = None,
        counter: TokenCounter | None = None,
    ) -> None:
        self.cfg = cfg or default_config()
        self.retriever = retriever or Retriever(cfg=self.cfg)
        self.evaluator = evaluator or SupportEvaluator(cfg=self.cfg)
        self.generator = generator or Generator(cfg=self.cfg)
        self.counter = counter or shared_counter(self.cfg)
        self._verifier = None
        # Reranking reuses the support scorer's cross-encoder rather than loading a
        # second copy of the same weights.
        self.reranker = (
            Reranker(cfg=self.cfg, scorer=getattr(self.evaluator, "scorer", None))
            if bool(self.cfg.get_path("rerank.enabled", False))
            else None
        )

    def _choose_abstain_branch(
        self, evidence: list[ClaimEvidence], unsupported: list[str]
    ) -> tuple[Branch, str]:
        """Decide between 'retrieve more' and 'clarify'.

        If nothing retrieved for the failing claims scored well, the knowledge base
        probably lacks the material. If retrieval looked fine but support still failed
        across several claims, the question itself is likely underspecified.
        """
        low_score = float(self.cfg.get_path("abstain.low_retrieval_score", 0.25))
        ambiguous_count = int(self.cfg.get_path("abstain.ambiguous_claim_count", 3))

        failing = [e for e in evidence if e.claim in set(unsupported)]
        best_scores = [
            max((u.retrieval_score for u in e.units), default=0.0) for e in failing
        ]

        if best_scores and max(best_scores) < low_score:
            return "abstain_retrieve", (
                f"Retrieved evidence scored below {low_score:.2f}, so the knowledge "
                "base likely does not cover this."
            )
        if len(evidence) >= ambiguous_count and len(unsupported) > 1:
            return "abstain_clarify", (
                "Evidence was retrieved but did not support several parts of the "
                "question."
            )
        return "abstain_retrieve", (
            "Evidence was retrieved but did not support the question."
        )

    @property
    def verifier(self):
        """NLI scorer for answer-aware verification, built on first use.

        Lazy because it is a second model (~1.4 GB) that only one restoration policy
        needs; loading it for every run would tax the arms that never use it.
        """
        if self._verifier is None:
            from .verification import entailment_scorer

            self._verifier = entailment_scorer(self.cfg)
        return self._verifier

    def _verify_and_restore(
        self,
        query: str,
        compressed: list,
        final_evidence: list[ClaimEvidence],
        trace: PipelineTrace,
    ) -> bool:
        """Draft an answer, check the evidence entails it, and restore if it does not.

        This is the only check that asks about the *answer* rather than the topic, which
        is why it is the only one that can notice the failure the others miss: evidence
        that is still on-topic after the passage carrying the answer has been removed.

        The *trigger* is whole-query, because a draft answer is a property of the query.
        The *response* is per claim: once the draft is found ungrounded, the comparative
        signal picks which claims actually lost support, so a single unsupported sentence
        does not force every claim to hand back its entire saving. Measured, the
        all-or-nothing version restored 30% of questions for no F1 gain over the topical
        trigger, which is what motivated splitting the two decisions.
        """
        from .generation import build_prompt
        from .verification import verify

        style = getattr(self.generator, "style", {})
        instruction = style.get("instruction") if isinstance(style, dict) else None
        draft = self.generator.complete(
            build_prompt(query, final_evidence, instruction=instruction)
        )

        units = [u for item in final_evidence for u in item.units]
        report = verify(
            draft,
            units,
            self.verifier,
            query=query,
            cfg=self.cfg,
            contextualize_units=self.evaluator.contextualize,
        )
        trace.grounding = report.to_dict()
        trace.draft_answer = draft

        if report.grounded:
            return False

        if report.vacuous:
            # The draft made no checkable assertion — a refusal, or a bare yes/no whose
            # entailment score is meaningless. Verification abstains rather than guessing,
            # and the comparative signal decides instead. Treating "unverifiable" as
            # "unsupported" would restore on every yes/no question and forfeit the saving.
            return any(
                self._relative_says_restore(comp) for comp in compressed if comp.dropped
            )

        # The draft is ungrounded. Restore the claims whose support actually fell, not
        # every claim that happened to drop something.
        restored = False
        targeted = [
            i for i, comp in enumerate(compressed)
            if comp.dropped and self._relative_says_restore(comp)
        ]
        # If the comparative signal blames nothing, the loss is invisible to it but the
        # answer is still ungrounded — so fall back to restoring everything dropped
        # rather than silently doing nothing about a detected failure.
        if not targeted:
            targeted = [i for i, comp in enumerate(compressed) if comp.dropped]

        for index in targeted:
            current, _ = self._restore_and_recheck(compressed[index])
            final_evidence[index] = current
            restored = True
        return restored

    def _restore_and_recheck(self, comp):
        """Reinstate dropped evidence, escalating only as far as sufficiency requires.

        With `restoration.step: 0` this is the original all-at-once behaviour. With a
        positive step it climbs a ladder — restore the n best-ranked dropped units,
        re-check, escalate — whose final rung is always full restoration. So it can never
        recover less than the policy it replaces; it can only stop earlier and keep the
        savings the remaining units would have cost.

        The trade is scoring passes for tokens: each rung costs one more evaluation.
        """
        # Cheapest rung first: a dropped chunk sitting next to a kept one is usually the
        # context that chunk was severed from, and recovering it needs no scoring at all.
        if bool(self.cfg.get_path("restoration.neighbours_first", False)):
            current = restore_neighbours(comp)
            if len(current.units) > len(comp.kept):
                verdict = self.evaluator.evaluate(comp.claim, current.units)
                if verdict.is_sufficient:
                    return current, verdict

        step = int(self.cfg.get_path("restoration.step", 0))
        if step <= 0:
            current = restore(comp)
            return current, self.evaluator.evaluate(comp.claim, current.units)

        current = restore(comp)
        verdict = None
        for n in restoration_ladder(comp, step):
            current = restore_partial(comp, n)
            verdict = self.evaluator.evaluate(comp.claim, current.units)
            if verdict.is_sufficient:
                break
        if verdict is None:  # nothing was dropped; ladder was empty
            verdict = self.evaluator.evaluate(comp.claim, current.units)
        return current, verdict

    def _should_restore(self, comp, original: ClaimEvidence, verdict) -> bool:
        """Decide whether compression removed something this claim needed.

        `absolute` asks whether the compressed evidence clears the abstain threshold.
        That conflates two questions — "is this claim supportable at all?" and "did
        compression break it?" — and answers the second badly: the support signal is
        topical, so removing the one sentence carrying the answer usually leaves enough
        on-topic text to keep the score high. Measured on QASPER, it fired on 0 of 6
        cases where compression removed every trace of the marked answer evidence.

        `relative` compares against the full evidence set's own score, asking whether
        compression *changed* the support rather than whether the result clears some
        absolute bar. Self-calibrating, and it needs no per-corpus constant.
        """
        if not comp.dropped:
            # Nothing was removed, so there is nothing to put back regardless of policy.
            return False

        policy = self.cfg.get_path("restoration.policy", "absolute")
        if policy == "absolute":
            return not verdict.is_sufficient

        if policy == "answer_aware":
            # Handled after generation, over the whole query — see _verify_and_restore.
            # Restoring per claim here as well would double-count the token cost.
            return False

        if policy != "relative":
            raise ValueError(
                f"unknown restoration.policy {policy!r}; expected 'absolute', "
                "'relative' or 'answer_aware'"
            )

        return self._relative_says_restore(comp, verdict)

    def _relative_says_restore(self, comp, verdict=None) -> bool:
        """Has compression moved this claim's support materially below the full set's?

        Shared by the `relative` policy and by `answer_aware` when a draft answer turns
        out to be unverifiable, so the two cannot drift apart.
        """
        compressed_units = to_claim_evidence(comp).units
        if verdict is None:
            verdict = self.evaluator.evaluate(comp.claim, compressed_units)

        full_units = list(comp.kept) + list(comp.dropped)
        full = self.evaluator.evaluate(comp.claim, full_units)
        if full.support_score <= 0.0:
            # The full set supports nothing either, so compression cannot be the cause.
            # Fall back to the absolute test rather than restoring on a meaningless ratio.
            return not verdict.is_sufficient

        retain = float(self.cfg.get_path("restoration.retain_fraction", 0.9))
        return verdict.support_score < retain * full.support_score

    def _record_prompt_cost(
        self,
        trace: PipelineTrace,
        query: str,
        final_evidence: list[ClaimEvidence],
        full_evidence: list[ClaimEvidence],
    ) -> None:
        """Measure the real prompt, and the prompt the baseline would have sent.

        Best-effort: a stub generator in tests has no tokenizer, and a provenance-style
        measurement must never be the reason an experiment dies. Zeros mean "not
        measured", and `prompt_reduction` returns 0.0 rather than inventing a ratio.
        """
        counter = getattr(self.generator, "count_prompt_tokens", None)
        if counter is None:
            return
        style = getattr(self.generator, "style", {})
        instruction = style.get("instruction") if isinstance(style, dict) else None
        try:
            trace.prompt_tokens = counter(
                build_prompt(query, final_evidence, instruction=instruction)
            )
            trace.prompt_tokens_uncompressed = counter(
                build_prompt(query, full_evidence, instruction=instruction)
            )
        except Exception:  # noqa: BLE001 - measurement must not break the run
            logger.debug("prompt token measurement failed", exc_info=True)

    def run(
        self,
        query: str,
        compression_mode: str | None = None,
        restrict_to: str | None = None,
        oracle_spans: set | None = None,
        seed: int | None = None,
    ) -> tuple[Answer, PipelineTrace]:
        """Run one query end to end, returning the answer and its trace.

        `restrict_to` confines retrieval to a single document, for single-paper
        benchmarks where the question is only meaningful relative to a given paper.

        `oracle_spans` and `seed` exist for the evaluation harness only. `oracle_spans`
        carries human-marked answer evidence and is rejected by every mode except
        `oracle`, so it cannot leak into a measured system by accident; `seed` lets the
        random arm be run many times instead of once.
        """
        trace = PipelineTrace(query=query, token_counts_exact=self.counter.is_exact)

        # Stages 1-3: claims, retrieval, claim-wise evidence.
        claims = decompose(query, cfg=self.cfg)
        evidence = map_evidence(
            claims, self.retriever, cfg=self.cfg, counter=self.counter,
            restrict_to=restrict_to,
        )

        # Stage 1b: rerank candidates, so compression drops in cross-encoder order rather
        # than in bi-encoder order. No-op unless `rerank.enabled`.
        rerank_evidence(evidence, cfg=self.cfg, reranker=self.reranker)

        # Stage 4: compression (uncertainty is estimated and stored inside).
        compressed = compress(
            evidence,
            cfg=self.cfg,
            mode=compression_mode,
            oracle_spans=oracle_spans,
            seed=seed,
        )

        # Stages 5-6: evaluate, and restore only the claims that failed.
        final_evidence: list[ClaimEvidence] = []
        verdicts = []
        restoration_triggered = False

        for original, comp in zip(evidence, compressed):
            claim_trace = ClaimTrace(
                claim=comp.claim,
                n_candidates=len(original.units),
                uncertainty=comp.uncertainty,
                keep_ratio=comp.keep_ratio,
                n_kept=len(comp.kept),
                n_dropped=len(comp.dropped),
                baseline_tokens=comp.original_token_count,
                compressed_tokens=comp.compressed_token_count,
                kept=_record(comp.kept),
                dropped=_record(comp.dropped),
            )

            current = to_claim_evidence(comp)
            verdict = self.evaluator.evaluate(comp.claim, current.units)
            claim_trace.support_score = verdict.support_score
            claim_trace.is_sufficient = verdict.is_sufficient

            if self._should_restore(comp, original, verdict):
                # Stage 6.1 + 6.2 — restore this claim and re-check.
                restoration_triggered = True
                current, verdict = self._restore_and_recheck(comp)
                claim_trace.restored_support_score = verdict.support_score
                claim_trace.restored_sufficient = verdict.is_sufficient
                claim_trace.n_restored = len(current.units) - len(comp.kept)
                claim_trace.restored_tokens = sum(
                    self.counter.count(u.text) for u in current.units
                )

            final_evidence.append(current)
            verdicts.append(verdict)
            trace.claims.append(claim_trace)

        # Answer-aware verification runs once over the whole query, after per-claim
        # compression and before the final answer. Skipped when a claim already failed
        # the topical check, since that question is heading for abstention regardless and
        # drafting an answer for it would be wasted compute.
        if (
            self.cfg.get_path("restoration.policy", "absolute") == "answer_aware"
            and all(v.is_sufficient for v in verdicts)
            and any(c.dropped for c in compressed)
        ):
            if self._verify_and_restore(query, compressed, final_evidence, trace):
                restoration_triggered = True
                verdicts = [
                    self.evaluator.evaluate(item.claim, item.units)
                    for item in final_evidence
                ]
                for claim_trace, verdict in zip(trace.claims, verdicts):
                    claim_trace.restored_support_score = verdict.support_score
                    claim_trace.restored_sufficient = verdict.is_sufficient

        trace.restoration_triggered = restoration_triggered
        unsupported = [v.claim for v in verdicts if not v.is_sufficient]

        if self.cfg.get_path("abstain.gate", "support") == "retrieval":
            # Abstain on *retrieval* score instead of support score.
            #
            # The support scorer cannot carry this decision. Measured on a real uploaded
            # paper, the sentence literally containing the answer scored 0.00085 — and it
            # was still the highest-scoring unit retrieved. The ranking is informative;
            # the absolute scale is not, so any fixed cut either refuses everything or
            # refuses nothing. Refusing a question whose answer is sitting at rank 1 is
            # the worst failure this interface can produce.
            #
            # Cosine similarity is bounded, comparable across corpora, and calibrated:
            # "nothing retrieved is even close" is a judgement it can actually support.
            floor = float(self.cfg.get_path("abstain.low_retrieval_score", 0.25))
            best = max(
                (u.retrieval_score for item in final_evidence for u in item.units),
                default=0.0,
            )
            unsupported = [] if best >= floor else [e.claim for e in final_evidence]

        # End-to-end prompt cost, measured against the same prompt built from the *full*
        # evidence. Both are needed: the difference between them is the reduction actually
        # paid for, as opposed to the reduction in evidence text, which ignores every
        # incompressible part of the prompt.
        self._record_prompt_cost(trace, query, final_evidence, evidence)

        # Stage 7: generate, or abstain.
        if unsupported:
            branch, reason = self._choose_abstain_branch(final_evidence, unsupported)
            answer = abstain(query, final_evidence, verdicts, branch, reason)
        else:
            branch = "restored" if restoration_triggered else "direct"
            answer = generate_answer(
                query,
                final_evidence,
                verdicts,
                branch,
                generator=self.generator,
                cfg=self.cfg,
            )

        trace.branch = answer.branch
        trace.abstained = answer.abstained
        trace.answer = answer.text
        return answer, trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one query through the pipeline.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--trace", action="store_true", help="print the stage trace")
    parser.add_argument("--json", action="store_true", help="print the trace as JSON")
    parser.add_argument(
        "--mode",
        choices=["uncertainty_guided", "identity"],
        default="uncertainty_guided",
        help="identity = no-compression baseline",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    answer, trace = Pipeline().run(args.query, compression_mode=args.mode)

    if args.json:
        print(trace.to_json())
        return 0

    if args.trace:
        print(trace.summary())
        print()
    print(answer.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
