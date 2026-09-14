"""Answer-aware verification — does the evidence actually support what we said?

WHY THIS EXISTS

The support check this replaces asks *"is this evidence about the topic?"*. Measured
against QASPER's human-marked answer evidence, that question is nearly useless as a safety
signal: when compression removed every trace of the passage supporting the reference
answer, the topical check failed to notice **77.9%** of the time, and in one case scored
0.985 on evidence containing none of it. Topical relevance survives the removal of the
answer, because the neighbouring sentences are still on-topic.

So this module asks the other question: *does the evidence entail the claims our answer
actually makes?* That requires the answer first, which is the cost — one extra generation
pass — and it is why the check could not be built from query-derived claims. Query claims
are topics; answer claims are assertions, and only assertions can be entailed.

This is the switch `evaluation.scorer` was documented as waiting for:

    "If claim decomposition ever switches to draft-answer decomposition, switch
     evaluation.scorer to nli — those claims *are* assertions."   (CLAUDE.md)

CIRCULARITY, AND WHY IT IS NOT A PROBLEM HERE

Verifying text the model just produced sounds circular, and it would be if the model's
own confidence were the signal. It is not: the hypothesis comes from the model, but the
*judgement* comes from a separate NLI model reading the retrieved evidence. The question
being asked is "is this claim in the evidence?", never "does the generator believe it?".

What it genuinely cannot catch is an omission — an answer that is fully grounded but
incomplete. That is a real limitation and is why this supplements rather than replaces
the abstain path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import Config, default_config
from .types import EvidenceUnit

# Sentence boundaries. Deliberately simple: draft answers here are short, and a heavier
# splitter would be a dependency for no measured gain.
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

# A bare polarity answer. Measured: entailment scores these near zero whether or not the
# supporting evidence survived compression ("Yes" -> 0.001 with evidence intact, 0.934
# with it entirely removed), because the token carries no proposition of its own. Judging
# them by entailment produced 4 of 8 false alarms in the calibration probe, so they are
# reported as unverifiable rather than as unsupported — a distinction that matters,
# because "unsupported" triggers a restoration that costs the whole saving.
_POLARITY = re.compile(r"^\s*(?:yes|no|true|false)\s*[.!]?\s*$", re.IGNORECASE)

# Text that asserts nothing about the world and must not be verified as if it did.
# Scoring "I cannot answer this" against evidence produces a low entailment score and
# would be read as "unsupported claim", triggering restoration on a refusal.
_NON_ASSERTIONS = re.compile(
    r"^\s*(?:"
    r"i (?:cannot|can't|could not|couldn't|don't|do not)\b"
    r"|(?:the )?(?:evidence|context|passage|text|document)s? (?:do(?:es)? not|don't)\b"
    r"|there is (?:no|insufficient)\b"
    r"|unsupported:"
    r"|unfortunately\b"
    r"|based on the (?:provided )?(?:evidence|context)\s*[,:]?\s*$"
    r")",
    re.IGNORECASE,
)

# Leading filler that adds no verifiable content but drags the hypothesis off-distribution
# for an NLI model trained on plain declaratives.
_LEAD_IN = re.compile(
    r"^\s*(?:based on the (?:provided )?(?:evidence|context|text)\s*,?\s*"
    r"|according to the (?:evidence|paper|text)\s*,?\s*"
    r"|the (?:evidence|text|paper) (?:shows|states|says|indicates) that\s+"
    r"|it (?:is|was) (?:stated|shown|reported) that\s+)",
    re.IGNORECASE,
)


@dataclass
class AnswerClaim:
    """One assertion the draft answer makes, and whether the evidence entails it."""

    text: str
    entailment: float = 0.0
    supported: bool = False
    best_unit_index: int = -1


@dataclass
class GroundingReport:
    claims: list[AnswerClaim] = field(default_factory=list)
    # True when the answer asserted nothing checkable — a refusal, or an empty draft.
    # Distinguished from "grounded" so a refusal is never counted as a verified answer.
    vacuous: bool = False

    @property
    def grounded(self) -> bool:
        return bool(self.claims) and all(c.supported for c in self.claims)

    @property
    def unsupported(self) -> list[str]:
        return [c.text for c in self.claims if not c.supported]

    @property
    def weakest(self) -> float:
        """An answer is only as grounded as its least supported claim."""
        return min((c.entailment for c in self.claims), default=0.0)

    def to_dict(self) -> dict:
        return {
            "vacuous": self.vacuous,
            "grounded": self.grounded,
            "weakest_entailment": round(self.weakest, 4),
            "claims": [
                {
                    "text": c.text,
                    "entailment": round(c.entailment, 4),
                    "supported": c.supported,
                }
                for c in self.claims
            ],
        }


def extract_answer_claims(
    answer: str, query: str = "", max_claims: int = 6, min_words: int = 2
) -> list[str]:
    """Split a draft answer into individually checkable assertions.

    Two answer shapes have to work, because the benchmark and the web UI ask for
    different ones (`generation.style`):

      prose      — "The model uses BERT. Training took three days."  -> two claims
      extractive — "Stanford NER, spaCy 2.0"                         -> one claim

    An extractive answer is a fragment, not a sentence, and a bare fragment is a weak NLI
    hypothesis. Where a query is available the fragment is attached to the query's topic to
    form something an entailment model can actually judge.
    """
    if not answer or not answer.strip():
        return []

    # A yes/no answer cannot be entailed on its own terms. Verifying "Yes" against the
    # evidence measures nothing, so it is reported as having no checkable claims and the
    # caller falls back to a signal that does apply.
    if _POLARITY.match(answer.strip()):
        return []

    claims: list[str] = []
    for sentence in _SENTENCE.split(answer.strip()):
        sentence = _LEAD_IN.sub("", sentence).strip().rstrip(".")
        if not sentence or _NON_ASSERTIONS.match(sentence):
            continue
        if len(sentence.split()) < min_words and not query:
            continue
        claims.append(sentence)
        if len(claims) >= max_claims:
            break

    # A short extractive answer carries no subject of its own. "Europarl; MultiUN" entails
    # nothing on its own terms; "datasets they experiment with: Europarl; MultiUN" does.
    if query and claims and all(len(c.split()) <= 6 for c in claims):
        from .claims import to_topic_phrase

        topic = to_topic_phrase(query) or query.strip().rstrip("?")
        claims = [f"{topic}: {c}" for c in claims]

    return claims


def verify(
    answer: str,
    units: list[EvidenceUnit],
    scorer,
    query: str = "",
    cfg: Config | None = None,
    contextualize_units: bool = True,
) -> GroundingReport:
    """Check every assertion in `answer` against `units` by entailment.

    Each claim is scored against every unit independently and takes its best — the same
    max-aggregation used for topical support, and for the same measured reason:
    concatenating a claim's units dilutes the signal because irrelevant neighbours drown
    the one that matters.

    A claim is supported if *some* piece of evidence entails it. That is the citation
    question — "can this be attributed to a retrieved passage?" — rather than the weaker
    "is the evidence collectively on-topic?".
    """
    cfg = cfg or default_config()
    report = GroundingReport()

    max_claims = int(cfg.get_path("verification.max_claims", 6))
    threshold = float(cfg.get_path("verification.entailment_threshold", 0.5))

    claim_texts = extract_answer_claims(answer, query=query, max_claims=max_claims)
    if not claim_texts:
        report.vacuous = True
        return report

    if not units:
        report.claims = [AnswerClaim(text=t) for t in claim_texts]
        return report

    from .evidence_mapping import contextualize

    premises = [contextualize(u, contextualize_units) for u in units]

    for text in claim_texts:
        scores = scorer.score(premises, text)
        if not scores:
            report.claims.append(AnswerClaim(text=text))
            continue
        best = max(range(len(scores)), key=scores.__getitem__)
        report.claims.append(
            AnswerClaim(
                text=text,
                entailment=scores[best],
                supported=scores[best] >= threshold,
                best_unit_index=best,
            )
        )

    return report


def entailment_scorer(cfg: Config | None = None):
    """Build the NLI scorer used for verification, independent of `evaluation.scorer`.

    Kept separate on purpose. `evaluation.scorer` answers "is this evidence relevant to
    this query-derived topic?" and relevance is right for that. Verification answers "does
    this evidence entail this assertion?", which is what NLI is for. Tying them together
    would force one model to do two jobs it scores on different scales.
    """
    from .evaluation import _SCORERS
    from .config import resolve_device

    cfg = cfg or default_config()
    name = cfg.get_path("verification.scorer", "nli")
    settings = cfg.get_path(f"evaluation.scorers.{name}") or {}
    if "model" not in settings:
        raise KeyError(f"verification.scorer is {name!r} but evaluation.scorers.{name} is missing")

    return _SCORERS[name](
        model_id=settings["model"],
        threshold=float(cfg.get_path("verification.entailment_threshold", 0.5)),
        device=resolve_device(cfg.get_path("models.device", "auto")),
    )
