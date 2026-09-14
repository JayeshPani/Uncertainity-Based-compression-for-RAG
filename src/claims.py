"""Pipeline stage 2a — decompose a query into independently verifiable claims.

Claims are derived from the *query*, not from a draft answer. That avoids the circularity
of verifying text the model just invented, at the cost that a claim set may not cover
everything the final answer asserts (see CLAUDE.md / README for the tradeoff).

One empirical finding drives the design here. The downstream NLI evaluator scores a
claim as a *hypothesis*, and it handles a raw interrogative very poorly — measured on
the seeded KB with a known-relevant premise:

    hypothesis form                  P(entailment)
    "What are the tunable...?"           0.033     <- raw question, unusable
    "the tunable parameters of BM25"     0.987     <- topic phrase

So claims are emitted as declarative topic phrases with the subject retained, not as
questions. Stripping the interrogative frame is all that is required — full
question-to-statement rewriting is not needed and would need a model call.
"""

from __future__ import annotations

import re

from .config import Config, default_config

# Sub-question boundaries. Deliberately conservative: splitting on a bare "and" would
# shred noun phrases like "k1 and b" into meaningless claims.
_SPLIT_PATTERN = re.compile(
    r"(?:\?+\s+)|(?:\s*;\s*)|(?:\s+and\s+also\s+)|(?:\s*,\s+and\s+)", re.IGNORECASE
)

# Leading interrogative frame: wh-word plus optional auxiliary.
_WH_AUX = re.compile(
    r"^\s*(?:what|which|who|whom|whose|where|when|why|how)\s+"
    r"(?:is|are|was|were|do|does|did|can|could|will|would|should|has|have|had)\s+",
    re.IGNORECASE,
)

# "how many/much X", "what kind of X" — strip the quantifier frame, keep the noun.
_WH_QUANT = re.compile(
    r"^\s*(?:how\s+(?:many|much)|what\s+(?:kind|sort|type)s?\s+of)\s+", re.IGNORECASE
)

# Bare wh-word with no auxiliary, e.g. "what causes X".
_WH_BARE = re.compile(
    r"^\s*(?:what|which|who|whom|whose|where|when|why|how)\s+", re.IGNORECASE
)

# Imperative prompts.
_IMPERATIVE = re.compile(
    r"^\s*(?:explain|describe|define|list|summari[sz]e|outline|compare|"
    r"tell\s+me\s+about|what\s+about)\s+",
    re.IGNORECASE,
)

_TRAILING_AUX = re.compile(
    r"\s+(?:is|are|was|were|do|does|did|can|could|will|would|should)\s*$",
    re.IGNORECASE,
)

# Undo subject-auxiliary inversion left behind after the wh-frame is stripped:
# "parameters does BM25 have" -> "BM25 have parameters". Without this the topic phrase
# keeps its interrogative word order, which reads as a question to the NLI evaluator.
_INVERSION = re.compile(
    r"^(?P<object>.+?)\s+(?:do|does|did)\s+(?P<subject>.+?)\s+(?P<verb>\w+)$",
    re.IGNORECASE,
)


def to_topic_phrase(question: str) -> str:
    """Strip the interrogative frame, leaving a topic phrase usable as an NLI hypothesis.

    The subject is deliberately preserved — "the tunable parameters of BM25" keeps
    "BM25", which is what lets the evaluator tell relevant evidence from irrelevant.
    """
    text = question.strip().rstrip("?").strip()
    if not text:
        return ""

    for pattern in (_IMPERATIVE, _WH_QUANT, _WH_AUX):
        stripped = pattern.sub("", text, count=1)
        if stripped != text:
            text = stripped
            break
    else:
        text = _WH_BARE.sub("", text, count=1)

    inverted = _INVERSION.match(text)
    if inverted:
        text = (
            f"{inverted['subject']} {inverted['verb']} {inverted['object']}".strip()
        )

    # "does BM25 use" -> "BM25 use": drop auxiliaries left dangling at the end.
    while True:
        stripped = _TRAILING_AUX.sub("", text).strip()
        if stripped == text:
            break
        text = stripped
    return text.rstrip(".,;: ").strip()


def _heuristic_decompose(query: str, max_claims: int) -> list[str]:
    parts = [p.strip() for p in _SPLIT_PATTERN.split(query) if p and p.strip()]
    if not parts:
        parts = [query]

    claims: list[str] = []
    for part in parts:
        topic = to_topic_phrase(part)
        # Single-word topics are legitimate ("chunking", "calibration") — only drop
        # genuinely empty results.
        if not topic:
            continue
        if topic.lower() not in {c.lower() for c in claims}:
            claims.append(topic)
        if len(claims) >= max_claims:
            break

    if not claims:
        # Never return an empty claim set — fall back to the whole query as one claim.
        fallback = to_topic_phrase(query) or query.strip().rstrip("?").strip()
        claims = [fallback]
    return claims


def _llm_decompose(query: str, max_claims: int, cfg: Config) -> list[str]:
    """LLM-based decomposition (Fall extension). Falls back to heuristic on any failure."""
    from .generation import Generator

    prompt = (
        "Break the question into atomic, independently checkable factual topics.\n"
        "Write each as a short declarative noun phrase, one per line, no numbering.\n"
        f"Question: {query}\nTopics:"
    )
    try:
        # Pinned to `prose`: this call wants a multi-line list of topics, and the
        # `extractive` style explicitly forbids saying more than the answer itself.
        # The answer style belongs to answering, not to decomposition.
        raw = Generator(cfg=cfg, style="prose").complete(prompt, max_new_tokens=128)
    except Exception:  # noqa: BLE001 - decomposition must never break the pipeline
        return _heuristic_decompose(query, max_claims)

    claims = [
        re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip()
        for line in raw.splitlines()
        if line.strip()
    ]
    claims = [c for c in claims if len(c.split()) >= 2][:max_claims]
    return claims or _heuristic_decompose(query, max_claims)


def decompose(query: str, cfg: Config | None = None) -> list[str]:
    """Decompose a query into claims. Strategy comes from config."""
    cfg = cfg or default_config()
    strategy = cfg.get_path("claims.strategy", "heuristic")
    max_claims = int(cfg.get_path("claims.max_claims", 6))

    if strategy == "llm":
        return _llm_decompose(query, max_claims, cfg)
    return _heuristic_decompose(query, max_claims)
