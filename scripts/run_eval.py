"""Measure the compressed pipeline against the no-compression baseline.

This script produces the project's headline result. The claim under test is "compressing
evidence saves tokens without losing the ability to answer", so it reports cost and
quality side by side — a token reduction reported alone would be meaningless, since
dropping all evidence would score 100%.

Four arms, identical in every other respect:

    identity            every retrieved unit goes into the prompt (baseline)
    uncertainty_guided  keep budget scales with retrieval uncertainty (the proposal)
    fixed_ratio         constant keep fraction, top-ranked first
    random              constant keep fraction, chosen at random

The last two are the ablation. Beating `identity` only shows that dropping evidence
saves tokens, which is true of any compression. `fixed_ratio` is what tests whether
*uncertainty* is doing the work, and `random` separates that from whether retrieval
ranking matters at all. The ablation arms are calibrated to the budget the uncertainty
arm actually spent, so the arms are compared at equal cost rather than equal settings.

Reported per arm:

    tokens              baseline -> actually sent to the generator
    reduction           the headline number
    restoration rate    how often compression removed something it needed back
    abstain rate        split into answerable questions (bad) and unanswerable (good)
    keyword recall      coarse answer-quality proxy on answerable questions

Usage:
    python scripts/run_eval.py                    # both arms, with generation
    python scripts/run_eval.py --no-generate      # token/branch metrics only, fast
    python scripts/run_eval.py --limit 5          # first 5 questions
    python scripts/run_eval.py --json out.json    # machine-readable results
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.compression import compress  # noqa: E402
from src.config import load_config  # noqa: E402
from src.provenance import collect as collect_provenance, dataset_record  # noqa: E402
from src.gold_evidence import align as align_evidence, recall as evidence_recall  # noqa: E402
from src.evidence_mapping import map_evidence  # noqa: E402
from src.claims import decompose  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.restoration import restore, restore_from_corpus  # noqa: E402

DEFAULT_QUESTIONS = REPO_ROOT / "data" / "eval" / "questions.yaml"

# Order matters: uncertainty_guided runs before the ablation arms so its realised budget
# can be measured and handed to them.
MODES = ["identity", "uncertainty_guided", "fixed_ratio", "random"]

# Available but not run by default. `oracle` reads the answer key and is a headroom
# measurement rather than a system; `fixed_tokens` needs a token budget set explicitly.
# Keeping them out of the default keeps the headline ablation to arms that are all
# deployable and all budget-matched.
EXTRA_MODES = ["fixed_tokens", "oracle"]
ALL_MODES = MODES + EXTRA_MODES

# Below this, a narrow confidence interval is not trustworthy enough to declare a bounded
# null. The interval is built from a variance *estimated from the same small sample*, and
# on a lucky draw that estimate is too small, producing a confident "no effect" from data
# that cannot support one. A smoke test must not be able to close a research question.
MIN_N_FOR_NULL = 100


def apply_override(cfg, assignment: str) -> None:
    """Set one dotted config path from a `path=value` string, in place.

    Values are parsed as YAML so `0.5`, `true` and `extractive` all arrive as the right
    type — a threshold silently set to the *string* "0.5" would compare wrongly against
    floats rather than failing, which is the sort of quiet miscalibration this project
    has already lost an experiment cycle to.
    """
    path, _, raw = assignment.partition("=")
    if not _:
        raise ValueError(f"--set expects PATH=VALUE, got {assignment!r}")
    parts = path.strip().split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise KeyError(f"--set {path}: no config section {part!r}")
        # Copy rather than mutate: cfg is an lru_cached default in some call paths.
        node[part] = dict(node[part])
        node = node[part]
    if parts[-1] not in node:
        raise KeyError(f"--set {path}: no such config key")
    node[parts[-1]] = yaml.safe_load(raw)


class _SpanView:
    """Adapts a trace `EvidenceRecord` to the `.span` shape gold_evidence expects.

    The trace records offsets because the GUI needs them; gold_evidence works in spans.
    Rather than widen either interface for the sake of one caller, they meet here.
    """

    __slots__ = ("span",)

    class _Span:
        __slots__ = ("doc_id", "start", "end")

        def __init__(self, doc_id, start, end):
            self.doc_id, self.start, self.end = doc_id, start, end

    def __init__(self, record):
        self.span = self._Span(record.doc_id, record.start, record.end)


def evidence_outcome(trace, gold_spans, doc_id) -> dict | None:
    """Did compression drop evidence the answer needed, and did restoration notice?

    Blame is scoped deliberately. Recall is measured against what **retrieval actually
    returned**, not against all marked evidence: if retrieval never surfaced the passage,
    losing it is a retrieval failure and charging it to the compressor would make a good
    compressor look bad on a corpus with weak retrieval.

    So the comparison is `kept` against `kept + dropped`, and the interesting cell is:

        evidence was retrieved, compression dropped it, restoration did NOT fire

    which is the only case where the safety net was needed and missed. That is the number
    the reversibility claim lives or dies on.
    """
    if not gold_spans:
        return None

    kept = [_SpanView(r) for c in trace.claims for r in c.kept]
    dropped = [_SpanView(r) for c in trace.claims for r in c.dropped]

    retrieved_recall = evidence_recall(gold_spans, kept + dropped, doc_id)
    if retrieved_recall == 0.0:
        # Retrieval never found the marked evidence. Nothing to say about compression.
        return {"retrieval_found": False}

    kept_recall = evidence_recall(gold_spans, kept, doc_id)
    # Tolerance guards against a character or two lost to chunk-boundary rounding being
    # reported as a dropped passage.
    lost = kept_recall < retrieved_recall - 1e-9
    # Reported separately because "any loss" is a strict test: QASPER marks whole
    # paragraphs and we chunk by sentence, so trimming one sentence of a marked paragraph
    # counts as a loss even though the answer may survive in a neighbouring chunk.
    # Complete loss admits no such objection — every retrieved trace of the supporting
    # passage is gone.
    lost_completely = lost and kept_recall == 0.0

    return {
        "retrieval_found": True,
        "retrieval_recall": retrieved_recall,
        "kept_recall": kept_recall,
        "evidence_dropped": lost,
        "evidence_lost_completely": lost_completely,
        "restoration_fired": trace.restoration_triggered,
        "dangerous": lost and not trace.restoration_triggered,
    }


class NullGenerator:
    """Stands in for the generator when --no-generate is used."""

    def complete(self, prompt, max_new_tokens=None):
        return "[generation skipped]"


@dataclass
class ArmResult:
    mode: str
    baseline_tokens: int = 0
    final_tokens: int = 0
    # End-to-end prompt cost. `*_tokens` above count evidence text only.
    prompt_tokens: int = 0
    prompt_tokens_uncompressed: int = 0
    n_questions: int = 0
    n_answerable: int = 0
    restorations: int = 0
    abstained_answerable: int = 0
    abstained_unanswerable: int = 0
    keyword_hits: int = 0
    keyword_total: int = 0
    f1_scores: list[float] = field(default_factory=list)
    ceilings: list[float] = field(default_factory=list)
    pred_words: list[int] = field(default_factory=list)
    gold_words: list[int] = field(default_factory=list)
    # Evidence-level accounting. Populated only for questions whose marked evidence both
    # aligned to spans and was actually retrieved — see evidence_outcome().
    n_evidence_scored: int = 0
    evidence_dropped: int = 0
    evidence_dropped_restored: int = 0
    evidence_dropped_missed: int = 0
    evidence_lost_completely: int = 0
    evidence_lost_completely_missed: int = 0
    kept_recalls: list[float] = field(default_factory=list)
    retrieval_recalls: list[float] = field(default_factory=list)
    elapsed: float = 0.0
    # What the saving cost to obtain: scorer forward passes and generator calls. A token
    # reduction reported without these is only half a cost claim.
    scorer_calls: int = 0
    scorer_premises: int = 0
    generator_calls: int = 0
    peak_rss_mb: float = 0.0
    rows: list[dict] = field(default_factory=list)
    # Populated when an arm is averaged over several seeds; records the spread so the
    # report can show a distribution rather than implying a single draw is definitive.
    seed_spread: dict | None = None

    @property
    def reduction(self) -> float:
        if self.baseline_tokens == 0:
            return 0.0
        return (self.baseline_tokens - self.final_tokens) / self.baseline_tokens

    @property
    def prompt_reduction(self) -> float:
        """Reduction in tokens actually sent to the model, template and all.

        Always lower than `reduction`, because the prompt scaffolding does not compress.
        This is the number a cost claim has to be made on.
        """
        if not self.prompt_tokens_uncompressed:
            return 0.0
        saved = self.prompt_tokens_uncompressed - self.prompt_tokens
        return saved / self.prompt_tokens_uncompressed

    @property
    def restoration_rate(self) -> float:
        return self.restorations / self.n_questions if self.n_questions else 0.0

    @property
    def false_abstain_rate(self) -> float:
        """Abstaining on a question the KB *can* answer. Lower is better."""
        return (
            self.abstained_answerable / self.n_answerable if self.n_answerable else 0.0
        )

    @property
    def correct_abstain_rate(self) -> float:
        """Abstaining on a question the KB cannot answer. Higher is better."""
        n_unanswerable = self.n_questions - self.n_answerable
        return self.abstained_unanswerable / n_unanswerable if n_unanswerable else 0.0

    @property
    def keyword_recall(self) -> float:
        return self.keyword_hits / self.keyword_total if self.keyword_total else 0.0

    @property
    def mean_f1(self) -> float:
        return sum(self.f1_scores) / len(self.f1_scores) if self.f1_scores else 0.0

    @property
    def mean_ceiling(self) -> float:
        """Mean length-implied F1 ceiling. `mean_f1` is only readable against this."""
        return sum(self.ceilings) / len(self.ceilings) if self.ceilings else 0.0

    @property
    def f1_vs_ceiling(self) -> float:
        """Fraction of the achievable F1 actually reached — length-independent quality."""
        return self.mean_f1 / self.mean_ceiling if self.mean_ceiling else 0.0

    @property
    def gold_evidence_recall(self) -> float:
        """Marked evidence surviving compression, before restoration."""
        return (
            sum(self.kept_recalls) / len(self.kept_recalls) if self.kept_recalls else 0.0
        )

    @property
    def retrieval_evidence_recall(self) -> float:
        """The ceiling retrieval achieved — compression cannot beat this."""
        return (
            sum(self.retrieval_recalls) / len(self.retrieval_recalls)
            if self.retrieval_recalls
            else 0.0
        )

    @property
    def restoration_detector_recall(self) -> float:
        """P(restoration fired | compression dropped needed evidence). Higher is better."""
        return (
            self.evidence_dropped_restored / self.evidence_dropped
            if self.evidence_dropped
            else 0.0
        )

    @property
    def dangerous_false_negative_rate(self) -> float:
        """P(restoration did NOT fire | compression dropped needed evidence).

        The safety-critical number. Every one of these is a question answered from
        evidence known to be missing the passage that supported the reference answer,
        with the mechanism designed to catch exactly that staying silent.
        """
        return (
            self.evidence_dropped_missed / self.evidence_dropped
            if self.evidence_dropped
            else 0.0
        )

    @property
    def total_loss_missed_rate(self) -> float:
        """Dangerous rate restricted to *complete* loss of the retrieved evidence.

        The claim-proof version of `dangerous_false_negative_rate`: no argument about
        chunk boundaries or partial paragraphs survives here, because nothing of the
        supporting passage remained.
        """
        return (
            self.evidence_lost_completely_missed / self.evidence_lost_completely
            if self.evidence_lost_completely
            else 0.0
        )

    @property
    def mean_pred_words(self) -> float:
        return sum(self.pred_words) / len(self.pred_words) if self.pred_words else 0.0

    @property
    def mean_gold_words(self) -> float:
        return sum(self.gold_words) / len(self.gold_words) if self.gold_words else 0.0

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "questions": self.n_questions,
            "baseline_tokens": self.baseline_tokens,
            "final_tokens": self.final_tokens,
            "reduction": round(self.reduction, 4),
            "prompt_tokens": self.prompt_tokens,
            "prompt_tokens_uncompressed": self.prompt_tokens_uncompressed,
            "prompt_reduction": round(self.prompt_reduction, 4),
            "restoration_rate": round(self.restoration_rate, 4),
            "false_abstain_rate": round(self.false_abstain_rate, 4),
            "correct_abstain_rate": round(self.correct_abstain_rate, 4),
            "keyword_recall": round(self.keyword_recall, 4),
            "mean_f1": round(self.mean_f1, 4),
            "mean_f1_ceiling": round(self.mean_ceiling, 4),
            "f1_vs_ceiling": round(self.f1_vs_ceiling, 4),
            "mean_answer_words": round(self.mean_pred_words, 1),
            "mean_gold_words": round(self.mean_gold_words, 1),
            "evidence": {
                "n_scored": self.n_evidence_scored,
                "retrieval_recall": round(self.retrieval_evidence_recall, 4),
                "kept_recall": round(self.gold_evidence_recall, 4),
                "dropped": self.evidence_dropped,
                "dropped_and_restored": self.evidence_dropped_restored,
                "dropped_and_missed": self.evidence_dropped_missed,
                "detector_recall": round(self.restoration_detector_recall, 4),
                "dangerous_false_negative_rate": round(
                    self.dangerous_false_negative_rate, 4
                ),
                "lost_completely": self.evidence_lost_completely,
                "lost_completely_missed": self.evidence_lost_completely_missed,
                "total_loss_missed_rate": round(self.total_loss_missed_rate, 4),
            },
            "elapsed_seconds": round(self.elapsed, 1),
            "cost": {
                "scorer_calls": self.scorer_calls,
                "scorer_premises": self.scorer_premises,
                "generator_calls": self.generator_calls,
                "peak_rss_mb": round(self.peak_rss_mb, 1),
            },
            "seed_spread": self.seed_spread,
            "rows": self.rows,
        }


def check_round_trip(pipeline: Pipeline, queries: list[str], cfg) -> tuple[int, int]:
    """Re-verify reversibility on real retrieved evidence, not just fixtures.

    The unit tests prove the property on synthetic data; this proves it on whatever the
    retriever actually returns, which is the case that matters for a reported result.
    """
    checked = failures = 0
    for query in queries:
        evidence = map_evidence(decompose(query, cfg=cfg), pipeline.retriever, cfg=cfg)
        for comp in compress(evidence, cfg=cfg):
            fast = [(u.span.doc_id, u.span.start, u.span.end, u.text) for u in restore(comp).units]
            from_corpus = [
                (u.span.doc_id, u.span.start, u.span.end, u.text)
                for u in restore_from_corpus(comp, pipeline.retriever.corpus).units
            ]
            checked += 1
            if sorted(fast) != sorted(from_corpus):
                failures += 1
    return checked, failures


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^a-z0-9 ]")


def normalise(text: str) -> list[str]:
    """SQuAD-style normalisation: lowercase, drop punctuation and articles."""
    text = _PUNCT.sub(" ", text.lower())
    return _ARTICLES.sub(" ", text).split()


def answer_f1(prediction: str, golds: list[str]) -> float:
    """Unigram F1 against the best-matching gold answer — QASPER's own metric.

    Replaces keyword recall because it is **lower variance**: a partially correct answer
    scores partially instead of hit-or-miss on two or three keywords. That buys
    statistical power at no extra sample cost, which is the binding constraint on
    deciding the ablation.
    """
    predicted = normalise(prediction)
    best = 0.0
    for gold in golds:
        reference = normalise(gold)
        if not predicted or not reference:
            continue
        common = Counter(predicted) & Counter(reference)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(predicted)
        recall = overlap / len(reference)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def length_ceiling(prediction: str, golds: list[str]) -> float:
    """Highest F1 the answer's *length* allows, regardless of whether it is correct.

    Unigram F1 with p predicted words and g gold words can never exceed
    `2*min(p,g)/(p+g)`, because at most `min(p,g)` tokens can overlap. A 40-word answer
    against a 7-word gold is capped at 0.30 even if every gold word appears in it.

    Reported next to F1 because without it a low score is unreadable: it cannot be told
    apart from a wrong answer. This is the diagnostic that turned "our QASPER F1 is 0.03,
    the system is broken" into "our QASPER F1 is 0.03 and 0.30 was the most it could have
    been" — a measurement problem, not a system problem.
    """
    predicted = len(normalise(prediction))
    best = 0.0
    for gold in golds:
        reference = len(normalise(gold))
        if predicted and reference:
            best = max(best, 2 * min(predicted, reference) / (predicted + reference))
    return best


def two_proportion_p(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Two-sided p-value for a difference in two proportions (normal approximation).

    Exists because the first run of this ablation reported "uncertainty guidance shows
    no advantage" from a 3-keyword difference out of 30 — a gap well inside noise. A
    verdict the sample cannot support is worse than no verdict, so the report now has to
    clear this bar before it claims anything.
    """
    if n_a == 0 or n_b == 0:
        return 1.0
    p_a, p_b = hits_a / n_a, hits_b / n_b
    pooled = (hits_a + hits_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 1.0
    z = abs(p_a - p_b) / se
    # Two-sided tail of the standard normal.
    return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


def paired_p(a: list[float], b: list[float]) -> tuple[float, float]:
    """Paired test on per-question scores. Returns (mean difference, p-value).

    Every arm answers the same questions, so pairing is the right design: it removes
    question difficulty — by far the largest source of variance — and leaves only the
    effect of the arm. An unpaired test on the same data throws that away and needs far
    more questions to reach the same power.

    Normal approximation to the t-distribution; fine at the sample sizes used here.
    """
    if len(a) != len(b) or len(a) < 2:
        return 0.0, 1.0
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    mean = sum(diffs) / n
    variance = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if variance == 0:
        return mean, 0.0 if mean else 1.0
    se = math.sqrt(variance / n)
    z = abs(mean) / se
    return mean, 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


def paired_interval(a: list[float], b: list[float]) -> tuple[float, float, float, float]:
    """Paired difference with a 95% confidence interval. Returns (mean, p, lo, hi).

    A p-value alone cannot tell "no effect" apart from "not enough data", and those are
    opposite conclusions: one closes the question, the other says collect more. The
    interval separates them. If it is wide, the study is underpowered. If it is narrow and
    contains zero, the effect is genuinely absent *and bounded* — a real negative result
    rather than a shrug.

    This distinction is the whole point of running 276 questions instead of 30.
    """
    mean, p = paired_p(a, b)
    if len(a) != len(b) or len(a) < 2:
        return mean, p, 0.0, 0.0
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    variance = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(variance / n)
    return mean, p, mean - 1.96 * se, mean + 1.96 * se


def cluster_bootstrap(
    a: list[float],
    b: list[float],
    clusters: list[str],
    n_boot: int = 10000,
    seed: int = 20260803,
) -> tuple[float, float, float]:
    """Paired difference with a CI that resamples **papers**, not questions.

    QASPER asks several questions about each paper, and they are not independent: they
    share a document, a vocabulary, and whatever the retriever does well or badly on that
    document. Treating 276 questions as 276 independent observations counts information
    that is not there and reports an interval narrower than the data supports.

    Resampling whole papers respects that structure. The interval widens, which is the
    honest direction — an effect that survives this is one a reviewer cannot dismiss as
    pseudo-replication, and an effect that does not survive it was never real.

    Returns (mean difference, 2.5th percentile, 97.5th percentile).
    """
    if len(a) != len(b) or len(a) != len(clusters) or len(a) < 2:
        return 0.0, 0.0, 0.0

    by_cluster: dict[str, list[float]] = {}
    for x, y, cluster in zip(a, b, clusters):
        by_cluster.setdefault(cluster, []).append(x - y)

    names = sorted(by_cluster)
    if len(names) < 2:
        return 0.0, 0.0, 0.0

    observed = [d for name in names for d in by_cluster[name]]
    mean = sum(observed) / len(observed)

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        pooled: list[float] = []
        for _ in names:
            pooled.extend(by_cluster[names[rng.randrange(len(names))]])
        if pooled:
            means.append(sum(pooled) / len(pooled))

    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[min(int(0.975 * len(means)), len(means) - 1)]
    return mean, lo, hi


def cluster_permutation_p(
    a: list[float],
    b: list[float],
    clusters: list[str],
    n_perm: int = 10000,
    seed: int = 20260803,
) -> float:
    """Sign-flip permutation test at the paper level.

    Under the null the two arms are interchangeable, so flipping which arm is which is
    valid — but the flip must apply to a whole paper at once, for the same reason the
    bootstrap resamples papers. Flipping individual questions would assume the very
    independence the clustering denies.
    """
    if len(a) != len(b) or len(a) != len(clusters) or len(a) < 2:
        return 1.0

    by_cluster: dict[str, list[float]] = {}
    for x, y, cluster in zip(a, b, clusters):
        by_cluster.setdefault(cluster, []).append(x - y)

    names = sorted(by_cluster)
    total = sum(sum(by_cluster[n]) for n in names)
    count = sum(len(by_cluster[n]) for n in names)
    if count == 0:
        return 1.0
    observed = abs(total / count)

    rng = random.Random(seed)
    at_least_as_extreme = 0
    for _ in range(n_perm):
        flipped = sum(
            (1 if rng.random() < 0.5 else -1) * sum(by_cluster[n]) for n in names
        )
        if abs(flipped / count) >= observed - 1e-15:
            at_least_as_extreme += 1
    # +1 in both terms: a permutation test can never report p = 0, since the observed
    # arrangement is itself one of the possibilities being counted.
    return (at_least_as_extreme + 1) / (n_perm + 1)


def realised_keep_fraction(result: ArmResult) -> float:
    """Fraction of candidate units the arm actually kept.

    This is what the ablation arms are calibrated to. Comparing at equal *settings*
    would be meaningless — an arm that simply keeps more evidence will look better on
    quality and worse on cost. Comparing at equal *budget* isolates the only thing
    under test: whether allocating that budget by uncertainty beats spreading it evenly.
    """
    kept = sum(r["n_kept"] for r in result.rows)
    total = sum(r["n_kept"] + r["n_dropped"] for r in result.rows)
    return kept / total if total else 1.0


def align_all(questions: list[dict], corpus) -> tuple[dict[int, list], dict]:
    """Align every question's marked evidence once, before any arm runs.

    Done up front rather than per arm because alignment is arm-independent and would
    otherwise be recomputed four times over identical inputs. Returns the spans keyed by
    question index, plus the alignment statistics — which are reported rather than hidden,
    since a metric computed over 84% of the evidence must say that it is.
    """
    spans: dict[int, list] = {}
    stats = {"evidence": 0, "aligned": 0, "table_or_figure": 0, "unmatched": 0,
             "questions_with_evidence": 0, "questions_usable": 0}

    for i, item in enumerate(questions):
        marked = item.get("evidence") or []
        if not marked:
            continue
        stats["questions_with_evidence"] += 1
        document = corpus.get(item.get("paper", ""))
        if document is None:
            continue
        result = align_evidence(marked, document.text)
        stats["evidence"] += result.n_evidence
        stats["aligned"] += result.n_aligned
        stats["table_or_figure"] += result.n_table_or_figure
        stats["unmatched"] += result.n_unmatched
        if result.usable:
            stats["questions_usable"] += 1
            spans[i] = result.spans
    return spans, stats


def summarise_seeds(draws: list[ArmResult]) -> ArmResult:
    """Collapse several random draws into one arm, averaging per question.

    Averaging *per question* rather than across whole-arm totals keeps the result paired
    with the other arms: the significance tests compare question i against question i, and
    they would be meaningless against a mean taken over a different set of questions.

    The spread across seeds is kept in `seed_spread` so the report can show that random
    selection is being represented by a distribution and not by one lucky or unlucky draw.

    One caveat to state when reporting: averaging over seeds estimates the *expected*
    behaviour of random selection, which is the right quantity to compare ranking against,
    but a later paired test treats that average as fixed and so ignores its own sampling
    error. With 20 seeds that error is small, and `seed_spread` carries the raw range so a
    reader can judge it rather than take the point estimate on trust.
    """
    if len(draws) == 1:
        return draws[0]

    base = draws[0]
    merged = ArmResult(mode=base.mode)
    n = len(draws)

    merged.n_questions = base.n_questions
    merged.n_answerable = base.n_answerable
    merged.baseline_tokens = round(sum(d.baseline_tokens for d in draws) / n)
    merged.final_tokens = round(sum(d.final_tokens for d in draws) / n)
    merged.prompt_tokens = round(sum(d.prompt_tokens for d in draws) / n)
    merged.prompt_tokens_uncompressed = round(
        sum(d.prompt_tokens_uncompressed for d in draws) / n
    )
    merged.restorations = round(sum(d.restorations for d in draws) / n)
    merged.abstained_answerable = round(sum(d.abstained_answerable for d in draws) / n)
    merged.abstained_unanswerable = round(
        sum(d.abstained_unanswerable for d in draws) / n
    )
    merged.keyword_hits = round(sum(d.keyword_hits for d in draws) / n)
    merged.keyword_total = base.keyword_total
    merged.elapsed = sum(d.elapsed for d in draws)

    for attribute in ("f1_scores", "ceilings", "kept_recalls", "retrieval_recalls"):
        columns = [getattr(d, attribute) for d in draws]
        if columns and all(len(c) == len(columns[0]) for c in columns):
            setattr(
                merged,
                attribute,
                [sum(values) / n for values in zip(*columns)],
            )

    merged.pred_words = base.pred_words
    merged.gold_words = base.gold_words
    for attribute in (
        "n_evidence_scored", "evidence_dropped", "evidence_dropped_restored",
        "evidence_dropped_missed", "evidence_lost_completely",
        "evidence_lost_completely_missed",
    ):
        setattr(merged, attribute, round(sum(getattr(d, attribute) for d in draws) / n))

    merged.rows = base.rows
    merged.seed_spread = {
        "n_seeds": n,
        "reduction": [round(d.reduction, 4) for d in draws],
        "mean_f1": [round(d.mean_f1, 4) for d in draws],
        "restoration_rate": [round(d.restoration_rate, 4) for d in draws],
    }
    return merged


def run_arm(
    pipeline: Pipeline,
    questions: list[dict],
    mode: str,
    gold_spans: dict[int, list] | None = None,
    seed: int | None = None,
) -> ArmResult:
    result = ArmResult(mode=mode)
    gold_spans = gold_spans or {}
    start = time.time()

    scorer = getattr(pipeline.evaluator, "scorer", None)
    before_scorer_calls = getattr(scorer, "calls", 0)
    before_scorer_premises = getattr(scorer, "premises_scored", 0)
    before_generator_calls = getattr(pipeline.generator, "calls", 0)

    for index, item in enumerate(questions):
        query = item["query"]
        answerable = item.get("answerable", True)
        expected = [k.lower() for k in item.get("expect", [])]
        # QASPER sets carry `gold` reference answers; the hand-built set carries
        # `expect` keywords. Supporting both keeps the original result reproducible.
        golds = [g for g in item.get("gold", []) if g]

        spans = gold_spans.get(index)
        answer, trace = pipeline.run(
            query,
            compression_mode=mode,
            restrict_to=item.get("paper"),
            seed=seed,
            # Only the oracle arm ever receives the answer key, and `compress()` rejects
            # it for every other mode. Passed even when empty — a question whose evidence
            # is unaligned or absent leaves the oracle nothing to select on, where it
            # falls back to top-1 like any other arm's floor. Those questions are excluded
            # from the evidence metrics anyway, so the ceiling is not measured from them.
            oracle_spans=(
                {(item.get("paper", ""), s.start, s.end) for s in (spans or [])}
                if mode == "oracle"
                else None
            ),
        )

        result.n_questions += 1
        result.baseline_tokens += trace.baseline_tokens
        result.final_tokens += trace.final_tokens
        result.prompt_tokens += trace.prompt_tokens
        result.prompt_tokens_uncompressed += trace.prompt_tokens_uncompressed
        if trace.restoration_triggered:
            result.restorations += 1

        hits = f1 = ceiling = None
        if answerable:
            result.n_answerable += 1
            if answer.abstained:
                result.abstained_answerable += 1
            elif expected:
                text = answer.text.lower()
                hits = sum(1 for k in expected if k in text)
                result.keyword_hits += hits
                result.keyword_total += len(expected)

            if golds:
                # An abstention scores 0 rather than being excluded: refusing to answer
                # an answerable question is a failure, and dropping it from the average
                # would reward an arm for abstaining its way out of hard questions.
                f1 = 0.0 if answer.abstained else answer_f1(answer.text, golds)
                result.f1_scores.append(f1)

                # Length diagnostics, recorded only for answered questions: an
                # abstention has no answer whose length could be judged, and folding
                # its zero into the ceiling would understate what was achievable.
                if not answer.abstained:
                    ceiling = length_ceiling(answer.text, golds)
                    result.ceilings.append(ceiling)
                    result.pred_words.append(len(normalise(answer.text)))
                    result.gold_words.append(
                        min(len(normalise(g)) for g in golds if normalise(g))
                        if any(normalise(g) for g in golds)
                        else 0
                    )
        elif answer.abstained:
            result.abstained_unanswerable += 1

        outcome = evidence_outcome(trace, gold_spans.get(index), item.get("paper", ""))
        if outcome and outcome.get("retrieval_found"):
            result.n_evidence_scored += 1
            result.kept_recalls.append(outcome["kept_recall"])
            result.retrieval_recalls.append(outcome["retrieval_recall"])
            if outcome["evidence_dropped"]:
                result.evidence_dropped += 1
                if outcome["restoration_fired"]:
                    result.evidence_dropped_restored += 1
                else:
                    result.evidence_dropped_missed += 1
            if outcome["evidence_lost_completely"]:
                result.evidence_lost_completely += 1
                if not outcome["restoration_fired"]:
                    result.evidence_lost_completely_missed += 1

        result.rows.append(
            {
                "query": query,
                "evidence_outcome": outcome,
                # Recorded so the dev/test split used by calibrate_threshold.py can be
                # reapplied to these rows afterwards. The support threshold is picked on
                # dev, and while it is shared identically by all four arms and so cannot
                # favour one, absolute scores on the dev half are mildly optimistic.
                "paper": item.get("paper", ""),
                "branch": answer.branch,
                "abstained": answer.abstained,
                "answerable": answerable,
                "baseline_tokens": trace.baseline_tokens,
                "final_tokens": trace.final_tokens,
                "reduction": round(trace.reduction, 4),
                "uncertainty": [round(c.uncertainty, 3) for c in trace.claims],
                "support": [round(c.support_score, 3) for c in trace.claims],
                "n_kept": sum(c.n_kept for c in trace.claims),
                "n_dropped": sum(c.n_dropped for c in trace.claims),
                # Per-question quality, so a whole-arm difference can be traced back to
                # the questions that caused it rather than taken on faith.
                "keyword_hits": hits,
                "keyword_total": len(expected) if answerable else 0,
                "f1": None if f1 is None else round(f1, 4),
                "f1_ceiling": None if ceiling is None else round(ceiling, 4),
                "answer": answer.text,
            }
        )

    result.elapsed = time.time() - start
    result.scorer_calls = getattr(scorer, "calls", 0) - before_scorer_calls
    result.scorer_premises = getattr(scorer, "premises_scored", 0) - before_scorer_premises
    result.generator_calls = getattr(pipeline.generator, "calls", 0) - before_generator_calls
    result.peak_rss_mb = peak_rss_mb()
    return result


def peak_rss_mb() -> float:
    """Process high-water-mark memory. Monotonic, so it is a run-level ceiling, not a
    per-arm figure — reported once rather than attributed to whichever arm ran last."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS bytes.
        return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
    except Exception:  # noqa: BLE001
        return 0.0


def print_report(
    arms: list[ArmResult],
    round_trip: tuple[int, int],
    exact: bool,
    generated: bool = True,
    min_effect: float = 0.02,
) -> None:
    baseline = next((a for a in arms if a.mode == "identity"), None)

    print("\n" + "=" * 74)
    print("UNCERTAINTY-GUIDED COMPRESSION — BASELINE COMPARISON")
    print("=" * 74)

    checked, failures = round_trip
    status = "PASS" if failures == 0 else f"FAIL ({failures}/{checked})"
    print(f"\nReversibility on real retrieved evidence: {status}  [{checked} claims checked]")
    if not exact:
        print("WARNING: token counts are ESTIMATED, not tokenizer-exact.")

    width = 20
    line = "-" * (26 + width * len(arms))

    print(f"\n{'metric':<26}" + "".join(f"{a.mode:>{width}}" for a in arms))
    print(line)

    def row(label, fn, fmt="{:.1%}"):
        print(f"{label:<26}" + "".join(f"{fmt.format(fn(a)):>{width}}" for a in arms))

    row("questions", lambda a: a.n_questions, "{:d}")
    row("evidence tokens (baseline)", lambda a: a.baseline_tokens, "{:d}")
    row("evidence tokens (sent)", lambda a: a.final_tokens, "{:d}")
    row("evidence reduction", lambda a: a.reduction)
    if any(a.prompt_tokens_uncompressed for a in arms):
        # The number a cost claim rests on. Lower than the evidence reduction, because
        # the preamble, labels, question and chat template do not compress.
        row("PROMPT tokens (sent)", lambda a: a.prompt_tokens, "{:d}")
        row("PROMPT reduction", lambda a: a.prompt_reduction)
    row("restoration rate", lambda a: a.restoration_rate)
    if generated and any(a.f1_scores for a in arms):
        row("answer F1 (vs gold)", lambda a: a.mean_f1)
        # F1 alone is unreadable: a low score could be a wrong answer or merely a long
        # one. The ceiling says which, and `F1 / ceiling` is the length-independent
        # quality number that can be compared across answer styles.
        row("  length-implied ceiling", lambda a: a.mean_ceiling)
        row("  F1 as % of ceiling", lambda a: a.f1_vs_ceiling)
        row("  answer words (mean)", lambda a: a.mean_pred_words, "{:.1f}")
        row("  gold words (mean)", lambda a: a.mean_gold_words, "{:.1f}")
    if generated and any(a.keyword_total for a in arms):
        row("keyword recall", lambda a: a.keyword_recall)
    if not generated:
        # Without generation there is no answer text to search, so a 0% here would read
        # as a quality failure rather than "not measured".
        print(f"{'keyword recall':<26}" + "".join(f"{'n/a (--no-generate)':>23}" for _ in arms))
    if any(a.n_evidence_scored for a in arms):
        print(line)
        row("gold evidence retrieved", lambda a: a.retrieval_evidence_recall)
        row("  ...surviving compression", lambda a: a.gold_evidence_recall)
        row("needed evidence dropped", lambda a: a.evidence_dropped, "{:d}")
        row("  restoration caught it", lambda a: a.restoration_detector_recall)
        row("  DANGEROUS: missed", lambda a: a.dangerous_false_negative_rate)
        row("evidence lost ENTIRELY", lambda a: a.evidence_lost_completely, "{:d}")
        row("  ...and missed", lambda a: a.total_loss_missed_rate)
        print(line)

    row("false abstain (answerable)", lambda a: a.false_abstain_rate)
    row("correct abstain (unansw.)", lambda a: a.correct_abstain_rate)
    row("elapsed (s)", lambda a: a.elapsed, "{:.1f}")
    if any(a.scorer_calls for a in arms):
        # What the token saving cost. `relative` restoration roughly doubles these.
        row("scorer forward passes", lambda a: a.scorer_calls, "{:d}")
        row("  premises scored", lambda a: a.scorer_premises, "{:d}")
        row("generator calls", lambda a: a.generator_calls, "{:d}")
    if arms and arms[0].peak_rss_mb:
        print(f"\npeak process memory: {max(a.peak_rss_mb for a in arms):.0f} MB "
              "(run-level high-water mark, not per arm)")

    # A low ceiling means the F1 column is measuring answer length, not answer quality —
    # which is exactly how a real result was nearly misread as total system failure.
    worst = min((a.mean_ceiling for a in arms if a.ceilings), default=1.0)
    if generated and worst and worst < 0.6:
        print(
            f"\nWARNING: length-implied F1 ceiling is only {worst:.2f} — answers are far "
            f"longer than\n         the gold answers, so F1 here mostly measures "
            f"verbosity. Re-run with\n         --answer-style extractive before reading "
            "these numbers as quality."
        )

    compressed = next((a for a in arms if a.mode == "uncertainty_guided"), None)
    fixed = next((a for a in arms if a.mode == "fixed_ratio"), None)

    if baseline and compressed:
        saved = baseline.final_tokens - compressed.final_tokens
        print("\n" + line)
        quality = (
            f"keyword recall {compressed.keyword_recall - baseline.keyword_recall:+.1%} "
            "vs baseline"
            if generated
            else "answer quality not measured (--no-generate)"
        )
        print(
            f"COST: {saved} tokens saved ({compressed.reduction:.1%} reduction), "
            f"{quality}."
        )
        if compressed.false_abstain_rate > baseline.false_abstain_rate:
            print(
                "NOTE: compression increased false abstentions "
                f"({baseline.false_abstain_rate:.1%} -> "
                f"{compressed.false_abstain_rate:.1%}) — evidence needed to answer is "
                "being dropped and not always recovered."
            )

    # The ablation verdict — the question the whole study exists to answer.
    if compressed and fixed:
        gap_tokens = fixed.final_tokens - compressed.final_tokens
        matched = abs(gap_tokens) <= 0.05 * max(fixed.final_tokens, 1)
        print(line)
        print("ABLATION — does uncertainty guidance do the work?")
        print(
            f"  budgets {'match' if matched else 'DIFFER'}: "
            f"uncertainty {compressed.final_tokens} tokens vs fixed {fixed.final_tokens}"
            + ("" if matched else "  <- comparison is not budget-matched, treat with care")
        )

        p, delta, metric, n = 1.0, 0.0, "", 0
        lo = hi = 0.0
        if generated and compressed.f1_scores and fixed.f1_scores:
            # Paired on the same questions — the powerful comparison.
            delta, p, lo, hi = paired_interval(compressed.f1_scores, fixed.f1_scores)
            metric, n = "answer F1", len(compressed.f1_scores)
            direction = "better" if delta > 0 else ("worse" if delta < 0 else "the same")
            print(
                f"  answer F1: uncertainty {abs(delta):.4f} {direction} "
                f"({compressed.mean_f1:.4f} vs {fixed.mean_f1:.4f}, "
                f"paired n={n}, p={p:.4f})"
            )
            print(f"  95% CI, questions independent: [{lo:+.4f}, {hi:+.4f}]")

            # The interval that actually governs the verdict. Questions cluster within
            # papers, so this one is wider and is the one reported in the paper.
            papers = [
                r["paper"] for r in compressed.rows if r.get("f1") is not None
            ]
            if papers and len(papers) == len(compressed.f1_scores):
                _, clo, chi = cluster_bootstrap(
                    compressed.f1_scores, fixed.f1_scores, papers
                )
                cp = cluster_permutation_p(
                    compressed.f1_scores, fixed.f1_scores, papers
                )
                n_papers = len(set(papers))
                print(
                    f"  95% CI, clustered by paper: [{clo:+.4f}, {chi:+.4f}]  "
                    f"(p={cp:.4f}, {n_papers} papers)  <- primary"
                )
                lo, hi, p = clo, chi, cp
        elif generated:
            delta = compressed.keyword_recall - fixed.keyword_recall
            p = two_proportion_p(
                compressed.keyword_hits, compressed.keyword_total,
                fixed.keyword_hits, fixed.keyword_total,
            )
            metric, n = "keyword recall", compressed.keyword_total
            direction = "better" if delta > 0 else ("worse" if delta < 0 else "the same")
            print(
                f"  keyword recall: uncertainty {abs(delta):.1%} {direction} "
                f"({compressed.keyword_hits}/{compressed.keyword_total} vs "
                f"{fixed.keyword_hits}/{fixed.keyword_total}, p={p:.2f})"
            )
        abstain_delta = fixed.false_abstain_rate - compressed.false_abstain_rate
        print(f"  false abstain: uncertainty {abs(abstain_delta):.1%} "
              f"{'better' if abstain_delta > 0 else 'worse' if abstain_delta < 0 else 'the same'}")

        if not generated:
            print("  VERDICT: cannot judge without generation (--no-generate).")
        elif p > 0.05 and lo > -min_effect and hi < min_effect and n >= MIN_N_FOR_NULL:
            # The interval rules out an effect worth having. This is a *result*, not a
            # failure to get one, and it must not be reported as "inconclusive" —
            # inconclusive invites collecting more data, and more data will not help.
            print(
                f"  VERDICT: NO EFFECT, and the study is large enough to say so.\n"
                f"           The 95% CI [{lo:+.4f}, {hi:+.4f}] excludes any difference\n"
                f"           bigger than {min_effect:.3f} {metric} in either direction "
                f"(n={n}).\n"
                "           Uncertainty guidance does not beat a fixed budget at equal\n"
                "           cost. Collecting more questions will not change this."
            )
        elif p > 0.05 and n < MIN_N_FOR_NULL:
            print(
                f"  VERDICT: INCONCLUSIVE — SAMPLE TOO SMALL TO CLOSE THE QUESTION.\n"
                f"           p={p:.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], n={n}. The interval\n"
                f"           may look tight, but below n={MIN_N_FOR_NULL} it rests on a\n"
                "           variance estimated from the same few points. Run the full set."
            )
        elif p > 0.05:
            print(
                f"  VERDICT: INCONCLUSIVE — UNDERPOWERED. The difference is not\n"
                f"           distinguishable from noise (p={p:.4f}, {metric}, n={n}), and\n"
                f"           the 95% CI [{lo:+.4f}, {hi:+.4f}] is too wide to rule out an\n"
                f"           effect of {min_effect:.3f}. The set is too small to decide."
            )
        elif delta > 0:
            print("  VERDICT: uncertainty guidance beats a fixed budget at equal cost.")
        else:
            print(
                "  VERDICT: a fixed budget beats uncertainty guidance at equal cost —\n"
                "           the saving comes from compressing at all, not from\n"
                "           allocating adaptively."
            )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="skip answer generation; token and branch metrics only",
    )
    parser.add_argument("--modes", nargs="+", choices=ALL_MODES, default=MODES)
    parser.add_argument(
        "--random-seeds",
        type=int,
        default=1,
        help=(
            "run the random arm this many times and report the distribution. One draw is "
            "an anecdote: any claim of the form 'ranked selection beats random' is only "
            "as strong as the spread of the arm it beats. 20 is a reasonable default for "
            "a reported result"
        ),
    )
    parser.add_argument(
        "--generation-model",
        default=None,
        help="override models.generation (e.g. a smaller model for a quicker run)",
    )
    parser.add_argument(
        "--no-contextualize",
        action="store_true",
        help="do not prefix evidence with its document title (right for real papers)",
    )
    parser.add_argument(
        "--kb",
        default=None,
        help="override kb.path, e.g. data/qasper/kb for the real-paper evaluation",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help=(
            "override any config key, e.g. --set compression.max_keep=0.5. Exists for "
            "budget sweeps: the ablation was run at a realised keep of ~0.65, and "
            "whether the arms separate under a budget that actually binds is the main "
            "open question. Repeatable"
        ),
    )
    parser.add_argument(
        "--min-effect",
        type=float,
        default=0.02,
        help=(
            "smallest answer-F1 difference worth caring about, declared up front. If the "
            "95%% CI excludes it in both directions the verdict is a real null rather "
            "than 'inconclusive'. Default 0.02 — on a base F1 near 0.16 that is a ~12%% "
            "relative gain, a fair bar for a technique to clear to justify itself"
        ),
    )
    parser.add_argument(
        "--support-threshold",
        type=float,
        default=None,
        help=(
            "override the active scorer's threshold. PER-CORPUS — calibrate with "
            "scripts/calibrate_threshold.py. The default 0.15 was set on the hand-built "
            "KB and abstains on ~half of answerable QASPER questions"
        ),
    )
    parser.add_argument(
        "--answer-style",
        default=None,
        help=(
            "override generation.style. Use 'extractive' whenever --gold answers are "
            "short (QASPER's are median 7 words); 'prose' answers are penalised on "
            "unigram-F1 precision for words the gold answer simply does not contain"
        ),
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    questions = yaml.safe_load(args.questions.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    cfg = load_config()
    for assignment in args.overrides:
        apply_override(cfg, assignment)
        print(f"Config override: {assignment}")
    if args.generation_model:
        cfg["models"] = dict(cfg["models"], generation=args.generation_model)
    if args.kb:
        cfg["kb"] = dict(cfg["kb"], path=args.kb)
        print(f"Knowledge base: {args.kb}")
    if args.no_contextualize:
        cfg["evaluation"] = dict(cfg["evaluation"], contextualize=False)
        print("Title prefixing: off")
    if args.answer_style:
        cfg["generation"] = dict(cfg["generation"], style=args.answer_style)
    if args.support_threshold is not None:
        scorer_name = cfg.get_path("evaluation.scorer", "relevance")
        scorers = dict(cfg["evaluation"]["scorers"])
        scorers[scorer_name] = dict(
            scorers[scorer_name], threshold=args.support_threshold
        )
        cfg["evaluation"] = dict(cfg["evaluation"], scorers=scorers)
        print(f"Support threshold: {args.support_threshold} (overridden)")
    pipeline = Pipeline(cfg=cfg)
    if args.no_generate:
        pipeline.generator = NullGenerator()

    print(f"Loaded {len(questions)} questions. Building index...")
    print(f"Generation model: {cfg.require('models.generation')}")
    if not args.no_generate:
        print(f"Answer style: {cfg.get_path('generation.style', 'prose')}")
    pipeline.retriever.build()

    round_trip = check_round_trip(pipeline, [q["query"] for q in questions], cfg)

    gold_spans, alignment = align_all(questions, pipeline.retriever.corpus)
    if alignment["evidence"]:
        n = alignment["evidence"]
        print(
            f"Gold evidence: {alignment['aligned']}/{n} passages aligned "
            f"({alignment['aligned'] / n:.1%}); "
            f"{alignment['table_or_figure']} table/figure, "
            f"{alignment['unmatched']} unmatched. "
            f"{alignment['questions_usable']} questions scorable."
        )

    arms = []
    calibrated: float | None = None
    base_seed = int(cfg.get_path("compression.random_seed", 0))
    for mode in args.modes:
        # Calibrate the ablation arms to the budget uncertainty_guided actually spent,
        # so the comparison isolates allocation rather than amount.
        if mode in ("fixed_ratio", "random") and calibrated is not None:
            cfg["compression"] = dict(cfg["compression"], fixed_keep=calibrated)
            print(f"Running arm: {mode} (keep={calibrated:.3f}, budget-matched) ...")
        else:
            print(f"Running arm: {mode} ...")

        if mode == "random" and args.random_seeds > 1:
            # One random draw is an anecdote. The claim "ranked selection beats random"
            # is only as strong as the spread of the arm it beats, so run it many times
            # and report the distribution.
            draws = []
            for i in range(args.random_seeds):
                draws.append(
                    run_arm(
                        pipeline, questions, mode, gold_spans=gold_spans,
                        seed=base_seed + i,
                    )
                )
            arm = summarise_seeds(draws)
            print(
                f"  random over {len(draws)} seeds: reduction "
                f"{min(d.reduction for d in draws):.1%}–"
                f"{max(d.reduction for d in draws):.1%}, F1 "
                f"{min(d.mean_f1 for d in draws):.4f}–{max(d.mean_f1 for d in draws):.4f}"
            )
        else:
            arm = run_arm(pipeline, questions, mode, gold_spans=gold_spans)
        arms.append(arm)

        if mode == "uncertainty_guided":
            calibrated = realised_keep_fraction(arm)

    print_report(
        arms,
        round_trip,
        pipeline.counter.is_exact,
        generated=not args.no_generate,
        min_effect=args.min_effect,
    )

    if args.json:
        payload = {
            # Provenance first, so the file opens on what produced it rather than on
            # 900 kB of per-question rows.
            "provenance": collect_provenance(
                cfg,
                argv=["python", "scripts/run_eval.py"] + (argv or sys.argv[1:]),
                extra={"dataset": dataset_record(args.questions, args.kb)},
            ),
            "round_trip_checked": round_trip[0],
            "round_trip_failures": round_trip[1],
            "token_counts_exact": pipeline.counter.is_exact,
            "generation_skipped": args.no_generate,
            "arms": [a.to_dict() for a in arms],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")

    return 1 if round_trip[1] else 0


if __name__ == "__main__":
    sys.exit(main())
