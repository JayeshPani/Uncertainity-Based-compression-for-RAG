"""Calibrate the support threshold on a dev split, so it is not tuned on the test set.

WHY THIS EXISTS

`evaluation.scorers.relevance.threshold` was 0.15, chosen on the 24-document hand-built
corpus where "the lowest answerable claim scored 0.205 and the highest unanswerable
scored 0.001". On real papers the scale is completely different: best-support on
*answerable* QASPER questions has a median of 0.080, so a 0.15 threshold abstains on more
than half of the questions the paper demonstrably answers.

That is not a small quality loss. Every arm of the ablation abstains on the same
questions and emits the same refusal text, so all four arms score identically and the
ablation cannot resolve anything — which is exactly what the first QASPER run showed
(four arms, identical F1 0.0332). The threshold was the reason the experiment could not
run, not a finding about compression.

METHOD

Split by *paper*, not by question. Questions about one paper share its retrieval
behaviour and its vocabulary, so splitting by question would leak dev information into
test and report a threshold that looks better than it is.

Objective is balanced accuracy, not accuracy: the set is 276 answerable to 11
unanswerable, so a threshold of 0 scores 96% accuracy by never abstaining. Balanced
accuracy weights the two classes equally and cannot be gamed that way.

HONEST LIMIT — READ BEFORE USING THE OUTPUT

11 unanswerable questions across the whole set means roughly 5 per split. The
false-abstain side of this curve rests on ~138 questions and is solid; the correct-abstain
side rests on ~5 and is not. Treat the recommended threshold as "the value that stops the
scorer from abstaining on answerable questions", which is what it is measured well enough
to say, and not as a calibrated abstention policy.

Usage:
    python scripts/calibrate_threshold.py --kb data/qasper/kb \
        --questions data/qasper/questions.yaml --no-contextualize
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.claims import decompose  # noqa: E402
from src.config import load_config  # noqa: E402
from src.evaluation import SupportEvaluator  # noqa: E402
from src.evidence_mapping import map_evidence  # noqa: E402
from src.retrieval import Retriever  # noqa: E402


def split_of(paper: str, dev_fraction: float = 0.5) -> str:
    """Deterministic paper-level split — stable across runs and machines."""
    digest = hashlib.sha256(paper.encode("utf-8")).hexdigest()
    return "dev" if int(digest[:8], 16) / 0xFFFFFFFF < dev_fraction else "test"


def collect_scores(questions: list[dict], cfg) -> list[dict]:
    """Best support score per question — the quantity the threshold is applied to."""
    retriever = Retriever(cfg=cfg)
    retriever.build()
    evaluator = SupportEvaluator(cfg=cfg)

    records = []
    for i, item in enumerate(questions, 1):
        claims = decompose(item["query"], cfg=cfg)
        evidence = map_evidence(
            claims, retriever, cfg=cfg, restrict_to=item.get("paper")
        )
        # The pipeline abstains when *no* claim clears the threshold, so the per-question
        # quantity is the weakest claim's best unit — mirror that here rather than
        # calibrating against a number the pipeline never computes.
        per_claim = []
        for ce in evidence:
            scores = evaluator.score_units(ce.claim, ce.units)
            per_claim.append(max(scores) if scores else 0.0)

        records.append(
            {
                "query": item["query"],
                "paper": item.get("paper", ""),
                "answerable": bool(item.get("answerable", True)),
                "split": split_of(item.get("paper", item["query"])),
                "weakest_claim_support": min(per_claim) if per_claim else 0.0,
            }
        )
        if i % 25 == 0:
            print(f"  scored {i}/{len(questions)}")
    return records


def sweep(records: list[dict], thresholds: list[float]) -> list[dict]:
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]

    rows = []
    for t in thresholds:
        answered = sum(1 for r in answerable if r["weakest_claim_support"] >= t)
        abstained = sum(1 for r in unanswerable if r["weakest_claim_support"] < t)
        true_answer = answered / len(answerable) if answerable else 0.0
        true_abstain = abstained / len(unanswerable) if unanswerable else 0.0
        rows.append(
            {
                "threshold": t,
                "answer_rate_on_answerable": true_answer,
                "false_abstain_rate": 1 - true_answer,
                "correct_abstain_rate": true_abstain,
                "balanced_accuracy": (true_answer + true_abstain) / 2,
                "n_answerable": len(answerable),
                "n_unanswerable": len(unanswerable),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--kb", default=None)
    parser.add_argument("--no-contextualize", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="reuse a previous run's per-question scores instead of recomputing",
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.kb:
        cfg["kb"] = dict(cfg["kb"], path=args.kb)
    if args.no_contextualize:
        cfg["evaluation"] = dict(cfg["evaluation"], contextualize=False)

    if args.scores and args.scores.exists():
        records = json.loads(args.scores.read_text(encoding="utf-8"))
        print(f"reusing {len(records)} cached scores from {args.scores}")
    else:
        questions = yaml.safe_load(args.questions.read_text(encoding="utf-8"))
        if args.limit:
            questions = questions[: args.limit]
        print(f"scoring {len(questions)} questions...")
        records = collect_scores(questions, cfg)
        if args.scores:
            args.scores.write_text(json.dumps(records, indent=2), encoding="utf-8")

    thresholds = [round(0.01 * i, 2) for i in range(0, 51)]
    dev = [r for r in records if r["split"] == "dev"]
    test = [r for r in records if r["split"] == "test"]

    print(f"\ndev: {len(dev)} questions / {len({r['paper'] for r in dev})} papers")
    print(f"test: {len(test)} questions / {len({r['paper'] for r in test})} papers")

    dev_rows = sweep(dev, thresholds)
    best = max(dev_rows, key=lambda r: (r["balanced_accuracy"], -r["threshold"]))

    print(f"\n{'threshold':>10}{'answered':>12}{'false abst':>12}{'corr abst':>12}{'bal acc':>10}")
    print("-" * 56)
    for row in dev_rows:
        if row["threshold"] in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50):
            mark = "  <- best" if row["threshold"] == best["threshold"] else ""
            print(
                f"{row['threshold']:>10.2f}"
                f"{row['answer_rate_on_answerable']:>11.1%}"
                f"{row['false_abstain_rate']:>12.1%}"
                f"{row['correct_abstain_rate']:>12.1%}"
                f"{row['balanced_accuracy']:>10.1%}{mark}"
            )

    current = next(r for r in dev_rows if r["threshold"] == 0.15)
    print(f"\nDEV  recommends threshold {best['threshold']:.2f} "
          f"(balanced accuracy {best['balanced_accuracy']:.1%}, "
          f"false abstain {best['false_abstain_rate']:.1%})")
    print(f"     current 0.15 gives balanced accuracy {current['balanced_accuracy']:.1%}, "
          f"false abstain {current['false_abstain_rate']:.1%}")

    # Held-out confirmation. The dev split chose the number; test only reports it.
    held = sweep(test, [best["threshold"], 0.15])
    print("\nTEST (held out, threshold chosen on dev — reported, not tuned):")
    for row in held:
        print(f"  threshold {row['threshold']:.2f}: "
              f"false abstain {row['false_abstain_rate']:.1%}, "
              f"correct abstain {row['correct_abstain_rate']:.1%} "
              f"(n={row['n_answerable']} answerable, {row['n_unanswerable']} unanswerable)")

    n_unans = best["n_unanswerable"]
    if n_unans < 20:
        print(
            f"\nCAVEAT: only {n_unans} unanswerable questions in dev. The false-abstain "
            f"column is\n        well determined; the correct-abstain column is not. "
            "This threshold is\n        calibrated to stop over-abstention, not to set an "
            "abstention policy."
        )

    if args.json:
        args.json.write_text(
            json.dumps(
                {"dev": dev_rows, "test": sweep(test, thresholds),
                 "recommended": best["threshold"]},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
