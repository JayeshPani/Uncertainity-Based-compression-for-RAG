# Paper outline

Working title:

> **Detect your own losses: reversible evidence compression turns selection quality into a
> cost curve, and retrieval confidence is not the signal you need**

All numbers below come from `artifacts/qasper-main/` — 287 QASPER questions over 110
papers, budgets matched, random averaged over 20 seeds, intervals bootstrapped over papers.

---

## The claim we can defend

**Headline.** Reversible compression converts selection errors into lost savings rather
than wrong answers — *in proportion to how well the system detects those errors*. Improve
the detector and the value of good selection roughly doubles:

| restoration trigger | detector miss rate (random) | ranked − random reduction |
|---|---|---|
| `absolute` (topical threshold) | 55.6% | **+14.2 pp** |
| `relative` (vs full-set support) | 23.2% | **+26.3 pp** |

Random's realised reduction collapses from 39.0% to 17.9% under the better detector,
because its restoration rate rises from 48.8% to 76.0%. Ranked selection still returns
44.2%. Per question, clustered over 109 papers: +19.4 pp (CI [+15.3, +23.6]) and +31.6 pp
(CI [+24.6, +38.5]), both p=0.0001.

Supporting results:

1. **Ranking buys tokens, not accuracy.** Ranked vs random answer F1: +0.0024,
   CI [−0.0230, +0.0265], p=0.85. In a reversible system that is the expected shape — good
   selection should show up in cost.

2. **Uncertainty-guided allocation adds nothing.** +0.0052 F1 vs a fixed ratio,
   clustered CI [−0.0047, +0.0164], p=0.39, at a matched budget; and the sign flips between
   budget regimes. Bounded well inside the ±0.02 declared in advance. The founding
   premise — *confident retrieval implies redundant evidence* — is false here: retrieval
   confidence carries almost no information about whether evidence answers the question
   (top-ranked unit contains the gold answer 22% of the time vs 12% chance; correlation
   +0.045).

3. **The detector misses most losses.** Against human-marked evidence, `absolute` fails to
   fire on 76.5% of questions where compression dropped needed evidence, and on 77.9% of
   those where it dropped *all* of it. `relative` reduces this to 58.8% / 60.3%. Still bad;
   reported as the project's open problem.

4. **The oracle says the headroom is in cost, not quality.** Perfect selection reaches
   F1 0.1785 vs ranked 0.1649 — a gap that is not significant (CI [−0.0100, +0.0371],
   p=0.27). The generator, not the selector, limits answer quality. But the oracle hits
   baseline quality at 48.9% reduction with a **0%** dangerous-miss rate, which is the
   target a detector should be measured against.

5. **Compression costs quality, and how much depends on the generator.** identity −
   uncertainty_guided is +0.0120 F1 (p=0.42, not significant) on Qwen2.5-0.5B but
   **+0.0540, CI [+0.0300, +0.0798], p=0.0001** on Qwen2.5-1.5B. The smaller model cannot
   exploit the evidence being removed, so it cannot miss it. **Report the generator with
   every compression number**, and treat an identity arm scoring near the floor as a
   warning that the experiment cannot detect what it is measuring.

## What we must not claim

- **Not** "compression is free." It costs 5.4 F1 points on a competent generator. The
  free-lunch version of this claim survived three weeks only because the generator was too
  weak to notice, and it is exactly the claim a reviewer with a bigger model would break.
- **Not** "uncertainty-guided compression works." It does not, and we tested it properly.
- **Not** "the system knows when it doesn't know." 11 unanswerable questions cannot
  support that, and the evidence-level results argue against it.
- **Not** "reversibility guarantees safety." Reversibility guarantees *recoverability*.
  Detection is separate and is where the system is weakest.
- **Not** open-corpus RAG. See scope below.

---

## Structure

### 1. Introduction

Adaptive evidence compression assumes a system can tell which evidence it can afford to
drop. We test that assumption directly and it fails — but the *reversible* framing that
was supposed to make adaptivity safe turns out to be the interesting part, for a reason we
did not anticipate: it changes where selection errors land.

### 2. Scope — state this early and plainly

**This is known-document long-document QA, not open-corpus RAG.** QASPER questions are
asked about one specific paper and are not self-identifying ("which datasets did *they*
experiment with?"). Retrieval is restricted to the named paper (`restrict_to`,
`src/pipeline.py`). A deployed open-corpus system must first *find* the document, and we
do not evaluate that step.

This is not a detail. Run without the restriction, the baseline scored ~1% F1 and abstained
on 83% of answerable questions — retrieval confidently returned evidence from the wrong
paper. Document selection is a hard, separate problem and our numbers assume it solved.

### 3. Method

- Span-based extractive compression: evidence is `(doc_id, start, end)`, never a string.
  Compression drops spans from the prompt, never from the index.
- Restoration re-resolves dropped spans **from the corpus**, verified against the retained
  copies. Round-trip integrity checked on all 287 questions, both paths agreeing.
- Claim decomposition from the query; support scored by relevance, not entailment
  (query-derived claims are topics, not assertions — NLI scores them near zero).

### 4. Experimental design

| arm | selection | ablates |
|---|---|---|
| `identity` | keep everything | baseline |
| `uncertainty_guided` | budget scales with retrieval uncertainty | — (the proposal) |
| `fixed_ratio` | constant fraction, top-ranked | adaptivity |
| `random` × 20 seeds | constant fraction, random | adaptivity **and** ranking |
| `oracle` | units overlapping human-marked evidence | headroom ceiling |

Design decisions worth defending in text:

- **Budget-matched, not setting-matched.** Ablation arms are calibrated to the budget the
  uncertainty arm actually spent. Comparing at equal settings would let an arm that simply
  keeps more evidence look better on quality and worse on cost.
- **Paper-level inference.** Questions cluster within papers; a paired test over questions
  is pseudo-replication. CIs come from a bootstrap resampling **papers**, p-values from a
  paper-level sign-flip permutation test.
- **Held-out split primary.** The support threshold is calibrated on dev papers only.
- **Random is a distribution.** Over seeds its reduction ranged 34.8–58.0% on a smoke test;
  a single draw would have been an anecdote in either direction.
- **A minimum effect declared in advance** (0.02 F1, ~12% relative) so a null can be
  reported as a bounded null rather than as "inconclusive".

### 5. Results

#### 5.1 Cost — lead with the prompt number

**53.0% of evidence tokens is 38.6% of the prompt.** The preamble, per-claim labels, title
prefixes, question and chat template do not compress. Report both; only the second is a
cost claim, and it is the one to put in the abstract.

`absolute` policy, tight budget (`artifacts/qasper-main/tight_absolute.json`):

| arm | evidence red. | **prompt red.** | F1 | restoration | gold kept | dangerous miss |
|---|---|---|---|---|---|---|
| identity | 0.0% | 0.0% | 0.1769 | 0.0% | 55.1% | — |
| uncertainty_guided | 53.0% | 38.6% | 0.1649 | 29.6% | 24.0% | 76.5% |
| fixed_ratio | 53.2% | 38.7% | 0.1597 | 31.4% | 22.3% | 72.0% |
| random ×20 | 39.0% | 28.5% | 0.1561 | 48.8% | 13.4% | 55.6% |
| oracle | 48.9% | 35.9% | 0.1785 | 41.1% | 55.1% | 0.0% |

#### 5.2 The ablation — adaptivity does nothing

+0.0052 F1, clustered CI [−0.0047, +0.0164], p=0.39, 109 papers, budgets matched to 0.4%.
Sign flips at the milder budget. Bounded inside the ±0.02 declared in advance, so this is a
**null with a bound**, not an underpowered shrug — and `run_eval.py` refuses to declare one
below n=100.

#### 5.3 Ranking and the detector — the headline

The table in "The claim we can defend". Causal chain: poor selection → dropped evidence →
detector fires → restoration → forfeited saving. Each link measured. The comparison across
the two triggers is what isolates the detector's role.

#### 5.4 Evidence-level safety

Against human-marked evidence: gold-evidence recall before restoration (55.1% retrieved →
24.0% surviving compression), detector recall, and the **dangerous false-negative rate**
(76.5% `absolute`, 58.8% `relative`). Restricted to total losses: 77.9% and 60.3%.

Alignment succeeds on 83.9% of marked passages; 9.8% are table/figure captions absent from
a text-only corpus and are excluded rather than scored as compression failures.

#### 5.5 Oracle ceiling

F1 0.1785 at 48.9% reduction with 0% dangerous misses. The quality gap to ranked selection
is not significant (p=0.27) — so the remaining headroom is in **cost and safety**, not
accuracy, and the generator limits quality regardless of selection.

#### 5.6 Seed sensitivity

Across 20 seeds the random arm's reduction ranged **27.8% – 52.7%**. A single unlucky draw
would have put random level with ranked selection. Our own earlier single-seed aggregate
(19.1 pp) was inflated; the true aggregate is 14.2 pp. Report distributions for stochastic
baselines.

### 6. Limitations — write these, do not let a reviewer find them

1. **One generator (Qwen2.5-0.5B), one retriever (MiniLM), one dataset (QASPER).** A
   robustness run at 1.5B addresses the "generator too weak to reward better evidence"
   objection; a second retriever and dataset are not attempted.
2. **Abstention is not evaluated.** 11 unanswerable questions. We report what we have and
   claim nothing. The clean fix is *more real QASPER papers*, not invented negatives —
   authoring our own negatives is precisely the failure mode that produced a +0.99 result
   from a corpus we wrote ourselves.
3. **12% of marked evidence is table or figure captions**, absent from our text-only
   corpus. Those questions are unanswerable from our KB through no fault of the
   compressor, and they inflate the false-abstain rate. Reported separately.
4. **Evidence alignment succeeds on ~84% of marked passages.** Evidence-level metrics are
   computed over that subset and say so.
5. **Answer F1 is a weak proxy for grounding.** One case scored F1 0.33 having lost 100%
   of the marked evidence — the generator answered from other text or from parametric
   knowledge. This is the reason §5.4 exists.
6. **Restoration and abstention were entangled** through one threshold; separating them is
   new and only the absolute policy is used by the historical results.

### 6b. The improvement study — six independent tests of one hypothesis

Seven improvements were implemented and measured. They sort by a single criterion —
**whether they depend on the support scorer's judgement** — and the sorting predicts the
outcome exactly.

| improvement | uses the support signal? | result |
|---|---|---|
| Qwen2.5-1.5B generator | no | **works** — F1 0.177 → 0.250 on full evidence |
| scorer result caching | no | **works** — 141 forward passes → 0, identical numbers |
| answer-aware verification | **replaces** it with entailment | **partial** — total-loss misses 92.9% → 64.3% |
| cross-encoder reranking | yes | no effect — retention 20.0% → 21.1%, F1 −1.2 pp |
| neighbour restoration | yes | **no effect on any metric** |
| more candidates (top_k 8→16) | yes | no effect once budget-matched |
| graded restoration | yes | marginal — +1.2 pp reduction for +31% compute |

This is the paper's strongest structural argument. Six changes designed independently, all
bounded by the same ceiling, is far better evidence about *where* the bottleneck is than
any of the individual negative results. The support signal — which locates gold evidence
22% of the time against a 12% chance baseline, correlation +0.045 — is the binding
constraint on the entire design.

It also predicts the one partial success: answer-aware verification is the only change
that replaces the signal rather than consuming it, and it is the only gated-on-judgement
change that improved anything.

### 7. What would change our mind

Stated so the negative result is falsifiable rather than merely asserted:

- A different uncertainty estimator. We tested retrieval-score uncertainty only; logprob
  and self-consistency estimators drop into the same interface and are untested. **This
  should be pre-declared before running**, not chosen after seeing results.
- A budget harsh enough that allocation dominates. We tested keep ≈ 0.65 and ≈ 0.29 and
  saw no adaptivity effect at either, but the space is not exhausted.
- A stronger support signal. Every failure in this paper traces back to the same place:
  the scorer cannot tell answer-bearing evidence from on-topic evidence. Fix that and the
  allocation policy, the abstention gate and the restoration trigger all improve at once.
  This is now demonstrated rather than argued — see §6b, where six independent attempts to
  improve the system around that signal all hit the same ceiling.

---

## Reproducibility

Every result JSON in `artifacts/` embeds its commit, resolved config, dataset hashes,
resolved model revisions and environment. `scripts/make_bundle.py --check` refuses to
certify a run made from a dirty tree.

```bash
python scripts/make_bundle.py artifacts/qasper-main --check
```
