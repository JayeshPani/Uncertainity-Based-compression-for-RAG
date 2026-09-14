# CLAUDE.md

This file provides context for Claude (or Claude Code) when working in this repository.

## Project Overview

This is a research project building an **Uncertainty-Guided Reversible Evidence Compression
RAG system**. The core idea: compress retrieved evidence aggressively to save tokens, but do
so *reversibly* and *uncertainty-aware*, so evidence can be restored if the compressed version
turns out insufficient to support a claim.

The baseline pipeline (stages 1–7) is implemented and measured. Everything runs locally —
no API keys, no per-query cost.

## Pipeline Stages (in order)

1. `retrieval.py` — retrieves candidate documents from the KB for a user query
2. `evidence_mapping.py` — maps retrieved evidence to individual claims (claim-wise mapping)
3. Initial evidence set — full, high-token evidence per claim (output of stage 2, not its own module)
4. `compression.py` — uncertainty-guided compression of evidence (reduces tokens);
   uncertainty itself is estimated in `uncertainty.py`
5. `evaluation.py` — claim support evaluation; decides if compressed evidence is "enough"
6. If insufficient:
   - `restoration.py` — restores fuller evidence (reverses compression)
   - Re-run `evaluation.py` — re-check sufficiency
7. `generation.py` — final output generation once evidence is sufficient, OR abstain/clarify/retrieve-more if still insufficient after restoration

`pipeline.py` orchestrates all of it. `claims.py` (stage 2a) decomposes the query into
claims before mapping.

`server.py` + `web/index.html` are a local web interface over the same pipeline
(`python -m src.server`), and `ingest.py` handles adding papers. The UI is built on
stdlib `http.server` with no web framework — keep it that way; the dependency rule below
applies to Flask/FastAPI/Streamlit/Gradio too. It is a thin layer: it must not reimplement
pipeline logic, only present what `PipelineTrace` already records.

## Load-Bearing Design Decisions

Change these only deliberately — each is the answer to a problem that was measured, and the
measurements are recorded in the relevant module docstring.

- **Spans, not strings.** Evidence is `(doc_id, start, end)` offsets into a source document.
  This is what makes compression reversible. `chunk.text` must always equal
  `corpus[doc_id].text[start:end]`; `tests/test_kb.py` enforces it.
- **Support scoring is relevance, not entailment — because claims come from the query.**
  Query-derived claims are topics, not assertions, and NLI scores them near zero even when
  the KB plainly answers them ("dense retrieval differ from lexical retrieval" → 0.009).
  If claim decomposition ever switches to draft-answer decomposition, switch
  `evaluation.scorer` to `nli` — those claims *are* assertions.
- **Evidence is title-prefixed before scoring — but only on some corpora.** Chunks are
  anaphoric, and on the hand-built KB the title resolves the subject (0.004 → 0.997 on the
  same chunk). On QASPER's real papers the effect **reverses** (0.176 → 0.004), because a
  60-character paper-title slug repeated on every chunk disambiguates nothing and drowns
  short passages. Set `evaluation.contextualize` per corpus; never carry it over.
- **Score thresholds are corpus-specific, full stop.** Two have now been measured not to
  transfer (title prefixing above, and the support threshold below). Assume none do.
  `evaluation.scorers.relevance.threshold: 0.15` is right for the hand-built KB and
  abstains on ~50% of *answerable* QASPER questions, whose median best-support is 0.080.
  Calibrate with `scripts/calibrate_threshold.py` and pass `--support-threshold`; the
  measurement belongs in the config comment next to the value.
- **Answer length is scored, so it is configuration, not presentation.** `generation.style`
  picks between `prose` (web UI) and `extractive` (benchmarks). Answer-F1 charges a
  precision penalty per extra word, so the style has to match what the eval set expects.
  `run_eval.py` reports a `length-implied ceiling` beside F1 for exactly this reason — a
  low F1 is unreadable without it, since a wrong answer and a verbose one look identical.
- **Units are scored individually, aggregated with max.** Concatenating a claim's evidence
  dilutes the signal (0.997 → 0.510).
- **`identity` compression mode is not dead code.** It is the no-compression baseline arm
  that `scripts/run_eval.py` measures reduction against.
- **Restoration and abstention are different questions and now use different signals.**
  `restoration.policy: absolute` fires when compressed support falls under the abstain
  threshold; measured against human-marked evidence it misses **88.9%** of real losses,
  and 6 of 6 total losses, because the support signal is topical and dropping the answer
  sentence leaves other on-topic sentences scoring just as high. `relative` compares
  against the full evidence set's own score, halves the miss rate, and needs no
  per-corpus constant. Both are kept: `absolute` is what every result before 2026-08-03
  used.
- **Gold evidence is evaluation-only.** `src/gold_evidence.py` must never be imported by
  a pipeline module — `tests/test_gold_evidence.py` enforces it. The `oracle` compression
  mode is the one deliberate exception, it requires spans passed explicitly, and it is a
  headroom measurement, never a configuration.
- **Two token numbers, and only one is a cost claim.** `final_tokens` counts evidence text;
  `prompt_tokens` counts what the model actually received, chat template included. The
  scaffolding does not compress, so `prompt_reduction` is always lower and is the honest
  figure.

## Conventions

- Keep each pipeline stage an independent, testable module in `src/`
- Uncertainty scores are logged/returned alongside compression decisions for reproducibility
  (see `trace.py` — every run records uncertainty, keep ratio, token counts, and branch)
- Explicit config (`config/default.yaml`) over hardcoded thresholds — there should be no
  bare numeric threshold anywhere in `src/`
- Greedy decoding by default so runs are reproducible

## Testing

```bash
pytest tests/ -v                     # 174 tests
pytest tests/ -m "not integration"   # skip tests needing model weights
```

`tests/test_compression.py::TestRoundTrip` is the gate for the whole project: compression is
only legitimate if it can be undone exactly. If it fails, nothing downstream is trustworthy.
`scripts/run_eval.py` re-verifies the same property on real retrieved evidence.

## Current Priorities

- [x] Fix mid-August deadline deliverable / define concrete outcome
- [x] Get baseline pipeline (stages 1–7) working end-to-end
- [x] Measure against a no-compression baseline (43.3% token reduction, keyword recall
      56.7% -> 60.0%, 0% false abstain, reversibility PASS)
- [ ] Expand the evaluation set — 16 questions is a smoke test, not a benchmark
- [ ] Plan Fall '26–'27 extension for conference/journal submission

## When helping in this repo

- Dependencies are deliberately minimal: `torch`, `transformers`, `numpy`, `pyyaml`,
  `pytest`. Ask before introducing anything else (LangChain, LlamaIndex, FAISS,
  sentence-transformers) — FAISS and sentence-transformers were evaluated and dropped
  on purpose, not overlooked
- **Verify large model downloads by sha256, never by file size.** Twice now a Qwen 1.5B
  download has ended as a file of *exactly* the right size (3 087 467 144 bytes) with the
  wrong contents. HF's resume logic sees a complete file and never re-fetches, so the
  download sits at 0% CPU forever and no error is ever printed. The blob filename is its
  expected sha256 — compare against it, and delete the `.incomplete` blob before retrying
- Flag anywhere token-count assumptions (high-token vs reduced-token) aren't made explicit
  in code. `tokens.py` reports `is_exact`; a reduction measured with estimated counts must
  say so
- Preserve reversibility of compression — don't suggest lossy compression without a
  restoration path
- When changing a threshold or model, re-run `scripts/run_eval.py` and update the results
  table in `README.md`. Numbers in docstrings are measurements, not decoration
- **Treat identical results across arms as a bug, not a null result.** A run where
  `identity`, `uncertainty_guided`, `fixed_ratio` and `random` all scored exactly 0.0332
  was read as "compression makes no difference"; the real cause was that 80% of questions
  abstained, and every arm emits the same refusal text. Check the arms actually diverge
  before reading a verdict off them
- **Never measure a component with its own output.** "Restoration recovered every
  question" was inferred from abstention rates matching — but abstention is decided by the
  same support scorer whose blind spot was the thing being tested. Safety claims need an
  external label; here that is QASPER's evidence annotations
- **Report cost and quality against clustered intervals.** Questions from one paper are
  correlated. `cluster_bootstrap` and `cluster_permutation_p` in `run_eval.py` resample
  papers; the naive per-question interval is reported alongside but is not the verdict
- **Any experiment whose numbers may be quoted must write `--json` into `artifacts/`.**
  Provenance is embedded automatically; `scripts/make_bundle.py --check` fails a run made
  from a dirty tree
