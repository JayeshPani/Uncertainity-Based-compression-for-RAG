"""Answer-style tests.

These exist because of a measurement failure, not a crash. QASPER gold answers are a
median of 7 words; our generator wrote multi-sentence prose; unigram F1 charges a
precision penalty for every extra word. The benchmark reported F1 ~0.033 when the answer
length alone capped it near 0.30 — so the number was reading verbosity, not quality.

Styles fix that by letting the benchmark and the web UI ask for different answer shapes
from one pipeline. The risk being pinned here is silent regression: if `extractive` ever
stops reaching the generator, F1 quietly collapses again with nothing visibly broken.
"""

from __future__ import annotations

import pytest

from src.config import load_config
from src.generation import Generator, answer_style, build_prompt
from src.types import ClaimEvidence, EvidenceUnit, Span


@pytest.fixture
def evidence():
    unit = EvidenceUnit(
        span=Span(doc_id="bert_paper", start=0, end=34),
        text="We fine-tune BERT on the SQuAD set.",
        retrieval_score=0.9,
        token_count=9,
    )
    return [ClaimEvidence(claim="model used", units=[unit], token_count=9)]


class TestAnswerStyle:
    def test_default_style_resolves(self, cfg):
        style = answer_style(cfg)
        assert "system" in style

    def test_both_styles_are_defined(self, cfg):
        for name in ("prose", "extractive"):
            style = answer_style(cfg, name)
            assert style["system"].strip()
            assert style["instruction"].strip()

    def test_extractive_asks_for_fewer_tokens_than_prose(self, cfg):
        """The token cap is the backstop for when the instruction is not obeyed."""
        assert (
            answer_style(cfg, "extractive")["max_new_tokens"]
            < answer_style(cfg, "prose")["max_new_tokens"]
        )

    def test_unknown_style_fails_loudly(self, cfg):
        """A typo must not silently fall back to prose and quietly halve the F1."""
        with pytest.raises(KeyError, match="unknown generation.style"):
            answer_style(cfg, "concice")


class TestPromptCarriesTheStyle:
    def test_instruction_reaches_the_prompt(self, evidence):
        prompt = build_prompt("Which model?", evidence, instruction="Answer briefly.")
        assert prompt.startswith("Answer briefly.")

    def test_extractive_instruction_states_the_length_requirement(self, cfg, evidence):
        prompt = build_prompt(
            "Which model?",
            evidence,
            instruction=answer_style(cfg, "extractive")["instruction"],
        )
        assert "as few words as possible" in prompt

    def test_evidence_is_unchanged_by_style(self, cfg, evidence):
        """Style controls the answer, never which evidence is sent — that is compression's
        job, and confusing the two would corrupt the token accounting."""
        prose = build_prompt("Which model?", evidence, instruction="A")
        extractive = build_prompt("Which model?", evidence, instruction="B")
        assert "We fine-tune BERT on the SQuAD set." in prose
        assert prose[1:] == extractive[1:]


class TestGeneratorWiring:
    def test_generator_selects_the_requested_style(self, cfg):
        assert Generator(cfg=cfg, style="extractive").style["max_new_tokens"] == 48

    def test_config_style_is_honoured_without_an_explicit_argument(self):
        cfg = load_config()
        cfg["generation"] = dict(cfg["generation"], style="extractive")
        assert Generator(cfg=cfg).style == answer_style(cfg, "extractive")

    def test_explicit_argument_beats_config(self):
        """Claim decomposition relies on this to pin itself to prose."""
        cfg = load_config()
        cfg["generation"] = dict(cfg["generation"], style="extractive")
        assert Generator(cfg=cfg, style="prose").style == answer_style(cfg, "prose")
