"""Pipeline stage 7 — final answer generation, or a deliberate abstention.

The abstain path is a first-class output, not an error case. "Still insufficient after
restoration" is a real branch of the design: a system that declines when the evidence
does not support an answer is the point of the uncertainty machinery, so it returns a
structured refusal rather than quietly generating something plausible.
"""

from __future__ import annotations

import logging

from .config import Config, default_config, resolve_device
from .evidence_mapping import document_title
from .types import Answer, Branch, ClaimEvidence, ClaimVerdict

logger = logging.getLogger(__name__)


def answer_style(cfg: Config | None = None, name: str | None = None) -> dict:
    """Resolve the active answer style — system prompt, prompt instruction, token cap.

    Answer length is a scored property, not a presentation detail. Answer F1 is unigram
    overlap against gold answers that are a median of 7 words on QASPER, so a correct but
    verbose answer is penalised on precision: ~40 words against a 7-word gold caps F1
    near 0.30 however right it is. Styles exist so the benchmark and the web UI can ask
    for different things from the same pipeline instead of one silently mis-serving the
    other. See config/default.yaml for the definitions.
    """
    cfg = cfg or default_config()
    name = name or cfg.get_path("generation.style", "prose")
    style = cfg.get_path(f"generation.styles.{name}")
    if not isinstance(style, dict):
        available = sorted(cfg.get_path("generation.styles", {}))
        raise KeyError(f"unknown generation.style {name!r}; available: {available}")
    return style


class Generator:
    """Local instruction-tuned generator."""

    def __init__(self, cfg: Config | None = None, style: str | None = None) -> None:
        self.cfg = cfg or default_config()
        self.model_id = self.cfg.require("models.generation")
        self.device = resolve_device(self.cfg.get_path("models.device", "auto"))
        self.style = answer_style(self.cfg, style)
        self._model = None
        self._tokenizer = None
        # Cost instrumentation — see SupportScorer for why this is tracked.
        self.calls = 0
        self.prompt_tokens_total = 0

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("loading generation model %s on %s", self.model_id, self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            # float16 on accelerators halves memory (~3GB vs ~6GB for 1.5B params) and is
            # markedly faster; CPU keeps float32 since float16 there is emulated and slower.
            dtype = torch.float32 if self.device == "cpu" else torch.float16
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=dtype
            ).to(self.device)
            self._model.eval()
            self._torch = torch
        return self._model, self._tokenizer

    def count_prompt_tokens(self, prompt: str) -> int:
        """Tokens in the prompt as the model receives it, chat template included.

        Counting the evidence text alone understates the bill by everything that wraps it:
        the system turn, the role markers, the instruction, the per-claim labels and the
        question. None of that shrinks when evidence is compressed, so a reduction
        measured on evidence only is not the reduction anyone pays.
        """
        _, tokenizer = self._load()
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.style["system"]},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return len(tokenizer(text)["input_ids"])

    def complete(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """Run the chat template over a single user turn and return the reply."""
        model, tokenizer = self._load()
        torch = self._torch
        limit = max_new_tokens or int(
            self.style.get("max_new_tokens")
            or self.cfg.get_path("generation.max_new_tokens", 256)
        )
        temperature = float(self.cfg.get_path("generation.temperature", 0.0))

        messages = [
            {"role": "system", "content": self.style["system"]},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        self.calls += 1
        self.prompt_tokens_total += int(inputs["input_ids"].shape[1])

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=limit,
                # Greedy by default: reproducible runs, per CLAUDE.md.
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()


_DEFAULT_INSTRUCTION = "Answer the question using only the evidence below."


def build_prompt(
    query: str, evidence: list[ClaimEvidence], instruction: str | None = None
) -> str:
    """Assemble the evidence prompt, grouped claim-wise.

    Only the units present in `evidence` are included — this is where compression
    actually converts into saved tokens.

    `instruction` comes from the answer style. Small instruct models follow the user turn
    more reliably than the system turn, so the length requirement is stated in both.
    """
    blocks: list[str] = []
    for item in evidence:
        lines = [f"Evidence for: {item.claim}"]
        for unit in item.units:
            lines.append(f"  - [{document_title(unit.span.doc_id)}] {unit.text}")
        blocks.append("\n".join(lines))

    return (
        (instruction or _DEFAULT_INSTRUCTION).strip()
        + "\n\n"
        + "\n\n".join(blocks)
        + f"\n\nQuestion: {query}\nAnswer:"
    )


def generate_answer(
    query: str,
    evidence: list[ClaimEvidence],
    verdicts: list[ClaimVerdict],
    branch: Branch,
    generator: Generator | None = None,
    cfg: Config | None = None,
) -> Answer:
    """Generate a grounded answer from sufficient evidence."""
    cfg = cfg or default_config()
    generator = generator or Generator(cfg=cfg)
    # A stub generator in tests has no `.style`; fall back to the plain instruction.
    style = getattr(generator, "style", {})
    text = generator.complete(
        build_prompt(query, evidence, instruction=style.get("instruction"))
    )
    return Answer(
        query=query,
        text=text,
        branch=branch,
        abstained=False,
        claims=[e.claim for e in evidence],
        verdicts=verdicts,
    )


def abstain(
    query: str,
    evidence: list[ClaimEvidence],
    verdicts: list[ClaimVerdict],
    branch: Branch,
    reason: str,
) -> Answer:
    """Return a structured refusal naming the claims that evidence could not support."""
    unsupported = [v.claim for v in verdicts if not v.is_sufficient]
    if branch == "abstain_retrieve":
        suggestion = (
            "The knowledge base does not appear to contain evidence for this. "
            "Try rephrasing, or add sources covering it."
        )
    else:
        suggestion = (
            "The question is ambiguous as posed. Could you narrow it down "
            "or specify which part you mean?"
        )

    detail = "; ".join(unsupported) if unsupported else "the question as asked"
    return Answer(
        query=query,
        text=(
            f"I cannot answer this from the available evidence. {reason}\n"
            f"Unsupported: {detail}\n{suggestion}"
        ),
        branch=branch,
        abstained=True,
        claims=[e.claim for e in evidence],
        verdicts=verdicts,
        unsupported_claims=unsupported,
    )
