'Prompt builder for the open-domain QA rows of the GRPO-extraction line.\n\n\nThe QA mix rides on FineWeb2-HQ-QA pairs (question/answer grounded in a web\ndocument; the collector applies the quality/language/PII/grounding filters).\nTwo prompt flavours, deterministic per row (seeded share / seeded per row):\n\n* CLOSED-book — the question alone: trains parametric knowledge recall;\n* OPEN-book — the source document is prepended as context: trains reading\n  comprehension with verifiable grounding\n\nThe row carries ``reference = answer`` (the ONE reference-bearing class of\nthis line — everything else stays label-free; the standard scorer scores it\nby SemSim-vs-reference + fluency, the QA decision of the plan).\n\nOutput contract spelled inline (the extraction convention — no server-side\nstructured decoding): a SHORT factual answer, no meta-commentary, no\nenumeration of the context.'

from __future__ import annotations

from jinja2 import Environment, StrictUndefined

# Single source of truth for the target-language prompt values

from llm_training.extraction.prompts import LANGUAGE_VALUES

# Jinja2 environment as in prompts.py (StrictUndefined, trim/lstrip).
_ENV = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)

# : Task class of the QA rows
QA_TASK_CLASS = "fineweb_qa"

#: Target-language prompt values — re-exported from the single source of
#: truth (:data:`llm_training.extraction.prompts.LANGUAGE_VALUES`); see the
#: import at the top of the module.



QA_SYSTEM = """
You are a precise question-answering system.
Answer the user's question with a SHORT factual answer: one to three
sentences, directly addressing the question.

## RULES
- Answer ONLY from established facts; if the question is unanswerable,
  say so in one sentence.
- No meta-commentary ("as an AI...", "the question asks..."), no lists of
  alternatives, no restating the question.
- Keep the answer self-contained: names and numbers spelled out.
{% if text %}
- The context document may not contain the answer — use it when relevant,
  but you may answer from your own knowledge.
{% endif %}

Provide the answer in the following language: {{ language }}
Return ONLY the answer text, no JSON, no preamble.
"""

COMPACT_QA_SYSTEM = """
Answer the question in 1-3 short factual sentences. No meta-commentary,
no lists. Answer in {{ language }}. Return ONLY the answer text.
"""


def build_qa_prompt(
    question: str,
    text: str | None = None,
    language: str = "Russian",
    variant: int = 0,
) -> list[dict[str, str]]:
    'Build the QA prompt (closed- or open-book).\n\n    → closed-book (question only); a non-empty ``text`` is prepended as a\n    context block (open-book). ``variant=1`` selects the compact system.'
    q = (question or "").strip()
    if not q:
        raise ValueError(
            'QA prompt requires a non-empty question.'
        )
    template = QA_SYSTEM if variant == 0 else COMPACT_QA_SYSTEM
    system = _ENV.from_string(template).render(
        language=language, text=bool((text or "").strip())
    ).rstrip()
    if (text or "").strip():
        user = f"Context document:\n{text.strip()}\n\nQuestion: {q}"
    else:
        user = f"Question: {q}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
