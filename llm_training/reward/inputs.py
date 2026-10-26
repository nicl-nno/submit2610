'Per-task-class extraction of the **user question** for the judge.\n\nPer-class\n\nThe dispatcher / reward service feed the judge\'s ``user_input`` field. RAGAS\ntreats ``user_input`` as the **question** (the few-shot examples in\n``StatementGeneratorPrompt`` / ``CorrectnessClassifier`` are short questions),\nso we must extract just the question from the conversational prompt — not the\nwhole ``user`` message, which for RAG classes (``menon_qa``, ``multihop_rag``,\n``mtrageval``) also carries the retrieved documents / chunks / context.\n\n``user_input``\n\nBefore this module existed, the dispatcher used :func:`last_user_text`, which\nreturns the entire last ``user`` message. For ``menon_qa`` that meant the judge\nreceived up to ~14 000 gpt-4o-mini tokens of documents + abbreviations +\ninstruction as the "question", which (a) inflated judge prompts ~5× and\n(b) was wildly out-of-distribution for RAGAS prompts (their few-shot examples\nare ~10-token questions).\n\n\nEverything here is a pure function over ``(prompt, extra)`` → ``str`` so it is\nunit-tested with no mocks'

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Callable

from llm_training.tokenizer_utils import last_user_text

if TYPE_CHECKING:  # pragma: no cover
    from llm_training.schema import Message

# Menon-QA query sections are delimited by runs of '=' (the source uses exactly
# 10 '='). A section is ``==========\nHEADER\n==========\n\nCONTENT``. The
# question lives in the section whose header is ``CURRENT QUESTION``; the
# instruction (``INSTRUCTION``) is appended when present so the judge sees the
# same task framing the policy saw.


_MENONQA_DELIM_RE = re.compile(r"(?:\n|^)={5,}[ \t]*\n")
_MENONQA_QUESTION_HEADER_RE = re.compile(r"^\s*CURRENT\s+QUESTION\s*$", re.IGNORECASE)
_MENONQA_INSTRUCTION_HEADER_RE = re.compile(r"^\s*INSTRUCTION\s*$", re.IGNORECASE)

# MultiHopRAG query embeds the retrieved chunks after a fixed marker. Everything
# BEFORE the marker is the question (with framing); everything after is the
# retrieved context (not part of the question).

_MULTIHOP_CHUNK_MARKER = "The chunk list:"

# DocSem (docsem_quant) prompts embed the document blocks FIRST and the
# question LAST, after a fixed ``Question:`` line marker (the champion
# pipeline's user-message layout, preserved verbatim by the loader). The
# question is the text AFTER the LAST marker (a block body may itself
# contain the word "Question:").


_DOCSEM_QUESTION_MARKER = "\nQuestion:"


def _menon_qa_question(prompt: "list[Message]", extra: dict[str, Any]) -> str:
    'Return the ``CURRENT QUESTION`` section (+ optional ``INSTRUCTION``).\n\n\n    Falls back to the full last ``user`` text when the section structure is not\n    found (e.g. chitchat examples assembled without the standard delimiters).'
    query = last_user_text(prompt)
    if not query:
        return ""
    tokens = _MENONQA_DELIM_RE.split(query)
    question = ""
    instruction = ""
    # tokens[0] is the text before the first delimiter (usually empty); headers
    # are at odd indices, their content at the following even index.
    # tokens[0]
    
    i = 1
    while i + 1 < len(tokens):
        header = tokens[i].strip()
        content = tokens[i + 1].strip()
        if _MENONQA_QUESTION_HEADER_RE.match(header) and content:
            question = content
        elif _MENONQA_INSTRUCTION_HEADER_RE.match(header) and content:
            instruction = content
        i += 2
    if not question:
        # No structured question section — return the raw user text unchanged.
        
        return query
    return f"{question}\n\n{instruction}" if instruction else question


def _multihop_rag_question(prompt: "list[Message]", extra: dict[str, Any]) -> str:
    'Return the text **before** the ``The chunk list:`` marker.\n\n\n    The marker separates the question framing from the retrieved chunks. Falls\n    back to the full user text when the marker is absent.'
    query = last_user_text(prompt)
    if not query:
        return ""
    idx = query.find(_MULTIHOP_CHUNK_MARKER)
    if idx < 0:
        return query
    return query[:idx].rstrip()


def _mtrageval_question(prompt: "list[Message]", extra: dict[str, Any]) -> str:
    'Return the user question with the rendered context stripped out.\n\n\n    The mtrageval loader variably places ``context_text`` either in ``system``\n    or appended to the final ``user`` message. When placed in ``user``, we strip\n    it back out so the judge sees only the question. When placed in ``system``\n    (or absent), the user message is already clean and we return it verbatim.'
    text = last_user_text(prompt)
    ctx = (extra or {}).get("context_text") or ""
    if not text:
        return ""
    if not ctx:
        return text
    # The loader appends ``"\n\n" + context_text`` to the user content. Strip
    # exactly that trailing block if present; leave the text untouched otherwise
    # (defensive: don't accidentally truncate a question that happens to contain
    # the context text by coincidence).
    
    
    sep = "\n\n" + ctx
    if text.endswith(sep):
        return text[: -len(sep)].rstrip()
    return text


def _docsem_quant_question(prompt: "list[Message]", extra: dict[str, Any]) -> str:
    'Return the text after the last ``Question:`` marker.\n\n\n    The docsem_quant user message is ``Document blocks:\\n{…}\\n\\nQuestion:\n    {query}`` — the blocks (up to ~1.5k chars of OCR text) are external\n    context, not part of the question. Falls back to the full user text when\n    the marker is absent.'
    text = last_user_text(prompt)
    if not text:
        return ""
    idx = text.rfind(_DOCSEM_QUESTION_MARKER)
    if idx < 0:
        return text
    return text[idx + len(_DOCSEM_QUESTION_MARKER):].strip()


def _fineweb_qa_question(prompt: "list[Message]", extra: dict[str, Any]) -> str:
    "Return the bare QA question (never the open-book context block).\n\n    the open-book user message prepends the whole source document\n    (``Context document:\\n…\\n\\nQuestion: {q}``), which would otherwise\n    inflate the judge's ``user_input`` by the full document. Falls back to the\n    text after the last ``Question:`` marker, then to the full user text."
    question = str(((extra or {}).get("qa") or {}).get("question") or "").strip()
    if question:
        return question
    text = last_user_text(prompt)
    if not text:
        return ""
    idx = text.rfind(_DOCSEM_QUESTION_MARKER)
    if idx < 0:
        return text
    return text[idx + len(_DOCSEM_QUESTION_MARKER):].strip()


def _default_question(prompt: "list[Message]", extra: dict[str, Any]) -> str:
    'Default: return the full last ``user`` message text.\n\n\n    For non-RAG classes the whole user message is the question already.'
    return last_user_text(prompt)


#: Per-class question extractors. Classes not listed here use
#: :func:`_default_question` (the whole last ``user`` message).
# : Per-class

QUESTION_EXTRACTORS: dict[
    str, Callable[["list[Message]", dict[str, Any]], str]
] = {
    "menon_qa": _menon_qa_question,
    "multihop_rag": _multihop_rag_question,
    "mtrageval": _mtrageval_question,
    "docsem_quant": _docsem_quant_question,
    "fineweb_qa": _fineweb_qa_question,
}


def extract_user_question(
    task_class: str,
    prompt: "list[Message]",
    extra: dict[str, Any] | None = None,
) -> str:
    "Extract the user-facing question for the judge's ``user_input`` field.\n\n\n    For RAG classes the last ``user`` message carries documents / chunks alongside\n    the question; this function returns just the question so the judge sees a\n    short, in-distribution prompt. For all other classes it returns the whole\n    last ``user`` message. Never raises."
    extractor = QUESTION_EXTRACTORS.get(task_class)
    if extractor is None:
        return _default_question(prompt, extra or {})
    return extractor(prompt, extra or {})


#: Default character budget for :func:`build_rubric_context`. The rubric judge
#: call needs the task + the recent context, not the whole (potentially ~30k
#: token) running prompt — 6 000 chars ≈ 1.5–2k judge tokens per anchor example.


DEFAULT_RUBRIC_CONTEXT_CHARS = 6000


def build_rubric_context(
    prompt: "list[Message]", max_chars: int = DEFAULT_RUBRIC_CONTEXT_CHARS,
) -> str:
    'Build the input-context string for the reasoning-turn rubric criteria.\n\n\n    The second rubric criterion ("consistency of the generated turn with its\n    input context") needs the context the policy actually conditioned on —\n    which is NOT the extracted question (for SGR classes that is just the date\n    preamble) but the running conversation state. This helper joins, in order:\n    the ``system`` message (the SGR task), the last ``user`` message (the task\n    framing), and the TAIL of everything else (folded ``role:"tool"`` results\n    and prior assistant turns — the most recent state first matters most),\n    truncated to ``max_chars`` characters total. Pure function, never raises.\n\n    ``role:"tool"``\n    ``max_chars``'
    system = ""
    user = ""
    rest: list[str] = []
    for m in prompt or []:
        role = m.get("role")
        content = m.get("content") or ""
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "system":
            system = content
        elif role == "user":
            user = content  # keep the LAST user message
        else:
            rest.append(f"[{role}]\n{content}")
    head_parts = []
    if system:
        head_parts.append(f"[system]\n{system}")
    if user:
        head_parts.append(f"[user]\n{user}")
    head = "\n\n".join(head_parts)
    rest_text = "\n\n".join(rest)
    if not head and not rest_text:
        return ""
    # Budget: keep the head (task) verbatim; the remaining budget goes to the
    # TAIL of the running context (the freshest tool results / turns).
    
    
    budget = max_chars - len(head) - (2 if head and rest_text else 0)
    if rest_text and budget < len(rest_text):
        rest_text = ("…" if budget > 1 else "") + rest_text[-(max(0, budget - 1)):]
    parts = [p for p in (head, rest_text) if p]
    return "\n\n".join(parts)[:max_chars]
