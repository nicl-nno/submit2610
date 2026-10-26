'Judge-annotated pseudo-gold for the extraction anchor pool (annotate once).\n\n\nThe ctrl-point **recall** signal of the GRPO-extraction line (the ``(c)``\nanti-collapse component the design settled on): the judge — a model ≠ policy,\nthe same convention as Answer Correctness — annotates the FROZEN anchor val\npool ONCE, and every ctrl-point afterwards compares the policy\'s extraction\nagainst the cached annotation (``llm_training.reward.extraction.pseudo_gold_score``).\nBecause the pool is frozen, the annotation is computed a single time and\ncached in ``extra["extraction"]["pseudo_gold"]`` — each anchor after the\nfirst is nearly free, and the measuring instrument stays byte-identical\nacross steps (the frozen-methodology rule).\n\nSince the 8-class extension the module also owns the **summary-reference\nmode** (:func:`annotate_summary_references`): for consolidation rows\n(``extra["summary"]``) the judge solves the row\'s OWN prompt once and the\nreply is cached in ``extra["summary"]["reference"]`` — the anchor then\ncompares the policy\'s prose against it with plain clipped-cosine SemSim.\n\nImplementation notes\n\n* A **custom judge prompt** (NOT a ragas metric): the ragas ``ContextRecall``\n  collections API exists only on ragas ≥ 0.4 while the vllm_env stack carries\n  0.2.15 — a third ``_ragas_compat`` adapter would add version churn for\n  nothing. The rubrics precedent (``_RU_LITERACY_RUBRICS``): a bespoke prompt\n  through the SAME judge client, sanitizer pipeline and resilience knobs.\n* The reply goes through the full judge cleaning pipeline\n  (``_clean_judge_response_text``: ``<think>`` blocks, markdown fences, prose\n  extraction, invalid-escape repair) and the ``reasoning_content`` fallback —\n  thinking judges (GLM on Z.AI) are supported.\n* Every request carries ``response_format={"type": "json_object"}`` — the\n  server-side JSON guarantee where the backend honours it, a no-op where it\n  does not (the client-side pipeline remains the fallback).'

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from llm_training.extraction.ontology import get_ontology
from llm_training.reward.extraction import parse_extraction_output
from llm_training.reward.judge import _clean_judge_response_text
from llm_training.reward.summarization import (
    SUMMARY_REFERENCE_KEY,
    summary_output_text,
)
from llm_training.runtime.endpoints import OpenAIEndpoint

#: The judge-annotator system prompt (entity/relation lists are rendered in).

PSEUDO_GOLD_SYSTEM = """You are a meticulous information-extraction annotator.
Given a source text and the allowed entity / relation types, produce the
COMPLETE gold annotation of the text:

1. Every entity of the allowed types that appears in the text (or is clearly
   referred to). Use the normalized canonical name: base (lemma) form,
   nominative case, proper capitalization; resolve coreferences; merge
   duplicates and aliases into one entry.
2. Every relation between the annotated entities that the text states or
   clearly implies. Both endpoints MUST be annotated entities; use only the
   allowed relation types; descriptions must be grounded in the text.

Entity types: {entity_types}
Relation types: {relation_types}

Return ONLY a valid JSON object of exactly this shape (no markdown fences,
no extra text):
{{"entities": [{{"entity_name": str, "entity_type": str, "description": str}}], "relations": [{{"source_entity": str, "target_entity": str, "relation_type": str, "description": str, "relationship_strength": int}}]}}"""

#: The extra key the annotation is cached under.

PSEUDO_GOLD_KEY = "pseudo_gold"


@dataclass
class PseudoGoldReport:
    'Summary of one annotation run.\n\n    ``label`` distinguishes the two annotation modes in the operator log:\n    ``"pseudo-gold"`` (extraction) and ``"summary-ref"`` (consolidation\n    references). ``label``'

    n_rows: int
    n_annotated: int
    n_failed: int
    model: str
    label: str = "pseudo-gold"

    def format_lines(self) -> list[str]:
        return [
            f"{self.label}: {self.n_annotated}/{self.n_rows} annotated "
            f"(failed: {self.n_failed}) by {self.model}"
        ]


def _annotator_system(payload: dict[str, Any]) -> str:
    ontology = get_ontology(payload.get("ontology"))
    entity_types = ontology.render_entity_types(
        list(payload.get("entity_types") or ())
    )
    relation_types = ontology.render_relation_types(
        list(payload.get("relation_types") or ())
    )
    return PSEUDO_GOLD_SYSTEM.format(
        entity_types=entity_types or "(infer freely)",
        relation_types=relation_types or "(infer freely)",
    )


def _message_text(message: Any) -> str:
    """``content`` with the ``reasoning_content`` thinking fallback."""
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    reasoning = getattr(message, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    return content if isinstance(content, str) else ""


async def _annotate_one(
    client: Any,
    model: str,
    payload: dict[str, Any],
    max_tokens: int,
    retries: int,
    retry_sleep_s: float,
) -> dict[str, Any] | None:
    """Annotate one row; ``None`` on failure after retries (never raises)."""
    messages = [
        {"role": "system", "content": _annotator_system(payload)},
        {"role": "user", "content": f"Text:\n{payload.get('text') or ''}"},
    ]
    last_error: str = ""
    for attempt in range(retries + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            text = _message_text(response.choices[0].message)
            cleaned = _clean_judge_response_text(text)
            parsed = parse_extraction_output(cleaned)
            if parsed is not None and parsed["entities"]:
                return parsed
            last_error = "empty or unparseable annotation"
        except Exception as exc:  # noqa: BLE001 - a failed row is skipped
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries and retry_sleep_s > 0:
            await asyncio.sleep(retry_sleep_s * (2**attempt))
    if last_error:
        # Surface the reason in the caller's log without failing the run.
        
        print(f"pseudo-gold: annotation failed ({last_error})")
    return None


async def annotate_rows(
    rows: list[dict[str, Any]],
    endpoint: OpenAIEndpoint,
    concurrency: int = 8,
    max_tokens: int = 8192,
    retries: int = 2,
    retry_sleep_s: float = 2.0,
    timeout_s: float = 600.0,
) -> tuple[list[dict[str, Any]], PseudoGoldReport]:
    'Annotate the frozen anchor rows in place; returns ``(rows, report)``.\n\n    ``extra["extraction"]["pseudo_gold"]`` are skipped (idempotent re-runs —\n    the annotation must stay byte-identical once computed).'
    client = endpoint.async_client(max_retries=2, timeout=timeout_s)
    # The judge sanitizer wrapper (fences / think-blocks / escape repair) is
    # applied manually per reply here; a wrapped client would also work, but
    # the explicit call keeps this module independent of instructor wiring.
    sem = asyncio.Semaphore(max(1, concurrency))
    report = PseudoGoldReport(
        n_rows=len(rows), n_annotated=0, n_failed=0, model=endpoint.model
    )
    pending: list[asyncio.Task[dict[str, Any] | None]] = []

    async def guarded(payload: dict[str, Any]) -> dict[str, Any] | None:
        async with sem:
            return await _annotate_one(
                client, endpoint.model, payload, max_tokens, retries, retry_sleep_s
            )

    for row in rows:
        payload = (row.get("extra") or {}).get("extraction")
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get(PSEUDO_GOLD_KEY), dict):
            report.n_annotated += 1
            continue
        pending.append(asyncio.create_task(guarded(payload)))

    results = list(await asyncio.gather(*pending)) if pending else []
    # asyncio.gather preserves the submission order; walk the rows again and
    # attach the results to the payloads that requested them.
    # asyncio.gather
    it = iter(results)
    for row in rows:
        payload = (row.get("extra") or {}).get("extraction")
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get(PSEUDO_GOLD_KEY), dict):
            continue
        annotation = next(it, None)
        if annotation is not None:
            payload[PSEUDO_GOLD_KEY] = {
                "entities": annotation["entities"],
                "relations": annotation["relations"],
                "model": endpoint.model,
            }
            report.n_annotated += 1
        else:
            report.n_failed += 1
    return rows, report


# ---------------------------------------------------------------------------
# Summary-reference mode
# ---------------------------------------------------------------------------
_SUMMARY_KINDS = ("entity", "relation", "community")


def _summary_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    "The row's own prompt as chat messages; ``None`` when unusable.\n\n    The judge solves the SAME consolidation task the policy faces — no\n    separate annotator template to drift out of sync with the task prompts\n    (the schema stays inline in the prompt, the established deviation)."
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        return None
    messages: list[dict[str, str]] = []
    for m in prompt:
        if isinstance(m, dict) and isinstance(m.get("content"), str):
            messages.append(
                {"role": str(m.get("role") or "user"), "content": m["content"]}
            )
    return messages or None


async def _annotate_summary_one(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    kind: str,
    max_tokens: int,
    retries: int,
    retry_sleep_s: float,
) -> str | None:
    """Judge-solve one consolidation prompt; ``None`` on failure (no raise)."""
    last_error: str = ""
    for attempt in range(retries + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            text = _message_text(response.choices[0].message)
            cleaned = _clean_judge_response_text(text)
            # A reference is usable only when it parses into the kind's
            # envelope with a non-empty prose payload — the anchor compares
            # exactly that prose
            
            if summary_output_text(cleaned, kind):
                return cleaned
            last_error = "empty or unparseable summary reference"
        except Exception as exc:  # noqa: BLE001 - a failed row is skipped
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries and retry_sleep_s > 0:
            await asyncio.sleep(retry_sleep_s * (2**attempt))
    if last_error:
        print(f"summary-ref: annotation failed ({last_error})")
    return None


async def annotate_summary_references(
    rows: list[dict[str, Any]],
    endpoint: OpenAIEndpoint,
    concurrency: int = 8,
    max_tokens: int = 8192,
    retries: int = 2,
    retry_sleep_s: float = 2.0,
    timeout_s: float = 600.0,
) -> tuple[list[dict[str, Any]], PseudoGoldReport]:
    'Write the judge\'s reference summaries for consolidation rows (once).\n\n    The summary-reference mode: for every row carrying ``extra["summary"]``\n    the judge (≠ policy) solves the row\'s OWN consolidation prompt and the\n    reply is cached as ``extra["summary"]["reference"] = {"response",\n    "model"}``. The anchor afterwards compares the policy\'s prose against\n    this reference with plain clipped-cosine SemSim — judge-free at anchor\n    time (the pseudo-gold precedent; the reference is computed ONCE, the\n    measuring instrument stays frozen across the run series).\n\nIdempotency\n    counted and skipped (re-runs never rewrite a frozen annotation); rows\n    with a malformed prompt are skipped WITHOUT a failure (a broken row is a\n    data defect, not a judge error). ``report.n_rows`` counts the eligible\n    consolidation rows of this mode (unlike the extraction report, whose\n    denominator is the whole pool).'
    client = endpoint.async_client(max_retries=2, timeout=timeout_s)
    sem = asyncio.Semaphore(max(1, concurrency))
    report = PseudoGoldReport(
        n_rows=0, n_annotated=0, n_failed=0, model=endpoint.model,
        label="summary-ref",
    )
    pending: list[asyncio.Task[str | None]] = []

    async def guarded(
        messages: list[dict[str, str]], kind: str
    ) -> str | None:
        async with sem:
            return await _annotate_summary_one(
                client, endpoint.model, messages, kind, max_tokens,
                retries, retry_sleep_s,
            )

    for row in rows:
        payload = (row.get("extra") or {}).get("summary")
        if not isinstance(payload, dict):
            continue
        kind = str(payload.get("kind") or "")
        if kind not in _SUMMARY_KINDS:
            continue
        report.n_rows += 1
        if isinstance(payload.get(SUMMARY_REFERENCE_KEY), dict):
            report.n_annotated += 1
            continue
        messages = _summary_messages(row)
        if messages is None:
            continue
        pending.append(asyncio.create_task(guarded(messages, kind)))

    results = list(await asyncio.gather(*pending)) if pending else []
    # asyncio.gather preserves the submission order; walk the rows again and
    # attach the results to the payloads that requested them (the same
    # two-pass alignment as annotate_rows).
    # asyncio.gather
    it = iter(results)
    for row in rows:
        payload = (row.get("extra") or {}).get("summary")
        if not isinstance(payload, dict):
            continue
        kind = str(payload.get("kind") or "")
        if kind not in _SUMMARY_KINDS:
            continue
        if isinstance(payload.get(SUMMARY_REFERENCE_KEY), dict):
            continue
        if _summary_messages(row) is None:
            continue
        reference = next(it, None)
        if reference is not None:
            payload[SUMMARY_REFERENCE_KEY] = {
                "response": reference,
                "model": endpoint.model,
            }
            report.n_annotated += 1
        else:
            report.n_failed += 1
    return rows, report
