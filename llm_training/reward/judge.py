'RAGAS-backed :class:`Scorer` — the real judge implementation.\n\n\nThis module is the **only** place that imports ``ragas`` (lazily), so the rest of\nthe reward layer stays importable in offline/dev environments without the\n``train`` extra. The :class:`~llm_training.reward.config.Scorer` protocol is what\n:mod:`llm_training.reward.dispatch` depends on; tests inject a fake scorer, while\nthe real evaluation / training path builds this one via :func:`make_ragas_scorer`.\n\n\nReward shape (revised)\n\nFor every example, regardless of task class\n\n* **Semantic Similarity** (RAGAS :class:`~ragas.metrics.SemanticSimilarity`;\n  embeddings only, no LLM) — always computed. This is the cheap backbone that\n  makes the reward usable in the GRPO loop (one embedding call per response).\n  (We use ``SemanticSimilarity`` rather than its ``AnswerSimilarity`` subclass\n  because the latter is not exported from ``ragas.metrics.collections`` in\n  ragas 0.4.3 despite the DeprecationWarning advising that path; the two are\n  behaviourally identical.)\n* **Answer Correctness** (RAGAS :class:`~ragas.metrics.AnswerCorrectness`; LLM)\n  — only when ``cfg.judge is not None`` AND ``cfg.compute_answer_correctness``.\n  This is the semantic anchor used in run_eval.py and at GRPO ctrl-points.\n* **Judge rubrics** (RAGAS ``RubricsScoreWithoutReference``; LLM) — only when\n  ``cfg.judge is not None`` AND ``cfg.enable_judge_rubrics``, and only for SGR\n  **reasoning turns** (where AC is structurally skipped): two reference-free\n  criteria — Russian-language literacy and consistency with the input context\n  — combined into ``ScoredMetrics.rubric_score`` via the strict harmonic mean.\n  The ctrl-point companion of AC: full val-pool coverage by the judge with a\n  methodology different from the in-loop reward (which uses the deterministic\n  ``reasoning_check`` + the small grammar-classifier fluency signal there).\n* **Format check** (:func:`llm_training.reward.scoring.format_check`; pure\n  Python) — only for tool-use classes; ``None`` (excluded from the harmonic\n  combine) for everyone else.\n\nThe active metrics are combined via :func:`llm_training.reward.scoring.harmonic_mean`.\n``Faithfulness`` and ``NoiseSensitivity`` were removed — see\n:mod:`llm_training.reward.config` for the rationale.'

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any

from llm_training.reward.config import RewardConfig, ScoredMetrics
from llm_training.reward.docsem import (
    docsem_answer_text,
    docsem_check,
    docsem_check_applies,
)
from llm_training.reward.fluency import (
    FluencyClassifier,
    FluencyScorer,
    make_fluency_scorer,
)
from llm_training.reward.inputs import build_rubric_context
from llm_training.reward.language import language_match_factor
from llm_training.reward.nerel_f1 import nerel_f1
from llm_training.reward.rerank_em import rerank_em, rerank_em_applies
from llm_training.reward.rummlu_em import rummlu_em
from llm_training.reward.schema_check import schema_check
from llm_training.reward.scoring import (
    code_check,
    extract_final_answer,
    format_check,
    harmonic_mean,
    reasoning_check,
    safe_score,
    split_prose_and_code,
)

logger = logging.getLogger(__name__)

#: Likert (1–5) rubric descriptions for the **Russian-language literacy**
#: criterion of SGR reasoning turns. The judge's score is normalized to [0, 1]
#: as ``(score − 1) / 4`` (a hard 1 → 0.0 — the strict harmonic mean then
#: zeroes the example, a strong ctrl-point signal).

# 4.
_RU_LITERACY_RUBRICS: dict[str, str] = {
    "score1_description": (
        "Текст на русском языке практически нечитаем: множество грубых "
        "орфографических, пунктуационных и грамматических ошибок, слова "
        "склеены или искажены, смысл отдельных фраз восстановить невозможно."
    ),
    "score2_description": (
        "Много ошибок (орфография, пунктуация, согласование, управление); "
        "они заметно мешают чтению, часть предложений приходится перечитывать,"
        " чтобы понять смысл."
    ),
    "score3_description": (
        "Текст в целом читаем, но содержит несколько заметных ошибок "
        "орфографии, пунктуации или грамматики; стиль небрежный, встречаются "
        "неестественные формулировки."
    ),
    "score4_description": (
        "Грамотный текст: встречаются лишь единичные незначительные ошибки "
        "(опечатка, запятая), не мешающие восприятию; стиль в целом ровный."
    ),
    "score5_description": (
        "Безупречный литературный русский язык: ошибок нет, текст изложен "
        "ясно, гладко и стилистически аккуратно."
    ),
}

#: Likert (1–5) rubric descriptions for the **consistency with the input
#: context** criterion of SGR reasoning turns (the context is rebuilt from the
#: original message-list prompt by :func:`build_rubric_context`).

_CONTEXT_CONSISTENCY_RUBRICS: dict[str, str] = {
    "score1_description": (
        "Рассуждение никак не связано с поставленной задачей и контекстом: "
        "речь идёт о постороннем предмете либо содержание противоречит "
        "исходным данным."
    ),
    "score2_description": (
        "Рассуждение в основном не согласуется с задачей или контекстом: "
        "задача понята неверно, важные факты контекста игнорируются или "
        "искажаются."
    ),
    "score3_description": (
        "Рассуждение связано с задачей лишь частично: встречаются заметные "
        "несоответствия контексту или утверждения об отсутствующих в нём "
        "данных."
    ),
    "score4_description": (
        "Рассуждение соответствует задаче и контексту; возможны лишь мелкие "
        "неточности, не влияющие на ход решения."
    ),
    "score5_description": (
        "Рассуждение полностью согласуется с задачей и всем входным "
        "контекстом: корректно учитывает условие, предыдущие шаги и "
        "результаты инструментов."
    ),
}


def _normalize_rubric(value: Any) -> float:
    'Normalize a 1–5 Likert rubric score to [0, 1].'
    try:
        v = (float(value) - 1.0) / 4.0
    except (TypeError, ValueError):
        return 0.0
    return safe_score(v)


def _import_ragas_metrics() -> tuple[Any, Any, Any]:
    'Lazily import the RAGAS metric classes we still use (version-adaptive).\n\n\n    Returns ``(AnswerCorrectness, SemanticSimilarity,\n    RubricsScoreWithoutReference)``. On ragas ≥ 0.4.0 these are the native\n    classes (we use :class:`SemanticSimilarity` — the base class of\n    ``AnswerSimilarity`` — because in ragas 0.4.3 ``AnswerSimilarity`` is\n    **not** exported from ``ragas.metrics.collections`` despite the\n    DeprecationWarning advising exactly that import path; the two classes are\n    behaviourally identical). On the older stacks (the ``vllm_env`` cluster\n    env carries ragas 0.2.15, where ``RubricsScoreWithoutReference`` does not\n    exist and the ``ascore``/embeddings interfaces differ)\n    :mod:`llm_training.reward._ragas_compat` returns adapter classes with the\n    SAME construction/call surface. Raises a clear, actionable error if\n    ``ragas`` is not installed.'
    try:
        from llm_training.reward._ragas_compat import import_ragas_metrics

        return import_ragas_metrics()
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "The 'ragas' package is required for the judge. Install the "
            "train extra: pip install -e '.[train]'."
            "\nДля работы судьи нужен пакет 'ragas': pip install -e '.[train]'."
        ) from e


def _import_ragas_factories() -> tuple[Any, Any]:
    'Lazily import ``llm_factory`` / ``embedding_factory`` (version-adaptive).\n\n\n    On ragas ≥ 0.4.0 the native factories; on the older stacks (vllm_env /\n    ragas 0.2.15) the :mod:`llm_training.reward._ragas_compat` adapters (the\n    native ``llm_factory`` there accepts neither ``client=`` nor\n    ``max_tokens=``, and ``embedding_factory`` builds OpenAI embeddings — see\n    the compat module docstring).'
    try:
        from llm_training.reward._ragas_compat import import_ragas_factories

        return import_ragas_factories()
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Could not import ragas llm/embedding factories. Install the "
            "train extra: pip install -e '.[train]'."
            "\nНе удалось импортировать фабрики ragas: pip install -e "
            "'.[train]'."
        ) from e


def strip_json_code_fences(text: str) -> str:
    'Strip a markdown code fence from around a JSON string, if present.\n\n\n    The judge LLM (observed: Qwen2.5 on vLLM 0.7.3 V1, which has no\n    structured-output support and silently ignores guided-decoding params)\n    habitually wraps structured answers in ```` ```json ... ``` ````.\n    RAGAS/instructor then call a STRICT ``model_validate_json`` on the raw\n    string and fail at char 0 ("Invalid JSON: expected value at line 1\n    column 1") — deterministically, at low temperature, so every retry fails\n    too. (The judge service now runs the V0 engine, where xgrammar enforces\n    ``json_object`` at the token level; this helper remains as defense in\n    depth.) It returns the inner JSON; plain unfenced text is returned\n    unchanged (the conservative guard: only strings starting with ```` ``` ````\n    are touched).\n    structured output, guided\n    ``model_validate_json``'
    s = text.strip()
    if not s.startswith("```"):
        return text
    if "\n" not in s:
        return text
    body = s.split("\n", 1)[1].strip()
    if body.endswith("```"):
        body = body[: -len("```")].strip()
    return body


def strip_think_blocks(text: str) -> str:
    'Remove ``<think>...</think>`` reasoning blocks from a judge reply.\n\n\n    Some OpenAI-compatible gateways inline a thinking model\'s chain of\n    thought into ``message.content`` (the DeepSeek-R1 style\n    ``<think>…</think>{"TP": ...}``) instead of the separate\n    ``reasoning_content`` field. RAGAS/instructor run a strict\n    ``model_validate_json`` and contain NO ``<think>`` handling — such a\n    reply fails at char 0. Closed blocks are removed anywhere in the text;\n    an UNCLOSED ``<think>`` (no matching ``</think>``) means everything from\n    it on is reasoning — the tail is dropped. Text without the tags is\n    returned unchanged.\n    ``<think>…</think>{"TP": ...}``\n    ``reasoning_content``. RAGAS/instructor'
    s = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    idx = s.find("<think>")
    if idx != -1:
        s = s[:idx]
    return s


def extract_json_payload(text: str) -> str:
    'Extract the outermost JSON object/array from a prose-wrapped reply.\n\n\n    When the text does not already start with ``{`` or ``[``, return the\n    substring from the first opening bracket to the LAST matching closing\n    bracket (``{``…``}`` or ``[…]`` — whichever opens earlier). Conservative:\n    text that already starts with a JSON bracket, or contains no bracket\n    pair, is returned unchanged — the heuristic can only rescue replies it\n    would otherwise have doomed, never break a parseable one.'
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return text
    candidates: list[tuple[int, int]] = []
    obj_start, obj_end = text.find("{"), text.rfind("}")
    if obj_start != -1 and obj_end > obj_start:
        candidates.append((obj_start, obj_end))
    arr_start, arr_end = text.find("["), text.rfind("]")
    if arr_start != -1 and arr_end > arr_start:
        candidates.append((arr_start, arr_end))
    if not candidates:
        return text
    start, end = min(candidates)
    return text[start : end + 1]


def _repair_invalid_json_escapes(text: str) -> str:
    'Double backslashes that do not start a valid JSON escape sequence.\n\n\n    JSON strings allow exactly ``\\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX``; a judge LLM\n    that copies LaTeX notation verbatim from the judged content (observed on\n    the vllm_env SFT anchors: ``"\\(x^4 - 5x^2 + 4 = 0\\)"`` in the\n    statements) emits ``\\(`` — an invalid escape that fails ``json.loads``\n    ("Invalid \\\\escape") deterministically, so every retry of a near-greedy\n    judge reproduces the same broken output. The walk scans left to right;\n    a backslash followed by a character OUTSIDE the valid set is doubled\n    (``\\(`` → ``\\\\(``, i.e. a literal backslash before the paren — the\n    semantics the judge meant), while already-valid escapes (including the\n    second backslash of an escaped ``\\\\`` pair) pass through untouched —\n    a naive ``re.sub`` would re-escape that second backslash and corrupt\n    valid JSON.'
    valid_next = '"\\/bfnrtu'
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            if text[i + 1] in valid_next:
                # A valid escape pair — consume BOTH characters, so the
                # second backslash of a ``\\`` pair is never re-examined
                # (and never re-doubled in front of the char that follows).
                
                
                
                out.append(ch)
                out.append(text[i + 1])
                i += 2
                continue
            out.append("\\\\")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _maybe_repair_json_escapes(text: str) -> str:
    'Return the escape-repaired text, but only when it rescues the parse.\n\n\n    Conservative rescue-only semantics (the :func:`extract_json_payload`\n    philosophy): the repair is applied IFF the original fails ``json.loads``\n    AND the repaired text parses — a reply that already parses, free-form\n    prose (e.g. the judge-preflight greeting) and truncated JSON (which no\n    escape repair can rescue — that failure class is handled by raising the\n    judge ``max_tokens``) are all returned unchanged. The judge therefore\n    sees exactly what it produced, except for the one deterministic\n    transformation that turns a doomed reply into the reply it meant.'
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        pass
    repaired = _repair_invalid_json_escapes(text)
    if repaired == text:
        return text
    try:
        json.loads(repaired)
    except (json.JSONDecodeError, ValueError):
        return text
    return repaired


def _clean_judge_response_text(text: str) -> str:
    'The full judge-reply cleaning pipeline (see the sanitizer docstring).'
    payload = extract_json_payload(strip_json_code_fences(strip_think_blocks(text)))
    return _maybe_repair_json_escapes(payload)


def sanitize_judge_client_json_fences(client: Any) -> Any:
    'Wrap ``client.chat.completions.create`` to clean judge JSON replies.\n\n\n    Post-processes every non-streaming response BEFORE instructor/RAGAS parse\n    it — their ``model_validate_json`` is strict and their OpenAI JSON-mode\n    handler reads ``choices[*].message.content`` exclusively, so any noise\n    around the JSON fails the parse at char 0 and triggers the retry storm\n    (instructor re-asks up to 4 times with an ever-growing "Correct your\n    JSON" conversation; the metric level retries 3 more times — up to 12\n    judge calls per broken metric, each re-generating a thinking model\'s\n    reasoning). The cleaning pipeline (``_clean_judge_response_text``):\n    (1) strip ``<think>…</think>`` blocks (:func:`strip_think_blocks`);\n    (2) strip markdown ```` ``` ````-fences (:func:`strip_json_code_fences`);\n    (3) extract the outermost JSON substring from prose-wrapped text\n    (:func:`extract_json_payload`); (4) repair invalid JSON escape\n    sequences — a backslash before a character outside\n    ``" \\ / b f n r t u`` is doubled, rescuing judge replies that copy\n    LaTeX ``\\(…\\)`` notation verbatim from the judged content\n    (:func:`_maybe_repair_json_escapes`, applied only when the repair\n    turns a failing parse into a passing one). Client-side cleaning is far\n    cheaper than any single retry. Apply BEFORE the client is handed to\n    ``llm_factory``: instructor captures the bound ``create`` at wiring\n    time, so it must capture the sanitized wrapper.\n    instructor/RAGAS\n    ``<think>…</think>`` (:func:`strip_think_blocks`); (2\n    (:func:`extract_json_payload`); (4\n\n    Thinking-model fallback: when ``message.content`` is empty but\n    ``message.reasoning_content`` is not, the reasoning text is used as the\n    content (before the cleaning pipeline). Cloud thinking models (GLM on\n    Z.AI; ``glm-5.3-flash`` cannot disable thinking — ``thinking.type``\n    supports only ``enabled``) occasionally return the whole answer in the\n    separate ``reasoning_content`` field with an empty ``content``, which\n    would deterministically kill every AC/rubric parse. Mirrors the\n    established fallback in ``scripts/calibrate_fluency.py``\n    (``_message_text``). A no-op for local vLLM judges: their messages carry\n    no ``reasoning_content`` attribute at all.\n    ``message.reasoning_content``\nvLLM'
    original_create = client.chat.completions.create

    async def _create_with_fence_sanitizer(*args: Any, **kwargs: Any) -> Any:
        response = await original_create(*args, **kwargs)
        if kwargs.get("stream") or not hasattr(response, "choices"):
            return response
        for choice in response.choices or []:
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None)
            if not (isinstance(content, str) and content.strip()):
                reasoning = getattr(message, "reasoning_content", None)
                if isinstance(reasoning, str) and reasoning.strip():
                    content = reasoning
            if isinstance(content, str):
                try:
                    message.content = _clean_judge_response_text(content)
                except Exception:  # noqa: BLE001 - never fail a judge call on sanitation
                    pass
        return response

    client.chat.completions.create = _create_with_fence_sanitizer  # type: ignore[assignment]
    return client


def _build_metrics_from_ragas(
    cfg: RewardConfig, *, base_embeddings: Any | None = None,
) -> tuple[Any, Any, tuple[Any, Any] | None]:
    "Lazily import RAGAS and wire the metrics to ``cfg``'s endpoints.\n\n\n    Returns ``(ac, semsim, rubrics)`` — the :class:`AnswerCorrectness` (needs\n    judge LLM + embeddings), the :class:`SemanticSimilarity` (embeddings only)\n    and the rubric pair (``(literacy, context_consistency)`` — two\n    ``RubricsScoreWithoutReference`` instances sharing the SAME judge LLM;\n    ``None`` when rubrics are not enabled). Extracted from the constructor as\n    a module function so it can be unit-tested in isolation (wiring of\n    ``max_tokens`` / ``max_retries``) and bypassed in tests via the\n    ``metrics`` injection point.\n\nArgs\n        cfg: reward configuration\n        base_embeddings: optional pre-built base embeddings object to REUSE\n            instead of calling ``embedding_factory`` — the sharing seam for\n            processes that build several scorers (grpo_train.py: the in-loop reward\n            + the judge anchors). Only the heavy ST model weights are shared;\n            each scorer still wraps the base in its OWN\n            :class:`CachingEmbeddings` + semaphore, so asyncio primitives\n            never cross event loops. ``None`` (default) → build a new one.\n\n    Embeddings are **always** computed in-process via ``sentence-transformers``\n    (RAGAS's ``HuggingFaceEmbeddings``) — no HTTP server, no cloud endpoint, no\n    connection management. The model shares the GPU with the policy; its\n    concurrency is capped by ``CachingEmbeddings``' semaphore (wired in the\n    scorer constructor). The judge LLM, when configured, stays an external\n    OpenAI-compatible HTTP endpoint.\n    in-process"
    AnswerCorrectness, SemanticSimilarity, Rubrics = _import_ragas_metrics()
    llm_factory, embedding_factory = _import_ragas_factories()

    # Silence the sentence-transformers "Batches:" tqdm bars: ST decides to
    # show them when its own logger level is INFO/DEBUG (our basicConfig is
    # INFO) and then passes `disable=not show_progress_bar` to tqdm
    # EXPLICITLY — env vars like TQDM_DISABLE only override defaults, so they
    # cannot suppress these bars. Raising the logger above INFO flips ST's
    # own heuristic off, deterministically, for every consumer of this module
    # (grpo_train.py, run_eval.py, the judge anchors). The trainer's own ETA bar
    # is NOT touched.
    
    
    
    
    
    # run_eval.py
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    # Embeddings: in-process via sentence-transformers (HuggingFaceEmbeddings).
    # Reuse the caller's base when given (shared weights); otherwise build.
    
    
    if base_embeddings is not None:
        embeddings = base_embeddings
    else:
        embeddings = embedding_factory(
            "huggingface",
            cfg.local_embedding_model,
            device=cfg.local_embedding_device,
        )

    # The judge LLM client is built when ANY judge-side metric is requested —
    # AC and/or the reasoning-turn rubrics — so one client, one fence
    # sanitizer and one timeout wiring serve all of them.
    
    
    ac_metric = None
    rubrics: tuple[Any, Any] | None = None
    if cfg.judge is not None and (
        cfg.compute_answer_correctness or cfg.enable_judge_rubrics
    ):
        # RAGAS's InstructorModelArgs defaults max_tokens to 1024, which is far
        # too low for the structured output of CorrectnessClassifier and
        # triggers "The output is incomplete due to a max_tokens length limit".
        # Pass cfg.judge_max_tokens explicitly (default 4096).
        
        judge_client = cfg.judge.async_client(
            max_retries=cfg.judge_max_retries,
            timeout=cfg.judge_timeout_s,
        )
        # Strip markdown JSON fences from judge responses before instructor/
        # RAGAS parse them (see sanitize_judge_client_json_fences) — without
        # this, fenced JSON fails the strict validate_json deterministically
        # and the example's AC is dropped after a 4×12-call retry storm.
        
        # instructor/RAGAS
        
        judge_client = sanitize_judge_client_json_fences(judge_client)
        judge_llm = llm_factory(
            cfg.judge.model,
            client=judge_client,
            max_tokens=cfg.judge_max_tokens,
        )
        if cfg.compute_answer_correctness:
            ac_metric = AnswerCorrectness(llm=judge_llm, embeddings=embeddings)
        if cfg.enable_judge_rubrics:
            # Two reference-free rubric criteria for SGR reasoning turns,
            # sharing the judge LLM with AC (one HTTP client, one semaphore).
            
            
            rubrics = (
                Rubrics(llm=judge_llm, rubrics=_RU_LITERACY_RUBRICS),
                Rubrics(llm=judge_llm, rubrics=_CONTEXT_CONSISTENCY_RUBRICS),
            )

    semsim_metric = SemanticSimilarity(embeddings=embeddings)
    return ac_metric, semsim_metric, rubrics


class CachingEmbeddings:
    'In-memory LRU cache with future-based concurrent-call dedup.\n\n    In-memory\n\n    Wraps a RAGAS embeddings object so that repeated calls with the same text\n    return a cached result instead of re-hitting the embedding endpoint. In\n    GRPO, each prompt has one reference answer but N generated completions —\n    :class:`~ragas.metrics.SemanticSimilarity` embeds that reference N times.\n    This cache eliminates the redundancy (up to 44 % fewer embedding calls per\n    GRPO step at ``num_generations=8``).\n\n\nFour mechanisms\n1. **Cache hit**\n    2. **Future dedup** — concurrent calls for the same text share one\n       :class:`asyncio.Future` (no thundering herd\n    3. **LRU eviction** — bounded :class:`~collections.OrderedDict` prevents\n       unbounded memory growth across GRPO steps\n    4. **Semaphore gating** — an optional semaphore is acquired only on a cache\n       miss, around the real base computation, so in-process embedding compute\n       (which shares the GPU with the policy) can be capped independently of the\n       judge endpoint. Cache hits and dedup-waiters skip it.\n\n    The cache is transparent: it delegates any attribute access (e.g.\n    ``model``, ``__repr__``) to the wrapped base object via ``__getattr__``.'

    #: Maximum number of cached embeddings (~40 MB at 4 KB per bge-m3 vector).
    
    MAX_CACHE_SIZE: int = 10_000

    def __init__(self, base: Any, *, sem: asyncio.Semaphore | None = None) -> None:
        'Args\n            base: the underlying RAGAS embeddings object being wrapped.\n            sem: optional semaphore acquired around each **cache-miss** base\n                call (i.e. the actual GPU/embedding computation). Cache hits and\n                dedup-waiters skip the semaphore, so it caps real compute\n                concurrency without throttling redundant requests. When ``None``\n                (default), no gating is applied'
        self._base = base
        self._sem = sem
        self._cache: OrderedDict[bytes, list[float]] = OrderedDict()
        self._pending: dict[bytes, asyncio.Future[list[float]]] = {}

    @staticmethod
    def _key(text: str) -> bytes:
        # Deterministic, collision-safe key (unlike hash(text), which is
        # process-randomised under PYTHONHASHSEED and theoretically collision-
        # prone). blake2b at 16 bytes is ~2^128 collision space — overkill here.
        
        return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        'Async embed with cache + dedup + optional semaphore.'
        key = self._key(text)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        pending = self._pending.get(key)
        if pending is not None:
            return await pending
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[list[float]] = loop.create_future()
        self._pending[key] = fut
        try:
            # Acquire the semaphore (if any) only on a cache miss, around the
            # actual base call. Dedup-waiters (awaiting ``fut``) never acquire
            # it, and cache hits skip it entirely — so the semaphore caps real
            # compute concurrency, not redundant requests.
            
            if self._sem is None:
                result = await self._base.aembed_text(text, **kwargs)
            else:
                async with self._sem:
                    result = await self._base.aembed_text(text, **kwargs)
            self._cache[key] = result
            if len(self._cache) > self.MAX_CACHE_SIZE:
                self._cache.popitem(last=False)
            fut.set_result(result)
            return result
        except BaseException as exc:
            if not fut.done():
                if isinstance(exc, Exception):
                    # Propagate the actual exception to concurrent waiters so
                    # that _ascore's ``except Exception`` can retry. Using
                    # fut.cancel() here would give waiters CancelledError
                    # (BaseException), which escapes all ``except Exception``
                    # handlers and aborts the entire asyncio.gather batch.
                    
                    
                    # CancelledError (BaseException
                    
                    fut.set_exception(exc)
                else:
                    # Genuine CancelledError or other BaseException — cancel.
                    
                    fut.cancel()
            raise
        finally:
            self._pending.pop(key, None)

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        'Sync embed with cache (no dedup — sync calls are sequential).'
        key = self._key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        result = self._base.embed_text(text, **kwargs)
        self._cache[key] = result
        if len(self._cache) > self.MAX_CACHE_SIZE:
            self._cache.popitem(last=False)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


class RAGASScorer:
    "Real :class:`Scorer` backed by RAGAS metrics (in-process embeddings).\n\n\n    Construction wires the in-process embedding model (always) and the optional\n    judge LLM (only when ``cfg.judge is not None`` and\n    ``cfg.compute_answer_correctness``) into the RAGAS metrics. :meth:`score`\n    is async and intended to be awaited inside the event loop owned by\n    :class:`~llm_training.runtime.reward_service.RewardService`.\n\nTwo independent semaphores control concurrency\n\n    * ``_embed_sem`` (``cfg.embedding_concurrency``) — limits concurrent\n      in-process embedding computations. In production it is wired into the\n      shared :class:`CachingEmbeddings` instance and acquired on each cache\n      miss, so it caps real GPU compute regardless of whether the call\n      originates from SemSim (every GRPO step) or from Answer Correctness\n      (ctrl-points). Cache hits and dedup-waiters skip it.\n    * ``_judge_sem`` (``cfg.judge_concurrency``) — limits concurrent judge LLM\n      requests (Answer Correctness, at ctrl-points only). The judge endpoint is\n      typically an external LLM with higher capacity.\n\n    AC's :meth:`_ascore` call passes ``_judge_sem`` (it gates the LLM call);\n    its internal embedding sub-calls are gated by ``_embed_sem`` inside the\n    shared cache. SemSim's :meth:`_ascore` call passes no outer semaphore — its\n    embedding calls are gated directly by ``_embed_sem`` inside the cache. This\n    avoids acquiring ``_embed_sem`` twice (a non-reentrant deadlock) and means\n    the two limits never throttle each other.\n\n    Both metrics share the **same** :class:`CachingEmbeddings` instance (built\n    once in :func:`_build_metrics_from_ragas`), so the reference-answer\n    embedding is computed once and reused across SemSim and AC and across all\n    completions of a GRPO prompt.\n\nArgs\n        cfg: reward configuration (endpoints, ``judge_max_tokens``,\n            ``judge_max_retries``, ``embedding_concurrency``, known tool names,\n            fluency model / allowlist / calibrator, ``enable_fluency`` /\n            ``enable_code_check``\n        metrics: optional ``(ac, semsim, rubrics)`` tuple to bypass the real\n            RAGAS wiring. When ``None`` (default), the metrics are built via\n            :func:`_build_metrics_from_ragas` — the only path used in\n            production. ``ac`` may be ``None`` to disable AC even when the\n            judge endpoint is set (mirrors ``compute_answer_correctness=\n            False``); ``rubrics`` is the ``(literacy, context_consistency)``\n            metric pair or ``None`` (mirrors ``enable_judge_rubrics=False``).\n            Tests inject fakes to exercise :meth:`_ascore` retry and\n            :meth:`score` failure-recording logic offline.\n            RAGAS-wiring'\n        fluency_scorer: optional :class:`FluencyScorer` to bypass the real\n            (model-backed) fluency wiring — the DI seam tests use to inject a\n            fake. When ``None`` (default), a real scorer is built via\n            :func:`make_fluency_scorer` iff ``cfg.enable_fluency`` is True\n            (the model itself loads lazily on the first :meth:`score` call, so\n            construction stays cheap). When ``enable_fluency`` is False no\n            scorer is built at all and fluency is skipped entirely.\n        base_embeddings: optional pre-built base embeddings object to reuse in\n            the production path (forwarded to\n            :func:`_build_metrics_from_ragas`) — the seam grpo_train.py uses to\n            share the heavy bge-m3 weights between the in-loop scorer and the\n            judge-anchor scorer instead of loading a second ~2.3 GB copy.\n            Ignored when ``metrics`` is injected\n        fluency_classifier: optional already-loaded fluency classifier shared\n            with another scorer (each scorer still builds its own\n            :class:`FluencyScorer` wrapper with its own semaphore/cache, so\n            asyncio primitives never cross event loops). ``None`` → the\n            classifier loads lazily as usual"

    def __init__(
        self,
        cfg: RewardConfig,
        *,
        metrics: tuple[Any | None, Any, tuple[Any, Any] | None] | None = None,
        fluency_scorer: FluencyScorer | None = None,
        base_embeddings: Any | None = None,
        fluency_classifier: FluencyClassifier | None = None,
    ) -> None:
        self._cfg = cfg
        self._max_retries = cfg.judge_max_retries
        self._known_tool_names = cfg.known_tool_names
        self._known_tool_schemas = cfg.known_tool_schemas
        self._embed_sem = asyncio.Semaphore(cfg.embedding_concurrency)
        self._judge_sem = asyncio.Semaphore(cfg.judge_concurrency)
        if metrics is not None:
            self._ac, self._semsim, self._rubrics = metrics
        else:
            self._ac, self._semsim, self._rubrics = _build_metrics_from_ragas(
                cfg, base_embeddings=base_embeddings,
            )
        # Wrap each distinct base embeddings object in a CachingEmbeddings that
        # carries _embed_sem, then repoint every metric that used it. In the
        # production path both metrics share ONE base → ONE cache (the reference
        # embedding is computed once and reused across SemSim and AC and across
        # all completions of a GRPO prompt). In the test path metrics may carry
        # distinct bases → each gets its own wrap. This is also where AC's
        # internal embedding calls get gated by _embed_sem (previously they
        # bypassed it, violating the per-endpoint limit).
        
        
        seen: dict[int, CachingEmbeddings] = {}
        for metric in (self._semsim, self._ac):
            if metric is None:
                continue
            _emb = getattr(metric, "embeddings", None)
            if _emb is None or isinstance(_emb, CachingEmbeddings):
                continue
            wrapped = seen.get(id(_emb))
            if wrapped is None:
                wrapped = CachingEmbeddings(_emb, sem=self._embed_sem)
                seen[id(_emb)] = wrapped
            metric.embeddings = wrapped
        # Fluency scorer: build the real (model-backed) one iff ``enable_fluency``.
        # The model itself loads lazily on the first score() call, so construction
        # is cheap. Tests inject a fake via the ``fluency_scorer`` DI seam; when
        # fluency is disabled, no scorer is built and the signal is skipped.
        
        
        if fluency_scorer is not None:
            self._fluency: FluencyScorer | None = fluency_scorer
        elif cfg.enable_fluency:
            self._fluency = (
                FluencyScorer(cfg, classifier=fluency_classifier)
                if fluency_classifier is not None
                else make_fluency_scorer(cfg)
            )
        else:
            self._fluency = None

    @property
    def base_embeddings(self) -> Any | None:
        "The unwrapped base embeddings object behind this scorer's cache.\n\n\n        Read-only accessor for cross-scorer weight sharing (grpo_train.py passes it\n        as ``base_embeddings=`` to the judge-anchor scorer). Returns the base\n        even though the metrics hold :class:`CachingEmbeddings` wrappers."
        emb = getattr(self._semsim, "embeddings", None)
        if isinstance(emb, CachingEmbeddings):
            return emb._base
        return emb

    @property
    def fluency_classifier(self) -> FluencyClassifier | None:
        "This scorer's fluency classifier (``None`` until lazily loaded).\n\n\n        After the first NL score the classifier is loaded; grpo_train.py hands the\n        instance to the judge-anchor scorer so the ~0.7 GB model is not\n        duplicated. ``None`` also when fluency is disabled."
        if self._fluency is None:
            return None
        return self._fluency.classifier

    async def _ascore(
        self, metric: Any, sem: asyncio.Semaphore | None = None, **kwargs: Any,
    ) -> float | None:
        'Run one RAGAS metric with retries, returning ``None`` on failure.\n\n\n        ``sem`` is an optional :class:`asyncio.Semaphore` that limits\n        concurrent calls to this metric (and hence to its underlying HTTP\n        endpoint). The semaphore is acquired **inside** the retry loop and\n        released before the backoff sleep, so waiting coroutines are never\n        blocked by a sleeping one. When ``None`` (e.g., direct test calls),\n        no limiting is applied.\n``sem``\n\n        A failed judge/embedding call on one example must not abort the whole\n        batch: it is retried up to ``cfg.judge_max_retries`` extra times (with\n        exponential backoff) for transient errors (HTTP 429/5xx, proxy 400,\n        network), then logged and mapped to ``None``; :func:`safe_score` turns\n        ``None`` into a conservative ``0.0`` contribution'
        guard = nullcontext() if sem is None else sem
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                async with guard:
                    result = await metric.ascore(**kwargs)
                return getattr(result, "value", result)
            except Exception as e:  # noqa: BLE001 - judge must stay resilient
                last_exc = e
                if attempt < self._max_retries:
                    await asyncio.sleep(2 ** attempt)
        logger.warning(
            "RAGAS metric %s failed after %d attempt(s): %s",
            type(metric).__name__,
            self._max_retries + 1,
            last_exc,
        )
        return None

    async def score(
        self,
        *,
        user_input: str,
        response: str,
        reference: str,
        task_class: str,
        extra: dict[str, Any] | None = None,
        prompt: list[dict[str, Any]] | None = None,
    ) -> ScoredMetrics:
        "Score one example: SemSim / NEREL F1 / docsem check as the workhorse.\n\nNEREL F1\n\n        The workhorse metric is chosen **by content**: the deterministic NEREL\n        F1 for the NEREL extraction sub-configs and the deterministic\n        executable DocSem check for ``docsem_quant`` (SemSim is skipped — not\n        computed at all for either), Semantic Similarity for everything else.\n        The active-\n        metric set is then extended **by content**: AC when a judge ran;\n        judge **rubrics** for SGR reasoning turns when\n        ``cfg.enable_judge_rubrics`` is on (two reference-free criteria —\n        Russian-language literacy and consistency with the input context —\n        from the same judge, combined into ``rubric_score`` by the strict\n        harmonic mean; ``prompt`` carries the original message list the\n        context is rebuilt from); ``format_check`` for tool-call turns;\n        ``code_check`` for the ``coder`` class (iff it emits a Python block);\n        ``fluency`` for the NL allowlist (incl. ``coder`` prose and the nerel\n        definition/normalization sub-configs, gated by\n        ``cfg.fluency_subtasks``), iff a scoreable sentence remains after\n        stripping non-prose. Active metrics are combined via the harmonic\n        mean.\n\n        ``failed_metrics`` records which metrics were **expected** but failed\n        due to a judge/embedding/fluency error (used by run_eval.py's failure\n        counter). Non-applicable metrics (AC without a judge; rubrics when the\n        flag is off or the turn is not a reasoning turn; format_score for\n        non-tool classes; fluency for a short answer; code_check for non-Python\n        coder answers) do NOT count as failures.\n        ``failed_metrics``"
        failed: list[str] = []
        _extra = extra or {}
        _is_final_answer = bool(_extra.get("is_final_answer"))
        # SGR reasoning turns are scored by fluency + an SGR-schema structural
        # check instead of SemSim (reasoning is open-ended — see
        # ``enable_reasoning_semsim``). Opt into SemSim with that flag.
        
        
        _is_reasoning = bool(_extra.get("is_reasoning_turn")) and not self._cfg.enable_reasoning_semsim
        # docsem_quant: the deterministic executable PoT check REPLACES SemSim
        # (the NEREL-F1 pattern); the CONTENT metrics (AC) score the extracted
        # effective answer against the GOLD answer (both live in extra — the
        # JSON envelope itself is scored by docsem_check on the raw response).
        # docsem_quant
        
        
        _is_docsem = docsem_check_applies(task_class)
        # meno_rerank: the deterministic JSON-verdict grading REPLACES SemSim
        # (cosine over a 15-character verdict is meaningless) AND AC is
        # structurally skipped for the class — a judge over a JSON verdict
        # would be pure noise (counted in ac_skipped, like reasoning turns).
        # / meno_rerank
        
        _is_rerank = rerank_em_applies(task_class)

        # For SGR final-answer turns the model's answer is buried inside a
        # finalanswertool ``<tool_call>``; extract it once so the CONTENT metrics
        # (SemSim / AC / fluency / code_check) score the answer rather than the
        # wrapping JSON. ``format_check`` still uses the RAW response below
        # (it validates the tool-call structure, which lives in the raw text).
        
        # format_check
        if _is_docsem:
            extracted = docsem_answer_text(response)
            content_response = extracted if extracted is not None else ""
        elif _is_final_answer:
            extracted = extract_final_answer(response)
            content_response = extracted if extracted is not None else ""
        else:
            content_response = response
        content_reference = (
            str(_extra.get("gold_answer") or "") if _is_docsem else reference
        )

        # --- Workhorse metric: exactly one of SemSim / NEREL F1 / docsem
        #     check / rummlu letter EM / rerank verdict EM / reasoning
        #     structural check drives the reward (chosen by content). ---
        
        semsim_val: float | None = None
        f1_val: float | None = None
        reasoning_val: float | None = None
        docsem_val: float | None = None
        em_val: float | None = None
        rerank_val: float | None = None
        if _is_reasoning:
            # Reasoning turn: SGR-schema structural anchor, no SemSim / F1.
            # F1.
            reasoning_val = reasoning_check(content_response)
        else:
            # NEREL deterministic F1 — for the extraction sub-configs it REPLACES
            # Semantic Similarity (SemSim over a newline-separated entity list is
            # meaningless). When active, SemSim is not computed at all.
            # NEREL F1
            if self._cfg.enable_nerel_f1:
                f1_val = nerel_f1(content_response, content_reference, task_class, _extra)
            # DocSem deterministic check — for docsem_quant it REPLACES SemSim
            # (SemSim over a JSON envelope would reward lexical overlap of
            # program text, not numeric correctness). When active, SemSim is
            # not computed at all — no embedding forward pass for the class.
            
            if self._cfg.enable_docsem_check and _is_docsem:
                try:
                    docsem_val = docsem_check(
                        response,
                        _extra.get("gold_answer"),
                        _extra.get("gold_evidence"),
                    )
                except Exception as e:  # noqa: BLE001 - record + continue
                    logger.warning(
                        "docsem_check failed for task_class=%s: %s", task_class, e
                    )
                    failed.append("docsem_check")
                    docsem_val = 0.0
            # ruMMLU-pro deterministic letter EM — the class REPLACES SemSim
            # (cosine over a single option letter barely separates the correct
            # choice from the wrong ones). When active, SemSim is skipped.
            # ruMMLU-pro
            
            if self._cfg.enable_rummlu_em:
                em_val = rummlu_em(content_response, content_reference, task_class)
            # Meno-stand rerank deterministic verdict grading — the class
            # REPLACES SemSim (0.3 valid JSON verdict + 0.7 graded by the
            # ORDINAL label distance: both the integer and the string label
            # spellings normalize to ints; gold lives in extra — the loader
            # pre-parses the heterogeneous reference strings).
            
            
            
            if self._cfg.enable_rerank_em and _is_rerank:
                rerank_val = rerank_em(content_response, task_class, _extra)
            if (
                f1_val is None
                and docsem_val is None
                and em_val is None
                and rerank_val is None
            ):
                # Semantic Similarity — the workhorse otherwise. No outer
                # semaphore: its embedding calls are gated directly by _embed_sem
                # inside the shared CachingEmbeddings.
                # SemSim
                semsim = await self._ascore(
                    self._semsim, None,
                    response=content_response,
                    reference=content_reference,
                )
                if semsim is None:
                    failed.append("semantic_similarity")
                semsim_val = safe_score(semsim)

        # 2. Answer Correctness — only if a judge endpoint is configured AND this
        #    is not a reasoning turn (reasoning is open-ended → AC would be noisy;
        #    the structural check + fluency handle reasoning turns) AND not a
        #    Meno-stand rerank row (a judge over a JSON verdict would be pure
        #    noise — the deterministic rerank_em IS the class's content signal;
        #    the skip lands in ac_skipped, mirroring reasoning turns).
        # _judge_sem gates the LLM call. AC
        
        ac_val: float | None = None
        if self._ac is not None and not _is_reasoning and not _is_rerank:
            ac = await self._ascore(
                self._ac, self._judge_sem,
                user_input=user_input,
                response=content_response,
                reference=content_reference,
            )
            if ac is None:
                failed.append("answer_correctness")
            else:
                ac_val = safe_score(ac)

        # 1b. Judge rubrics — SGR reasoning turns (AC is structurally skipped
        #     there): two reference-free criteria from the SAME judge —
        #     Russian-language literacy and consistency with the input context
        #     — combined via the strict harmonic mean of the normalized 1–5
        #     Likert scores. Gated on ``cfg.enable_judge_rubrics`` (grpo_train.py
        #     anchors / run_eval.py ``--judge-rubrics``); the GRPO in-loop
        #     scorer never enables it (judge=None there).
        
        
        
        rubric_val: float | None = None
        if self._rubrics is not None and _is_reasoning:
            rubric_context = (
                build_rubric_context(prompt) if prompt else user_input
            )
            lit = await self._ascore(
                self._rubrics[0], self._judge_sem,
                user_input=rubric_context, response=content_response,
            )
            ctx = await self._ascore(
                self._rubrics[1], self._judge_sem,
                user_input=rubric_context, response=content_response,
            )
            if lit is None or ctx is None:
                failed.append("rubrics")
            else:
                rubric_val = harmonic_mean(
                    [_normalize_rubric(lit), _normalize_rubric(ctx)]
                )

        # 3. Format check — only for tool-CALL turns. The loaders tag turns with
        #    ``is_call_turn`` (False for SGR reasoning turns and for
        #    tool_calls_multiturn final text answers); format_check is skipped
        #    then. Runs on the RAW response (the ``<tool_call>`` lives there).
        
        _is_call_turn = _extra.get("is_call_turn")
        _is_final_turn = _extra.get("is_final_turn")
        if _is_call_turn is False or _is_final_turn is True:
            fmt = None  # text-only turn — format_check not applicable
        else:
            fmt = format_check(response, task_class, self._known_tool_names)

        # 3b. Tool-call schema_check — the argument-level sibling of
        #     format_check (required fields 0.5 + declared types 0.5 against
        #     the tool's parameter JSON schema from the registry). Gated by
        #     ``cfg.enable_schema_check`` (default OFF: the anchors never set
        #     it — the frozen methodology; grpo_train enables it in the
        #     IN-LOOP reward only) and by the same turn-kind gating as
        #     format_check; runs on the RAW response, skips tools without a
        #     registered schema (unknown names are format_check's concern).
        # schema_check
        # format_check
        if (
            self._cfg.enable_schema_check
            and not (_is_call_turn is False or _is_final_turn is True)
        ):
            schema_val = schema_check(
                response, task_class, self._known_tool_schemas
            )
        else:
            schema_val = None

        # 3c. Format epsilon floor — an IN-LOOP-ONLY shaping knob (the anchor
        #     / run_eval configs keep it at 0.0 = strict): floor the format
        #     component so a broken envelope no longer zeroes the WHOLE
        #     example (the r2 zero-variance lesson). The floored value is
        #     what enters the harmonic combine and what gets reported — the
        #     same in-place-transform precedent as the fluency guard.
        
        
        if fmt is not None and self._cfg.format_epsilon_floor > 0.0:
            fmt = max(fmt, self._cfg.format_epsilon_floor)

        # 4. Code check — only for the ``coder`` class (Python ``ast``). Runs on
        #    the (possibly extracted) content_response: for coder's SGR final turn
        #    that is the finalanswertool ``answer`` field, so a ``python`` block
        #    inside the answer is detected. ``None`` for non-Python / non-coder.
        # code_check
        if self._cfg.enable_code_check:
            cc = code_check(content_response, task_class)
        else:
            cc = None

        # 5. Fluency — for the NL allowlist + ``coder`` prose (model-backed,
        #    in-process; gated by ``cfg.enable_fluency``). ``extra`` threads the
        #    sub-config allowlist through; content_response is the text scored
        #    (the extracted answer for final turns). A model failure → None +
        #    ``failed_metrics``; an answer with no scoreable sentence → None (N/A).
        # fluency
        fluency_val: float | None = None
        if self._fluency is not None:
            try:
                fluency_val = await self._fluency.score(
                    content_response, task_class, extra=_extra,
                )
            except Exception as e:  # noqa: BLE001 - record + continue
                logger.warning(
                    "Fluency scorer failed for task_class=%s: %s", task_class, e
                )
                failed.append("fluency")

        # 5b. Fluency guard — optional degradation brake (OFF by default).
        #     With the grammar classifier the raw P(grammatical) is well-spread
        #     and used directly as an improvement signal (the RuCoLa saturation
        #     that motivated the brake is gone), so
        #     ``fluency_guard_threshold`` defaults to None. When a value is set,
        #     ``min(1, val / threshold)`` re-enables the brake for ablations.
        
        
        if (
            fluency_val is not None
            and self._cfg.fluency_guard_threshold is not None
        ):
            fluency_val = min(1.0, fluency_val / self._cfg.fluency_guard_threshold)

        # 5c. Fluency language guard — reward-hacking defence (IN-LOOP only;
        #     default OFF). The grammar classifier is English-trained and may
        #     score an English answer slightly higher than an equally good
        #     Russian one, so the policy could drift to answering every
        #     bilingual task in English. When enabled, the fluency score is
        #     multiplied by the confidence of the example's target language
        #     (``extra["target_language"]``, computed at build time) over the
        #     CODE-STRIPPED prose (the same prose fluency scores) — a confident
        #     wrong-language answer drives the factor to ~0 and the strict
        #     harmonic mean zeroes the example. Language detection is a no-op
        #     (factor 1.0) on terse / non-lingual text or when the target
        #     language is unknown. The anchors never enable it (frozen
        #     methodology); grpo_train.py enables it in the in-loop reward only.
        
        
        if (
            fluency_val is not None
            and self._cfg.enable_language_guard
            and _extra.get("target_language")
        ):
            _, prose = split_prose_and_code(content_response)
            _lang = _extra.get("target_language")
            fluency_val = fluency_val * language_match_factor(prose, _lang)

        # Combine the active metrics via the harmonic mean. The workhorse is
        # exactly one of SemSim / NEREL F1 / docsem check / reasoning_check
        # (by content); AC / format / code_check / fluency join only when they
        # produced a value.
        
        # SemSim/F1/docsem/reasoning.
        active: list[float] = []
        if semsim_val is not None:
            active.append(semsim_val)
        if f1_val is not None:
            active.append(f1_val)
        if docsem_val is not None:
            active.append(docsem_val)
        if reasoning_val is not None:
            active.append(reasoning_val)
        if em_val is not None:
            active.append(em_val)
        if rerank_val is not None:
            active.append(rerank_val)
        if ac_val is not None:
            active.append(ac_val)
        if rubric_val is not None:
            active.append(rubric_val)
        if fmt is not None:
            active.append(fmt)
        if schema_val is not None:
            active.append(schema_val)
        if cc is not None:
            active.append(cc)
        if fluency_val is not None:
            active.append(fluency_val)
        combined = harmonic_mean(active)

        return ScoredMetrics(
            semantic_similarity=semsim_val,
            answer_correctness=ac_val,
            format_score=fmt,
            fluency=fluency_val,
            code_check=cc,
            nerel_f1=f1_val,
            docsem_score=docsem_val,
            rubric_score=rubric_val,
            rummlu_em=em_val,
            schema_check=schema_val,
            rerank_em=rerank_val,
            combined=combined,
            failed_metrics=tuple(failed),
        )


def make_ragas_scorer(
    cfg: RewardConfig,
    *,
    base_embeddings: Any | None = None,
    fluency_classifier: FluencyClassifier | None = None,
) -> RAGASScorer:
    'Build the real RAGAS-backed scorer (lazy ragas import happens here).\n\n\n    Returns an object satisfying the :class:`~llm_training.reward.config.Scorer`\n\nArgs\n        base_embeddings: optional pre-built base embeddings object to reuse\n(shared weights\n        fluency_classifier: optional loaded fluency classifier to share\n            (see :class:`RAGASScorer`'
    return RAGASScorer(
        cfg,
        base_embeddings=base_embeddings,
        fluency_classifier=fluency_classifier,
    )
