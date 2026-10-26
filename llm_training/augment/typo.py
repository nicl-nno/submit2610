'Keyboard-typo augmentation of user-typed prompt spans (augmentex wrapper).\n\n\n``grpo_train.py`` calls :func:`augment_pool` on the TRAIN split right after\n:func:`llm_training.build.build_unified_dataset` and before\n:func:`build_training_rows` — at that point prompts are still message lists,\nso the per-class typed/external boundaries (see :func:`typed_spans`) are\nexact; a rendered TEXT prompt could not be re-parsed reliably (AGENTS.md).\nThe ``question`` column is then derived from the augmented messages, so the\njudge-side metadata stays consistent with what the policy actually saw.\n\n``grpo_train.py``\n:func:`llm_training.build.build_unified_dataset`\n:func:`build_training_rows`\n\nBoundary table (typed = typo-able; everything else is untouchable context):\n\n===========================  =====================================\nTask class                   Typed spans\nTask class\n===========================  =====================================\nSGR 9 classes                first/subsequent ``user`` messages\n                             minus the ``Current Date:`` preamble\n                             (the whole task line, including inline\n                             quoted payload — no reliable delimiter)\n``menon_qa``                 ``CURRENT QUESTION`` + ``INSTRUCTION``\n                             section contents (documents untouchable)\n``mtrageval``                user text minus the attached context\n                             block (``extra["context_text"]``)\n``multihop_rag``             text before ``The chunk list:`` marker\n``docsem_quant``             text after the last ``Question:`` marker\n                             (the document blocks are untouchable\n                             context; the question is English → PC layout)\n``self_cognition``           all user messages\n``rummlu_pro``               all user messages; answer-option\n``B)`` …) protected\n``tool_calls_multiturn``     ``user`` messages (tool results arrive\n``opencode_gen``             as ``role:"tool"`` and are skipped)\n===========================  =====================================\n\nClasses without a known boundary (``asr_summarization`` — the ASR transcript\nis the payload; ``ragu_*`` — template + document; ``nerel_instruct``) are\n**not augmentable**: :func:`typed_spans` returns no spans for them and\n:func:`resolve_classes` rejects them.\n\n\naugmentex support matrix (v1.3.1): PC keyboard — Russian and English;\nmobile keyboard — **Russian only**. English spans therefore always fall\nback to the PC layout regardless of the platform draw.'

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Sequence

from llm_training.loaders.sgr_agent import CLASSES as _SGR_CLASSES
from llm_training.reward.inputs import (
    _DOCSEM_QUESTION_MARKER,
    _MENONQA_DELIM_RE,
    _MENONQA_INSTRUCTION_HEADER_RE,
    _MENONQA_QUESTION_HEADER_RE,
    _MULTIHOP_CHUNK_MARKER,
)
from llm_training.schema import UnifiedExample

if TYPE_CHECKING:  # pragma: no cover
    from llm_training.schema import Message

#: Classes with a KNOWN typed/external boundary — the default augmentation
#: allowlist AND the set :func:`typed_spans` knows how to split. Everything
#: else (``asr_summarization``, ``ragu_*``, ``nerel_instruct``) has no safe
#: boundary and is never augmented.


DEFAULT_TYPO_CLASSES: frozenset[str] = frozenset(
    {*_SGR_CLASSES,
     "menon_qa",
     "mtrageval",
     "multihop_rag",
     "docsem_quant",
     "rummlu_pro",
     "self_cognition",
     "tool_calls_multiturn",
     "opencode_gen"}
)

#: Classes whose SYSTEM prompt is a plain instruction safe to corrupt with
#: light typos (the instruction-noise-robustness augmentation). Excluded on
#: purpose: ``tool_calls_multiturn``/``opencode_gen`` (the system carries the
#: ``<tools>`` block), ``docsem_quant`` (the JSON output format is spelled out
#: in the system), ``mtrageval``/``ragu_*``/``asr_summarization`` (the system
#: may carry external context), the Meno-stand classes (the system carries the
#: task-kind markers that define what to do). ``nerel_instruct`` is included:
#: its system is the task instruction, and the reward dispatches by
#: ``extra["config"]`` (never by parsing the system text).



SYSTEM_TYPO_CLASSES: frozenset[str] = frozenset(
    {*_SGR_CLASSES, "menon_qa", "rummlu_pro", "self_cognition", "nerel_instruct"}
)

#: SGR framework preamble injected as an extra user turn — machine-generated,
#: never typed by the user.

_SGR_DATE_PREAMBLE_PREFIX = "Current Date"

# ``B)`` …)
#: hitting the option LETTER would make the task unsolvable; every attempt
#: that changes any marker is reverted (see :func:`apply_typos`).


_OPTION_MARKER_RE = re.compile(r"(?m)^[ \t]*[A-Za-zА-Яа-яЁё][.)][ \t\n]")

#: Structured regions of a SYSTEM prompt that a typo must never touch: the
#: ``<tools>…</tools>`` block (tool_calls_multiturn keeps the schema there) and
#: fenced code/JSON blocks (e.g. a spelled-out output format).


_SYSTEM_TOOLS_RE = re.compile(r"<tools>.*?</tools>", re.DOTALL)
_SYSTEM_FENCE_RE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)

#: One typed span: (message index, start offset, end offset) inside
#: ``UnifiedExample.prompt``.

Span = tuple[int, int, int]

#: augmentex CharAug instances keyed by (lang, platform, unit_prob, min_aug,
#: max_aug) — built lazily, reused across calls (the stat tables load once).

_AUG_CACHE: dict[tuple[Any, ...], Any] = {}

_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class TypoConfig:
    'Configuration of the offline keyboard-typo augmentation.\n\n\nAttributes\n        share: exact fraction of eligible train prompts that end up carrying\n            ≥1 visible typo (in-place replacement; the clean share is exactly\n            ``1 - share``). ``0.0`` disables the augmentation entirely.\nshare``). ``0.0``\n        seed: RNG seed; every random choice derives from\n            ``f"{seed}:{task_class}:{source}#{occurrence}"``.\n        unit_prob: augmentex ``unit_prob`` (fraction of candidate positions\n            before the min/max clip; with ``min_aug=1, max_aug=2`` the clip\n            dominates, keeping the corruption light).\n        min_aug / max_aug: hard bounds on character substitutions per typed\n            span (the user-facing guarantee: one-two chars per phrase).\n        platforms: platform draw for Russian spans (``("pc", "mobile")``);\n            English always falls back to ``"pc"`` (augmentex has no\n            mobile+eng statistics).\n        classes: the augmentation allowlist (subset of\n            :data:`DEFAULT_TYPO_CLASSES` — classes without a known boundary\n            are rejected).\n        max_attempts: retry-until-changed budget per span (deterministic\n            reseeds ``...:a{attempt}``); a span unchanged after all attempts\n            leaves the example clean.\n        copies: number of ADDITIONAL augmented copies of the pool appended\n            after the clean original (multi-variant scheme: the output holds\n            ``(copies + 1) * N`` examples — the original stays clean, each\n            copy independently corrupts ``share`` of its eligible prompts).\n            ``0`` keeps the legacy in-place behaviour (``share`` of the pool\n            corrupted, size unchanged).'

    share: float = 0.0
    seed: int = 42
    unit_prob: float = 0.02
    min_aug: int = 1
    max_aug: int = 2
    platforms: tuple[str, ...] = ("pc", "mobile")
    classes: frozenset[str] = DEFAULT_TYPO_CLASSES
    max_attempts: int = 3
    copies: int = 0
    #: The instruction-noise-robustness dimension (a SEPARATE share/allowlist
    #: from ``share``/``classes``): the fraction of eligible SYSTEM prompts
    #: carrying ≥1 typo. ``0.0`` = off (default).
    
    
    system_share: float = 0.0
    #: The system-augmentation allowlist (subset of :data:`SYSTEM_TYPO_CLASSES`).
    
    system_classes: frozenset[str] = SYSTEM_TYPO_CLASSES

    def __post_init__(self) -> None:
        if not 0.0 <= self.share <= 1.0:
            raise ValueError(
                f'share must be in [0, 1], got {self.share}'
            )
        if self.min_aug < 1:
            raise ValueError(
                f'min_aug must be >= 1, got {self.min_aug}'
            )
        if self.max_aug < self.min_aug:
            raise ValueError(
                f'max_aug ({self.max_aug}) must be >= min_aug ({self.min_aug}'
            )
        if self.max_attempts < 1:
            raise ValueError(
                f'max_attempts must be >= 1, got {self.max_attempts}'
            )
        if self.copies < 0:
            raise ValueError(
                f'copies must be >= 0, got {self.copies}'
            )
        bad_platforms = set(self.platforms) - {"pc", "mobile"}
        if bad_platforms or not self.platforms:
            raise ValueError(
                f"platforms must be a non-empty subset of {{'pc', 'mobile'}}, got {self.platforms}"
            )
        unknown = set(self.classes) - DEFAULT_TYPO_CLASSES
        if unknown:
            raise ValueError(
                f'classes contain task classes with no known typed/external boundary: {sorted(unknown)}; allowed: {sorted(DEFAULT_TYPO_CLASSES)}'
            )
        if not 0.0 <= self.system_share <= 1.0:
            raise ValueError(
                f'system_share must be in [0, 1], got {self.system_share}'
            )
        unknown_system = set(self.system_classes) - SYSTEM_TYPO_CLASSES
        if unknown_system:
            raise ValueError(
                f'system_classes contain classes whose system prompt is not a safe plain instruction: {sorted(unknown_system)}; allowed: {sorted(SYSTEM_TYPO_CLASSES)}'
            )


def resolve_classes(
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> frozenset[str]:
    'Resolve the CLI allowlist: explicit include set minus exclusions.\n\n\n    With no ``include`` the default allowlist is used; ``exclude`` removes\n    from either. Unknown include entries raise (a typo in a class name must\n    not silently narrow the augmentation); excluding a non-member is a no-op.'
    base = frozenset(include) if include else DEFAULT_TYPO_CLASSES
    unknown = base - DEFAULT_TYPO_CLASSES
    if unknown:
        raise ValueError(
            f'unknown --typo-class entries (no known typed/external boundary or not a task class): {sorted(unknown)}; allowed: {sorted(DEFAULT_TYPO_CLASSES)}'
        )
    return base - frozenset(exclude or ())


def resolve_system_classes(
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> frozenset[str]:
    'Resolve the CLI system-augmentation allowlist (include minus exclude).\n\n\n    Mirrors :func:`resolve_classes` over :data:`SYSTEM_TYPO_CLASSES`.'
    base = frozenset(include) if include else SYSTEM_TYPO_CLASSES
    unknown = base - SYSTEM_TYPO_CLASSES
    if unknown:
        raise ValueError(
            f'unknown --system-typo-class entries (system prompt not a safe plain instruction, or not a task class): {sorted(unknown)}; allowed: {sorted(SYSTEM_TYPO_CLASSES)}'
        )
    return base - frozenset(exclude or ())


def _menonqa_typed_ranges(text: str) -> list[tuple[int, int]]:
    "Character ranges of the QUESTION/INSTRUCTION section contents.\n\n\n    Mirrors the parsing of\n    :func:`llm_training.reward.inputs._menon_qa_question`: sections are\n    delimited by runs of ``=``; the question lives under ``CURRENT\n    QUESTION``, the optional instruction under ``INSTRUCTION``. All other\n    sections (documents, abbreviation dictionary, dialogue history) are\n    external context and stay out of the returned ranges. Without the\n    structured delimiters the whole text is typed (the extractor's\n    fallback, e.g. chitchat records)."
    matches = list(_MENONQA_DELIM_RE.finditer(text))
    if len(matches) < 2:
        return [(0, len(text))]
    ranges: list[tuple[int, int]] = []
    i = 0
    while i + 1 < len(matches):
        header = text[matches[i].end():matches[i + 1].start()]
        start = matches[i + 1].end()
        end = matches[i + 2].start() if i + 2 < len(matches) else len(text)
        header_is_typed = bool(
            _MENONQA_QUESTION_HEADER_RE.match(header.strip())
            or _MENONQA_INSTRUCTION_HEADER_RE.match(header.strip())
        )
        if header_is_typed and end > start:
            # Trim the surrounding whitespace so the span is exactly the
            # section content (the delimiters and blank lines stay intact).
            
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if end > start:
                ranges.append((start, end))
        i += 2
    return ranges


def _last_user_idx(prompt: list["Message"]) -> int:
    """Index of the last ``user`` message (``-1`` if none)."""
    for i in range(len(prompt) - 1, -1, -1):
        if prompt[i].get("role") == "user":
            return i
    return -1


def _mtrageval_spans(example: UnifiedExample) -> list[Span]:
    'User text minus the attached context block (both loader placements).\n\n\n    The loader prepends ``context_text + "\\n\\n"`` to the final user message\n    (context-then-question); the reward extractor also tolerates the\n    appended form. Both are handled: only the question part is typed.\n    When the context sits in ``system`` (or is absent) the user message is\n    fully typed.'
    ctx = (example.extra or {}).get("context_text") or ""
    last = _last_user_idx(example.prompt)
    spans: list[Span] = []
    for i, msg in enumerate(example.prompt):
        if msg.get("role") != "user":
            continue
        content = msg.get("content") or ""
        if not content:
            continue
        if i == last and ctx:
            if content.startswith(ctx + "\n\n"):
                spans.append((i, len(ctx) + 2, len(content)))
                continue
            if content.endswith("\n\n" + ctx):
                spans.append((i, 0, len(content) - len(ctx) - 2))
                continue
        spans.append((i, 0, len(content)))
    return spans


def typed_spans(example: UnifiedExample) -> list[Span]:
    'Find the user-typed spans of a prompt (the typo-able regions).\n\n\n    Pure function over the message list; returns ``[]`` for classes without\n    a known typed/external boundary and for prompts without user turns.\n    Never raises, never mutates the example.'
    task_class = example.task_class
    if task_class not in DEFAULT_TYPO_CLASSES:
        return []
    prompt = example.prompt or []
    last = _last_user_idx(prompt)
    if last < 0:
        return []
    if task_class == "menon_qa":
        content = prompt[last].get("content") or ""
        return [(last, s, e) for s, e in _menonqa_typed_ranges(content)]
    if task_class == "mtrageval":
        return _mtrageval_spans(example)
    if task_class == "multihop_rag":
        # Only messages with the chunk marker have an exact boundary; a
        # marker-less user message is treated as unparseable context and
        # skipped (conservative; the reward extractor's whole-text fallback
        # is a question-extraction concern, not a corruption-safety one).
        
        
        spans: list[Span] = []
        for i, msg in enumerate(prompt):
            if msg.get("role") != "user":
                continue
            content = msg.get("content") or ""
            idx = content.find(_MULTIHOP_CHUNK_MARKER)
            if idx > 0:
                spans.append((i, 0, idx))
        return spans
    if task_class == "docsem_quant":
        # Only the question line is typed: the user message is
        # ``Document blocks:\n{…}\n\nQuestion: {query}`` and the blocks are
        # external context. The span starts after the LAST ``\nQuestion:``
        # marker (a block body may itself contain the word "Question:");
        # a marker-less message is skipped (conservative, same as multihop).
        
        
        spans = []
        for i, msg in enumerate(prompt):
            if msg.get("role") != "user":
                continue
            content = msg.get("content") or ""
            idx = content.rfind(_DOCSEM_QUESTION_MARKER)
            if idx >= 0:
                start = idx + len(_DOCSEM_QUESTION_MARKER)
                # Trim the leading whitespace: the span is exactly the
                # question text (a typo in whitespace is invisible anyway).
                
                while start < len(content) and content[start].isspace():
                    start += 1
                if len(content) > start:
                    spans.append((i, start, len(content)))
        return spans
    # Generic branch: SGR 9 + self_cognition + rummlu_pro +
    # tool_calls_multiturn + opencode_gen — every genuine user message is
    # typed, minus the machine-generated SGR date preamble and any folded
    # tool results (defensive: tool results normally arrive as role:"tool").
    
    
    spans = []
    for i, msg in enumerate(prompt):
        if msg.get("role") != "user":
            continue
        content = msg.get("content") or ""
        if not content:
            continue
        if content.lstrip().startswith(_SGR_DATE_PREAMBLE_PREFIX):
            continue
        if "<tool_response>" in content:
            continue
        spans.append((i, 0, len(content)))
    return spans


def _system_typed_ranges(content: str) -> list[tuple[int, int]]:
    'Character ranges of a system prompt safe to corrupt with typos.\n\n\n    Everything EXCEPT the structured regions (:data:`_SYSTEM_TOOLS_RE` — the\n    ``<tools>…</tools>`` block — and :data:`_SYSTEM_FENCE_RE` — fenced\n    code/JSON blocks) is typed. A system prompt without protected regions\n    yields one whole-content range.'
    protected: list[tuple[int, int]] = []
    for m in _SYSTEM_TOOLS_RE.finditer(content):
        protected.append((m.start(), m.end()))
    for m in _SYSTEM_FENCE_RE.finditer(content):
        protected.append((m.start(), m.end()))
    protected.sort()
    ranges: list[tuple[int, int]] = []
    pos = 0
    for s, e in protected:
        if s > pos:
            ranges.append((pos, s))
        pos = max(pos, e)
    if pos < len(content):
        ranges.append((pos, len(content)))
    return ranges


def system_spans(example: UnifiedExample) -> list[Span]:
    'Find the typo-able spans of the FIRST system message (instruction noise).\n\n\n    Returns ``[]`` for classes outside :data:`SYSTEM_TYPO_CLASSES`, for\n    prompts without a system message, and for an empty system. Never raises,\n    never mutates the example.'
    if example.task_class not in SYSTEM_TYPO_CLASSES:
        return []
    prompt = example.prompt or []
    for i, msg in enumerate(prompt):
        if msg.get("role") == "system":
            content = msg.get("content") or ""
            if not content.strip():
                return []
            return [(i, s, e) for s, e in _system_typed_ranges(content)]
    return []


def _seed_int(tag: str) -> int:
    """Deterministic string → int seed (blake2b, process-salt-independent)."""
    digest = hashlib.blake2b(tag.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31)


def _levenshtein(a: str, b: str) -> int:
    """Levenshtein distance (C lib when available, pure-Python fallback)."""
    try:
        import Levenshtein  # noqa: PLC0415 - present with augmentex

        return int(Levenshtein.distance(a, b))
    except ImportError:  # pragma: no cover - envs without python-Levenshtein
        if len(a) < len(b):
            a, b = b, a
        previous = list(range(len(b) + 1))
        for i, char_a in enumerate(a, 1):
            current = [i]
            for j, char_b in enumerate(b, 1):
                current.append(
                    min(
                        previous[j] + 1,
                        current[j - 1] + 1,
                        previous[j - 1] + (char_a != char_b),
                    )
                )
            previous = current
        return previous[-1]


def _detect_lang(spans_text: str) -> str | None:
    """Dominant script of the typed text: ``"rus"`` / ``"eng"`` / ``None``."""
    cyrillic = len(_CYRILLIC_RE.findall(spans_text))
    latin = len(_LATIN_RE.findall(spans_text))
    if cyrillic == 0 and latin == 0:
        return None
    return "rus" if cyrillic >= latin else "eng"


def _augmentex_typo(
    text: str, lang: str, platform: str, seed_int: int, cfg: TypoConfig
) -> str:
    'One augmentex ``typo`` call with the global RNG fully controlled.\n\n\n    augmentex seeds the GLOBAL ``random`` / ``numpy.random`` in its\n    constructor and draws from them on every call — interleaving with other\n    consumers would break reproducibility. We therefore save both states,\n    reseed deterministically, call, and restore; instances are cached per\n    parameter tuple so the KartaSlov stat tables load only once.'
    import numpy as np
    from augmentex import CharAug

    key = (lang, platform, cfg.unit_prob, cfg.min_aug, cfg.max_aug)
    aug = _AUG_CACHE.get(key)
    if aug is None:
        aug = CharAug(
            unit_prob=cfg.unit_prob,
            min_aug=cfg.min_aug,
            max_aug=cfg.max_aug,
            lang=lang,
            platform=platform,
            random_seed=None,
        )
        _AUG_CACHE[key] = aug
    np_state = np.random.get_state()
    random_state = random.getstate()
    np.random.seed(seed_int)
    random.seed(seed_int)
    try:
        return str(aug.augment(text=text, action="typo"))
    finally:
        np.random.set_state(np_state)
        random.setstate(random_state)


def _markers_intact(orig: str, cand: str, task_class: str) -> bool:
    """Guard: protected markers (rummlu answer options) unchanged by cand."""
    if task_class != "rummlu_pro":
        return True
    return _OPTION_MARKER_RE.findall(orig) == _OPTION_MARKER_RE.findall(cand)


def _augment_spans(
    example: UnifiedExample,
    cfg: TypoConfig,
    spans: list[Span],
    *,
    occurrence: int = 0,
    copy_index: int | None = None,
    system: bool = False,
) -> tuple[UnifiedExample, int]:
    'Augment the given spans; return (example, n_changes).\n\n\n    Retry semantics: every span gets up to ``cfg.max_attempts`` deterministic\n    reseeds until it visibly changes within the Levenshtein ≤ ``max_aug``\n    budget without touching protected markers; an exhausted span stays\n    original. An example with zero total changes is returned **unchanged**.'
    if not spans:
        return example, 0
    joined = "".join(
        (example.prompt[mi].get("content") or "")[s:e] for mi, s, e in spans
    )
    lang = _detect_lang(joined)
    if lang is None:
        return example, 0
    base = (
        f"{cfg.seed}{':system' if system else ''}"
        f":{example.task_class}:{example.source}#{occurrence}"
    )
    # Platform draw per example (one keyboard per user); English has no
    # mobile statistics in augmentex and always falls back to PC.
    
    platform = random.Random(base).choice(cfg.platforms) if lang == "rus" else "pc"
    new_prompt = [dict(m) for m in example.prompt]
    # Spans are offsets into the ORIGINAL contents; multiple spans may share
    # one message, so splices accumulate per message. Only same-length
    # candidates are accepted, so the original offsets stay valid.
    
    updated_contents: dict[int, str] = {}
    total_changes = 0
    for span_no, (mi, start, end) in enumerate(spans):
        original_content = example.prompt[mi].get("content") or ""
        fragment = original_content[start:end]
        if not fragment.strip():
            continue
        changed: str | None = None
        for attempt in range(cfg.max_attempts):
            candidate = _augmentex_typo(
                fragment,
                lang,
                platform,
                _seed_int(f"{base}:s{span_no}:a{attempt}"),
                cfg,
            )
            if candidate == fragment:
                continue  # no visible change → reseed and retry
            if len(candidate) != len(fragment):
                # A length change would invalidate the other spans' offsets.
                
                continue
            if _levenshtein(fragment, candidate) > cfg.max_aug:
                continue  # defensive: over budget → never keep
            if not _markers_intact(fragment, candidate, example.task_class):
                continue  # an answer-option marker was hit → retry
            changed = candidate
            break
        if changed is None:
            continue
        total_changes += _levenshtein(fragment, changed)
        spliced = updated_contents.get(mi, original_content)
        updated_contents[mi] = spliced[:start] + changed + spliced[end:]
        new_prompt[mi] = {
            **new_prompt[mi],
            "content": updated_contents[mi],
        }
    if total_changes == 0:
        return example, 0
    extra = dict(example.extra or {})
    extra["typo_augmented" if not system else "system_typo_augmented"] = True
    extra["typo_lang"] = lang
    extra["typo_platform"] = platform
    extra["typo_changes"] = total_changes
    if copy_index is not None:
        extra["typo_copy"] = copy_index
    return replace(example, prompt=new_prompt, extra=extra), total_changes


def apply_typos(
    example: UnifiedExample,
    cfg: TypoConfig,
    *,
    occurrence: int = 0,
    copy_index: int | None = None,
) -> tuple[UnifiedExample, int]:
    'Augment one example\'s typed (user) spans; return (example, n_changes).\n\n\n    ``copy_index`` (the multi-variant copy number, 0-based) is stored in\n    ``extra["typo_copy"]`` for observability; ``None`` (the single-pool path)\n    leaves the extra untouched.'
    if cfg.share <= 0.0 or example.task_class not in cfg.classes:
        return example, 0
    return _augment_spans(
        example, cfg, typed_spans(example),
        occurrence=occurrence, copy_index=copy_index, system=False,
    )


def apply_system_typos(
    example: UnifiedExample,
    cfg: TypoConfig,
    *,
    occurrence: int = 0,
) -> tuple[UnifiedExample, int]:
    "Augment one example's SYSTEM prompt (instruction noise); return (example, n).\n\n\n    Gated by ``cfg.system_share`` / ``cfg.system_classes`` — a dimension\n    orthogonal to :func:`apply_typos` (user-typed spans)."
    if cfg.system_share <= 0.0 or example.task_class not in cfg.system_classes:
        return example, 0
    return _augment_spans(
        example, cfg, system_spans(example),
        occurrence=occurrence, system=True,
    )


def augment_pool(
    examples: Sequence[UnifiedExample], cfg: TypoConfig
) -> tuple[list[UnifiedExample], dict[str, Any]]:
    'Apply typo augmentation to the pool (in-place or multi-variant).\n\n\n    Eligible = examples of an allowlisted class with at least one typed\n    span. Two modes, selected by :attr:`TypoConfig.copies`:\n\n    * **In-place** (``copies == 0``, the legacy behaviour): exactly\n      ``ceil(share * n_eligible)`` examples are selected with\n      ``random.Random(f"{seed}:typo")`` and replaced by their augmented\n      copies; the pool size is unchanged.\n    * **Multi-variant** (``copies > 0``): the output is the CLEAN original\n      pool followed by ``copies`` additional copies — ``(copies + 1) * N``\n      examples in total. Each copy ``k`` independently selects\n      ``ceil(share * n_eligible)`` of the (eligible) examples via\n      ``random.Random(f"{seed}:typo:c{k}")`` and corrupts them with\n      ``apply_typos(..., copy_index=k)``; non-eligible examples are\n      duplicated byte-identically (no typos are possible on them). Every\n      copy\'s corruption is seeded through the shared occurrence counter, so\n      distinct copies of the same ``source`` (final-turn boost included)\n      get independent typos.\n\n    Duplicates sharing a ``source`` are seeded by occurrence number; examples\n    that refuse to change after all retries stay original and are reported as\n    ``reverted_clean`` — the effective corrupted share can therefore be\n    marginally below ``share`` only for spans with no keyboard statistics.'
    stats: dict[str, Any] = {
        "enabled": cfg.share > 0.0,
        "share": cfg.share,
        "copies": cfg.copies,
        "eligible": 0,
        "selected": 0,
        "augmented": 0,
        "reverted_clean": 0,
        "per_class": {},
    }
    if cfg.share <= 0.0:
        return list(examples), stats
    eligible = [
        i
        for i, ex in enumerate(examples)
        if ex.task_class in cfg.classes and typed_spans(ex)
    ]
    stats["eligible"] = len(eligible)
    if not eligible and cfg.copies <= 0:
        return list(examples), stats

    if cfg.copies <= 0:
        return _augment_inplace(examples, cfg, stats, eligible)

    # Multi-variant: the clean original followed by ``copies`` copies.
    
    stats["n_in"] = len(examples)
    out: list[UnifiedExample] = list(examples)
    occurrences: Counter = Counter()
    for ex in examples:
        occurrences[(ex.task_class, ex.source)] += 1
    per_copy: list[int] = []
    for k in range(cfg.copies):
        if eligible:
            n_select = math.ceil(cfg.share * len(eligible))
            selected = set(
                random.Random(f"{cfg.seed}:typo:c{k}").sample(eligible, n_select)
            )
        else:
            n_select = 0
            selected: set[int] = set()
        stats["selected"] += n_select
        copy_aug = 0
        for i, ex in enumerate(examples):
            occ = occurrences[(ex.task_class, ex.source)]
            occurrences[(ex.task_class, ex.source)] += 1
            if i not in selected:
                out.append(ex)
                continue
            new_ex, n_changes = apply_typos(
                ex, cfg, occurrence=occ, copy_index=k
            )
            per_class = stats["per_class"].setdefault(
                ex.task_class, {"selected": 0, "augmented": 0}
            )
            per_class["selected"] += 1
            if n_changes > 0:
                per_class["augmented"] += 1
                stats["augmented"] += 1
                copy_aug += 1
                out.append(new_ex)
            else:
                stats["reverted_clean"] += 1
                out.append(ex)
        per_copy.append(copy_aug)
    stats["per_copy"] = per_copy
    stats["n_out"] = len(out)
    return out, stats


def _augment_inplace(
    examples: Sequence[UnifiedExample],
    cfg: TypoConfig,
    stats: dict[str, Any],
    eligible: list[int],
) -> tuple[list[UnifiedExample], dict[str, Any]]:
    'Legacy in-place path (``copies == 0``): replace a share of the pool.'
    n_select = math.ceil(cfg.share * len(eligible))
    selected = set(random.Random(f"{cfg.seed}:typo").sample(eligible, n_select))
    stats["selected"] = n_select
    occurrences: Counter = Counter()
    out: list[UnifiedExample] = []
    for i, ex in enumerate(examples):
        if i not in selected:
            out.append(ex)
            occurrences[(ex.task_class, ex.source)] += 1
            continue
        occ = occurrences[(ex.task_class, ex.source)]
        occurrences[(ex.task_class, ex.source)] += 1
        new_ex, n_changes = apply_typos(ex, cfg, occurrence=occ)
        per_class = stats["per_class"].setdefault(
            ex.task_class, {"selected": 0, "augmented": 0}
        )
        per_class["selected"] += 1
        if n_changes > 0:
            per_class["augmented"] += 1
            stats["augmented"] += 1
            out.append(new_ex)
        else:
            stats["reverted_clean"] += 1
            out.append(ex)
    return out, stats


def augment_pool_system(
    examples: Sequence[UnifiedExample], cfg: TypoConfig
) -> tuple[list[UnifiedExample], dict[str, Any]]:
    'Apply SYSTEM-prompt (instruction-noise) augmentation to the pool.\n\n\n    In-place, share-based: exactly ``ceil(system_share * n_eligible)`` examples\n    of an allowlisted class with a non-empty system prompt are replaced by\n    their system-augmented copies; the pool size is unchanged. Distinct\n    duplicates of one source get independent typos via the shared occurrence\n    counter.'
    stats: dict[str, Any] = {
        "enabled": cfg.system_share > 0.0,
        "system_share": cfg.system_share,
        "eligible": 0,
        "selected": 0,
        "augmented": 0,
        "reverted_clean": 0,
    }
    if cfg.system_share <= 0.0:
        return list(examples), stats
    eligible = [
        i
        for i, ex in enumerate(examples)
        if ex.task_class in cfg.system_classes and system_spans(ex)
    ]
    stats["eligible"] = len(eligible)
    if not eligible:
        return list(examples), stats
    n_select = math.ceil(cfg.system_share * len(eligible))
    selected = set(
        random.Random(f"{cfg.seed}:system-typo").sample(eligible, n_select)
    )
    stats["selected"] = n_select
    occurrences: Counter = Counter()
    out: list[UnifiedExample] = []
    for i, ex in enumerate(examples):
        if i not in selected:
            out.append(ex)
            occurrences[(ex.task_class, ex.source)] += 1
            continue
        occ = occurrences[(ex.task_class, ex.source)]
        occurrences[(ex.task_class, ex.source)] += 1
        new_ex, n_changes = apply_system_typos(ex, cfg, occurrence=occ)
        if n_changes > 0:
            stats["augmented"] += 1
            out.append(new_ex)
        else:
            stats["reverted_clean"] += 1
            out.append(ex)
    return out, stats
