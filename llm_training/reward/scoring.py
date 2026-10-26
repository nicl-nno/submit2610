'Pure score-combining math + format-checks — no LLM, no network, deterministic.\n\n\nThe reward layer combines up to four signals per example\n\n* **Semantic Similarity** (always; embeddings-only, cheap, deterministic).\n* **Answer Correctness** (only when a judge endpoint is configured; the LLM\n  anchor). Answer Correctness\n* **Format check** (only for tool-use classes; verifies that ``<tool_call>``\n  blocks are well-formed\n* **Code check** (only for the ``coder`` class; verifies that ``python``\n  fenced blocks parse and are non-trivial / free of common anti-patterns via\n  :mod:`ast`). ``code_check``\n\nThis module also hosts the **pure helpers shared by the fluency signal**:\n:func:`split_prose_and_code` (extracts fenced code and strips non-prose spans),\n:func:`count_words` (the per-sentence ``razdel`` word-count gate) and\n:func:`aggregate_fluency` (mean of per-sentence scores → one answer-level\nscore). The model-backed fluency computation (grammar classifier + ``razdel``)\nlives in :mod:`llm_training.reward.fluency`\n:func:`count_words`\n:func:`aggregate_fluency`\nfluency\n\nThey are combined via the harmonic mean of the **active** metrics (the strict\nconvention: a low score on any single dimension drags the combined reward down,\nand a zero on any active metric — e.g. an invalid tool-call — zeroes the whole\nreward).\n\n\nEverything here is a pure function so it is unit-tested with no mocks.'

from __future__ import annotations

import ast
import json
import re
from collections.abc import Sequence
from typing import Any

#: Task classes whose references are tool-call blocks that must pass structural
#: validation. Only these get a non-``None`` :func:`format_check` score. Includes
#: the OpenCode / tool-calls-multiturn classes AND the 9 SGR-agent classes
#: (every SGR turn is a tool call: action tools or the ``finalanswertool``).

# : SGR-agent
TOOL_USE_CLASSES: frozenset[str] = frozenset(
    {
        "opencode_gen",
        "tool_calls_multiturn",
        # SGR-agent (distilled from sgr-agent-core; tool-call-faithful).
        "coder",
        "creative_writer",
        "data_analyst",
        "deep_research",
        "factual_qa",
        "instruction_following",
        "math_reasoner",
        "rewriter",
        "summarizer",
    }
)

# : The SGR framework's finalization tool name.
FINAL_ANSWER_TOOL: str = "finalanswertool"

# Weights for the three orthogonal dimensions of :func:`format_check` for
# tool-use classes (must sum to 1.0). A tool-using response must parse as JSON,
# use a known tool name, and carry the required fields — each dimension earns a
# fraction of the format score so partial credit propagates a useful gradient
# (a binary 0/1 would tell GRPO only "broken / ok" and waste signal).

_FORMAT_WEIGHT_JSON = 0.4    # every <tool_call> block parses as valid JSON.
_FORMAT_WEIGHT_NAME = 0.3    # every tool name is in the known registry.
_FORMAT_WEIGHT_FIELDS = 0.3  # every call carries both ``name`` and ``arguments``.

# Matches ``<tool_call>...</tool_call>`` blocks (non-greedy, DOTALL).

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def safe_score(value: float | None) -> float:
    'Coerce a metric value into ``[0.0, 1.0]`` (``None`` → ``0.0``).\n\n\n    RAGAS scores live in ``[0, 1]``; this guards against NaN/None/out-of-range\n    values coming back from the judge (a failed judge call must not poison the'
    if value is None:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v != v or v in (float("inf"), float("-inf")):  # NaN / inf
        return 0.0
    return max(0.0, min(1.0, v))


def harmonic_mean(values: Sequence[float]) -> float:
    'Harmonic mean of non-negative values; ``0.0`` if any value is ``0``.\n\n\n    For ``n`` values ``H = n / sum(1/x_i)``. If any ``x_i == 0`` the harmonic\n    mean is ``0`` by the standard limit convention used in score combination\n    (a fully unfaithful answer scores zero overall).\n\n``H = n'
    n = len(values)
    if n == 0:
        # Invariant: harmonic_mean is invoked on the *active* metrics of an
        # example. The caller always pre-collects at least one (Semantic
        # Similarity is unconditionally active), so this branch is a defensive
        # fallback — never a legitimate code path in production. We keep it
        # rather than raising so that an accidental misuse cannot abort a
        # GRPO batch over a single malformed example.
        
        
        
        return 0.0
    cleaned = [safe_score(v) for v in values]
    if any(x <= 0.0 for x in cleaned):
        return 0.0
    return n / sum(1.0 / x for x in cleaned)


def format_check(
    response: str,
    task_class: str,
    known_tool_names: frozenset[str] = frozenset(),
) -> float | None:
    'Structural score [0, 1] for tool-use responses; ``None`` when N/A.\n\n\n    Returns ``None`` for non-tool-use classes (the metric is not applicable and\n    is excluded from the harmonic combine — it must NOT drag the score down).\n\n    For tool-use classes (:data:`TOOL_USE_CLASSES`) the score is a weighted sum\n    of three orthogonal dimensions:\n\n    * ``0.4`` — every ``<tool_call>`` block parses as valid JSON / Python dict.\n    * ``0.3`` — every parsed call uses a tool name from ``known_tool_names``.\n      When ``known_tool_names`` is empty the check is skipped (no registry was\n      collected) and the 0.3 share is awarded neutrally.\n    * ``0.3`` — every parsed call carries both ``name`` and ``arguments`` keys.\n\n    A response with no ``<tool_call>`` blocks scores 0.0 (a tool-use turn that\n    makes no calls is structurally wrong); a perfectly-formed response scores 1.0.'
    if task_class not in TOOL_USE_CLASSES:
        return None

    if not response or not response.strip():
        return 0.0

    blocks = _TOOL_CALL_BLOCK_RE.findall(response)
    if not blocks:
        # A tool-use turn that produced no tool_call block at all is structurally
        # broken (text-only answer when a tool call was expected).
        # tool-use
        return 0.0

    json_ok = True
    fields_ok = True
    names_ok = True
    parsed_calls: list[dict] = []
    for raw in blocks:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            # Fall back to Python-literal parsing (``True``/``None``/single
            # quotes) — the Qwen2.5 template sometimes emits Python repr in
            # ``<tool_call>`` blocks, mirroring the loader convention.
            
            try:
                import ast

                data = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                json_ok = False
                continue
        if not isinstance(data, dict):
            json_ok = False
            continue
        parsed_calls.append(data)
        if "name" not in data or "arguments" not in data:
            fields_ok = False

    # Defensive: if NO block parsed into a dict, there is nothing to validate
    # names or fields against — both sub-checks must fail (a response where
    # every <tool_call> is unparseable cannot earn the fields/name credit).
    
    
    if not parsed_calls:
        fields_ok = False
        names_ok = False

    # Tool-name check: only enforced when a registry was provided AND at least
    # one call parsed.
    if known_tool_names:
        for call in parsed_calls:
            if call.get("name") not in known_tool_names:
                names_ok = False
                break

    score = _FORMAT_WEIGHT_JSON if json_ok else 0.0
    score += _FORMAT_WEIGHT_NAME if names_ok else 0.0
    score += _FORMAT_WEIGHT_FIELDS if fields_ok else 0.0
    return safe_score(score)


#: SGR reasoning-schema markers — the four sections the sgr-agent-core SGR
#: framework's reasoning phase emits (English, regardless of the task language —
#: they come from the framework template, stable across the distilled corpus).

_SGR_REASONING_MARKERS: tuple[str, ...] = (
    "reasoning steps",
    "current situation",
    "plan status",
    "remaining steps",
)


def reasoning_check(text: str) -> float:
    "Structural score [0, 1] for an SGR reasoning turn (CoT text).\n\n\n    Returns the fraction of the SGR reasoning-schema markers present\n    (:data:`_SGR_REASONING_MARKERS`). A turn emitting all four sections\n    (``Reasoning steps`` / ``Current situation`` / ``Plan status`` / ``Remaining\n    steps``) scores 1.0; a partial turn scores less; an empty / unstructured\n    turn scores 0.0. This is the structural anchor used INSTEAD of Semantic\n    Similarity for reasoning turns (see ``RewardConfig.enable_reasoning_semsim``):\n    reasoning is open-ended, so SemSim would reward paraphrase-imitation of the\n    teacher's wording rather than the Schema-Guided-Reasoning structure that\n    defines the SGR framework"
    if not text or not text.strip():
        return 0.0
    low = text.lower()
    present = sum(1 for marker in _SGR_REASONING_MARKERS if marker in low)
    return safe_score(present / len(_SGR_REASONING_MARKERS))


def extract_final_answer(response: str) -> str | None:
    "Pull the ``answer`` field out of a ``finalanswertool`` tool-call.\n\n\n    Scans the ``<tool_call>`` blocks in a model's response for one whose name is\n    :data:`FINAL_ANSWER_TOOL` and returns its ``arguments.answer`` string.\n    Returns ``None`` when there is no valid final-answer call (the caller then\n    treats the answer as empty/degenerate → low SemSim, no fluency). Used by the\n    reward dispatcher so SemSim/fluency/``code_check`` score the answer **content**\n    rather than the wrapping tool-call JSON. ``arguments`` is accepted as either a\n    dict (the rendered form) or a JSON string (defensive)."
    if not response:
        return None
    blocks = _TOOL_CALL_BLOCK_RE.findall(response)
    for raw in blocks:
        data: dict | None = None
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            try:
                data = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                data = None
        if not isinstance(data, dict) or data.get("name") != FINAL_ANSWER_TOOL:
            continue
        args = data.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        if isinstance(args, dict):
            return args.get("answer") or ""
        return ""
    return None


def content_response_text(
    text: str | None,
    extra: dict[str, Any] | None,
    task_class: str | None = None,
) -> str | None:
    'Return the text the CONTENT metrics (fluency / ``code_check``) should score.\n\n\n    Mirrors the ``content_response`` selection in\n    :meth:`llm_training.reward.judge.RAGASScorer.score` so that offline analysis\n    (the Phase-0 scripts ``measure_signal_variance`` / ``calibrate_fluency``)\n    measures the **same population** the reward scores, not the raw wire text:\n\n    * ``is_final_answer`` turns → the ``answer`` field extracted from the\n      ``finalanswertool`` call (:func:`extract_final_answer`); ``None`` when\n      the response carries no valid call — the reward-time caller maps that to\n      ``""`` (an empty answer → poor scores, a strong gradient), while the\n      offline analysis **skips** such samples (there is no prose to measure).\n      Without this, a final turn\'s raw ``<tool_call>`` JSON would be stripped\n      of the tool span and contribute nothing — the extracted answer prose\n      would never be measured.\n    * ``docsem_quant`` (``task_class`` argument) → the effective answer\n      extracted from the JSON envelope (:func:`docsem_answer_text` — the\n      executed program value, stated ``answer`` fallback); ``None`` when the\n      JSON is unparseable (same skip semantics as final turns).\n    * every other turn (reasoning texts, action-turn content lines, plain\n      non-tool answers) → the raw text as-is: any ``<tool_call>`` span inside\n      is stripped downstream by :func:`split_prose_and_code`, exactly as the\n      reward does.'
    if text is None or not text.strip():
        return None
    if (extra or {}).get("is_final_answer"):
        return extract_final_answer(text)
    if task_class == "docsem_quant":
        from llm_training.reward.docsem import docsem_answer_text

        return docsem_answer_text(text)
    return text


# ---------------------------------------------------------------------------
# Code check (Python ast)
# ---------------------------------------------------------------------------
#: Task class for which :func:`code_check` is active. Only ``coder`` emits
#: free-form code that benefits from ``ast`` validation (other code-bearing
#: classes are tool-call JSON, handled by :func:`format_check`).

CODE_CHECK_CLASS: str = "coder"

#: Fenced-block language tags that count as Python (case-insensitive, matched
# : against the lower-cased tag).
_PYTHON_TAGS: frozenset[str] = frozenset({"python", "py", "python3"})

# Weights for the three orthogonal dimensions of :func:`code_check` (must sum
# to 1.0). A coder answer must parse as valid Python, do something non-trivial,
# and avoid common anti-patterns — each dimension earns a fraction so partial
# credit propagates a useful gradient (a binary 0/1 wastes signal).

_CODE_WEIGHT_PARSE = 0.5       # every python block parses as valid Python.
_CODE_WEIGHT_NONTRIVIAL = 0.3  # contains real constructs (def/class/statements).
_CODE_WEIGHT_ANTIPATTERN = 0.2  # free of bare except / star import / mutable default / exec-eval.

#: ``ast`` node types that count as "meaningful statements" for the
#: non-triviality check (docstrings, ``pass``, bare ``Expr`` strings excluded).

_MEANINGFUL_STMT_TYPES: tuple[type, ...] = (
    ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Return,
    ast.If, ast.For, ast.While, ast.With, ast.AsyncWith,
    ast.Raise, ast.Try,
)

#: Mutable-literal node types in default arguments (the classic mutable-default
# : bug).
_MUTABLE_DEFAULT_TYPES: tuple[type, ...] = (ast.List, ast.Set, ast.Dict)

#: Builtins whose call in a default arg is also a mutable-default smell.
# :
_MUTABLE_DEFAULT_CALLS: frozenset[str] = frozenset({"list", "set", "dict"})


# ---------------------------------------------------------------------------
# Prose / code extraction (shared by fluency and code_check) /

# ---------------------------------------------------------------------------
#: Matches a fenced block ```` ```lang\n<body>``` ````. The language tag is
#: optional and lower-cased; the body (non-greedy, DOTALL) is captured as-is.

_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+.-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)

#: ``<tool_call>…</tool_call>`` and ``<tool_response>…</tool_response>`` spans,
#: stripped from prose (tool mechanics, not natural language).

_TOOL_STRIP_RE = re.compile(
    r"<tool_call>.*?</tool_call>|<tool_response>.*?</tool_response>",
    re.DOTALL,
)

# : Inline `` `code` `` spans, stripped from prose.
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")


def split_prose_and_code(text: str) -> tuple[str, list[tuple[str, str]]]:
    'Split an answer into ``(prose, code_blocks)`` for fluency / code_check.\n\ncode_check.\n\n    * ``prose``: the text with fenced code blocks, ``<tool_call>`` /\n      ``<tool_response>`` spans, and inline `` `code` `` removed and whitespace\n      collapsed — the natural-language part the fluency scorer segments into\n    * ``code_blocks``: list of ``(lang, body)`` for fenced blocks (``lang`` is\n      the lower-cased tag, possibly ``""`` for an unlabelled fence) — the input\n      to :func:`code_check`, which filters to Python\n\n    Never raises. Empty/``None``-ish input → ``("", [])``.'
    if not text:
        return "", []
    code_blocks: list[tuple[str, str]] = []

    def _grab(m: re.Match[str]) -> str:
        lang = (m.group(1) or "").strip().lower()
        code_blocks.append((lang, m.group(2)))
        return " "  # replace the fenced block with a space in the prose

    prose = _FENCE_RE.sub(_grab, text)
    prose = _TOOL_STRIP_RE.sub(" ", prose)
    prose = _INLINE_CODE_RE.sub(" ", prose)
    prose = " ".join(prose.split())
    return prose, code_blocks


def _has_nontrivial_content(trees: list[ast.AST]) -> bool:
    'True iff the parsed trees define a function/class or have ≥3 statements.'
    has_def = False
    count = 0
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                has_def = True
            elif isinstance(node, _MEANINGFUL_STMT_TYPES):
                count += 1
    return has_def or count >= 3


def _has_no_antipatterns(trees: list[ast.AST]) -> bool:
    'True iff no common Python anti-pattern is present.\n\n\n    Detects: bare ``except:``, star ``from x import *``, mutable default args\n    (``[]``, ``{}``, ``set()``, …), and ``exec``/``eval`` calls.'
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                return False  # bare except
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        return False  # star import
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defaults = list(node.args.defaults) + list(node.args.kw_defaults)
                for default in defaults:
                    if isinstance(default, _MUTABLE_DEFAULT_TYPES):
                        return False  # mutable literal default
                    if (
                        isinstance(default, ast.Call)
                        and isinstance(default.func, ast.Name)
                        and default.func.id in _MUTABLE_DEFAULT_CALLS
                    ):
                        return False  # mutable call default (list()/set()/dict())
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("exec", "eval")
            ):
                return False  # exec/eval usage
    return True


def code_check(response: str, task_class: str) -> float | None:
    'Structural score [0, 1] for ``coder`` answers; ``None`` when N/A.\n\n\n    Returns ``None`` (excluded from the harmonic combine) when:\n    * ``task_class`` is not :data:`CODE_CHECK_CLASS` — the metric is not\n      applicable to non-coder classes; or\n    * the answer contains no Python fenced block — a coder answer about\n      Rust/SQL/Bash is structurally fine and must not be penalised for the\n      absence of Python (the metric is simply not applicable to it).\n\n    For a ``coder`` answer with ≥1 ``python`` fenced block, the score is a\n    weighted sum of three orthogonal dimensions (mirrors :func:`format_check`):\n\n    * ``0.5`` — every Python block parses via :func:`ast.parse` (syntactic\n      validity). **Any** broken block zeroes this dimension AND, since the AST\n      is unavailable, the other two as well → the whole ``code_check`` is\n      ``0.0`` (strong gradient towards compilable code).\n    * ``0.3`` — non-trivial content (≥1 function/class definition, or ≥3\n      meaningful statements; comments/``pass``/docstrings alone do not count).\n    * ``0.2`` — free of common anti-patterns (bare ``except:``, star import,\n      mutable default arg, ``exec``/``eval``).\n\n    Verified empirically on the reference ``coder`` answers: every Python block\n    parses (100 %), so a generated answer that fails to parse is genuinely a\n    regression below reference quality. Empirically ~39 % of ``coder`` records\n    carry Python; the rest get ``None``.'
    if task_class != CODE_CHECK_CLASS:
        return None
    if not response or not response.strip():
        return None  # degenerate empty answer — N/A rather than 0
    _, blocks = split_prose_and_code(response)
    py_blocks = [body for lang, body in blocks if lang in _PYTHON_TAGS]
    if not py_blocks:
        return None  # non-Python coder answer — N/A

    trees: list[ast.AST] = []
    for body in py_blocks:
        try:
            trees.append(ast.parse(body))
        except (SyntaxError, ValueError):
            # A broken Python block: no reliable AST → zero the whole check.
            
            return 0.0

    score = _CODE_WEIGHT_PARSE
    if _has_nontrivial_content(trees):
        score += _CODE_WEIGHT_NONTRIVIAL
    if _has_no_antipatterns(trees):
        score += _CODE_WEIGHT_ANTIPATTERN
    return safe_score(score)


# ---------------------------------------------------------------------------
# Fluency word-count + aggregation (pure; model-backed part lives in
# reward.fluency)

# ---------------------------------------------------------------------------
#: Minimum number of word-tokens in a prose sentence for it to carry a fluency
#: signal (counted by :func:`count_words` via ``razdel``). A sentence with
# : fewer alphabetic tokens ("
# : on and is dropped BEFORE classification. A single full sentence ("
# the grammar classifier is a sentence-level model, so
# : one well-formed sentence is in-distribution.

MIN_FLUENCY_WORDS: int = 3


def count_words(text: str) -> int:
    'Count word-tokens in ``text`` (``razdel``) that contain ≥1 letter.\n\n\n    Punctuation and purely-numeric tokens are excluded; hyphenated tokens such\n    them intact). This is the per-sentence gate the fluency scorer applies\nBEFORE loading the model'
    if not text:
        return 0
    from razdel import tokenize as _tokenize  # lazy: razdel is a train-extra dep.

    n = 0
    for tok in _tokenize(text):
        t = getattr(tok, "text", "") or ""
        if any(ch.isalpha() for ch in t):
            n += 1
    return n


def aggregate_fluency(per_sentence_scores: Sequence[float]) -> float | None:
    'Aggregate per-sentence fluency scores into one answer score.\n\n\n    Returns the arithmetic mean of the (``safe_score``-clamped) per-sentence\n    scores, or ``None`` when the list is empty (no sentence survived the\n    word-count gate → the metric is excluded from the harmonic combine, NOT\n    zeroed). Any calibrator transform (identity by default with the grammar\n    classifier) is applied PER SENTENCE before this aggregation, at the\n    model-backed fluency layer.'
    if not per_sentence_scores:
        return None
    vals = [safe_score(v) for v in per_sentence_scores]
    return sum(vals) / len(vals)
