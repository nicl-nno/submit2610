'Deterministic NEREL F1 — set-level F1 over normalized entity/relation lines.\n\n\nThe NEREL_Instruct extraction sub-configs (``entity_recognition``,\n``relation_extraction``, ``relation_keywords``) are deterministic tasks whose\nreference is a newline-separated list of entities / related entities / keywords.\nSemantic Similarity over such a list is meaningless (it rewards lexical overlap\nof paraphrases, not set correctness), so for these sub-configs the reward\nreplaces SemSim with a deterministic set-level F1 ported from the EleutherAI\nNEREL-bench (`lm_eval/tasks/nerel-bench/utils.py`_, MIT). The\n``entity_normalization`` / definition sub-configs are NOT handled here — they\nstay on the default SemSim (+ fluency) path.\nNEREL-bench (MIT).\n\nRecipe\n\n1. :func:`normalize_answer` — lowercase → ``nltk.wordpunct_tokenize`` → keep\n   tokens that are non-empty AND ``isalnum()`` → Russian Snowball stemmer → join\n   with spaces. Stemming collapses inflection ("\n   one-token normalization like "\n   lowercase + alnum\n2. Each prediction / reference line is normalized into one canonical string;\n   the two **sets** of canonical strings are compared with precision / recall /\n   F1 (the standard set-F1\n\n``nltk.wordpunct_tokenize`` and ``RussianStemmer`` need **no** ``nltk.download``\ndata — both are rule-based. The lazy imports keep this module importable without\nthe ``train`` extra.\n``wordpunct_tokenize``\n\n.. _lm_eval/tasks/nerel-bench/utils.py:\n   https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/nerel-bench/utils.py'

from __future__ import annotations

from typing import Any

from llm_training.reward.scoring import safe_score

#: NEREL_Instruct sub-configs scored by the deterministic set-F1 (the pure
#: extraction tasks). Definitions / normalization stay on the SemSim path.

NEREL_F1_CLASS: str = "nerel_instruct"

NEREL_F1_CONFIGS: frozenset[str] = frozenset(
    {
        "entity_recognition",
        "relation_extraction",
        "relation_keywords",
    }
)

# : Module-level cached stemmer (built once on first use).
_ru_stemmer: Any = None


def _stemmer() -> Any:
    'Lazily build + cache the Russian Snowball stemmer.'
    global _ru_stemmer
    if _ru_stemmer is None:
        from nltk.stem.snowball import RussianStemmer  # no nltk data needed.

        _ru_stemmer = RussianStemmer()
    return _ru_stemmer


def normalize_answer(sentence: str, apply_stemmer: bool = True) -> str:
    'Normalize one line for set comparison (verbatim EleutherAI recipe).\n\n\n    Lowercase → ``wordpunct_tokenize`` → keep non-empty alnum tokens → optional\n    Russian Snowball stemmer → space-joined. Returns ``""`` for a line that\n    normalizes to nothing (pure punctuation / whitespace).'
    from nltk import wordpunct_tokenize  # lazy: nltk is a train-extra dep.

    sentence_ = (sentence or "").lower().strip()
    if len(sentence_) == 0:
        return ""
    words = list(
        filter(
            lambda it2: (len(it2) > 0) and it2.isalnum(),
            [it1.strip() for it1 in wordpunct_tokenize(sentence_)],
        )
    )
    if apply_stemmer:
        words = list(
            filter(
                lambda it2: len(it2) > 0, [_stemmer().stem(it1) for it1 in words]
            )
        )
    if len(words) == 0:
        return ""
    return " ".join(words).strip()


def _set_f1(predicted_lines: list[str], reference_lines: list[str]) -> float:
    'Set-level precision/recall F1 over normalized lines (single pair).\n\n    Set-F1 (precision/recall'
    predicted = set(
        filter(
            lambda s: len(s) > 0,
            [normalize_answer(it) for it in predicted_lines],
        )
    )
    reference = set(
        filter(
            lambda s: len(s) > 0,
            [normalize_answer(it) for it in reference_lines],
        )
    )
    if not reference:
        # Nothing to match against → undefined; caller maps this to N/A.
        
        return 0.0
    common = predicted & reference
    num_same = len(common)
    if num_same == 0:
        return 0.0
    precision = num_same / len(predicted) if predicted else 0.0
    recall = num_same / len(reference)
    return (2.0 * precision * recall) / (precision + recall)


def nerel_f1_applies(task_class: str, extra: dict[str, Any] | None) -> bool:
    'True iff this example should be scored by the deterministic NEREL F1.'
    if task_class != NEREL_F1_CLASS:
        return False
    config = (extra or {}).get("config")
    return config in NEREL_F1_CONFIGS


def nerel_f1(
    response: str,
    reference: str,
    task_class: str,
    extra: dict[str, Any] | None = None,
) -> float | None:
    'Deterministic set-F1 ∈ [0, 1] for NEREL extraction sub-configs; else None.\n\n\n    Returns ``None`` (not applicable — excluded from the harmonic combine) when\n    ``task_class`` is not ``nerel_instruct`` or ``extra["config"]`` is not one of\n    :data:`NEREL_F1_CONFIGS`. Returns ``None`` when the reference normalizes to\n    an empty set (no entity to match against). Otherwise returns the set-F1 in\n    ``[0, 1]``: an empty / wrong prediction scores ``0.0`` (a strong gradient\n    toward the correct entity set).'
    if not nerel_f1_applies(task_class, extra):
        return None
    if not reference or not reference.strip():
        return None
    predicted_lines = (response or "").split("\n")
    reference_lines = reference.split("\n")
    # If the reference normalizes to nothing the metric is undefined → N/A.
    
    if not any(normalize_answer(l) for l in reference_lines):
        return None
    return safe_score(_set_f1(predicted_lines, reference_lines))
