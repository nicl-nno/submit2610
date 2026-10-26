'The GRPO drift probe — a synthetic instruction-following mini-benchmark.\n\n\nThe 2026-09 benchmark attribution showed the SFT/GRPO parent checkpoints lose\n~15 pp on IFEval-style instruction following while the in-loop reward never\nmeasures that dimension (nothing in the reward checks instruction\ncompliance of free-form text). The drift probe closes the MEASUREMENT gap at\nctrl-points: a small fixed set of synthetic prompts, each carrying one\nmachine-checkable instruction (word count, JSON keys, forbidden character,\nkeyword frequency, language), generated + checked deterministically at every\nanchor — WITHOUT the judge and WITHOUT touching the frozen ``val`` selection\nmeasure (the probe is a separate ANCHOR-log column and a WARN trigger, not an\nearly-stop criterion).\n\nWhy SYNTHETIC prompts (and never the IFEval dataset itself): the project\ndiscipline keeps training-time signals disjoint from the final-benchmarking\ntasks (the merge-evolution invariant) — scoring benchmark PROMPTS during\ntraining would train on the test distribution. The producer\n(:mod:`scripts.build_drift_probe`) is seeded and offline; its output\n``src_data/drift_probe.jsonl`` is committed like every other src_data set.\n\nRecord format\n\n    {"id": "probe-0007", "lang": "ru", "check": {"type": "min_words", "n": 40},\n\nThe generation mirrors the anchor loop (:func:`grpo_train.run_anchor_scoring`):\nHF ``.generate`` on the training model, one example at a time, greedy, with\nthe KV-cache temporarily re-enabled and the envelope-preserving\n:func:`~llm_training.tokenizer_utils.decode_generation` decode.'

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

#: Check types with their parameter dicts — the schema of ``check``.

CHECK_TYPES: tuple[str, ...] = (
    "min_words",
    "max_words",
    "exact_words",
    "no_commas",
    "json_keys",
    "keyword_min",
    "keyword_max",
    "language_ru",
    "language_en",
)


@dataclass(frozen=True)
class ProbeItem:
    'One probe prompt + its machine-checkable instruction.'

    id: str
    prompt: str
    check: dict[str, Any]
    lang: str = "ru"


def load_probe(path: str | Path) -> list[ProbeItem]:
    'Load the probe set from ``src_data/drift_probe.jsonl``.'
    items: list[ProbeItem] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            items.append(
                ProbeItem(
                    id=str(rec["id"]),
                    prompt=str(rec["prompt"]),
                    check=dict(rec["check"]),
                    lang=str(rec.get("lang", "ru")),
                )
            )
    return items


def _words(text: str) -> list[str]:
    'Whitespace-split tokens carrying ≥1 alphanumeric character.'
    return [t for t in (text or "").split() if any(c.isalnum() for c in t)]


def _strip_json_fences(text: str) -> str:
    'Strip markdown fences around a JSON payload (judge-observed habit).'
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        if first_nl != -1:
            cleaned = cleaned[first_nl + 1 :]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    return cleaned.strip()


def check_response(check: dict[str, Any], text: str) -> bool:
    'Run one deterministic instruction check over a generated response.\n\n\n    Unknown check types fail CLOSED (``False``): a probe record the runtime\n    cannot interpret must never silently pass — the producer and the runtime\n    ship together, a mismatch is a bug to surface, not to hide.'
    ctype = str(check.get("type"))
    if ctype == "min_words":
        return len(_words(text)) >= int(check["n"])
    if ctype == "max_words":
        return len(_words(text)) <= int(check["n"])
    if ctype == "exact_words":
        return len(_words(text)) == int(check["n"])
    if ctype == "no_commas":
        return "," not in (text or "")
    if ctype == "json_keys":
        try:
            payload = json.loads(_strip_json_fences(text))
        except (json.JSONDecodeError, ValueError):
            return False
        if not isinstance(payload, dict):
            return False
        return set(check["keys"]) <= set(payload.keys())
    if ctype == "keyword_min":
        lowered = (text or "").lower()
        return lowered.count(str(check["keyword"]).lower()) >= int(check["n"])
    if ctype == "keyword_max":
        lowered = (text or "").lower()
        return lowered.count(str(check["keyword"]).lower()) <= int(check["n"])
    if ctype == "language_ru":
        letters = [c for c in (text or "") if c.isalpha()]
        if not letters:
            return False
        return sum("а" <= c.lower() <= "я" or c.lower() == "ё" for c in letters) / len(
            letters
        ) >= 0.8
    if ctype == "language_en":
        letters = [c for c in (text or "") if c.isalpha()]
        if not letters:
            return False
        return sum(c.isascii() for c in letters) / len(letters) >= 0.8
    return False


def score_responses(
    items: Sequence[ProbeItem], responses: Sequence[str]
) -> dict[str, Any]:
    'Score check pass-rates over the generated responses (pure, no model).'
    per_check: dict[str, list[int]] = {}
    failed_ids: list[str] = []
    n = min(len(items), len(responses))
    for item, resp in zip(items[:n], responses[:n]):
        ok = check_response(item.check, resp)
        per_check.setdefault(str(item.check.get("type")), []).append(int(ok))
        if not ok:
            failed_ids.append(item.id)
    score = (
        sum(sum(v) for v in per_check.values())
        / max(1, sum(len(v) for v in per_check.values()))
    )
    return {
        "score": score,
        "n": n,
        "per_check": {
            k: {"pass_rate": sum(v) / len(v), "n": len(v)}
            for k, v in sorted(per_check.items())
        },
        "failed_ids": failed_ids[:20],
    }


def run_drift_probe(
    model: Any,
    tokenizer: Any,
    items: Sequence[ProbeItem],
    *,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    'Generate + check the probe set on the training model (anchor-style).\n\n\n    Mirrors the anchor generation loop: HF ``.generate``, one example at a\n    time, greedy, KV-cache temporarily re-enabled (restored in ``finally``),\n    decoded via :func:`llm_training.tokenizer_utils.decode_generation` so\n    special tokens never leak into the word counts. Returns the\n    :func:`score_responses` breakdown.'
    import torch

    from llm_training.tokenizer_utils import decode_generation

    responses: list[str] = []
    was_training = model.training
    model.eval()
    prev_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = True
    try:
        with torch.no_grad():
            for item in items:
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": item.prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = tokenizer(
                    prompt_text, return_tensors="pt", add_special_tokens=False
                ).to(model.device)
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                responses.append(
                    decode_generation(
                        tokenizer, out[0][inputs["input_ids"].shape[1]:]
                    )
                )
    finally:
        if prev_use_cache is not None:
            model.config.use_cache = prev_use_cache
        if was_training:
            model.train()
    return score_responses(items, responses)
