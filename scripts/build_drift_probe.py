'Build the synthetic drift-probe set (src_data/drift_probe.jsonl), offline.\n\n\nThe producer behind :mod:`llm_training.runtime.drift_probe`: combinatorially\nenumerates single-instruction prompts (RU/EN) whose compliance is checkable by\npure Python (word counts, JSON keys, a forbidden character, keyword frequency,\nanswer language). Everything is DETERMINISTIC (a fixed enumeration order, no\nRNG): re-running the script reproduces the file byte-for-byte — the probe set\nmust stay stable across a training run series, otherwise the ctrl-point drift\ncolumn measures the set change, not the model change.\n\nWhy SYNTHETIC (never the IFEval dataset): the project discipline keeps\ntraining-time signals disjoint from the final-benchmarking tasks — scoring\nbenchmark prompts during training would train on the test distribution.\n\nUsage\n\n    python scripts/build_drift_probe.py --out src_data/drift_probe.jsonl'

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Neutral topics shared by the word-count / keyword recipes.

_TOPICS: tuple[tuple[str, str], ...] = (
    ("городской транспорт", "public transport"),
    ("настольные игры", "board games"),
    ("горные походы", "mountain hiking"),
    ("морская кухня", "seafood cuisine"),
    ("библиотеки", "libraries"),
    ("метеорология", "meteorology"),
    ("фотография", "photography"),
    ("садоводство", "gardening"),
    ("мосты", "bridges"),
    ("чайная культура", "tea culture"),
    ("аквариумы", "aquariums"),
    ("нарды", "backgammon"),
)

#: Fixed JSON key sets for the json_keys recipe.

_JSON_KEY_SETS: tuple[tuple[str, ...], ...] = (
    ("name", "age"),
    ("title", "items"),
    ("city", "country"),
    ("question", "answer"),
)


def _recipes() -> list[dict]:
    'Enumerate every probe record (deterministic order, no RNG).'
    records: list[dict] = []

    def add(lang: str, prompt: str, check: dict) -> None:
        records.append(
            {
                "id": f"probe-{len(records) + 1:04d}",
                "lang": lang,
                "check": check,
                "prompt": prompt,
            }
        )

    topic_cycle = 0

    def topic_pair() -> tuple[str, str]:
        nonlocal topic_cycle
        pair = _TOPICS[topic_cycle % len(_TOPICS)]
        topic_cycle += 1
        return pair

    # --- min_words / max_words / exact_words -------------------------------
    for n in (20, 40, 60):
        for _ in range(4):
            ru, en = topic_pair()
            add(
                "ru",
                f"Напиши связный текст о теме «{ru}». Ответ должен содержать "
                f"не менее {n} слов.",
                {"type": "min_words", "n": n},
            )
            add(
                "en",
                f"Write a coherent text about {en}. Your answer must contain "
                f"at least {n} words.",
                {"type": "min_words", "n": n},
            )
    for n in (25, 50):
        for _ in range(3):
            ru, en = topic_pair()
            add(
                "ru",
                f"Расскажи коротко про {ru}. Ответ — не более {n} слов.",
                {"type": "max_words", "n": n},
            )
            add(
                "en",
                f"Write briefly about {en}. Your answer must be at most "
                f"{n} words long.",
                {"type": "max_words", "n": n},
            )
    for n in (15, 30):
        ru, en = topic_pair()
        add(
            "ru",
            f"Опиши {ru} ровно {n} словами — ни одним словом больше или "
            "меньше.",
            {"type": "exact_words", "n": n},
        )
        add(
            "en",
            f"Describe {en} in exactly {n} words — not one word more or "
            "less.",
            {"type": "exact_words", "n": n},
        )

    # --- no_commas ----------------------------------------------------------
    for _ in range(4):
        ru, en = topic_pair()
        add(
            "ru",
            f"Составь рассказ про {ru}. В ответе запрещено использовать "
            "запятые.",
            {"type": "no_commas"},
        )
        add(
            "en",
            f"Write a story about {en}. Your entire answer must not contain "
            "any commas.",
            {"type": "no_commas"},
        )

    # --- json_keys ----------------------------------------------------------
    for keys in _JSON_KEY_SETS:
        keys_str = ", ".join(f'"{k}"' for k in keys)
        add(
            "ru",
            "Ответь одним JSON-объектом с ключами " + keys_str +
            " (значения — любые осмысленные строки), без пояснений.",
            {"type": "json_keys", "keys": list(keys)},
        )
        add(
            "en",
            "Answer with a single JSON object with the keys " + keys_str +
            " (any meaningful string values), no explanations.",
            {"type": "json_keys", "keys": list(keys)},
        )

    # --- keyword_min / keyword_max -----------------------------------------
    for kw_ru, kw_en in (("данные", "data"), ("модель", "model")):
        for n in (2, 3):
            ru, en = topic_pair()
            add(
                "ru",
                f"Напиши про {ru}. Слово «{kw_ru}» должно встретиться в "
                f"ответе не менее {n} раз.",
                {"type": "keyword_min", "keyword": kw_ru, "n": n},
            )
            add(
                "en",
                f"Write about {en}. The word \"{kw_en}\" must appear in "
                f"your answer at least {n} times.",
                {"type": "keyword_min", "keyword": kw_en, "n": n},
            )
    for kw_ru, kw_en in (("облако", "cloud"), ("сеть", "network")):
        ru, en = topic_pair()
        add(
            "ru",
            f"Расскажи про {ru}. Слово «{kw_ru}» встречается в вопросе, но "
            "в ответе его использовать нельзя.",
            {"type": "keyword_max", "keyword": kw_ru, "n": 0},
        )
        add(
            "en",
            f"Write about {en}. The word \"{kw_en}\" appears in this prompt "
            "but must NOT appear in your answer.",
            {"type": "keyword_max", "keyword": kw_en, "n": 0},
        )

    # --- language_ru / language_en ------------------------------------------
    for _ in range(4):
        ru, en = topic_pair()
        add(
            "en",
            f"Write a short text about {en}. You MUST answer in Russian.",
            {"type": "language_ru"},
        )
        add(
            "ru",
            f"Напиши короткий текст про {ru}. Отвечать нужно на английском "
            "языке.",
            {"type": "language_en"},
        )

    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Build the synthetic drift-probe set (deterministic, no RNG).'
        )
    )
    parser.add_argument(
        "--out",
        default="src_data/drift_probe.jsonl",
        help='Output JSONL path (default src_data/drift_probe.jsonl).',
    )
    args = parser.parse_args()

    records = _recipes()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    per_type: dict[str, int] = {}
    for rec in records:
        per_type[rec["check"]["type"]] = per_type.get(rec["check"]["type"], 0) + 1
    print(f"OK: {len(records)} probe items -> {out_path}")
    print(f'per check type')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
