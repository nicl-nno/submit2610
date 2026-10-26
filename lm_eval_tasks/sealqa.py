'SealQA (arXiv:2506.01062, ICLR 2026)\n\n(GPT-4o-mini auto-rater\n\n``dataset_kwargs.revision``\n\nThe SealQA tasks (seal_0 / seal_hard) for lm-eval 0.4.13.\n\nSealQA (arXiv:2506.01062, ICLR 2026) is a challenge benchmark of\nfact-seeking questions where web search returns conflicting, noisy, or\nunhelpful results. The ``seal_0`` (111 questions) and ``seal_hard`` (254)\nsubsets are a pure ``generate_until`` over ``question`` without documents\n(the "w/o search" protocol). Upstream lm-eval 0.4.13 has no such task, so\nit lives in this directory and is plugged in via ``--include_path\nlm_eval_tasks`` (the same mechanism as the scrolls shadow tasks and\nhotpotqa).\n\nThe metric is a deterministic PROXY of the official "relaxed exact match"\n(the GPT-4o-mini auto-rater, see the paper): string normalization (lower →\npunctuation → the a/an/the articles → whitespace collapse) + a check that\nthe normalized gold is a substring of the normalized prediction. This is a\nboolean ``contains`` metric (uniform with fda/swde) — fast and judge-free,\nbut it is only a proxy: a contradictory long answer that happens to\nmention the gold string is scored as correct by mistake. The official\njudge-based rating is offline-only, outside the CMA-ES loop.\n\nSealQA is dynamic (fast/slow-changing answers, "Next update …"): for\nreproducibility the dataset revision is pinned in the yaml via\n``dataset_kwargs.revision``, and the canary string (the ``canary`` field)\ngoes into the report. Across dataset-version breaks the metrics are not\ncomparable.'

from __future__ import annotations

import re
import string

__all__ = [
    "doc_to_text",
    "doc_to_target",
    "process_results",
    "normalize_answer",
    "relaxed_contains_score",
]

_INSTRUCTION = (
    "Answer the question with the exact answer only (a short phrase or a "
    "number), do not explain."
)


def doc_to_text(doc) -> str:
    'The prompt of a single example: the instruction + the question.'
    return f"{_INSTRUCTION}\n\nQuestion: {doc['question']}"


def doc_to_target(doc) -> str:
    'The gold — the ``answer`` field (a short string).'
    return " " + doc["answer"]


def normalize_answer(text: str) -> str:
    'The normalization of the official HotpotQA/SQuAD script (the shared\n    path).'
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def relaxed_contains_score(prediction: str, gold: str) -> float:
    'The "relaxed exact match" proxy: the normalized gold is a substring of\n    the prediction. An empty prediction never matches (except an empty\n    gold — a defensive branch). The containment is over the substring of\n    the normalized strings, not over tokens: numbers like "378" must\n    survive being surrounded by prose.'
    norm_pred = normalize_answer(prediction)
    norm_gold = normalize_answer(gold)
    if not norm_gold:
        return float(norm_pred == norm_gold)
    if not norm_pred:
        return 0.0
    return 1.0 if norm_gold in norm_pred else 0.0


def process_results(doc, results):
    'The boolean ``contains`` of the answer against the gold.'
    prediction = results[0] if results else ""
    return {"contains": relaxed_contains_score(prediction, doc["answer"])}


def warm_up(limit: int = 20) -> None:
    'A no-GPU warm-up: load a slice and print one example.\n\n    Called manually before a run (like hotpotqa.warm_up): network/\n    environment failures surface before the vLLM init.'
    from datasets import load_dataset

    for name in ("seal_0", "seal_hard"):
        ds = load_dataset("vtllms/sealqa", name, split=f"test[:{limit}]")
        doc = ds[0]
        print(f"[sealqa:{name}] test slice: {len(ds)} rows loaded")
        print(f"[sealqa:{name}] question: {doc['question']}")
        print(f"[sealqa:{name}] gold answer: {doc['answer']}")
        print(f"[sealqa:{name}] canary: {doc['canary']}")


if __name__ == "__main__":
    warm_up()
