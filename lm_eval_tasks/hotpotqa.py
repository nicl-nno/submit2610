'``hotpotqa/hotpot_qa``\n``trust_remote_code``. Test\ngreedy\n\n(normalize_answer: lower →\nprecision/recall\nyes/no\n\nThe HotpotQA task (distractor) for lm-eval 0.4.13 — the English skills axis\nof the scaling ladder of the "Built to Read, Not to Recall" paper.\n\nUpstream lm-eval 0.4.13 has NO ``hotpotqa`` task (verified against the\nv0.4.13 tag tree: only longbench/ruler derivatives measuring something\nelse), so the task lives in this directory and is plugged in via\n``--include_path lm_eval_tasks`` (the same mechanism as the scrolls shadow\ntasks). The ``hotpotqa/hotpot_qa`` dataset (config ``distractor``) on HF is\npublished as parquet files without a loading script — the node\'s datasets\n3.2.0 loads it natively, no ``trust_remote_code``. The test split is hidden,\nso validation is scored (7905 questions; the ladder runs slice it with\n``--limit 1000``, seed 42, greedy — the W1 ladder protocol, NOT the official\nMERA protocol).\n\nThe metrics are SQuAD-style EM/F1 following the official HotpotQA evaluation\nscript (normalize_answer: lower → punctuation removal → English article\na/an/the removal → whitespace collapse; F1 over token bags with\nprecision/recall; an empty prediction equals only an empty gold, otherwise\n0). yes/no answers go through the common string path — no special cases.'

from __future__ import annotations

import collections
import re
import string

__all__ = [
    "doc_to_text",
    "doc_to_target",
    "process_results",
    "normalize_answer",
    "exact_match_score",
    "f1_score",
]

_INSTRUCTION = (
    "Read the passages and answer the question. "
    "Answer with the exact answer only (an entity name or a short phrase), "
    "do not explain."
)


def _paragraphs(context) -> str:
    '…».\n\n    list[list[str]]}\n\n    Build the distractor-mode context into a "Title: …\n…" text.\n\n    The ``context`` field is a struct {title: list[str], sentences:\n    list[list[str]]} (arrives as a dict from pyarrow); the paragraphs keep\n    the dataset order, the gold paragraphs are not marked in any way — that\n    is exactly the standard distractor setup.'
    titles = list(context["title"])
    sentences = list(context["sentences"])
    blocks = []
    for title, sents in zip(titles, sentences):
        body = "".join(sents).strip()
        blocks.append(f"Title: {title}\n{body}")
    return "\n\n".join(blocks)


def doc_to_text(doc) -> str:
    'The prompt of a single example: instruction + passages + question.'
    return (
        f"{_INSTRUCTION}\n\n{_paragraphs(doc['context'])}\n\n"
        f"Question: {doc['question']}"
    )


def doc_to_target(doc) -> str:
    'The gold — the ``answer`` field (a string).'
    return doc["answer"]


def normalize_answer(text: str) -> str:
    'The normalization of the official HotpotQA/SQuAD script.'
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction: str, gold: str) -> float:
    'EM over normalized strings.'
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    'Token-bag F1 over normalized tokens.'
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def process_results(doc, results):
    'EM/F1 of the answer against the gold.'
    prediction = results[0] if results else ""
    gold = doc["answer"]
    return {
        "em": exact_match_score(prediction, gold),
        "f1": f1_score(prediction, gold),
    }


def warm_up(limit: int = 20) -> None:
    'A no-GPU warm-up: load a validation slice and print one example.\n\n    Called manually before the ladder run (see paper/W2_RECALC_TASKS.md):\n    network/environment failures surface before the vLLM init.'
    from datasets import load_dataset

    ds = load_dataset(
        "hotpotqa/hotpot_qa", "distractor", split=f"validation[:{limit}]"
    )
    doc = ds[0]
    text = doc_to_text(doc)
    print(f"[hotpotqa] validation slice: {len(ds)} rows loaded")
    print(f"[hotpotqa] prompt chars: min sample = {len(text)}")
    print(f"[hotpotqa] question: {doc['question']}")
    print(f"[hotpotqa] gold answer: {doc['answer']}")


if __name__ == "__main__":
    warm_up()
