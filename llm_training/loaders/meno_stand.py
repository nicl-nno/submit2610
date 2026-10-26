'Loader: the Meno-stand distillation (GPT-4o through the AU "Menon") —\n``meno_rerank`` / ``meno_rewrite`` / ``meno_final``.\n\n``meno_rerank`` / ``meno_rewrite`` / ``meno_final``.\n\nThe source catalog ``src_data/meno_stand_distil/{train,test}.jsonl`` models\nTHREE production tasks of the Meno stand (the AU information portal — the\nvery system Meno-Lite serves; the ``menon_qa`` class comes from the same\nstand), distilled from GPT-4o (the operator\'s own distillation, the same\nprovenance philosophy as the SGR / opencode_gen distillates):\n\n* **rerank** (186 rows) — the RAG document-utility verdict: the system prompt\n  grades a candidate document as a factual basis for the question on the\n  0/1/2 scale; the reference is a compact JSON ``{"label": "N"}`` (the label\n  verdicts are GPT-4o\'s — the ordinal-distance reward pulls toward the\n  GPT-4o prior, an accepted distillation trade-off; the PROMPT asks for the\n  integer spelling — ``rerank_em`` normalizes both forms);\n* **rewrite** (9 rows) — the query rewriter: resolve anaphora against the\n  dialogue history and emit 1-2 search queries separated by ``\\n``;\n* **final** (9 rows) — the final Meno answer (the system prompt is the\n  Menon ``QA_SYSTEM_PROMPT`` up to a negligible edit; a close kin of\n  ``menon_qa``).\n\nThe record shape is the flat ``{system, query, response, history}`` (history\nis empty everywhere — single-turn). The task kind rides in the SYSTEM prompt\n(no dedicated field), so the loader detects it by stable prompt markers.\n\nThe ``test.jsonl`` rows carry ``extra["fixed_val"] = True``: per the operator\ndecision (r5) they are routed into the VAL pool by\n:func:`llm_training.build.build_unified_dataset` (before the per-class cap),\nwhile ``train.jsonl`` rows train — the classes thus participate in BOTH pools\nand the anchor val gains an in-domain interpolation component WITHOUT\nmemorizing the exact val rows. These classes must never be passed as\n``--val-task`` / ``--test-task`` (they already contribute fixed val rows;\nenforced in build.py).'

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Iterator

from llm_training.config import DataConfig
from llm_training.loaders.base import history_to_messages, read_jsonl
from llm_training.schema import UnifiedExample

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizer

CLASSES = ("meno_rerank", "meno_rewrite", "meno_final")

#: The source subdirectory inside ``cfg.data_dir``.

DATA_DIR_NAME = "meno_stand_distil"

#: Stable system-prompt markers for the task-kind detection (the kind rides
#: in the system prompt — there is no dedicated field).

_RERANK_MARKER = "насколько документ пригоден"
_REWRITE_MARKER = "Query Rewriter"

#: The label verdict pattern of a rerank reference (tolerant to the space
#: after the colon observed in 9/186 references).

_LABEL_RE = re.compile(r'"label"\s*:\s*"([012])"')


def _detect_kind(rec: dict[str, Any]) -> str:
    'Map one source record to its task kind by the system-prompt markers.'
    system = rec.get("system", "") or ""
    if _RERANK_MARKER in system:
        return "rerank"
    if _REWRITE_MARKER in system:
        return "rewrite"
    return "final"


def load_class(
    task_class: str,
    tokenizer: "PreTrainedTokenizer",
    cfg: DataConfig,
) -> Iterator[UnifiedExample]:
    "Load ONLY the rows of ``task_class``'s kind from both split files."
    kind = task_class.removeprefix("meno_")
    for split in ("train", "test"):
        path = cfg.data_dir / DATA_DIR_NAME / f"{split}.jsonl"
        for i, rec in enumerate(read_jsonl(path)):
            if not rec.get("response"):
                continue  # malformed; skip
            if _detect_kind(rec) != kind:
                continue
            prompt, ref_msgs = history_to_messages(
                rec.get("system", "") or "",
                rec.get("query", "") or "",
                rec["response"],
                rec.get("history") or [],
            )
            extra: dict[str, Any] = {"meno_task": kind}
            if kind == "rerank":
                m = _LABEL_RE.search(rec["response"])
                if m is None:
                    # A rerank reference without a parseable label is a data
                    # bug; the loader skips it (the metric requires gold).
                    
                    
                    continue
                extra["gold_label"] = int(m.group(1))
            if split == "test":
                extra["fixed_val"] = True
            yield UnifiedExample(
                task_class=task_class,
                prompt=prompt,
                reference=rec["response"],
                reference_messages=ref_msgs,
                source=f"{DATA_DIR_NAME}/{split}.jsonl#{i}",
                extra=extra,
            )


def load(tokenizer: "PreTrainedTokenizer", cfg: DataConfig) -> Iterator[UnifiedExample]:
    'Parameterized-loader registry entry (mirrors sgr_agent/ragu_icl).'  # pragma: no cover - wired via partial() in loaders/__init__
    yield from ()
