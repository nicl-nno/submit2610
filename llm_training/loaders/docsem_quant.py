'Loader: DocSem quantitative reasoning → ``docsem_quant``.\n\n\nDocument-grounded quantitative reasoning distilled from the DocSem shared\ntask (DocInsights 2026 @ EMNLP; data © Oracle, UPL 1.0 — see\n``src_data/docsem_quant.LICENSE``). Each example is a paraphrased question\nover the top-3 retrieved document blocks; the reference assistant turn is a\nchampion-trace JSON object ``{"reasoning", "program", "answer", "evidence"}``\n(Qwen3.8-27B + PoT, verified against the gold answer/evidence where a\nverified sample exists — see ``scripts/build_docsem_quant.py``).\n\nThe gold answer/evidence ride in ``extra`` (``gold_answer`` /\n``gold_evidence``): the deterministic :func:`llm_training.reward.docsem.docsem_check`\nreward scores the model\'s JSON against THEM (program executed in the safe\ninterpreter + canonical answer equality + evidence set-F1), replacing\nSemantic Similarity for this class (the NEREL-F1 pattern).\n\n``{"reasoning", "program", "answer", "evidence"}`` (Qwen3.8-27B + PoT,'

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from llm_training.config import DataConfig
from llm_training.loaders.base import read_jsonl
from llm_training.schema import UnifiedExample

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizer

CLASSES = ("docsem_quant",)
FILENAME = "docsem_quant.jsonl"


def load(tokenizer: "PreTrainedTokenizer", cfg: DataConfig) -> Iterator[UnifiedExample]:
    'Yield one single-turn example per DocSem check task.'
    path = cfg.data_dir / FILENAME
    for record in read_jsonl(path):
        prompt = [
            {"role": "system", "content": record["system"]},
            {"role": "user", "content": record["query"]},
        ]
        yield UnifiedExample(
            task_class="docsem_quant",
            prompt=prompt,
            reference=record["response"],
            tools=None,
            reference_messages=[
                {"role": "assistant", "content": record["response"]}
            ],
            source=record.get("instance_id", ""),
            extra={
                "instance_id": record.get("instance_id", ""),
                "gold_answer": record.get("gold_answer", ""),
                "gold_evidence": list(record.get("gold_evidence") or []),
                "trace_verified": bool(record.get("trace_verified")),
            },
        )
