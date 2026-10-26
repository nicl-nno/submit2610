'CPT data plumbing: record generators, tokenization, mixed batching.\n\n\nInput formats\n\n* SFT — the flat ``{"system", "query", "response", "history"}`` shape also\n  used by ``self_cognition`` (``history`` = prior ``[user, assistant]``\n  pairs); the reference is ``response``;\n* PT (unlabeled) — ``{"response"}`` with raw document text.\n\nSFT\n``self_cognition``; ``history``\n\nPre-tokenized rows\n    ``{"input_ids": int32[…], "n_prompt": int32, "kind": "sft"|"pt",\n    "source": str, "n_tokens": int32}``. Labels are NOT stored: the collator\n    rebuilds them on the fly from ``n_prompt`` (``-100`` on the prompt part;\n    PT rows carry ``n_prompt=0`` → loss on every token) — this halves the\n    cache size at the billion-token scale.\n\n    ``{"input_ids": int32[…], "n_prompt": int32, "kind": "sft"|"pt",\n    "source": str, "n_tokens": int32}``\n\nMixed mini-batches\n    :class:`MixedBatchSampler` guarantees that EVERY mini-batch contains\n    both SFT and PT rows (a fixed per-batch composition), reshuffling both\n    pools each epoch with a deterministic seed; the smaller pool cycles with\n    reshuffles (natural upsampling). An epoch passes over the LARGER pool\n    exactly once.\n\n    :class:`MixedBatchSampler`'

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Sampler

from llm_training.loaders.base import history_to_messages, read_jsonl
from llm_training.tokenizer_utils import render, tokenize

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizer

# : Row kinds
KIND_SFT = "sft"
KIND_PT = "pt"


def iter_sft_records(paths: Sequence[str | Path]) -> Iterator[dict[str, Any]]:
    'Yield normalized SFT records ``{system, query, response, history, source}``.\n\n\n    Malformed records (empty ``response``) are skipped; the loader shape is\n    the flat ``{system, query, response, history}`` one.'
    for path in paths:
        path = Path(path)
        for i, rec in enumerate(read_jsonl(path)):
            response = rec.get("response") or ""
            if not response.strip():
                continue
            history: list[list[str]] = []
            for pair in rec.get("history") or []:
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                history.append([str(pair[0] or ""), str(pair[1] or "")])
            yield {
                "system": str(rec.get("system") or ""),
                "query": str(rec.get("query") or ""),
                "response": response,
                "history": history,
                "source": f"{path}#{i}",
            }


def iter_pt_records(paths: Sequence[str | Path]) -> Iterator[dict[str, Any]]:
    'Yield normalized PT records ``{response, source}``.'
    for path in paths:
        path = Path(path)
        for i, rec in enumerate(read_jsonl(path)):
            response = rec.get("response") or ""
            if not response.strip():
                continue
            yield {"response": response, "source": f"{path}#{i}"}


def _empty_out() -> dict[str, list]:
    return {k: [] for k in ("input_ids", "n_prompt", "kind", "source", "n_tokens")}


def tokenize_sft_batch(
    batch: dict[str, list], tokenizer: "PreTrainedTokenizer", max_len: int
) -> dict[str, list]:
    'Batched map fn: SFT records → pre-tokenized rows with prompt masking.\n\n\n    The prompt/reference boundary uses the repo-wide **differential render**:\n    the prompt is rendered with ``add_generation_prompt=True``, the full\n    conversation with ``False``; the row keeps the full id list plus\n    ``n_prompt`` (the collator masks ``[:n_prompt]`` with ``-100``). Examples\n    whose renders break the prefix property or whose total length exceeds\n    ``max_len`` are dropped (repo convention: discard, never truncate).'
    out = _empty_out()
    for system, query, response, history, source in zip(
        batch["system"], batch["query"], batch["response"], batch["history"],
        batch["source"],
    ):
        prompt_messages, reference_messages = history_to_messages(
            system, query, response, history
        )
        full_messages = [*prompt_messages, *reference_messages]
        prompt_text = render(
            tokenizer, prompt_messages, add_generation_prompt=True
        )
        full_text = render(
            tokenizer, full_messages, add_generation_prompt=False
        )
        if not full_text.startswith(prompt_text):
            continue
        prompt_ids = tokenize(tokenizer, prompt_text)
        full_ids = tokenize(tokenizer, full_text)
        n_prompt = len(prompt_ids)
        if full_ids[:n_prompt] != prompt_ids:
            continue
        if len(full_ids) > max_len:
            continue
        out["input_ids"].append(full_ids)
        out["n_prompt"].append(n_prompt)
        out["kind"].append(KIND_SFT)
        out["source"].append(source)
        out["n_tokens"].append(len(full_ids))
    return out


def tokenize_pt_batch(
    batch: dict[str, list], tokenizer: "PreTrainedTokenizer", max_len: int
) -> dict[str, list]:
    'Batched map fn: PT records → chunked pre-tokenized rows.\n\n\n    Each document is tokenized, terminated with EOS and **chunked** into\n    consecutive segments of at most ``max_len`` tokens (PT documents longer\n    than the budget are split, not dropped); ``n_prompt=0`` → the loss covers\n    every token.'
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError(
            'The tokenizer has no eos_token_id; PT rows cannot terminate documents.'
        )
    out = _empty_out()
    for response, source in zip(batch["response"], batch["source"]):
        ids = tokenize(tokenizer, response)
        if not ids:
            continue
        ids = [*ids, eos]
        for start in range(0, len(ids), max_len):
            chunk = ids[start : start + max_len]
            out["input_ids"].append(chunk)
            out["n_prompt"].append(0)
            out["kind"].append(KIND_PT)
            out["source"].append(source)
            out["n_tokens"].append(len(chunk))
    return out


def build_val_rows(
    records: Sequence[dict[str, Any]], tokenizer: "PreTrainedTokenizer"
) -> list[dict[str, Any]]:
    'Build the (small) SFT validation rows for chrF++ scoring.\n\n\n    Each row carries the rendered prompt ids (generation prompt appended),\n    the raw prompt text (diagnostics), the reference string and its token\n    count (the train stage filters by the generation budget). Boundary-broken\n    examples are dropped.'
    rows: list[dict[str, Any]] = []
    for rec in records:
        prompt_messages, _ = history_to_messages(
            rec["system"], rec["query"], rec["response"], rec["history"]
        )
        prompt_text = render(
            tokenizer, prompt_messages, add_generation_prompt=True
        )
        prompt_ids = tokenize(tokenizer, prompt_text)
        ref_ids = tokenize(tokenizer, rec["response"])
        if not prompt_ids or not ref_ids:
            continue
        rows.append(
            {
                "input_ids": prompt_ids,
                "prompt_text": prompt_text,
                "reference": rec["response"],
                "source": rec["source"],
                "n_reference_tokens": len(ref_ids),
            }
        )
    return rows


@dataclass
class CptCollator:
    'Pad a batch of pre-tokenized rows; rebuild labels from ``n_prompt``.'

    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for f in features:
            ids = [int(t) for t in f["input_ids"]]
            n_prompt = int(f.get("n_prompt") or 0)
            pad = max_len - len(ids)
            input_ids.append([*ids, *[self.pad_token_id] * pad])
            attention_mask.append([*[1] * len(ids), *[0] * pad])
            labels.append([*[-100] * n_prompt, *ids[n_prompt:], *[-100] * pad])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class MixedBatchSampler(Sampler[list[int]]):
    'Every mini-batch mixes SFT and PT rows in a fixed composition.\n\n\n    Args:\n        kinds: the ``kind`` column of the train dataset (``"sft"``/``"pt"``).\n        batch_size: rows per mini-batch (the micro-batch each device sees).\n        sft_per_batch: how many of them must be SFT (the rest are PT); must\n            leave at least one slot for each kind.\n        seed: RNG seed (repo convention: log it alongside the run).\n\n    Both pools are reshuffled every epoch (each ``iter()`` advances the\n    generator deterministically); the smaller pool cycles with reshuffles,\n    so it is naturally upsampled. ``len()`` = enough mini-batches to pass\n    over the LARGER pool exactly once.'

    def __init__(
        self,
        kinds: Sequence[str],
        batch_size: int,
        sft_per_batch: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.sft_idx = [i for i, k in enumerate(kinds) if k == KIND_SFT]
        self.pt_idx = [i for i, k in enumerate(kinds) if k == KIND_PT]
        if not self.sft_idx or not self.pt_idx:
            raise ValueError(
                'MixedBatchSampler requires both SFT and PT rows in the dataset.'
            )
        if batch_size < 2:
            raise ValueError(
                'batch_size must be ≥ 2 to mix kinds.\nbatch_size'
            )
        if not 1 <= sft_per_batch <= batch_size - 1:
            raise ValueError(
                f'sft_per_batch must be within [1, {batch_size - 1}'
            )
        self.batch_size = batch_size
        self.sft_per_batch = sft_per_batch
        self.pt_per_batch = batch_size - sft_per_batch
        self.generator = torch.Generator().manual_seed(int(seed))
        self.n_batches = max(
            math.ceil(len(self.pt_idx) / self.pt_per_batch),
            math.ceil(len(self.sft_idx) / self.sft_per_batch),
        )

    def _shuffled(self, pool: list[int]) -> list[int]:
        order = torch.randperm(len(pool), generator=self.generator).tolist()
        return [pool[j] for j in order]

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[list[int]]:
        sft_order = self._shuffled(self.sft_idx)
        pt_order = self._shuffled(self.pt_idx)
        sft_ptr = pt_ptr = 0
        for _ in range(self.n_batches):
            batch: list[int] = []
            for _ in range(self.sft_per_batch):
                if sft_ptr == len(sft_order):
                    sft_order = self._shuffled(self.sft_idx)
                    sft_ptr = 0
                batch.append(sft_order[sft_ptr])
                sft_ptr += 1
            for _ in range(self.pt_per_batch):
                if pt_ptr == len(pt_order):
                    pt_order = self._shuffled(self.pt_idx)
                    pt_ptr = 0
                batch.append(pt_order[pt_ptr])
                pt_ptr += 1
            yield batch
