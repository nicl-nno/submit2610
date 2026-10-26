'Tests for the CPT data layer (offline; real Meno-Lite tokenizer fixture).'

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from llm_training.cpt.data import (
    CptCollator,
    MixedBatchSampler,
    build_val_rows,
    iter_pt_records,
    iter_sft_records,
    tokenize_pt_batch,
    tokenize_sft_batch,
)
from llm_training.tokenizer_utils import render, tokenize


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def _sft_record(query: str = "Что делать?", response: str = "Ответ.",
                system: str = "Ты - помощник.",
                history: list | None = None) -> dict:
    return {
        "system": system, "query": query, "response": response,
        "history": history or [],
    }


class TestRecordIterators:
    def test_sft_records_normalized(self, tmp_path):
        path = _write_jsonl(tmp_path / "s.jsonl", [
            _sft_record(history=[["u1", "a1"], ["bad", ]]),
            _sft_record(response="   "),
            {"query": "no response"},
        ])
        records = list(iter_sft_records([path]))
        assert len(records) == 1
        rec = records[0]
        assert rec["history"] == [["u1", "a1"]]
        assert rec["source"].endswith("#0")

    def test_pt_records_skip_empty(self, tmp_path):
        path = _write_jsonl(tmp_path / "p.jsonl", [
            {"response": "текст"},
            {"response": ""},
        ])
        records = list(iter_pt_records([path]))
        assert len(records) == 1
        assert records[0]["response"] == "текст"


class TestTokenizeSftBatch:
    def test_row_shape_and_mask_boundary(self, tmp_path, tokenizer):
        path = _write_jsonl(tmp_path / "s.jsonl", [_sft_record()])
        rec = next(iter(iter_sft_records([path])))
        batch = {k: [rec[k]] for k in
                 ("system", "query", "response", "history", "source")}
        rows = tokenize_sft_batch(batch, tokenizer, max_len=4096)
        assert len(rows["input_ids"]) == 1
        assert rows["kind"] == ["sft"]
        # The differential render: n_prompt == tokens of the rendered prompt
        # with the generation prompt appended; input_ids = prompt + reference.
        
        
        from llm_training.loaders.base import history_to_messages
        prompt_messages, reference_messages = history_to_messages(
            rec["system"], rec["query"], rec["response"], rec["history"]
        )
        full_messages = [*prompt_messages, *reference_messages]
        prompt_ids = tokenize(tokenizer, render(
            tokenizer, prompt_messages, add_generation_prompt=True
        ))
        full_ids = tokenize(tokenizer, render(
            tokenizer, full_messages, add_generation_prompt=False
        ))
        assert rows["n_prompt"][0] == len(prompt_ids)
        assert rows["input_ids"][0] == full_ids
        assert full_ids[:len(prompt_ids)] == prompt_ids

    def test_too_long_dropped(self, tmp_path, tokenizer):
        path = _write_jsonl(tmp_path / "s.jsonl", [
            _sft_record(query="в" * 400, response="р" * 400)
        ])
        rec = next(iter(iter_sft_records([path])))
        batch = {k: [rec[k]] for k in
                 ("system", "query", "response", "history", "source")}
        rows = tokenize_sft_batch(batch, tokenizer, max_len=16)
        assert rows["input_ids"] == []


class TestTokenizePtBatch:
    def test_chunking_and_eos(self, tokenizer):
        text = "Кабы не было зимы в городах и сёлах."
        ids = tokenize(tokenizer, text)
        max_len = 4
        rows = tokenize_pt_batch(
            {"response": [text], "source": ["p.jsonl#0"]},
            tokenizer, max_len=max_len,
        )
        expected_total = len(ids) + 1  # + EOS
        assert sum(rows["n_tokens"]) == expected_total
        assert all(len(chunk) <= max_len for chunk in rows["input_ids"])
        assert rows["input_ids"][-1][-1] == tokenizer.eos_token_id
        assert rows["n_prompt"] == [0] * len(rows["input_ids"])
        assert set(rows["kind"]) == {"pt"}

    def test_short_doc_single_chunk(self, tokenizer):
        rows = tokenize_pt_batch(
            {"response": ["ок"], "source": ["p#0"]}, tokenizer, max_len=128
        )
        assert len(rows["input_ids"]) == 1


class TestMixedBatchSampler:
    def test_every_batch_mixes_kinds(self):
        kinds = ["sft"] * 7 + ["pt"] * 23
        sampler = MixedBatchSampler(kinds, batch_size=4, sft_per_batch=1,
                                    seed=42)
        batches = list(sampler)
        assert len(batches) == max(math.ceil(23 / 3), math.ceil(7 / 1))
        for batch in batches:
            in_batch = [kinds[i] for i in batch]
            assert in_batch.count("sft") == 1
            assert in_batch.count("pt") == 3
        # The smaller SFT pool is revisited across the epoch (upsampling).
        
        sft_seen = [i for b in batches for i in b if kinds[i] == "sft"]
        assert len(sft_seen) > 7

    def test_same_seed_same_sequence(self):
        kinds = ["sft"] * 5 + ["pt"] * 11
        a = MixedBatchSampler(kinds, 4, 1, seed=7)
        b = MixedBatchSampler(kinds, 4, 1, seed=7)
        assert list(a) == list(b)

    def test_epoch_reshuffle(self):
        kinds = ["sft"] * 7 + ["pt"] * 21
        sampler = MixedBatchSampler(kinds, 4, 1, seed=7)
        first = [i for b in iter(sampler) for i in b]
        second = [i for b in iter(sampler) for i in b]
        assert first != second  # reshuffled

    def test_single_kind_rejected(self):
        with pytest.raises(ValueError):
            MixedBatchSampler(["sft"] * 10, 4, 1, seed=42)

    def test_bad_composition_rejected(self):
        kinds = ["sft"] * 4 + ["pt"] * 4
        with pytest.raises(ValueError):
            MixedBatchSampler(kinds, 4, 0, seed=42)
        with pytest.raises(ValueError):
            MixedBatchSampler(kinds, 4, 4, seed=42)


class TestCollator:
    def test_padding_and_labels(self):
        collator = CptCollator(pad_token_id=0)
        rows = [
            {"input_ids": [5, 6, 7, 8], "n_prompt": 2},
            {"input_ids": [9, 10], "n_prompt": 0},
        ]
        out = collator(rows)
        assert out["input_ids"].shape == (2, 4)
        assert out["input_ids"][1].tolist() == [9, 10, 0, 0]
        assert out["attention_mask"][1].tolist() == [1, 1, 0, 0]
        # Prompt part and padding are masked; content keeps its ids.
        
        assert out["labels"][0].tolist() == [-100, -100, 7, 8]
        assert out["labels"][1].tolist() == [9, 10, -100, -100]

    def test_output_is_long_tensors(self):
        out = CptCollator(pad_token_id=0)(
            [{"input_ids": [1, 2], "n_prompt": 1}]
        )
        for key in ("input_ids", "attention_mask", "labels"):
            assert out[key].dtype == torch.long


class TestValRows:
    def test_build_val_rows(self, tmp_path, tokenizer):
        path = _write_jsonl(tmp_path / "v.jsonl", [_sft_record()])
        rec = next(iter(iter_sft_records([path])))
        rows = build_val_rows([rec], tokenizer)
        assert len(rows) == 1
        row = rows[0]
        assert row["input_ids"] == tokenize(tokenizer, row["prompt_text"])
        assert row["reference"] == rec["response"]
        assert row["n_reference_tokens"] == len(tokenize(tokenizer, rec["response"]))
