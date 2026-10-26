'Tests for the batched anchor generation (LEFT padding, packing, trimming).\n\n\nOffline with fakes: a char-tokenizing fake tokenizer with real left/right\npadding semantics and a scripted fake model. Covers the pure packing\n(determinism, KV token budget, row cap, the legacy escape), the\nleft-padding mechanics (``padding_side`` temporarily flipped and restored,\npad rows trimmed at the first pad token, results re-sorted into the\nORIGINAL example order) and the model-state contract (``eval``/``train``,\n``use_cache`` restoration).\n``padding_side``\n``use_cache``).'

from __future__ import annotations

from types import SimpleNamespace

from llm_training.runtime.anchor_generation import (
    DEFAULT_ANCHOR_GEN_BATCH_TOKENS,
    DEFAULT_ANCHOR_GEN_MAX_BATCH,
    generate_anchor_responses,
    plan_generation_batches,
)


class _FakeIds:
    'Indexable (B, L) ids over plain lists.'

    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.rows), len(self.rows[0]) if self.rows else 0)

    def __getitem__(self, i: int) -> list[int]:
        return self.rows[i]


class _FakeBatch(dict):
    'Tokenizer batch: a mapping (for ``**inputs``) with a no-op ``.to()``.'

    def __init__(self, input_ids: _FakeIds) -> None:
        super().__init__(input_ids=input_ids)

    def to(self, device: str) -> "_FakeBatch":
        return self


class _FakeTokenizer:
    'Char-tokenizes texts, pads by ``padding_side``, scripts generations.'

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, gens_by_len: dict[int, list[int]] | None = None) -> None:
        self.padding_side = "right"
        self._gens_by_len = gens_by_len or {}
        self.padding_sides_seen: list[str] = []

    def _encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def __call__(
        self,
        texts,
        return_tensors: str | None = None,
        padding: bool = False,
        add_special_tokens: bool = False,
    ) -> _FakeBatch:
        if isinstance(texts, str):
            texts = [texts]
        rows = [self._encode(t) for t in texts]
        if padding:
            self.padding_sides_seen.append(self.padding_side)
            width = max(len(r) for r in rows)
            if self.padding_side == "left":
                rows = [
                    [self.pad_token_id] * (width - len(r)) + r for r in rows
                ]
            else:
                rows = [
                    r + [self.pad_token_id] * (width - len(r)) for r in rows
                ]
        return _FakeBatch(_FakeIds(rows))

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(chr(t) for t in ids)

    def gen_for(self, prompt_len: int) -> list[int]:
        return list(self._gens_by_len[prompt_len])


class _FakeOutput:
    def __init__(self, rows: list[list[int]]) -> None:
        self._rows = rows

    def __getitem__(self, i: int) -> list[int]:
        return self._rows[i]


class _FakeModel:
    def __init__(self, tokenizer: _FakeTokenizer) -> None:
        self._tokenizer = tokenizer
        self.config = SimpleNamespace(use_cache=False)
        self.device = "cpu"
        self.modes: list[str] = []
        self.batch_sizes: list[int] = []
        self.left_padded_rows: list[bool] = []

    def eval(self) -> None:
        self.modes.append("eval")

    def train(self) -> None:
        self.modes.append("train")

    def generate(
        self,
        input_ids: _FakeIds | None = None,
        attention_mask=None,
        pad_token_id: int | None = None,
        **kwargs,
    ) -> _FakeOutput:
        rows = input_ids.rows if input_ids is not None else []
        self.batch_sizes.append(len(rows))
        out_rows: list[list[int]] = []
        for row in rows:
            real_len = sum(1 for t in row if t != pad_token_id)
            self.left_padded_rows.append(row[0] == pad_token_id)
            out_rows.append(list(row) + self._tokenizer.gen_for(real_len))
        return _FakeOutput(out_rows)


def _render(ex: SimpleNamespace, tokenizer) -> str:
    return ex.prompt_text


def _examples(*texts: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(prompt_text=t) for t in texts]


class TestPlanGenerationBatches:
    def test_budget_packing_sorted_desc(self):
        # (10+10)*3 would exceed 50 → the long pair forms its own package;
        # the short triple fits: (3+10)*3 = 39 <= 50.
        
        
        batches = plan_generation_batches(
            [10, 9, 3, 3, 1], max_new_tokens=10, batch_tokens=50, max_batch=8,
        )
        assert batches == [[0, 1], [2, 3, 4]]

    def test_row_cap(self):
        batches = plan_generation_batches(
            [5, 5, 5, 5, 5], max_new_tokens=1, batch_tokens=1000, max_batch=2,
        )
        assert batches == [[0, 1], [2, 3], [4]]

    def test_ties_keep_original_index_order(self):
        batches = plan_generation_batches(
            [3, 3, 3], max_new_tokens=1, batch_tokens=1000, max_batch=8,
        )
        assert batches == [[0, 1, 2]]

    def test_legacy_escape_singletons_in_original_order(self):
        # batch_tokens <= 0 (and max_batch <= 1) = per-example packages in
        # the ORIGINAL order — token inputs identical to the former loop.
        # batch_tokens <= 0
        
        assert plan_generation_batches(
            [1, 9, 3], max_new_tokens=10, batch_tokens=0, max_batch=8,
        ) == [[0], [1], [2]]
        assert plan_generation_batches(
            [1, 9, 3], max_new_tokens=10, batch_tokens=1000, max_batch=1,
        ) == [[0], [1], [2]]

    def test_defaults(self):
        assert DEFAULT_ANCHOR_GEN_BATCH_TOKENS == 65536
        assert DEFAULT_ANCHOR_GEN_MAX_BATCH == 8


class TestGenerateAnchorResponses:
    @staticmethod
    def _ids(text: str) -> list[int]:
        return [ord(c) for c in text]

    def _wire(self, prompts: list[str], gens: list[list[int]]):
        'Unique prompt lengths; scripted generations keyed by length.'
        gens_by_len = {len(p): g for p, g in zip(prompts, gens)}
        tokenizer = _FakeTokenizer(gens_by_len)
        model = _FakeModel(tokenizer)
        return tokenizer, model

    def test_original_order_pad_trim_and_state_restored(self):
        # Prompts of different lengths; every scripted generation embeds a
        # finished-row pad tail that the trim must cut. The packing sorts by
        # length desc, but the OUTPUT must come back in the original order.
        
        
        
        prompts = ["aaa", "a", "bb"]
        tokenizer, model = self._wire(
            prompts,
            [
                self._ids("X0") + [0, 0],  # finished: padded after EOS
                self._ids("Y"),            # unfinished: no pads
                self._ids("Z") + [0],      # finished: one pad
            ],
        )
        out = generate_anchor_responses(
            model, tokenizer, _examples(*prompts),
            render_fn=_render, max_new_tokens=8,
        )
        assert out == ["X0", "Y", "Z"]
        # One package of 3 rows (the budget admits all), LEFT padded: the
        # shortest row starts with a pad token.
        
        
        assert model.batch_sizes == [3]
        assert tokenizer.padding_sides_seen == ["left"]
        assert any(model.left_padded_rows)
        assert tokenizer.padding_side == "right"  # restored in finally
        assert model.modes == ["eval", "train"]
        assert model.config.use_cache is False  # restored in finally

    def test_legacy_mode_singletons(self):
        prompts = ["aaa", "a", "bb"]
        tokenizer, model = self._wire(
            prompts, [self._ids("X0"), self._ids("Y"), self._ids("Z")],
        )
        out = generate_anchor_responses(
            model, tokenizer, _examples(*prompts),
            render_fn=_render, max_new_tokens=8, batch_tokens=0,
        )
        assert out == ["X0", "Y", "Z"]
        # Per-example packages in the ORIGINAL order (not length-sorted).
        
        assert model.batch_sizes == [1, 1, 1]
        assert not any(model.left_padded_rows)

    def test_budget_splits_into_packages_keeps_order(self):
        prompts = ["aaaa", "aaa", "aa", "a"]
        tokenizer, model = self._wire(
            prompts,
            [self._ids(c) for c in ("A", "B", "C", "D")],
        )
        # (4 + 2) * 1 = 6 <= 12, but (4 + 2) * 2 = 12 <= 12 and
        # (4 + 2) * 3 = 18 > 12 → packages of at most 2 rows.
        out = generate_anchor_responses(
            model, tokenizer, _examples(*prompts),
            render_fn=_render, max_new_tokens=2, batch_tokens=12,
        )
        assert out == ["A", "B", "C", "D"]
        assert model.batch_sizes == [2, 2]

    def test_empty_pool(self):
        tokenizer = _FakeTokenizer({})
        model = _FakeModel(tokenizer)
        assert (
            generate_anchor_responses(
                model, tokenizer, [], render_fn=_render, max_new_tokens=8,
            )
            == []
        )
        assert model.modes == []  # nothing touched on an empty pool
