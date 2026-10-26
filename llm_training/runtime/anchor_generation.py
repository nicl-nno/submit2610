"Anchor-pool generation for judge ctrl-points: batched HF ``.generate``.\n\n\nUntil 2026-09-25 every anchor generated the val pool ONE example at a time\n(greedy, EOS-stopped, the KV-cache temporarily on). Single-stream decode of a\n14B-class model is memory-bandwidth bound: a package of N left-padded\nsequences costs nearly the same WALL TIME as one sequence, so the anchor\ngeneration (hundreds of examples, up to ``max_new_tokens`` each) dominated\nthe ctrl-point latency ahead of the judge scoring. This module packs the\npool into deterministic length-bucketed packages under a KV-cache token\nbudget and generates them with HF ``.generate`` on the TRAINING model — the\nexact current weights, no vLLM weight-sync games (the original design\nconstraint, kept).\n\nShared by ``grpo_train.run_anchor_scoring`` and\n``sft_train.run_anchor_scoring``: the twin doctrine keeps the two aggregate\nfunctions as deliberate copies, but the padding/packing logic is exactly the\nerror-prone part that must live in ONE tested place.\n\nPacking invariants\n- **Deterministic\n  length desc, original index asc) and packed greedily — no RNG; the same\n  pool yields the same packages on every anchor and every run.\n- **Budget\n  batch_tokens`` — a conservative KV-cache bound (the worst case assumes\n  every row generates the FULL budget; greedy generation stops at EOS, so\n  the real transient is smaller). For a 14B Qwen2.5-class model the KV cache\n  is ~192 KiB/token, so the default 65536 bounds the transient at ~12 GB —\n  it fits next to the LoRA training peak on both target stacks.\n- **Cap\n  padding waste grows with the spread).\n- **Legacy escape\n  packs every example alone — token inputs identical to the former\n  one-at-a-time loop.\n\nLeft-padding mechanics\n- the tokenizer's ``padding_side`` is set to ``left`` for the packages and\n  restored in a ``finally`` block (training collators rely on the tokenizer\n  default — nothing may leak out of an anchor);\n- with left padding the generated tokens sit at the TAIL of every row:\n  ``out[row][padded_prompt_len:]``;\n- each row is trimmed at its first pad token (``generate`` pads finished\n  rows after EOS; ``decode_generation`` additionally strips\n  ``<|im_end|>``/``<|endoftext|>`` defensively, so the ``<tool_call>``\n  envelope survives intact);\n- results are returned in the ORIGINAL example order — the requests list and\n  the anchor aggregates never see the packing order.\n\nNumeric caveat\nbit-identical to single-stream ones, so a greedy token can occasionally flip\non a near-tie. Anchor series produced with packages are NOT token-identical\nto the batch=1 series; inside one run every anchor (including the step-0\nbaseline) uses the same packing, so the series stays self-consistent.\nEscape: ``--anchor-gen-batch-tokens 0``."

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

#: Default KV-cache token budget of one anchor-generation package:
#: (padded prompt + full generation budget) x rows <= 65536 tokens
#: (~12 GB of KV for a 14B Qwen2.5-class model at ~192 KiB/token).


DEFAULT_ANCHOR_GEN_BATCH_TOKENS = 65536

#: Hard row cap of one package (prefill waste grows with the spread).

DEFAULT_ANCHOR_GEN_MAX_BATCH = 8

__all__ = [
    "DEFAULT_ANCHOR_GEN_BATCH_TOKENS",
    "DEFAULT_ANCHOR_GEN_MAX_BATCH",
    "generate_anchor_responses",
    "plan_generation_batches",
]


def plan_generation_batches(
    lengths: Sequence[int],
    *,
    max_new_tokens: int,
    batch_tokens: int,
    max_batch: int,
) -> list[list[int]]:
    'Pack example indices into budgeted generation packages (pure, no RNG).\n\n\n    Indices are sorted by (length desc, original index asc) and packed\n    greedily: a new package starts whenever adding one more row would exceed\n    ``max_batch`` rows or the token budget\n    ``(padded_len + max_new_tokens) * rows > batch_tokens``. The budget uses\n    the FULL generation cap for every row — the conservative KV-cache worst\n    case. ``batch_tokens <= 0`` or ``max_batch <= 1`` is the legacy escape:\n    every example alone, in the ORIGINAL order (token inputs identical to\n    the former one-at-a-time loop).'
    n = len(lengths)
    if batch_tokens <= 0 or max_batch <= 1:
        return [[i] for i in range(n)]
    order = sorted(range(n), key=lambda i: (-int(lengths[i]), i))
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for i in order:
        ln = int(lengths[i])
        if cur and (
            len(cur) >= max_batch
            or (max(cur_max, ln) + max_new_tokens) * (len(cur) + 1)
            > batch_tokens
        ):
            batches.append(cur)
            cur = []
            cur_max = 0
        cur.append(i)
        cur_max = max(cur_max, ln)
    if cur:
        batches.append(cur)
    return batches


def _trim_at_pad(ids: Any, pad_id: Any) -> Any:
    'Cut a generated row at its first pad token (finished-row padding).\n\n\n    ``generate`` pads finished rows with ``pad_token_id`` after EOS while the\n    rest of the package keeps decoding; the pads must never reach the decode.\n    Works for torch tensor rows and plain lists alike (duck-typed iteration).'
    if pad_id is None:
        return ids
    for j, t in enumerate(ids):
        if t == pad_id:
            return ids[:j]
    return ids


def generate_anchor_responses(
    model: Any,
    tokenizer: Any,
    examples: Sequence[Any],
    *,
    render_fn: Callable[[Any, Any], str],
    max_new_tokens: int,
    batch_tokens: int = DEFAULT_ANCHOR_GEN_BATCH_TOKENS,
    max_batch: int = DEFAULT_ANCHOR_GEN_MAX_BATCH,
) -> list[str]:
    'Generate anchor responses on the training model; ORIGINAL-order output.\n\n\n    Greedy HF ``.generate`` over LEFT-padded, length-bucketed packages (see\n    :func:`plan_generation_batches`); the KV-cache is temporarily re-enabled\n    (``use_cache`` is ``False`` during training for gradient-checkpointing\n    compatibility) and every touched state — ``model.eval()/train()``,\n    ``use_cache``, ``padding_side`` — is restored in a ``finally`` block.\n    Every row is decoded via\n    :func:`llm_training.tokenizer_utils.decode_generation` so the\n    ``<tool_call>`` envelope survives (it consists of added special tokens).\n    :func:`plan_generation_batches`); KV\n    ``decode_generation``'
    import torch

    from llm_training.tokenizer_utils import decode_generation

    if not examples:
        return []

    prompt_texts: list[str] = [render_fn(ex, tokenizer) for ex in examples]
    encoded = tokenizer(prompt_texts, add_special_tokens=False)
    lengths = [len(ids) for ids in encoded["input_ids"]]
    batches = plan_generation_batches(
        lengths,
        max_new_tokens=max_new_tokens,
        batch_tokens=batch_tokens,
        max_batch=max_batch,
    )
    # NB: an explicit None check, NOT ``or`` — a legitimate pad id of 0 is
    # falsy and ``0 or eos`` would silently substitute the wrong token (the
    # latent bug of the former one-at-a-time loop, harmless there only
    # because the real Qwen tokenizer has a large pad id).
    
    # ``0 or eos``
    
    # Qwen).
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    generations: list[str | None] = [None] * len(examples)
    model.eval()
    prev_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = True
    prev_padding_side = getattr(tokenizer, "padding_side", None)
    if prev_padding_side is not None:
        tokenizer.padding_side = "left"
    try:
        with torch.no_grad():
            for batch in batches:
                texts = [prompt_texts[i] for i in batch]
                inputs = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ).to(model.device)
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                )
                padded_len = inputs["input_ids"].shape[1]
                for row, i in enumerate(batch):
                    gen_ids = _trim_at_pad(out[row][padded_len:], pad_id)
                    generations[i] = decode_generation(tokenizer, gen_ids)
    finally:
        if prev_padding_side is not None:
            tokenizer.padding_side = prev_padding_side
        if prev_use_cache is not None:
            model.config.use_cache = prev_use_cache
        model.train()

    rows = [len(b) for b in batches]
    logger.info(
        "Anchor generation: %d examples in %d package(s) "
        "(mean %.2f rows, max %d rows, budget=%d tokens)",
        len(examples), len(batches),
        sum(rows) / len(rows), max(rows), batch_tokens,
    )
    return [g if g is not None else "" for g in generations]
