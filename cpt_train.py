'CPT stage 2: continued pre-training on the mixed SFT+PT cache.\n\n\nRun AFTER ``cpt_tokenize.py`` (which builds the disk cache)\n``cpt_tokenize.py``\n\n    accelerate launch --multi_gpu --num_processes 10         --deepspeed --deepspeed_config_file deepspeed_zero2.json         cpt_train.py --model Qwen/Qwen2.5-7B-Instruct             --tokenizer anonymous/base-model --freeze-transformer             --cache-dir cpt_cache --output-dir checkpoints/cpt             --sft-share-per-batch 0.5 --patience 3 --seed 42\n\nKey properties\n\n* **Zero TRL / zero ``datasets`` API dependence**: plain\n  ``transformers.Trainer`` over a ``torch`` DataLoader of memory-mapped rows\n  (immune to the trl 0.15↔1.10 and datasets 3.2↔4.5 API drift between the\n  cluster nodes).\n* **Mixed mini-batches**: every mini-batch contains both SFT and PT rows\n  (:class:`~llm_training.cpt.data.MixedBatchSampler`, fixed per-batch\n  composition ``--sft-share-per-batch``); accelerate/DeepSpeed shard the\n  ready-made mixed micro-batches across ranks, so the guarantee survives\n  any GPU count and ``--grad-accum``.\n* **Foreign tokenizer** (``--tokenizer <model_id>``): the vocab-sized layers\n  are rebuilt per :mod:`llm_training.cpt.surgery` (wholesale copy when the\n  donor dimensions coincide; token-string remap / least-squares projection\n  otherwise; ``--freeze-transformer`` trains ONLY the two rebuilt layers\n  while gradients keep flowing through the frozen blocks).\n* **Validation = chrF++** on the SFT val rows (greedy generation vs the\n  reference, ``sacrebleu`` CHRF(word_order=2) — the same ``chrf_plus_plus``\n  the SFT warm-up uses); the best checkpoint by mean chrF++ is saved to\n  ``best/`` (plus ``last/`` at the end) and ``--patience`` stops the run\n  after that many consecutive evaluations without improvement.\n\n  ``transformers.Trainer``\n\nNode-specific notes (compute-node: CUDA 11.7, torch 2.5.1, transformers 4.49,\naccelerate 1.4.0, deepspeed 0.15.4; NO liger-kernel, NO flash-attn) /\naccelerate 1.4.0, deepspeed 0.15.4\n\n* default ``--attn sdpa`` (native torch SDPA backends, incl. the flash one,\n  need no external package); ``--liger`` raises a clear bilingual error when\n  the package is absent (it is an option for nodes that have it, e.g. compute-node);\n* without the fused linear+cross-entropy the ``[B, T, V]`` logits transient\n  (~1.2 GB bf16 + ~2.4 GB fp32 CE upcast per device at b=1, T=4096,\n  V≈145k) fits comfortably into 80 GB per A100 — keep ``--per-device-batch\n  1`` for very long ``--max-len``;\n* ``TrainingArguments`` kwargs are built adaptively by field introspection\n  (4.49 still has ``warmup_ratio``; a v5-era stack falls back to\n  ``warmup_steps`` as a float ratio; unknown fields are dropped).\n\n  ``--per-device-batch 1``\n* kwargs ``TrainingArguments``\n  ``warmup_steps``'

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any

from llm_training.runtime.checkpointing import (
    DEFAULT_MAX_SHARD_SIZE,
    sharded_checkpoint_saves,
)

logger = logging.getLogger("cpt_train")


def _seed_everything(seed: int) -> None:
    'Seed Python / NumPy / torch (CPU) RNGs; log the seed (repo convention).\n\nNumPy'
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a core dep
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            'CPT stage 2: continued pre-training on the pre-tokenized SFT+PT cache.'
        )
    )
    p.add_argument(
        "--model", default="anonymous/base-model",
        help='Model weights to train (transformer blocks stay; vocab-sized layers may be rebuilt).',
    )
    p.add_argument(
        "--tokenizer", default=None,
        help='Foreign tokenizer model id: the tokenizer AND the donor of the vocab-sized layers (must match cpt_tokenize.py).',
    )
    p.add_argument(
        "--cache-dir", default="cpt_cache",
        help='Disk cache built by cpt_tokenize.py.',
    )
    p.add_argument(
        "--freeze-transformer", action="store_true",
        help='Freeze everything except the embedding and classification layers (gradients still flow through the blocks).',
    )
    p.add_argument(
        "--output-dir", default="checkpoints/cpt",
        help='Where to save best/ and last/.',
    )
    p.add_argument(
        "--max-shard-size", default=DEFAULT_MAX_SHARD_SIZE, metavar="SIZE",
        help='pins the shard size across transformers versions: v5.2+ changed the library default to 50GB, which would write one monolithic ~30 GB file; also hard-checks bfloat16 before every save).',
    )
    p.add_argument(
        "--epochs", type=int, default=1,
        help='Number of epochs (an epoch = one pass over the LARGER pool).',
    )
    p.add_argument(
        "--max-steps", type=int, default=0,
        help='If >0, stop after N optimizer steps (overrides --epochs; smoke runs).',
    )
    p.add_argument(
        "--lr", type=float, default=1e-5,
        help='Learning rate.',
    )
    p.add_argument(
        "--per-device-batch", type=int, default=2,
        help='Rows per device mini-batch (the mixed-composition unit).',
    )
    p.add_argument(
        "--sft-share-per-batch", type=float, default=0.5,
        help='Share of SFT rows in every mini-batch (rest = PT).',
    )
    p.add_argument(
        "--grad-accum", type=int, default=8,
        help='Gradient accumulation steps.',
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help='RNG seed (logged).',
    )
    p.add_argument(
        "--optim", default="adamw_torch",
        help='Optimizer (bitsandbytes variants need the package).',
    )
    p.add_argument(
        "--attn", default="sdpa", choices=("sdpa", "flash_attention_2", "eager"),
        help='Attention implementation (sdpa needs no external package; flash_attention_2 needs flash-attn, absent on compute-node).',
    )
    p.add_argument(
        "--liger", action="store_true",
        help='Patch the model instance with Liger kernels (fused linear+cross-entropy; needs the liger-kernel package',
    )
    p.add_argument(
        "--eval-steps", type=int, default=0,
        help='Validate every N optimizer steps (0 = once per epoch).',
    )
    p.add_argument(
        "--patience", type=int, default=3,
        help='Early stopping: stop after N consecutive evaluations without chrF++ improvement.',
    )
    p.add_argument(
        "--val-max-new-tokens", type=int, default=512,
        help='Greedy generation budget per val example.',
    )
    p.add_argument(
        "--val-max-examples", type=int, default=0,
        help='Cap the val pool (0 = all; seeded subsample).',
    )
    p.add_argument(
        "--dataloader-num-workers", type=int, default=2,
        help='DataLoader workers per rank (rows are memory-mapped).',
    )
    return p


def _build_trainer_kwargs(
    training_args_cls: Any,
    *,
    output_dir: str,
    epochs: int,
    max_steps: int,
    lr: float,
    per_device_batch: int,
    grad_accum: int,
    optim: str,
    seed: int,
    dataloader_num_workers: int,
) -> dict[str, Any]:
    'Build TrainingArguments kwargs adaptive to the installed transformers.\n\n\n    Mirrors ``sft_warmup._build_sft_kwargs``: field introspection of the\n    installed class, ``warmup_ratio`` → ``warmup_steps`` (float ratio)\n    fallback for the v5 era, unknown kwargs dropped with a warning.'
    known = {f.name for f in dc_fields(training_args_cls)}
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "max_steps": max_steps if max_steps > 0 else -1,
        "learning_rate": lr,
        "per_device_train_batch_size": per_device_batch,
        "gradient_accumulation_steps": grad_accum,
        "lr_scheduler_type": "cosine",
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": 5,
        "save_strategy": "no",  # best/ and last/ are saved by the callback
        "report_to": [],
        "seed": seed,
        "remove_unused_columns": False,
        "optim": optim,
        "dataloader_num_workers": dataloader_num_workers,
    }
    if "warmup_ratio" in known:
        kwargs["warmup_ratio"] = 0.03
    elif "warmup_steps" in known:
        kwargs["warmup_steps"] = 0.03  # v5: float < 1 == ratio of steps
    else:
        logger.warning('TrainingArguments declares neither warmup_ratio nor warmup_steps; LR warmup is disabled.')
    unknown = sorted(set(kwargs) - known)
    if unknown:
        logger.warning("TrainingArguments dropped unsupported kwargs: %s", unknown)
        for key in unknown:
            kwargs.pop(key)
    return kwargs


def _apply_liger(model: Any) -> None:
    'Patch the model instance with Liger kernels (nodes that have them).\n\n\n    Qwen2-family only (Meno-Lite is a Qwen2 descendant) — the fused\n    linear+cross-entropy never materializes the ``[B, T, V]`` logits tensor.\n    Called AFTER the vocab surgery so the patched forward binds to the NEW\n    lm_head.'
    model_type = getattr(model.config, "model_type", "")
    if not model_type.startswith("qwen2"):
        raise SystemExit(
            f'--liger supports qwen2-family models, got model_type={model_type!r}'
        )
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2
    except ImportError as e:
        raise SystemExit(
            "--liger requires the 'liger-kernel' package (absent on compute-node): pip install -U liger-kernel, or drop the flag."
        ) from e
    apply_liger_kernel_to_qwen2(model=model)


def _exclusive_lock(path: Path):
    'Cross-process exclusive lock serializing the CPU-peak donor load.\n\n\n    Under a multi-rank launch every rank loads its own CPU copy of the\n    policy AND (for the vocab surgery) the donor model; letting the 10 donor\n    loads coincide would multiply the transient RAM peak. An flock on a\n    lockfile next to the output dir lets one rank at a time hold the donor;\n    on platforms without fcntl this degrades to a no-op (single-process\n    runs never block anyway).'
    import contextlib

    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-unix
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def _lock():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    return _lock()


def _kind_indices(kinds_column: Any, kind: str) -> list[int]:
    'Row indices of ``kind`` from an arrow string column (no Python objs).'
    import numpy as np
    import pyarrow.compute as pc

    mask = pc.equal(kinds_column, kind)
    return np.flatnonzero(mask.to_numpy(zero_copy_only=False)).tolist()


def filter_val_rows(
    rows: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    max_examples: int,
    seed: int,
) -> list[dict[str, Any]]:
    'Keep val rows whose reference fits the generation budget; optional cap.\n\n\n    Over-long references are dropped (regenerating them within the budget is\n    impossible, chrF++ against a truncated generation would understate the\n    model); ``max_examples > 0`` takes a seeded random subsample.'
    kept = [r for r in rows if int(r["n_reference_tokens"]) <= max_new_tokens]
    dropped = len(rows) - len(kept)
    if dropped:
        logger.info('Val: dropped %d rows with references longer than %d tokens.', dropped, max_new_tokens)
    if max_examples and max_examples > 0 and len(kept) > max_examples:
        rng = random.Random(f"{seed}:val-subsample")
        kept = rng.sample(kept, max_examples)
    return kept


def run_chrf_validation(
    model: Any,
    tokenizer: Any,
    val_rows: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    dump_path: Path | None = None,
    epoch: float | None = None,
    step: int | None = None,
) -> float:
    'Greedy-generate every val row and return the mean chrF++ vs reference.\n\n\n    Mirrors the SFT warm-up validation: the KV-cache is temporarily\n    re-enabled, the generation is decoded WITHOUT stripping special tokens\n    (``decode_generation`` — only the trailing stop tokens are removed) and\n    scored with ``chrf_plus_plus``. Each scored generation is appended to\n    ``dump_path`` for instant diagnosis.'
    import torch

    from llm_training.tokenizer_utils import decode_generation
    from llm_training.warmup.metrics import chrf_plus_plus

    scores: list[float] = []
    model.eval()
    prev_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = True
    device = next(model.parameters()).device
    dump_file = None
    try:
        if dump_path is not None:
            dump_file = open(dump_path, "a", encoding="utf-8")
        with torch.no_grad():
            for row in val_rows:
                input_ids = torch.tensor(
                    [list(row["input_ids"])], dtype=torch.long, device=device
                )
                attention = torch.ones_like(input_ids)
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
                gen = decode_generation(tokenizer, out[0][input_ids.shape[1]:])
                score = chrf_plus_plus(gen, str(row["reference"]))
                scores.append(score)
                if dump_file is not None:
                    dump_file.write(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "step": step,
                                "source": row["source"],
                                "generation": gen,
                                "score": score,
                                "reference": row["reference"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    dump_file.flush()
    finally:
        if dump_file is not None:
            dump_file.close()
        if prev_use_cache is not None:
            model.config.use_cache = prev_use_cache
        model.train()
    return sum(scores) / len(scores) if scores else 0.0


def _verify_cache(cache_dir: Path, tokenizer_id: str, vocab_size: int) -> dict:
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(
            f'No meta.json under {cache_dir}'
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("tokenizer") != tokenizer_id:
        raise SystemExit(
            f"Cache was tokenized for {meta.get('tokenizer')!r} but this run uses {tokenizer_id!r}"
        )
    if meta.get("vocab_size") != vocab_size:
        raise SystemExit(
            f"Cache vocab_size={meta.get('vocab_size')} != len(tokenizer)={vocab_size}"
        )
    return meta


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )
    args = build_arg_parser().parse_args(argv)
    logger.info("cpt_train: seed=%d", args.seed)
    _seed_everything(args.seed)

    try:
        import torch

        from datasets import load_from_disk
        from torch.utils.data import DataLoader
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainerCallback,
        )
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "CPT training needs 'torch', 'transformers' and 'datasets': pip install -e '.[train]'."
        ) from e

    from llm_training.cpt.data import CptCollator, MixedBatchSampler
    from llm_training.cpt.surgery import (
        apply_foreign_tokenizer,
        freeze_transformer_blocks,
    )

    tokenizer_id = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    cache_dir = Path(args.cache_dir)
    meta = _verify_cache(cache_dir, tokenizer_id, len(tokenizer))

    train_ds = load_from_disk(str(cache_dir / "train"))
    kinds_column = train_ds.data.column("kind")
    sft_idx = _kind_indices(kinds_column, "sft")
    pt_idx = _kind_indices(kinds_column, "pt")
    logger.info(
        "Cache: %d rows (%d SFT, %d PT), %d tokens; val=%d rows; "
        "max_len=%d.",
        len(train_ds), len(sft_idx), len(pt_idx),
        meta.get("counts", {}).get("total_tokens", -1),
        meta.get("counts", {}).get("val_rows", 0),
        meta.get("max_len", -1),
    )

    # torch_dtype is bf16 EXPLICITLY (not "auto"): the A100 cluster policy
    # requires every checkpoint in bfloat16, and "auto" would silently
    # follow the source checkpoint's config dtype (fp16/fp32 sources would
    # poison every save downstream; the tokenizer surgery preserves the
    # model dtype, so the guarantee survives the rebuild too).
    # torch_dtype
    
    # config.dtype
    
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation=args.attn,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    # The greedy chrF++ validation must not carry the checkpoint's sampling
    # defaults — silences the per-generate() "generation flags are not
    # valid" warning spam (greedy outputs are unaffected by the flags).
    
    
    
    from llm_training.runtime.generation_config import (
        unset_greedy_sampling_flags,
    )

    unset_greedy_sampling_flags(model)
    logger.info("Model loaded in %s.", model.dtype)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    surgery_report = None
    if args.tokenizer:
        logger.info('Acquiring the surgery lock (ranks serialize the CPU-side donor load)...')
        with _exclusive_lock(output_dir / ".surgery.lock"):
            surgery_report = apply_foreign_tokenizer(
                model, tokenizer, args.tokenizer, seed=args.seed
            )
    if args.freeze_transformer:
        freeze_transformer_blocks(model)
    if args.liger:
        _apply_liger(model)

    from transformers import TrainingArguments

    trainer_kwargs = _build_trainer_kwargs(
        TrainingArguments,
        output_dir=str(output_dir),
        epochs=args.epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        per_device_batch=args.per_device_batch,
        grad_accum=args.grad_accum,
        optim=args.optim,
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    collator = CptCollator(pad_token_id=pad_token_id)

    class _CptTrainer(Trainer):
        'Trainer with the mixed-batch dataloader (SFT+PT per mini-batch).'

        cpt_batch_sampler: Any = None
        cpt_num_workers: int = 0

        def get_train_dataloader(self):
            if self.cpt_batch_sampler is not None:
                dl = DataLoader(
                    self.train_dataset,
                    batch_sampler=self.cpt_batch_sampler,
                    collate_fn=self.data_collator,
                    num_workers=self.cpt_num_workers,
                    pin_memory=True,
                )
                return self.accelerator.prepare(dl)
            return super().get_train_dataloader()

    trainer = _CptTrainer(
        model=model,
        args=TrainingArguments(**trainer_kwargs),
        train_dataset=train_ds,
        data_collator=collator,
    )
    if sft_idx and pt_idx:
        batch_size = args.per_device_batch
        sft_per_batch = max(
            1, min(batch_size - 1, round(batch_size * args.sft_share_per_batch))
        )
        trainer.cpt_batch_sampler = MixedBatchSampler(
            kinds=train_ds.data.column("kind").to_pylist(),
            batch_size=batch_size,
            sft_per_batch=sft_per_batch,
            seed=args.seed,
        )
        trainer.cpt_num_workers = args.dataloader_num_workers
        logger.info('Mixed batches: %d SFT + %d PT rows per mini-batch, %d mini-batches per epoch.', sft_per_batch, batch_size - sft_per_batch, len(trainer.cpt_batch_sampler))
    else:
        logger.warning('Only one row kind in the cache (%s); the mixed-batch guarantee is impossible — falling back to the plain shuffled dataloader.', 'sft' if sft_idx else 'pt')

    class _ChrFValCallback(TrainerCallback):
        'Per-step/epoch chrF++ validation + best/ saving + patience stop.'

        def __init__(self) -> None:
            self.best = -1.0
            self.n_bad = 0
            self.has_val = bool(val_rows)

        def on_step_end(self, targs, state, control, **kwargs):
            if (
                self.has_val
                and args.eval_steps > 0
                and state.global_step > 0
                and state.global_step % args.eval_steps == 0
            ):
                self._run(targs, state, control, **kwargs)

        def on_epoch_end(self, targs, state, control, **kwargs):
            if self.has_val and args.eval_steps <= 0:
                self._run(targs, state, control, **kwargs)

        def _run(self, targs, state, control, **kwargs):
            payload: list[Any] = [None, False]
            if trainer.accelerator.is_main_process:
                m = kwargs.get("model") or trainer.model
                metric = run_chrf_validation(
                    m,
                    tokenizer,
                    val_rows,
                    max_new_tokens=args.val_max_new_tokens,
                    dump_path=output_dir / "val_generations.jsonl",
                    epoch=float(state.epoch or 0.0),
                    step=int(state.global_step),
                )
                if metric > self.best:
                    self.best = metric
                    self.n_bad = 0
                    best_dir = output_dir / "best"
                    logger.info(
                        "New best chrF++=%.4f → saving %s", metric, best_dir
                    )
                    with sharded_checkpoint_saves(trainer, args.max_shard_size):
                        trainer.save_model(str(best_dir))
                    tokenizer.save_pretrained(str(best_dir))
                else:
                    self.n_bad += 1
                stop = self.n_bad >= args.patience
                logger.info(
                    "CPT VAL step=%d epoch=%.2f chrf=%.4f best=%.4f "
                    "n_bad=%d/%d",
                    state.global_step, state.epoch or 0.0, metric, self.best,
                    self.n_bad, args.patience,
                )
                if stop:
                    logger.info('Early stopping: %d evaluations without improvement.', self.n_bad)
                payload = [metric, stop]
            if trainer.accelerator.num_processes > 1:
                payload = trainer.accelerator.broadcast_object_list(payload)
            _, stop = payload
            if payload[0] is None:
                return
            if stop:
                control.should_training_stop = True

    val_rows_raw = load_from_disk(str(cache_dir / "val")).to_list()
    val_rows = filter_val_rows(
        val_rows_raw,
        max_new_tokens=args.val_max_new_tokens,
        max_examples=args.val_max_examples,
        seed=args.seed,
    )
    if not val_rows:
        logger.warning('Val pool is empty: best/ will not be selected (last/ only), early stopping disabled.')

    trainer.add_callback(_ChrFValCallback())

    if trainer.accelerator.is_main_process:
        run_meta = {
            "args": vars(args),
            "cache_counts": meta.get("counts", {}),
            "surgery": surgery_report.to_dict() if surgery_report else None,
        }
        (output_dir / "run_meta.json").write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2), "utf-8"
        )
        if surgery_report is not None:
            (output_dir / "surgery_report.json").write_text(
                json.dumps(surgery_report.to_dict(), ensure_ascii=False,
                           indent=2),
                "utf-8",
            )

    trainer.train()

    if trainer.accelerator.is_main_process:
        last_dir = output_dir / "last"
        with sharded_checkpoint_saves(trainer, args.max_shard_size):
            trainer.save_model(str(last_dir))
        tokenizer.save_pretrained(str(last_dir))
        logger.info('CPT done. Checkpoints: %s, %s.', output_dir / 'best', last_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
