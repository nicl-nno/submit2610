'Checkpoint-save policy: pinned ``max_shard_size`` + a bfloat16 guarantee.\n\n\nNeither transformers 4.x nor v5 forwards ``max_shard_size`` through\n``Trainer.save_model``/``Trainer._save`` — both hard-code the inner\n``model.save_pretrained(output_dir, state_dict=...)`` call without it, and\nthe upstream default diverged across versions (``"5GB"`` in 4.x, ``"50GB"``\nsince v5.2). On the cluster stack (transformers v5.2+) a ~14B bf16 model\nwould therefore be saved as a single ~30 GB monolithic file, while the dev\nstack (4.55) shards at 5 GB. :func:`sharded_checkpoint_saves` temporarily\nwraps the unwrapped model\'s ``save_pretrained`` so that every save inside\nthe block — including ``Trainer._save`` under DeepSpeed/FSDP, which calls\n``save_pretrained`` on the same underlying model — passes an explicit\n``max_shard_size``. All Trainer semantics survive (the DeepSpeed state-dict\ngathering, ``training_args.bin``, the processing-class save).\n\nThe same context manager enforces the cluster checkpoint policy: **every\nsaved checkpoint must be bfloat16** (the A100 nodes train in bf16; fp16\nloses range, fp32 doubles disk). The check is a hard error raised BEFORE any\nfile is written — a wrong-dtype checkpoint is never persisted. The scripts\nguarantee it upstream by loading with ``torch_dtype=torch.bfloat16``, so the\ncheck is defence in depth (e.g. against a future surgery path that rebuilds\na layer in the wrong dtype).\n\n``Trainer.save_model``/``Trainer._save``\n``model.save_pretrained(output_dir, state_dict=...)``\n:func:`sharded_checkpoint_saves`\n``Trainer._save``\n\n``torch_dtype=torch.bfloat16``'

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

#: The default shard size for saved checkpoints. The string form is parsed
#: by transformers itself (``convert_file_size_to_int``: "5GB" = 5·10⁹
#: bytes); pass CLI values through verbatim.

# : transformers (``convert_file_size_to_int``: "5GB" = 5·10⁹

DEFAULT_MAX_SHARD_SIZE = "5GB"


def unwrap_trainer_model(trainer: Any) -> Any:
    "Return the core (unwrapped) model behind a Trainer-like object.\n\n\n    ``Trainer._save`` ultimately calls ``save_pretrained`` on the model\n    returned by ``accelerator.unwrap_model(trainer.model)`` (DeepSpeed\n    engine → the underlying ``PreTrainedModel``). Test fakes carry no\n    ``accelerator.unwrap_model``; then ``trainer.model`` is returned as-is\n    (and may be anything — the callers treat a missing ``save_pretrained``\n    as a no-op).\n\n    ``Trainer._save``\n    ``accelerator.unwrap_model``\n    no-op'"
    model = getattr(trainer, "model", None)
    accel = getattr(trainer, "accelerator", None)
    unwrap = getattr(accel, "unwrap_model", None)
    if callable(unwrap) and model is not None:
        return unwrap(model)
    return model


def assert_bfloat16_model(model: Any) -> None:
    'Hard-fail unless every parameter of ``model`` is bfloat16.\n\n\n    A wrong-dtype checkpoint must never be written: on the A100 cluster all\n    checkpoints are bf16 (the nodes train in bf16). Raises :exc:`TypeError`\n    naming the offending tensors; silently passes when the object carries no\n    ``named_parameters`` (test fakes, non-neural models).'
    # torch is imported lazily: the module stays importable in minimal
    # environments without the [train] extra (the parser tests need only
    # DEFAULT_MAX_SHARD_SIZE).
    
    
    # DEFAULT_MAX_SHARD_SIZE).
    import torch

    named = getattr(model, "named_parameters", None)
    if not callable(named):
        return
    bad = [name for name, p in named() if p.dtype != torch.bfloat16]
    if bad:
        raise TypeError(
            "Refusing to save a checkpoint with non-bfloat16 parameters "
            f"({len(bad)} tensors, first: {bad[:3]}). Every checkpoint must "
            "be bfloat16 (A100 cluster policy). Load the model with "
            "torch_dtype=torch.bfloat16.\n"
            "Отказ сохранять контрольную точку: параметры не в bfloat16 "
            f"({len(bad)} тензоров, первые: {bad[:3]}). Каждая контрольная "
            "точка обязана быть bfloat16 (дисциплина кластера A100). "
            "Загружайте модель с torch_dtype=torch.bfloat16."
        )


@contextlib.contextmanager
def sharded_checkpoint_saves(
    trainer: Any, max_shard_size: str | int | None = DEFAULT_MAX_SHARD_SIZE
) -> Iterator[None]:
    'Pin ``max_shard_size`` (and enforce bfloat16) for saves inside.\n\n\nUsage\n\n        with sharded_checkpoint_saves(trainer, args.max_shard_size):\n            trainer.save_model(str(best_dir))\n\n    Temporarily sets an instance-level ``save_pretrained`` wrapper on the\n    unwrapped model so that ``max_shard_size`` is passed (via\n    ``kwargs.setdefault`` — an explicit value from the caller, if a future\n    transformers ever forwards one, wins) to every save performed inside the\n    block; on exit the instance attribute is removed and the class method is\n    restored. The bfloat16 check runs on entry, BEFORE any save is written.\n    No-op (except the dtype check) when ``max_shard_size`` is falsy or the\n    model has no callable ``save_pretrained`` — the property that keeps the\n    offline test fakes (``object()`` models) working unchanged.\n\n``save_pretrained``'
    model = unwrap_trainer_model(trainer)
    assert_bfloat16_model(model)
    save_pretrained = getattr(model, "save_pretrained", None)
    patched = False
    if model is not None and max_shard_size and callable(save_pretrained):
        def _save_pretrained(save_directory, *args, **kwargs):
            kwargs.setdefault("max_shard_size", max_shard_size)
            return save_pretrained(save_directory, *args, **kwargs)

        model.save_pretrained = _save_pretrained
        patched = True
    try:
        yield
    finally:
        if patched:
            try:
                del model.save_pretrained
            except AttributeError:
                # Somebody removed our wrapper mid-block — restore the
                # original bound method rather than leaving a hole.
                
                
                model.save_pretrained = save_pretrained


def merge_adapter_into_base(
    base: Any,
    adapter_dir: str | Path,
    out_dir: str | Path,
    tokenizer: Any | None = None,
    max_shard_size: str | int | None = DEFAULT_MAX_SHARD_SIZE,
    oplora: bool = False,
    oplora_k: int = 128,
) -> Any:
    'Load a peft adapter onto an already-loaded base, merge, save full bf16.\n\n\n    The offline counterpart of the post-training merge in ``sft_train.py``\n    (``best-merged/``): wraps ``base`` with ``PeftModel.from_pretrained``,\n    optionally re-applies the OPLoRA invariant, calls ``merge_and_unload()``,\n    hard-checks bfloat16 and saves through the pinned shard size. With\n    ``oplora=False`` this is a plain peft merge — correct for adapters saved\n    AFTER an OPLoRA repair (the sft_train.py callbacks repair at every step\n    end BEFORE saving, so the persisted ``B @ A`` already lies in the\n    orthogonal complement and the plain merge produces exactly the projected\n    update). ``oplora=True`` additionally rebuilds the projection from the\n    frozen base weights (a deterministic truncated SVD — identical bases to\n    training time) and re-projects the loaded adapter: a mathematical no-op\n    for post-repair checkpoints, and a provenance guarantee for adapters of\n    unknown origin.\n\n    ``oplora=False``\n    no-op\n\n    ``base`` is modified IN PLACE by ``merge_and_unload`` and returned in its\n    merged (adapter-free) state; the caller owns its lifecycle. ``peft`` and\n    :mod:`llm_training.oplora` are imported lazily so the module stays\n    importable without the ``[train]`` extra.\n\n    :mod:`llm_training.oplora`'
    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    if oplora:
        from llm_training.oplora import OploraProjection

        projection = OploraProjection.from_peft_model(peft_model, k=oplora_k)
        projection.project_weights()
    merged = peft_model.merge_and_unload()
    assert_bfloat16_model(merged)
    if max_shard_size:
        merged.save_pretrained(str(out_dir), max_shard_size=max_shard_size)
    else:
        merged.save_pretrained(str(out_dir))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(out_dir))
    return merged
