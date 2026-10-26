'Offline dataset preparation cache: serialize the built/filtered dataset.\n\n\nThe two expensive, deterministic phases of the training scripts — building the\nunified dataset (loading every class + the ``mask_example`` length filter) and\nthe loss-based difficulty scoring (a bf16 forward pass of a separate model over\nthe train pool) — are moved into :mod:`prepare_dataset` (the offline prep\nstage, mirroring the ``cpt_tokenize.py`` → ``cpt_train.py`` split). This module\nowns the on-disk format those caches use, shared by the prep script (writer)\nand the training scripts (readers):\n\n* ``grpo`` pipeline — the FULL (uncapped) per-class train pools + the frozen\n  anchor val pool + the per-source difficulty scores;\n* ``warmup`` pipeline — the final warm-up train/val pools (post stratified\n  split + val-budget return) + the per-source difficulty scores.\n\nThe cache is plain **JSONL + a ``meta.json``** (no pyarrow, no ``datasets``) so\nthe nested message/tool/extra structures round-trip losslessly (the same\nheterogeneous-type landmine that forced pre-rendered TEXT rows in\n``sft_warmup.py`` is avoided entirely). Difficulty **scores** are cached, not\nthe filtered result, so the (instant) per-class quantile filter can be re-run\nwith different ``--difficulty-drop-*`` shares without re-scoring.\n\n``mask_example``\ntrain/val'

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from llm_training.schema import UnifiedExample

#: On-disk cache format version. Bumped on a breaking layout change; a cache
#: with a different version is rejected loudly (never silently mis-read).

FORMAT_VERSION = 1

# : Cache pipeline tags
PIPELINE_GRPO = "grpo"
PIPELINE_WARMUP = "warmup"


def example_to_dict(example: UnifiedExample) -> dict[str, Any]:
    'Flatten a :class:`UnifiedExample` into a plain-JSON dict.\n\n\n    Every field is JSON-native (verified against the loaders): ``prompt`` and\n    ``reference_messages`` are lists of message dicts, ``tools`` is ``None`` or\n    a list of tool-schema dicts, ``extra`` holds str/int/bool/list values.'
    return {
        "task_class": example.task_class,
        "prompt": list(example.prompt),
        "reference": example.reference,
        "tools": example.tools,
        "reference_messages": list(example.reference_messages),
        "source": example.source,
        "extra": dict(example.extra),
    }


def example_from_dict(data: dict[str, Any]) -> UnifiedExample:
    'Reconstruct a :class:`UnifiedExample` from :func:`example_to_dict` output.'
    return UnifiedExample(
        task_class=data["task_class"],
        prompt=list(data["prompt"]),
        reference=data["reference"],
        tools=data.get("tools"),
        reference_messages=list(data.get("reference_messages") or []),
        source=data.get("source") or "",
        extra=dict(data.get("extra") or {}),
    )


def write_examples_jsonl(path: Path, examples: Sequence[UnifiedExample]) -> None:
    'Write examples as one JSON object per line (UTF-8, no ASCII escaping).'
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(example_to_dict(ex), ensure_ascii=False) + "\n")


def read_examples_jsonl(path: Path) -> list[UnifiedExample]:
    'Read the examples written by :func:`write_examples_jsonl`.'
    examples: list[UnifiedExample] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            examples.append(example_from_dict(json.loads(line)))
    return examples


def write_json(path: Path, data: Any) -> None:
    'Write ``data`` as indented UTF-8 JSON.'
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    'Read a JSON file written by :func:`write_json`.'
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _check_meta(meta: dict[str, Any], expected_pipeline: str) -> None:
    'Validate the cache ``meta.json`` (version + pipeline) and fail loudly.'
    version = meta.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f'Cache format version {version!r} != {FORMAT_VERSION!r}'
        )
    pipeline = meta.get("pipeline")
    if pipeline != expected_pipeline:
        raise ValueError(
            f'Cache is a {pipeline!r} cache, not {expected_pipeline!r}'
        )


@dataclass(frozen=True)
class GrpoCache:
    "The loaded GRPO prep cache\n\n    ``train_pools`` is the FULL (uncapped, per-class) length-filtered pool —\n    difficulty scores ride separately (``scores``) so the (instant) quantile\n    filter can be re-applied with the training run's own ``--difficulty-drop-*``.\n    ``val`` is the frozen anchor val pool (fixed-val rows + the hold-out\n    classes). ``tool_names``/``tool_schemas`` feed the reward's ``format_check``\n    / ``schema_check``."

    train_pools: dict[str, list[UnifiedExample]]
    val: list[UnifiedExample]
    scores: dict[str, float]
    seed: int
    val_tasks: tuple[str, ...]
    include_network: bool
    max_per_class: int
    difficulty_model: str
    tool_names: frozenset[str]
    tool_schemas: dict[str, dict]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WarmupCache:
    'The loaded warm-up prep cache\n\n    ``train``/``val`` are the FINAL warm-up pools (post stratified split and\n    post val-budget return); ``scores`` rides separately so the hard-tail\n    difficulty filter can be re-applied with any ``--difficulty-drop-hard``.'

    train: list[UnifiedExample]
    val: list[UnifiedExample]
    scores: dict[str, float]
    seed: int
    tool_names: frozenset[str]
    meta: dict[str, Any] = field(default_factory=dict)


def save_grpo_cache(
    output_dir: Path,
    *,
    train_pools: dict[str, list[UnifiedExample]],
    val: Sequence[UnifiedExample],
    scores: dict[str, float],
    seed: int,
    val_tasks: Sequence[str],
    include_network: bool,
    max_per_class: int,
    difficulty_model: str,
    tool_names: "frozenset[str]",
    tool_schemas: dict[str, dict],
    stats: dict[str, Any],
) -> dict[str, Any]:
    'Write the GRPO prep cache under ``output_dir``; return ``meta``.'
    output_dir.mkdir(parents=True, exist_ok=True)
    all_train = [ex for cls in sorted(train_pools) for ex in train_pools[cls]]
    write_examples_jsonl(output_dir / "train_pools.jsonl", all_train)
    write_examples_jsonl(output_dir / "val.jsonl", list(val))
    write_json(output_dir / "scores.json", scores)
    meta: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "pipeline": PIPELINE_GRPO,
        "seed": seed,
        "val_tasks": list(val_tasks),
        "include_network": bool(include_network),
        "max_per_class": max_per_class,
        "difficulty_model": difficulty_model,
        "tool_names": sorted(tool_names),
        "tool_schemas": dict(tool_schemas),
        "n_train": len(all_train),
        "n_val": len(val),
        "n_scores": len(scores),
        "n_train_per_class": {
            cls: len(train_pools[cls]) for cls in sorted(train_pools)
        },
        "stats": stats,
    }
    write_json(output_dir / "meta.json", meta)
    return meta


def load_grpo_cache(output_dir: Path) -> GrpoCache:
    'Load a GRPO prep cache written by :func:`save_grpo_cache`.'
    meta = read_json(output_dir / "meta.json")
    _check_meta(meta, PIPELINE_GRPO)
    train_flat = read_examples_jsonl(output_dir / "train_pools.jsonl")
    train_pools: dict[str, list[UnifiedExample]] = {}
    for ex in train_flat:
        train_pools.setdefault(ex.task_class, []).append(ex)
    val = read_examples_jsonl(output_dir / "val.jsonl")
    scores = read_json(output_dir / "scores.json")
    return GrpoCache(
        train_pools=train_pools,
        val=val,
        scores={str(k): float(v) for k, v in scores.items()},
        seed=int(meta["seed"]),
        val_tasks=tuple(meta.get("val_tasks") or []),
        include_network=bool(meta.get("include_network")),
        max_per_class=int(meta.get("max_per_class", 0)),
        difficulty_model=str(meta.get("difficulty_model") or ""),
        tool_names=frozenset(meta.get("tool_names") or []),
        tool_schemas=dict(meta.get("tool_schemas") or {}),
        meta=meta,
    )


def save_warmup_cache(
    output_dir: Path,
    *,
    train: Sequence[UnifiedExample],
    val: Sequence[UnifiedExample],
    scores: dict[str, float],
    seed: int,
    difficulty_model: str,
    tool_names: "frozenset[str]",
    stats: dict[str, Any],
) -> dict[str, Any]:
    'Write the warm-up prep cache under ``output_dir``; return ``meta``.'
    output_dir.mkdir(parents=True, exist_ok=True)
    write_examples_jsonl(output_dir / "train.jsonl", list(train))
    write_examples_jsonl(output_dir / "val.jsonl", list(val))
    write_json(output_dir / "scores.json", scores)
    meta: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "pipeline": PIPELINE_WARMUP,
        "seed": seed,
        "difficulty_model": difficulty_model,
        "tool_names": sorted(tool_names),
        "n_train": len(train),
        "n_val": len(val),
        "n_scores": len(scores),
        "stats": stats,
    }
    write_json(output_dir / "meta.json", meta)
    return meta


def load_warmup_cache(output_dir: Path) -> WarmupCache:
    'Load a warm-up prep cache written by :func:`save_warmup_cache`.'
    meta = read_json(output_dir / "meta.json")
    _check_meta(meta, PIPELINE_WARMUP)
    train = read_examples_jsonl(output_dir / "train.jsonl")
    val = read_examples_jsonl(output_dir / "val.jsonl")
    scores = read_json(output_dir / "scores.json")
    return WarmupCache(
        train=train,
        val=val,
        scores={str(k): float(v) for k, v in scores.items()},
        seed=int(meta["seed"]),
        tool_names=frozenset(meta.get("tool_names") or []),
        meta=meta,
    )
