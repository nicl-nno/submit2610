'Metrics-only evaluation of a checkpoint on the extraction-line val pool.\n\n\nThe extraction sibling of ``run_eval.py``: the corpus rows are deliberately\nNOT ``UnifiedExample`` (the extraction/consolidation rows carry no reference\nby design), so they do not fit the unified-dataset pipeline — this script\nreads the corpus JSONL directly, generates with an in-process ``vLLM.LLM``\n(greedy or N samples) and scores ALL 8 classes with the SAME dispatch the\ntraining anchor uses (:func:`score_responses` mirrors\n``grpo_extract_train.run_extraction_anchor``, so the eval number stays\ncomparable to the anchor ``val``):\n\n* extraction rows — :func:`llm_training.reward.extraction.extraction_check`\n  + the cached judge pseudo-gold when the pool was annotated;\n* consolidation rows (``extra["summary"]``) —\n  :func:`llm_training.reward.summarization.summarization_check`, plus the\n  judge-reference clipped cosine (``harmonic(structural, semsim)``) when the\n  pool carries ``extra["summary"]["reference"]`` and an embedder is wired;\n* QA rows (``fineweb_qa``, the ONE reference-bearing class) — SemSim-vs-\n  reference + fluency in ONE batch through the in-process judge-free scorer\n  (built only when such rows exist — the embedder loads AFTER the vLLM\n  engine released its GPU memory).\n\nA JSON report with per-component means, a ``val_by_kind`` breakdown and a\n``--detailed`` per-row dump lands in ``--output``.\n\n in-process vLLM\n ``--output``.\n\nUsage\n\n    python run_extraction_eval.py --model checkpoints/grpo-extract/best-500         --val src_data/extraction/val_annotated.jsonl         --policy-gpu-ids 0 --policy-tp 1 --policy-gpu-memory-utilization 0.85         --seed 42 --output eval_extract.json --detailed'

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def generate_responses(
    val_rows: list[dict[str, Any]],
    model_path: str,
    gpu_ids: list[int],
    tp: int,
    gpu_memory_utilization: float,
    max_tokens_budget: int,
    generation_mode: str,
    num_samples: int,
    seed: int,
    tokenizer: Any = None,
) -> list[list[str]]:
    'Generate responses for every val row via an in-process vLLM engine.\n\n    ``skip_special_tokens=False`` pin mirrors :mod:`llm_training.inference`:\n    ``<tool_call>``-style special tokens must survive; here the envelope is\n    JSON, but the same guarantee applies. The per-example generation cap is\n    ``min(max_tokens_budget, max_model_len − prompt_tokens)`` — the vLLM\n    default of 16 tokens once silently truncated every completion.'
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(g) for g in gpu_ids))
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=None,  # derive from the model config
    )
    tokenizer = tokenizer or llm.get_tokenizer()

    prompts: list[str] = []
    meta: list[dict[str, Any]] = []
    for row in val_rows:
        prompt_text = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt_text)
        meta.append({"row": row, "n_prompt": len(tokenizer(prompt_text)["input_ids"])})

    if generation_mode == "greedy":
        params_list = [
            SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens_budget,
                seed=seed,
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
            )
            for _ in prompts
        ]
    else:
        params_list = [
            SamplingParams(
                temperature=1.0,
                top_p=1.0,
                n=num_samples,
                max_tokens=max_tokens_budget,
                seed=seed,
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
            )
            for _ in prompts
        ]

    outputs = llm.generate(prompts, params_list)
    grouped: list[list[str]] = []
    for out in outputs:
        grouped.append([o.text for o in out.outputs])
    # Release the engine BEFORE the scorer builds its in-process embedder on
    # the same GPU (main() loads bge-m3 only after this function returns).
    
    del outputs
    del llm
    import gc

    gc.collect()
    return grouped


def score_responses(
    val_rows: list[dict[str, Any]],
    groups: list[list[str]],
    service: Any | None = None,
    summary_semsim: Any | None = None,
) -> dict[str, Any]:
    'Score every response with the reward checks; aggregate a report.\n\n    dispatch MIRRORS the anchor of ``grpo_extract_train.py`` — the eval\n    number stays comparable to the anchor ``val``:\n\n    * extraction rows (``extra["extraction"]``) — the deterministic\n      ``extraction_check`` + the cached judge pseudo-gold F1;\n    * consolidation rows (``extra["summary"]``) — ``summarization_check``,\n      plus the judge-reference clipped cosine when ``summary_semsim`` is\n      wired and the pool carries a cached reference (the strict harmonic of\n      the two — the anchor semantics);\n    * QA rows (``fineweb_qa``) — SemSim-vs-reference + fluency in ONE batch\n      through ``service`` (a judge-free\n      :class:`~llm_training.runtime.reward_service.RewardService`); without\n      a service they measure 0.0 (visible in ``val_by_kind``/``qa``).\n\n    ``val`` = the mean of the per-row means (each row\'s samples averaged).\n(``val_by_kind``'
    from llm_training.extraction.qa_prompts import QA_TASK_CLASS
    from llm_training.reward.extraction import (
        extraction_check,
        parse_extraction_output,
        pseudo_gold_score,
    )
    from llm_training.reward.scoring import harmonic_mean
    from llm_training.reward.summarization import (
        SUMMARY_TASK_CLASSES,
        summarization_check,
        summary_reference_semsim,
    )
    from llm_training.runtime.reward_service import ScoreRequest

    def _kind_of(row: dict[str, Any]) -> str:
        tc = row.get("task_class") or ""
        if tc == QA_TASK_CLASS:
            return "qa"
        if tc in SUMMARY_TASK_CLASSES and isinstance(
            (row.get("extra") or {}).get("summary"), dict
        ):
            return "summary"
        return "extraction"

    qa_jobs: list[tuple[int, int, dict[str, Any], str]] = []
    per_row: list[dict[str, Any]] = []
    for ri, (row, responses) in enumerate(zip(val_rows, groups)):
        extra = row.get("extra") or {}
        payload = extra.get("extraction") or {}
        pseudo = payload.get("pseudo_gold")
        tc = row.get("task_class") or ""
        kind = _kind_of(row)
        samples: list[dict[str, Any]] = []
        for si, response in enumerate(responses):
            entry: dict[str, Any] = {"response": response}
            active: list[float] = []
            if kind == "summary":
                scored_s = summarization_check(tc, extra, response)
                if scored_s is not None:
                    entry.update(scored_s.as_dict())
                    active.append(scored_s.combined)
                    if summary_semsim is not None:
                        sem = summary_reference_semsim(
                            summary_semsim, extra["summary"], response
                        )
                        if sem is not None:
                            entry["summary_semsim"] = sem
                            active.append(sem)
                entry["val"] = harmonic_mean(active) if active else 0.0
            elif kind == "qa":
                # Scored in ONE batch at the end (the service is batched).
                qa_jobs.append((ri, si, row, response))
                entry["val"] = 0.0  # replaced after the batch below
            else:
                scored = extraction_check(response, tc, extra)
                pg = pseudo_gold_score(
                    parse_extraction_output(response), pseudo,
                    payload.get("language"), payload.get("text"),
                )
                if scored is not None:
                    entry.update(scored.as_dict())
                    active.append(scored.combined)
                if pg is not None and pg.combined is not None:
                    entry["pseudo_gold"] = pg.combined
                    active.append(pg.combined)
                entry["val"] = harmonic_mean(active) if active else 0.0
            samples.append(entry)
        per_row.append(
            {
                "id": row.get("id"),
                "style": (row.get("extra") or {}).get("style"),
                "stage": payload.get("stage"),
                "task_class": tc,
                # The kind rides explicitly: per-row report dicts carry no
                # ``extra``, so the classifier could not re-derive it.
                
                "kind": kind,
                "samples": samples,
                "val_mean": (
                    statistics.fmean(s["val"] for s in samples)
                    if samples
                    else 0.0
                ),
            }
        )

    # QA batch: SemSim-vs-reference + fluency, one service call (the
    # anchor's late-batch pattern).
    if qa_jobs:
        if service is None:
            print(
                f"WARNING: {len(qa_jobs)} QA samples scored 0.0 "
                "(no scorer service wired)."
            )
        else:
            requests = [
                ScoreRequest(
                    prompt=row["prompt"],
                    completion=response,
                    task_class=str(row.get("task_class") or ""),
                    reference=str(row.get("reference") or ""),
                    extra=row.get("extra") or {},
                    source=str(row.get("id") or ""),
                )
                for (_ri, _si, row, response) in qa_jobs
            ]
            try:
                metrics_list = service.score_batch(requests)
            except Exception as exc:  # noqa: BLE001 - never abort an eval
                print(f"WARNING: QA scorer crash ({exc}) — rows → 0.0")
                metrics_list = [None] * len(requests)
            for (ri, si, _row, _resp), metrics in zip(qa_jobs, metrics_list):
                combined = (
                    float(metrics.combined) if metrics is not None else 0.0
                )
                per_row[ri]["samples"][si]["qa"] = combined
                per_row[ri]["samples"][si]["val"] = combined
        for (ri, si, _row, _resp) in qa_jobs:
            per_row[ri]["samples"][si].setdefault("qa", 0.0)
            # Recompute the row mean: QA vals were placeholders before.
            row_samples = per_row[ri]["samples"]
            per_row[ri]["val_mean"] = (
                statistics.fmean(s["val"] for s in row_samples)
                if row_samples
                else 0.0
            )

    def _mean(key: str, kind: str | None = None) -> float | None:
        values = [
            s[key]
            for r in per_row
            if kind is None or r.get("kind") == kind
            for s in r["samples"]
            if s.get(key) is not None
        ]
        return statistics.fmean(values) if values else None

    val_by_kind: dict[str, float] = {}
    for kind in ("extraction", "summary", "qa"):
        means = [
            r["val_mean"] for r in per_row if r.get("kind") == kind
        ]
        if means:
            val_by_kind[kind] = statistics.fmean(means)

    report: dict[str, Any] = {
        "n_rows": len(per_row),
        "n_samples": sum(len(r["samples"]) for r in per_row),
        "val": statistics.fmean(r["val_mean"] for r in per_row) if per_row else 0.0,
        "val_by_kind": val_by_kind,
        # Extraction components (kind-restricted — no cross-kind averaging).
        
        "json_score": _mean("json_score", "extraction"),
        "schema_score": _mean("schema_score", "extraction"),
        "grounding": _mean("grounding", "extraction"),
        "closure": _mean("closure", "extraction"),
        "normalization": _mean("normalization", "extraction"),
        "coverage": _mean("coverage", "extraction"),
        "pseudo_gold": _mean("pseudo_gold", "extraction"),
        # Consolidation components + the judge-reference SemSim.
        "summary": _mean("combined", "summary"),
        "summary_json": _mean("json_score", "summary"),
        "summary_coverage": _mean("coverage", "summary"),
        "summary_semsim": _mean("summary_semsim", "summary"),
        # QA (the standard reference path).
        "qa": _mean("qa", "qa"),
        "per_row": per_row,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Metrics-only evaluation on the 8-class extraction-line pool.'
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy-gpu-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--policy-tp", type=int, default=1)
    parser.add_argument("--policy-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--generation-mode", choices=("greedy", "sampling"), default="greedy"
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--detailed", action="store_true",
        help="Keep per-sample responses in the report.",
    )
    parser.add_argument(
        "--local-embedding-model", default="BAAI/bge-m3",
        help='In-process embedder for QA SemSim',
    )
    parser.add_argument(
        "--local-embedding-device", default="cuda",
        help='Embedder device.',
    )
    parser.add_argument("--embedding-concurrency", type=int, default=4)
    parser.add_argument(
        "--no-fluency", action="store_true",
        help='Skip the grammar classifier (QA rows lose the fluency part).',
    )
    args = parser.parse_args(argv)

    _seed_everything(args.seed)
    val_rows: list[dict[str, Any]] = []
    with open(args.val, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                val_rows.append(json.loads(line))
    if not val_rows:
        raise SystemExit(f"{args.val}: no rows.")

    groups = generate_responses(
        val_rows,
        args.model,
        args.policy_gpu_ids,
        args.policy_tp,
        args.policy_gpu_memory_utilization,
        args.max_tokens,
        args.generation_mode,
        args.num_samples,
        args.seed,
    )

    # The scorer is built AFTER the vLLM engine released its GPU memory (the
    # in-process embedder + grammar classifier share the same card), and ONLY
    # when the pool actually needs it: QA rows → the batch service; cached
    # summary references → the clipped-cosine SemSim over the same embedder.
    
    from llm_training.extraction.qa_prompts import QA_TASK_CLASS
    from llm_training.reward.summarization import SUMMARY_REFERENCE_KEY

    need_qa = any(r.get("task_class") == QA_TASK_CLASS for r in val_rows)
    need_semsim = any(
        isinstance(
            ((r.get("extra") or {}).get("summary") or {}).get(
                SUMMARY_REFERENCE_KEY
            ),
            dict,
        )
        for r in val_rows
    )
    service = None
    summary_semsim = None
    if need_qa or need_semsim:
        from llm_training.reward.config import RewardConfig
        from llm_training.reward.judge import make_ragas_scorer

        reward_cfg = RewardConfig(
            judge=None,
            compute_answer_correctness=False,
            local_embedding_model=args.local_embedding_model,
            local_embedding_device=args.local_embedding_device,
            enable_fluency=not args.no_fluency,
            fluency_classes=frozenset({QA_TASK_CLASS}),
        )
        reward_scorer = make_ragas_scorer(reward_cfg)
        if need_qa:
            from llm_training.runtime.reward_service import RewardService

            service = RewardService(
                reward_scorer, concurrency=args.embedding_concurrency
            )
        if need_semsim:
            from llm_training.reward.summarization import make_summary_semsim

            summary_semsim = make_summary_semsim(reward_scorer.base_embeddings)
            if summary_semsim is None:
                print(
                    "WARNING: summary references found, but the scorer "
                    "exposes no sync embedder — structural only."
                )

    report = score_responses(
        val_rows, groups, service=service, summary_semsim=summary_semsim
    )
    if service is not None:
        service.close()
    if not args.detailed:
        for row in report["per_row"]:
            for sample in row["samples"]:
                sample.pop("response", None)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(
        f"OK: {report['n_rows']} rows / {report['n_samples']} samples -> {out}"
    )

    def _fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    print(
        f"val={report['val']:.4f} "
        + " ".join(
            f"{key}={_fmt(report[key])}"
            for key in ("qa", "summary", "summary_semsim", "pseudo_gold")
        )
    )
    if report["val_by_kind"]:
        print(
            "val_by_kind: "
            + " ".join(
                f"{kind}={value:.4f}"
                for kind, value in report["val_by_kind"].items()
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
