'Offline builder: DocSem check-181 → ``src_data/docsem_quant.jsonl``.\n\n\nDistils the DocSem shared-task (DocInsights 2026 @ EMNLP) ``check`` subset\ninto the ``docsem_quant`` GRPO task class: document-grounded quantitative\nreasoning answered as a single JSON object ``{reasoning, program, answer,\nevidence}`` (Program-of-Thought — the model writes a short Python program,\nthe reward executes it in the safe interpreter).\n\nSources (read-only, all on-disk artifacts of the ``ragu_for_docsem`` project):\n\n* the data snapshot (``tasks.jsonl`` / ``labels.jsonl`` of the ``train``\n  split; UPL 1.0, © Oracle — the license is copied next to the output);\n* ``experiments/dev_split.json`` — the ``check`` subset (181 tasks);\n* ``experiments/runs/<run>/details.jsonl`` — the champion run\'s per-sample\n  traces (top-3 retrieved blocks, 5 PoT programs, executed answers,\n  evidences, the winner\'s reasoning);\n* the parsed document blocks cache (``~/.cache/docsem_ragu/blocks``).\n\nThe reference assistant turn for a task is the champion sample whose program\nre-executes (safe interpreter) to the GOLD answer AND whose evidence matches\nthe GOLD evidence exactly (163/181 tasks); the remaining tasks keep a\ngold-only reference — the deterministic ``docsem_check`` reward scores them\nagainst ``extra["gold_answer"]`` / ``extra["gold_evidence"]`` regardless of\nthe reference trace.\n\n``docsem_quant``\n\n``extra["gold_evidence"]``.\n\nUsage\n\n    python scripts/build_docsem_quant.py         --docsem-root /path/to/user/sources/gitflic/ragu_for_docsem         --snapshot ~/datasets/docinsights_2026         --blocks-cache ~/.cache/docsem_ragu/blocks         --run check_keyents         --out src_data/docsem_quant.jsonl'

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from llm_training.reward.pot_interpreter import (
    ProgramError,
    execute_program,
    format_result,
)

logger = logging.getLogger("build_docsem_quant")

#: The only subset distilled: the ``check`` split of the internal dev split
#: (181 tasks with gold labels AND champion traces; stratified by document,
#: so every task comes from a distinct document — no leakage inside the class).

CHECK_KEY = "check"

#: The champion run whose details carry the top-3 blocks and the 5 PoT
#: samples (Qwen3.8-27B + PoT, k=5, T=0.8; answer 0.901 / evidence 1.000
# : on check).
DEFAULT_RUN = "check_keyents"

# : Generator provenance recorded in every record.
DISTILLED_FROM = "Qwen3.8-27B (ragu_for_docsem, run check_keyents)"

#: System prompt: the champion PoT prompt (docsem_ragu/generation.py,
#: SYSTEM_PROMPT) with the output format spelled out inline — the champion
#: relied on server-side structured output (json_schema), the policy must
#: learn the JSON envelope from the prompt text itself.



SYSTEM_PROMPT = """\
You answer quantitative questions strictly grounded in the numbered blocks of a document.

Rules:
- Reason only from the blocks given in the user message; never mix in numbers from other passages of the document.
- Several passages may look relevant: choose the single passage whose quantities, wording and time frame match the question. Do not combine facts from different passages unless the scenario explicitly requires it.
- Begin the reasoning by restating every quantity you will use from the cited block: its value, unit and time frame, exactly as written. Misreading a rate, a fraction base or a tariff structure is the main source of errors.
- "reasoning" is a short plan: the restated quantities and how they combine.
- "program" is a short Python program that computes the final answer. End it by assigning the result to a variable named answer. Use only numbers restated from the cited block. Allowed: arithmetic (+ - * / // % **), assignments, comparisons, conditional expressions (a if cond else b), and the functions min, max, round, abs, sum, ceil, floor, sqrt. Do not use any other functions, imports or loops.
- Give the answer in the unit the question asks for: if the question asks for a percentage, output the percent value (for example 25, not 0.25); do not convert units silently.
- "answer" must contain only the final value in canonical decimal notation: digits with an optional minus sign and decimal point; no thousands separators, no units, no explanatory text. It must match what the program computes.
- "evidence" must list the block identifiers sufficient to state the question and its numeric inputs. Copy identifiers character by character from the beginning of the block lines (the token before the colon).

Reply with a single JSON object and nothing else — no markdown fences, no text outside the JSON. The object has exactly these fields:
{"reasoning": "<short plan>", "program": "<python program>", "answer": "<final value>", "evidence": ["<block id>", ...]}"""


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _reexecute(program: str) -> str | None:
    'Re-execute a sample program; canonical string, ``None`` on failure.'
    if not program or not program.strip():
        return None
    try:
        return format_result(execute_program(program))
    except ProgramError:
        return None


def _pick_reference(
    detail: dict, gold_answer: str, gold_evidence: list[str]
) -> tuple[dict, bool]:
    'Select the reference trace: a sample that re-executes to the gold\n    answer with the gold evidence (prefer the voting winner, then a sample\n    with non-empty reasoning). Returns ``(trace, verified)``; for tasks with\n    no verified sample the trace is gold-only.'
    programs: list[str] = detail.get("programs") or []
    evidences: list[list[str]] = detail.get("evidences") or []
    executed: list[str | None] = detail.get("executed") or []
    # The details carry only the WINNER's reasoning. The champion votes by
    # strict majority over effective (executed) answers — replicate that to
    # decide whether the recorded reasoning belongs to a gold-correct winner
    # (only then is it coherent with a gold-verified program of the same
    # block and answer).
    
    winner_reasoning = detail.get("reasoning") or ""
    counts = Counter(e for e in executed if e)
    modal, modal_n = counts.most_common(1)[0] if counts else ("", 0)
    winner_is_gold = modal == gold_answer and modal_n * 2 > len(executed)
    # The reference trace: the first sample whose program re-executes to the
    # gold answer with the gold evidence (all such samples are equivalent —
    # same block, same answer; a working program + exact evidence).
    
    
    for i, program in enumerate(programs):
        reex = _reexecute(program)
        if reex is None or reex != gold_answer:
            continue
        if i >= len(evidences) or [str(e) for e in (evidences[i] or [])] != [
            str(e) for e in gold_evidence
        ]:
            continue
        return (
            {
                "reasoning": winner_reasoning if winner_is_gold else "",
                "program": program,
                "answer": gold_answer,
                "evidence": [str(e) for e in gold_evidence],
            },
            True,
        )
    return (
        {
            "reasoning": "",
            "program": "",
            "answer": str(gold_answer),
            "evidence": [str(e) for e in gold_evidence],
        },
        False,
    )


def build(
    docsem_root: Path,
    snapshot: Path,
    blocks_cache: Path,
    run: str,
    out_path: Path,
    license_out: Path | None = None,
) -> dict:
    'Build the JSONL; returns summary stats.'
    dev_split = json.loads(
        (docsem_root / "experiments" / "dev_split.json").read_text(encoding="utf-8")
    )
    check_ids = sorted(dev_split[CHECK_KEY])
    tasks = {
        r["instance_id"]: r
        for r in _read_jsonl(snapshot / "train" / "tasks.jsonl")
    }
    labels = {
        r["instance_id"]: r
        for r in _read_jsonl(snapshot / "train" / "labels.jsonl")
    }
    # details: last record wins on duplicates (re-run tails).
    details: dict[str, dict] = {}
    for r in _read_jsonl(
        docsem_root / "experiments" / "runs" / run / "details.jsonl"
    ):
        details[r["instance_id"]] = r

    records = []
    n_verified = 0
    n_unverified = 0
    for tid in check_ids:
        task = tasks[tid]
        gold = labels[tid]
        detail = details[tid]
        doc_path = snapshot / task["document_pdf"]
        md5 = _md5(doc_path)
        blocks_file = blocks_cache / f"{md5}.json"
        parsed = json.loads(blocks_file.read_text(encoding="utf-8"))
        blocks = {b["block_id"]: b["text"] for b in parsed["blocks"]}
        retrieved = [str(b) for b in (detail.get("retrieved") or [])]
        missing = [b for b in retrieved if b not in blocks]
        if missing:
            raise RuntimeError(
                f"{tid}: retrieved blocks missing from the parsed document: "
                f"{missing}"
            )
        rendered = "\n".join(blocks[b] for b in retrieved)
        query = f"Document blocks:\n{rendered}\n\nQuestion: {task['user_query']}"
        trace, verified = _pick_reference(
            detail, str(gold["answer"]), [str(e) for e in gold["evidence"]]
        )
        n_verified += int(verified)
        n_unverified += int(not verified)
        records.append(
            {
                "task_class": "docsem_quant",
                "instance_id": tid,
                "system": SYSTEM_PROMPT,
                "query": query,
                "response": json.dumps(trace, ensure_ascii=False),
                "gold_answer": str(gold["answer"]),
                "gold_evidence": [str(e) for e in gold["evidence"]],
                "distilled_from": DISTILLED_FROM,
                "trace_verified": verified,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if license_out is not None:
        license_src = snapshot / "LICENSE.txt"
        license_out.write_text(
            license_src.read_text(encoding="utf-8"), encoding="utf-8"
        )

    return {
        "records": len(records),
        "trace_verified": n_verified,
        "gold_only": n_unverified,
        "out": str(out_path),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description='Build src_data/docsem_quant.jsonl from the DocSem check subset (champion PoT traces + gold labels).'
    )
    p.add_argument(
        "--docsem-root", type=Path, required=True,
        help='Path to the ragu_for_docsem project.',
    )
    p.add_argument(
        "--snapshot", type=Path, required=True,
        help='DocSem data snapshot root (contains LICENSE.txt, train/).',
    )
    p.add_argument(
        "--blocks-cache", type=Path, required=True,
        help='Parsed document blocks cache (one <md5>.json per document).',
    )
    p.add_argument(
        "--run", default=DEFAULT_RUN,
        help=f'Champion run name under experiments/runs/ (default {DEFAULT_RUN}',
    )
    p.add_argument(
        "--out", type=Path, default=Path("src_data/docsem_quant.jsonl"),
        help='Output JSONL path.',
    )
    p.add_argument(
        "--license-out", type=Path, default=Path("src_data/docsem_quant.LICENSE"),
        help='Where to copy the UPL 1.0 license (data © Oracle).',
    )
    args = p.parse_args(argv)

    stats = build(
        docsem_root=args.docsem_root,
        snapshot=args.snapshot,
        blocks_cache=args.blocks_cache,
        run=args.run,
        out_path=args.out,
        license_out=args.license_out,
    )
    logger.info("docsem_quant built: %s", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
