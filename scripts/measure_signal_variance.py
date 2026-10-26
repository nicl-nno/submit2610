'Offline Phase-0 tool: measure the raw fluency / code_check signal variance.\n\ncode_check.\n\nThis script is the **go/no-go gate** before wiring the fluency / code_check\nsignals into the live reward. It reads an ``run_eval.py --detailed`` report\n(policy outputs + their task classes), computes the RAW (uncalibrated) fluency\n``P(grammatical)`` **per unique surviving sentence** and the ``code_check``\nscore per ``coder`` answer, and reports the score distribution. Final-answer\nturns contribute their ``answer`` field extracted from the ``finalanswertool``\ncall (:func:`content_response_text` — the same population the reward scores),\nnot the raw ``<tool_call>`` envelope. If the signal is saturated (e.g. all\nfluency scores ≈ 0.98, or every ``coder`` Python block already parses), it\ncarries no gradient in GRPO and should NOT be wired in.\n\n``run_eval.py --detailed``\n``code_check``\n\nUsage\n    python run_eval.py --model anonymous/base-model --detailed --output eval.json ...\n    python scripts/measure_signal_variance.py --report eval.json'

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from llm_training.reward.config import (
    DEFAULT_FLUENCY_CLASSES,
    DEFAULT_FLUENCY_SUBTASKS,
    DEFAULT_LOCAL_FLUENCY_MODEL,
    RewardConfig,
)
from llm_training.reward.fluency import raw_fluency_per_sentence
from llm_training.reward.scoring import code_check, content_response_text

logger = logging.getLogger("measure_signal_variance")


def _load_pairs(report_path: Path) -> list[tuple[str, str, dict]]:
    "Extract ``(task_class, analysis_text, extra)`` triples from a detailed eval report.\n\n\n    Each sample's text is passed through\n    :func:`llm_training.reward.scoring.content_response_text`, which mirrors the\n    reward's ``content_response`` selection: final-answer turns contribute the\n    ``answer`` field **extracted from the ``finalanswertool`` call** (a raw\n    ``<tool_call>`` envelope has no scoreable prose — before the Path-A\n    tool-call refactor this transform did not exist and final turns silently\n    vanished from the pool); reasoning / action / plain turns contribute their\n    raw text. Samples with no scoreable content (no valid final call, empty\n    text) are skipped and counted via the logger.\n    content_response"
    data = json.loads(report_path.read_text(encoding="utf-8"))
    # run_eval.py's ``_build_report`` nests the whole results block (including
    # the detailed per-sample dump) under the ``metrics`` key, so the examples
    # live at ``report["metrics"]["examples"]`` — not at the top level.
    # _build_report
    
    # ``report["metrics"]["examples"]``
    examples = data.get("metrics", {}).get("examples", [])
    pairs: list[tuple[str, str, dict]] = []
    skipped_no_content = 0
    for ex in examples:
        tc = ex.get("task_class", "")
        ex_extra = ex.get("extra") or {}
        for sample in ex.get("samples", []):
            text = content_response_text(
                sample.get("text") or "", ex_extra, task_class=tc,
            )
            if text:
                pairs.append((tc, text, ex_extra))
            elif sample.get("text"):
                skipped_no_content += 1
    if skipped_no_content:
        logger.info('Skipped %d samples with no scoreable content (final turns with no valid finalanswertool call).', skipped_no_content)
    return pairs


def _fluency_eligible(task_class: str, extra: dict) -> bool:
    'True iff this example is a candidate for the fluency signal.\n\n\n    Respects both the class allowlist (:data:`DEFAULT_FLUENCY_CLASSES`) and the\n    sub-config allowlist (:data:`DEFAULT_FLUENCY_SUBTASKS`): for a class like\n    ``nerel_instruct`` only the prose sub-configs (definitions / normalization)\n    qualify — the extraction sub-configs are entity lists, not prose.'
    if task_class not in DEFAULT_FLUENCY_CLASSES:
        return False
    subtask = DEFAULT_FLUENCY_SUBTASKS.get(task_class)
    if subtask is not None and (extra or {}).get("config") not in subtask:
        return False
    return True


def _stats(values: list[float]) -> dict[str, float | int]:
    import numpy as np

    n = len(values)
    if n == 0:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    qs = np.quantile(arr, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "n": n,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p10": float(qs[0]),
        "p25": float(qs[1]),
        "median": float(qs[2]),
        "p75": float(qs[3]),
        "p90": float(qs[4]),
        "max": float(arr.max()),
    }


def _fluency_verdict(values: list[float]) -> str:
    'Heuristic go/no-go on the raw fluency distribution.'
    if not values:
        return "NO DATA — no NL answers with >= 2 sentences found."
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std())
    p10, p90 = float(np.quantile(arr, 0.10)), float(np.quantile(arr, 0.90))
    frac_near_top = float((arr >= 0.95).mean())
    if std < 0.02:
        return (
            f"NO-GO (saturated): std={std:.4f} < 0.02 — almost no variance; the "
            f"gradient would be ~0. Calibrator cannot manufacture spread."
        )
    if p90 - p10 < 0.05:
        return (
            f"NO-GO (saturated): p90-p10={p90 - p10:.4f} < 0.05 — the middle 80% "
            f"span a negligible range."
        )
    if frac_near_top > 0.8:
        return (
            f"NO-GO (saturated): {frac_near_top:.0%} of scores are >= 0.95 — the "
            f"signal is pinned at the top."
        )
    return (
        f"GO (spread): std={std:.4f}, p10={p10:.3f}, p90={p90:.3f}, "
        f"frac>=0.95={frac_near_top:.0%} — the signal carries usable variance. "
        f"Proceed to scripts/calibrate_fluency.py."
    )


def _code_check_verdict(values: list[float]) -> str:
    if not values:
        return "NO DATA — no `coder` answers with a Python block found."
    frac_perfect = sum(1 for v in values if v >= 0.999) / len(values)
    frac_zero = sum(1 for v in values if v <= 1e-9) / len(values)
    if frac_perfect > 0.95:
        return (
            f"NO-GO (trivial): {frac_perfect:.0%} of coder answers score 1.0 — "
            f"the model already emits valid Python; little gradient."
        )
    if frac_zero == 0.0 and frac_perfect < 0.05:
        return (
            f"MARGINAL: no fully-broken (0.0) and few perfect answers — partial "
            f"credit dominates; signal is graded but possibly noisy."
        )
    return (
        f"GO (spread): {frac_zero:.0%} broken (0.0), {frac_perfect:.0%} perfect "
        f"(1.0) — code_check distinguishes compilable from broken code."
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )
    p = argparse.ArgumentParser(
        description=(
            'Measure raw fluency / code_check signal variance (Phase-0 go/no-go).\ncode_check (Phase-0).'
        )
    )
    p.add_argument("--report", required=True, help="run_eval.py --detailed JSON path.")
    p.add_argument("--fluency-model", default=None, help="Override fluency model id.")
    p.add_argument("--fluency-device", default=None, help="Override fluency device.")
    args = p.parse_args(argv)

    pairs = _load_pairs(Path(args.report))
    logger.info("Loaded %d (task_class, text) pairs.", len(pairs))
    if not pairs:
        print("No samples found in the report.", file=sys.stderr)
        return 1

    # --- Fluency over the NL allowlist (+ coder prose) ---
    nl_texts = [t for tc, t, e in pairs if _fluency_eligible(tc, e)]
    cfg = RewardConfig(
        local_fluency_model=args.fluency_model or DEFAULT_LOCAL_FLUENCY_MODEL,
        local_fluency_device=args.fluency_device,
    )
    logger.info("Computing raw fluency for %d NL texts...", len(nl_texts))
    # Per-sentence: the Phase-0 distribution is the raw P(grammatical) over
    # unique surviving sentences, not per-text means.
    
    
    sent_pairs = raw_fluency_per_sentence(nl_texts, cfg)
    fluency_vals = [score for _, score in sent_pairs]

    # --- code_check over coder answers ---
    coder_texts = [t for tc, t, e in pairs if tc == "coder"]
    cc_vals_all = [code_check(t, "coder") for t in coder_texts]
    cc_vals = [v for v in cc_vals_all if v is not None]

    report = {
        "fluency": {
            "n_texts": len(nl_texts),
            "n_unique_sentences": len(sent_pairs),
            "stats": _stats(fluency_vals),
            "verdict": _fluency_verdict(fluency_vals),
        },
        "code_check": {
            "n_coder_texts": len(coder_texts),
            "n_with_python": len(cc_vals),
            "stats": _stats(cc_vals),
            "verdict": _code_check_verdict(cc_vals),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
