'Offline Phase-0 tool: fit the one-shot fluency calibrator (GLM-5.2 labels).\n\n\nProducer of ``fluency_calibrator.json`` — the **strictly one-shot** artifact\nconsumed read-only by every reward caller (``run_eval.py``, the future\n``grpo_train.py``) via ``RewardConfig.fluency_calibrator_path``. Pipeline:\n\n\nNOTE\nmodel is the grammar classifier, whose raw ``P(grammatical)`` is already\nwell-spread and reality-matching — ``fluency_calibrator_path`` defaults to\n``None`` (identity calibration). Re-fit a calibrator only if the signal is\never re-saturated.\n\n1. Read policy outputs + task classes from an ``run_eval.py --detailed`` report\n   (final-answer turns contribute their ``answer`` field extracted from the\n   ``finalanswertool`` call — ``content_response_text``, the same population\n   the reward scores, not the raw ``<tool_call>`` envelope).\n2. Extract the UNIQUE surviving sentences (strip code → segment → drop those\n   with fewer than ``MIN_FLUENCY_WORDS`` words; deduplicate across all answers)\n   and compute the RAW per-sentence ``P(grammatical)`` (grammar classifier) —\n   the ``x`` for the fit. Calibration is fit at the SENTENCE level, then\n   applied per sentence at reward time; the online scorer aggregates the\n   per-sentence scores afterwards.\n3. Label each unique sentence with a large LLM (default GLM-5.2 via an OpenAI-\n   compatible endpoint) on an ordinal fluency bucket ``{0,1,2,3,4}`` → ``[0,1]``\n   — the ``y`` for the fit. This automates MERA\'s human-annotation step.\n   Deduplication keeps the labelling cost bounded.\n4. Fit ``sklearn.IsotonicRegression(raw_sentence → label)`` and serialise its\n   threshold tables to JSON (``{kind, x, y}``) — the shape :mod:`llm_training.\n   reward.calibration` loads and applies via ``numpy.interp`` at reward time (no\n   sklearn in the reward path).\n\nThe label prompt asks for a single integer (greedy, ``temperature=0``); the\nscheme is **buckets first**. If the residuals are large the operator should\nre-run with pairwise comparisons (Bradley-Terry) — out of scope for this script.\n\n\n**Endpoint convention\n**API root**, not the method URL — the OpenAI SDK appends ``chat/completions``\nitself. Z.AI\'s cURL docs show the already-expanded\n``https://api.z.ai/api/paas/v4/chat/completions``; pass only\n``https://api.z.ai/api/paas/v4/`` here (the script rejects a URL ending in\n``/chat/completions``).\n``--label-base-url``\n``chat/completions``\n\nUsage\n    python scripts/calibrate_fluency.py \\\n        --report eval.json --out fluency_calibrator.json \\\n        --label-base-url https://api.z.ai/api/paas/v4/ --label-model glm-5.2 \\\n        --label-api-key "$ZAI_API_KEY"'

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from llm_training.reward.config import (
    DEFAULT_FLUENCY_CLASSES,
    DEFAULT_FLUENCY_SUBTASKS,
    DEFAULT_LOCAL_FLUENCY_MODEL,
    RewardConfig,
)
from llm_training.reward.fluency import raw_fluency_per_sentence
from llm_training.reward.scoring import content_response_text

logger = logging.getLogger("calibrate_fluency")

# : Ordinal fluency buckets the labelling LLM emits.
_BUCKETS = 5

#: Output budget for one label. Kept small (we want a single digit) but not
#: razor-thin: some providers emit a leading space/newline, and with thinking
#: accidentally left on the digit may arrive after a short preamble.

_LABEL_MAX_TOKENS = 16

#: Placeholder key for local servers that do not validate it (vLLM et al.).

_LOCAL_API_KEY_SENTINEL = "EMPTY"

#: Russian labelling prompt — a single integer in [0, N-1]. The LLM judges the
#: natural-language fluency of the answer, ignoring any code/JSON it contains
#: (those are stripped before labelling too, via split_prose_and_code).

_LABEL_SYSTEM = (
    "Вы — эксперт-оценщик беглости русскоязычных ответов. Оцените, насколько "
    "ответ звучит естественно и грамотно для носителя языка, по шкале целым "
    "числом:\n"
    "0 — сломанный/неграмотный текст;\n"
    "1 — плохой, корявый;\n"
    "2 — приемлемый, но с шероховатостями;\n"
    "3 — беглый, естественный;\n"
    "4 — отличный, безупречный.\n"
    "Ответьте ТОЛЬКО одним числом от 0 до 4."
)


def _load_nl_texts(report_path: Path) -> list[str]:
    'Extract NL-answer texts from a detailed eval report.\n\n\n    Raises :class:`SystemExit` with an actionable message when the report is\n    missing or not valid JSON (a raw traceback is useless to the operator).'
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(
            f'Report not found: {report_path}. Produce it first with `run_eval.py --detailed --output {report_path}'
        ) from None
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise SystemExit(
            f'Report {report_path} is not valid JSON ({e}'
        ) from None
    if not isinstance(data, dict):
        raise SystemExit(
            f'Report {report_path}must be a JSON object.'
        )
    texts: list[str] = []
    skipped_no_content = 0
    # run_eval.py's ``_build_report`` nests the detailed per-sample dump under
    # the ``metrics`` key, so examples live at ``report["metrics"]["examples"]``.
    # _build_report
    
    for ex in data.get("metrics", {}).get("examples", []):
        tc = ex.get("task_class", "")
        if tc not in DEFAULT_FLUENCY_CLASSES:
            continue
        # Sub-config gate (e.g. nerel_instruct → only definition/normalization
        # prose; the extraction sub-configs are entity lists, not prose, and are
        # scored by a deterministic F1 instead).
        
        
        subtask = DEFAULT_FLUENCY_SUBTASKS.get(tc)
        ex_extra = ex.get("extra") or {}
        if subtask is not None and ex_extra.get("config") not in subtask:
            continue
        for sample in ex.get("samples", []):
            # The reward-equivalent population: final-answer turns contribute
            # the ``answer`` extracted from the finalanswertool call, not the
            # raw ``<tool_call>`` envelope (which has no scoreable prose).
            
            
            text = content_response_text(
                sample.get("text") or "", ex_extra, task_class=tc,
            )
            if text:
                texts.append(text)
            elif sample.get("text"):
                skipped_no_content += 1
    if skipped_no_content:
        logger.info('Skipped %d samples with no scoreable content (final turns with no valid finalanswertool call).', skipped_no_content)
    return texts


def _is_local_endpoint(base_url: str) -> bool:
    'True for localhost-style endpoints that do not validate the API key.'
    host = urlparse(base_url).hostname or ""
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _build_label_client(args: argparse.Namespace) -> tuple[Any, str]:
    'Build the OpenAI-compatible client for the labelling LLM.\n\n\n    ``base_url`` must be the **API root**, not the full method URL: the OpenAI\n    SDK appends ``chat/completions`` itself. For Z.AI / GLM-5.2 that root is\n    ``https://api.z.ai/api/paas/v4/`` (their cURL docs show the already-expanded\n    ``.../v4/chat/completions``, which must NOT be passed here).\n``base_url``\n    ``https://api.z.ai/api/paas/v4/``'
    base_url = args.label_base_url
    model = args.label_model
    if not base_url or not model:
        raise SystemExit(
            'A labelling LLM endpoint is required: pass --label-base-url and --label-model (or set LABEL_BASE_URL / LABEL_MODEL). Pass the API root (e.g. https://api.z.ai/api/paas/v4/ for GLM-5.2), not the full .../chat/completions URL.'
        )
    if base_url.rstrip("/").endswith("/chat/completions"):
        raise SystemExit(
            f'--label-base-url must be the API root, not the method URL: got {base_url!r}'
        )
    # A remote endpoint always needs a real key; sending the "EMPTY" sentinel
    # would fail deep inside the SDK with an opaque 401. Fail early and clearly.
    
    
    if not args.label_api_key and not _is_local_endpoint(base_url):
        raise SystemExit(
            'A labelling API key is required for a remote endpoint: pass --label-api-key (or set LABEL_API_KEY).'
        )
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "The 'openai' package is required: pip install -e '.[train]'."
        ) from e
    api_key = args.label_api_key or _LOCAL_API_KEY_SENTINEL
    return OpenAI(base_url=base_url, api_key=api_key), model


_INT_RE = re.compile(r"[0-4]")


def _label_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    'Build the provider-specific extra kwargs for a labelling call.\n\n\n    * ``thinking``: GLM-5.2 enables chain-of-thought by default. For a\n      single-integer bucket classification that only wastes latency, cost and\n      the tiny output budget (the reasoning may consume every allowed token and\n      leave ``content`` empty). ``auto``/``disabled`` therefore send\n      ``thinking={"type": "disabled"}`` via ``extra_body``; ``enabled`` leaves\n      the provider default alone.\n    * ``seed``: not part of the published Z.AI schema, so it is **opt-in**\n      (``--label-send-seed``). Determinism is already approximated by\n      ``temperature=0``.\nZ.AI'
    extra: dict[str, Any] = {}
    if args.label_thinking in ("auto", "disabled"):
        extra["extra_body"] = {"thinking": {"type": "disabled"}}
    if args.label_send_seed:
        extra["seed"] = args.seed
    return extra


def _extract_label_text(resp: Any) -> str:
    'Pull the answer text out of a chat-completion response.\n\n\n    Falls back to ``reasoning_content`` (GLM/Z.AI put the chain of thought\n    there; if ``content`` came back empty because the budget was spent on\n    reasoning, the digit is often still recoverable from it).'
    try:
        message = resp.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return ""
    text = (getattr(message, "content", None) or "").strip()
    if text:
        return text
    return (getattr(message, "reasoning_content", None) or "").strip()


def _label_one(
    client: Any, model: str, text: str, *, extra_kwargs: dict[str, Any],
) -> float | None:
    'Ask the labelling LLM for one ordinal bucket ∈ {0..4}/4; None on failure.\n\n\n    Provider-specific extras (``thinking``, ``seed``) are retried once without\n    them: a strict OpenAI-compatible server may reject unknown fields, and the\n    labelling run must not die on that.'
    def _call(kwargs: dict[str, Any]) -> Any:
        return client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=_LABEL_MAX_TOKENS,
            messages=[
                {"role": "system", "content": _LABEL_SYSTEM},
                {"role": "user", "content": text},
            ],
            **kwargs,
        )

    try:
        resp = _call(extra_kwargs)
    except Exception as e:  # noqa: BLE001 - skip on any labelling failure
        if extra_kwargs:
            # Retry once without provider-specific extras (unknown-field 400s).
            
            logger.warning(
                "Labelling call failed with provider extras (%s); retrying "
                "without them.", e,
            )
            try:
                resp = _call({})
            except Exception as e2:  # noqa: BLE001
                logger.warning("Labelling call failed: %s", e2)
                return None
        else:
            logger.warning("Labelling call failed: %s", e)
            return None
    raw = _extract_label_text(resp)
    m = _INT_RE.search(raw)
    if m is None:
        logger.warning("Unparseable LLM label: %r", raw[:40])
        return None
    return int(m.group(0)) / (_BUCKETS - 1)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )
    p = argparse.ArgumentParser(
        description=(
            'Fit the one-shot fluency calibrator (GLM-5.2 ordinal labels → IsotonicRegression).'
        )
    )
    p.add_argument("--report", required=True, help="run_eval.py --detailed JSON path.")
    p.add_argument("--out", default="fluency_calibrator.json",
                   help="Output calibrator JSON path.")
    p.add_argument("--fluency-model", default=None, help="Override fluency model id.")
    p.add_argument("--fluency-device", default=None, help="Override fluency device.")
    p.add_argument("--label-base-url", default=None,
                   help="the API ROOT, not the method URL (the OpenAI SDK appends 'chat/completions'). E.g. https://api.z.ai/api/paas/v4/ for GLM-5.2. Env: LABEL_BASE_URL.")
    p.add_argument("--label-model", default=None,
                   help="Labelling LLM model (e.g. glm-5.2). Env: LABEL_MODEL.")
    p.add_argument("--label-api-key", default=None,
                   help='Labelling LLM API key (required for remote endpoints). Env: LABEL_API_KEY.')
    p.add_argument("--label-thinking", choices=("auto", "disabled", "enabled"),
                   default="auto",
                   help="for a single-digit bucket, reasoning only burns latency/cost and can consume the whole output budget. 'enabled' leaves the provider default.")
    p.add_argument("--label-send-seed", action="store_true",
                   help='Also send `seed` to the labelling endpoint. Off by default: `seed` is not in the published Z.AI schema and a strict server may reject it (temperature=0 already approximates determinism).')
    p.add_argument("--seed", type=int, default=42, help='RNG/seed (logged).')
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap the number of labelled answers (for quick fits).")
    args = p.parse_args(argv)
    logger.info("calibrate_fluency: seed=%d", args.seed)

    import os

    args.label_base_url = args.label_base_url or os.environ.get("LABEL_BASE_URL")
    args.label_model = args.label_model or os.environ.get("LABEL_MODEL")
    args.label_api_key = args.label_api_key or os.environ.get("LABEL_API_KEY")

    texts = _load_nl_texts(Path(args.report))
    logger.info("Loaded %d NL texts from the report.", len(texts))
    if not texts:
        print("No NL texts found in the report.", file=sys.stderr)
        return 1

    # Validate + build the labelling client BEFORE the expensive raw-fluency
    # pass: loading the fluency model and scoring every sentence takes minutes,
    # and the operator must learn about a bad URL / missing key immediately
    # (fail fast).
    
    
    
    client, label_model = _build_label_client(args)
    extra_kwargs = _label_kwargs(args)

    # 1. Raw per-sentence fluency (x for the fit). Unique surviving sentences
    #    are deduplicated across all answers (boilerplate repeats a lot), so the
    #    labelling pass labels each distinct sentence once.
    
    
    cfg = RewardConfig(
        local_fluency_model=args.fluency_model or DEFAULT_LOCAL_FLUENCY_MODEL,
        local_fluency_device=args.fluency_device,
    )
    logger.info("Computing raw per-sentence fluency for %d texts...", len(texts))
    sent_pairs = raw_fluency_per_sentence(texts, cfg)
    if args.max_samples is not None:
        sent_pairs = sent_pairs[: args.max_samples]
        logger.info("Capped to %d sentences (--max-samples).", len(sent_pairs))
    logger.info(
        "%d unique scoreable sentences (%d NL texts scanned).",
        len(sent_pairs), len(texts),
    )
    if len(sent_pairs) < 10:
        print(
            f'Too few labelled points ({len(sent_pairs)}',
            file=sys.stderr,
        )
        return 1

    # 2. LLM labels (y for the fit) — one per unique sentence.
    
    logger.info(
        "Labelling %d sentences with %s (thinking=%s, send_seed=%s) ...",
        len(sent_pairs), label_model, args.label_thinking, args.label_send_seed,
    )
    xs: list[float] = []
    ys: list[float] = []
    skipped = 0
    for i, (sentence, raw_score) in enumerate(sent_pairs):
        y = _label_one(client, label_model, sentence, extra_kwargs=extra_kwargs)
        if y is None:
            skipped += 1
            continue
        xs.append(raw_score)
        ys.append(y)
        if (i + 1) % 25 == 0:
            logger.info("  labelled %d / %d", i + 1, len(sent_pairs))
    logger.info("Labelled %d sentences (%d unparseable/skipped).", len(xs), skipped)
    if len(xs) < 10:
        print(
            f"Too few successful labels ({len(xs)}); check the labelling endpoint.",
            file=sys.stderr,
        )
        return 1

    # 3. Fit IsotonicRegression.
    import numpy as np
    from sklearn.isotonic import IsotonicRegression  # type: ignore[import-not-found]

    xp = np.asarray(xs, dtype=np.float64)
    yp = np.asarray(ys, dtype=np.float64)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(xp, yp)
    pred = iso.predict(xp)
    mae = float(np.mean(np.abs(pred - yp)))

    artifact = {
        "kind": "isotonic",
        "x": [float(v) for v in iso.X_thresholds_],
        "y": [float(v) for v in iso.y_thresholds_],
        "fit_stats": {
            "n": int(len(xs)),
            "label_model": label_model,
            "mean_abs_residual": mae,
            "x_min": float(xp.min()),
            "x_max": float(xp.max()),
            "y_min": float(yp.min()),
            "y_max": float(yp.max()),
            "seed": args.seed,
        },
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    logger.info("Wrote calibrator to %s", out_path)
    print(
        json.dumps(
            {
                "out": str(out_path),
                "n_thresholds": len(artifact["x"]),
                "mean_abs_residual": mae,
                "verdict": (
                    "good fit" if mae < 0.15
                    else "mediocre — consider pairwise/BT labels (--label scheme)"
                ),
            },
            ensure_ascii=False, indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
