'Build a static-YaRN long-context catalog of a Qwen2-family model (inference-only).\n\n\nMeno-Lite descends from the Qwen2.5-7B architecture: the base family is\nbuilt for the YaRN x4 context extension (the Qwen2.5 model-card recipe),\nwhile the shipped ``config.json`` keeps the native 32,768-token window\n(``rope_scaling: null``). This script assembles a **YaRN catalog** for the\nlong-context benchmarking runs: symbolic links to every model file (no\nweight copies — ~0 bytes instead of ~16 GiB) plus a patched ``config.json``\nwith the static YaRN block exactly as the Qwen2.5 card prescribes::\n\n    "rope_scaling": {\n        "factor": 4.0,\n        "original_max_position_embeddings": 32768,\n        "type": "yarn"\n    }\n\n``max_position_embeddings`` itself stays untouched: vLLM derives the\neffective maximum as ``max_position_embeddings * factor`` (131,072 at the\ndefault factor), and transformers wants ``original_max_position_embeddings``\nto equal ``max_position_embeddings`` (otherwise it warns).\n\nMeno-Lite\n``config.json``\n``max_position_embeddings * factor`` (131 072\nmax_position_embeddings``\n\nUsage\n\n    python scripts/make_yarn_model.py --model /path/to/storage/models/meno-lite-0.1\n    # -> /path/to/storage/models/meno-lite-0.1-yarn\n    python scripts/make_yarn_model.py --model $MODEL --out $MODEL_YARN --factor 4.0\n\nInvariants\n\n* The catalog is for INFERENCE / BENCHMARKING ONLY: never feed it into\n  mergekit-evolve or the merge pool — rope_scaling would leak into the\n  CMA-ES fitness and into every merged descendant.\n  mergekit-evolve\n* vLLM supports only STATIC YaRN: the multiplier scales attention for ALL\n  positions (the Qwen2.5 card: "potentially impacting performance on\n  shorter texts") — hence the dedicated bench groups (``yarn`` /\n  ``yarn_ab`` of ``bench_slurm.sh``), never ``main``/``scrolls``/``ll``.\ntexts»)\n  ``bench_slurm.sh``\n* Idempotent: a valid existing catalog exits 0 without changes; a stale one\n  (config mismatch) is rebuilt — but only when it holds nothing besides\n  links and ``config.json`` (real weight files require ``--force``).\n  ``--force``).'

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# : The rope_scaling type of the Qwen2.5 recipe
YARN_TYPE = "yarn"

#: The card-default extension factor (32768 -> 131072).

DEFAULT_FACTOR = 4.0


def _load_json(path: Path) -> dict:
    'Load a JSON file, raising a bilingual SystemExit on failure.'
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(
            f'ERROR: file not found'
        ) from None
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f'ERROR: invalid JSON in'
        ) from None


def validate_source_config(cfg: dict) -> None:
    'Assert the source config is a Qwen2-family model with no rope scaling.'
    archs = [str(a) for a in (cfg.get("architectures") or [])]
    if not any(a.startswith("Qwen2") for a in archs):
        raise SystemExit(
            f'ERROR: not a Qwen2-family model (architectures={archs}'
        )
    if "max_position_embeddings" not in cfg:
        raise SystemExit(
            'ERROR: the config has no max_position_embeddings.'
        )
    if cfg.get("rope_scaling"):
        raise SystemExit(
            f"ERROR: rope_scaling is already set in the source config: {cfg['rope_scaling']}"
        )


def expected_config(cfg: dict, factor: float) -> dict:
    'Return the patched config: a copy with the static YaRN block set.'
    patched = dict(cfg)
    patched["rope_scaling"] = {
        "factor": float(factor),
        "original_max_position_embeddings": int(cfg["max_position_embeddings"]),
        "type": YARN_TYPE,
    }
    return patched


def _only_links_and_config(out: Path) -> bool:
    'True iff the directory holds nothing but links, config.json, empty dirs.'
    for p in out.iterdir():
        if p.is_symlink():
            continue
        if p.is_file() and p.name == "config.json":
            continue
        if p.is_dir() and not any(p.iterdir()):
            continue
        return False
    return True


def overlay_is_valid(model: Path, out: Path, expected: dict) -> bool:
    'True iff the existing catalog matches the source files and the target config.'
    try:
        have = _load_json(out / "config.json")
    except SystemExit:
        return False
    if have != expected:
        return False
    for f in model.iterdir():
        if not f.is_file() or f.name == "config.json":
            continue
        link = out / f.name
        if not link.is_symlink() or link.resolve() != f.resolve():
            return False
    return True


def build_overlay(
    model: Path, out: Path, *, factor: float = DEFAULT_FACTOR, force: bool = False,
) -> str:
    'Assemble (or reuse) the YaRN catalog; returns ``"built"`` or ``"reused"``.\n\n    ``"reused"``.'
    if float(factor) <= 1.0:
        raise SystemExit(
            'ERROR: --factor must be > 1.0'
        )
    model = Path(model)
    out = Path(out)
    cfg = _load_json(model / "config.json")
    validate_source_config(cfg)
    expected = expected_config(cfg, factor)

    model_r = model.resolve()
    out_r = out.resolve()
    if out_r == model_r or out_r in model_r.parents:
        raise SystemExit(
            f'ERROR: --out must differ from (and not contain) the model dir:\n  model = {model_r}\n  out   = {out_r}'
        )

    if out.exists():
        if not force and overlay_is_valid(model, out, expected):
            return "reused"
        if not force and not _only_links_and_config(out):
            raise SystemExit(
                f'ERROR: {out}holds real files (not just links + config.json); pass --force to rebuild over it.'
            )
        shutil.rmtree(out)

    out.mkdir(parents=True)
    linked = 0
    for f in sorted(model.iterdir()):
        if not f.is_file() or f.name == "config.json":
            continue
        (out / f.name).symlink_to(f.resolve())
        linked += 1
    (out / "config.json").write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return "built"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="make_yarn_model.py",
        description=(
            'Build a static-YaRN long-context catalog of a Qwen2-family model (symlinks + a patched config.json; inference/benchmarking ONLY).'
        ),
    )
    parser.add_argument("--model", required=True,
                        help='Source model directory')
    parser.add_argument("--out", default=None,
                        help='Output catalog (default: <model>-yarn)')
    parser.add_argument("--factor", type=float, default=DEFAULT_FACTOR,
                        help=f'YaRN extension factor (default: {DEFAULT_FACTOR}')
    parser.add_argument("--force", action="store_true",
                        help='Rebuild over a valid/existing catalog')
    args = parser.parse_args(argv)

    model = Path(args.model)
    out = Path(args.out) if args.out else Path(f"{model}-yarn")
    status = build_overlay(model, out, factor=args.factor, force=args.force)

    cfg = _load_json(out / "config.json")
    mpe = int(cfg["max_position_embeddings"])
    factor = float(cfg["rope_scaling"]["factor"])
    print(f"{status}: {out}")
    print(
        f"  rope_scaling: {json.dumps(_load_json(model / 'config.json').get('rope_scaling'))}"
        f" -> {json.dumps(cfg['rope_scaling'])}"
    )
    print(
        f"  max_position_embeddings: {mpe} (unchanged); the effective vLLM limit:"
        f" {mpe} x {factor:g} = {int(mpe * factor)}"
    )
    print(
        'REMINDER: inference/bench ONLY — never into mergekit-evolve or the merge pool; the static YaRN affects short contexts too — keep it out of main/scrolls/ll (bench_slurm.sh groups yarn / yarn_ab only).'
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
