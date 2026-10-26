'``sft_train.py``/``grpo_train.py`` (``llm_training/runtime/drift_probe.py``\n+ ``src_data/drift_probe.jsonl``\n\n``scripts/build_drift_probe.py``\n\n``llm_training/runtime/drift_probe.py``\n``tests/test_lm_eval_tasks/test_drift_probe.py``\n``--task-search-path``\n``sealqa_long.py``\n\nThe drift-probe task (drift_probe) for lm-eval 0.4.13: an instruction-\nfollowing regression guard for the evolutionary merging.\n\nThe twelfth task of the evolutionary selection: 76 synthetic\nmachine-checkable instruction-following prompts (RU/EN, 9 check types — word\ncounts, no commas, JSON keys, keyword frequency, the answer language) — the\nsame instrument as the degradation control of ``sft_train.py``/\n``grpo_train.py`` checkpoint selection (``llm_training/runtime/drift_probe.py``\n+ ``src_data/drift_probe.jsonl``). Motivation: the 2026-09 attribution\nshowed the SFT/GRPO parents lose ~15 pp on IFEval — an axis the evolution\nfitness did not guard.\n\nThere is no leak into the final benchmark: the prompts are synthetic (the\nproducer is ``scripts/build_drift_probe.py``, the seed is fixed) and by\nconstruction are not IFEval — the "evolution ≠ final benchmarking" invariant\nholds; the independent measure stays the final-bench IFEval. With this task\nthe probe becomes an OPTIMIZED objective of the merge line (it is no longer\nan independent post-merge check of that axis); the training-side probe gate\nis unchanged. The data (``src_data/drift_probe.jsonl``) is frozen for a run\nseries: regeneration breaks fitness comparability (the SealQA-revision\ndiscipline).\n\nThe module is SELF-CONTAINED: the checks are ported from\n``llm_training/runtime/drift_probe.py`` without logic changes, and the\nbehaviour equivalence is enforced by the test\n``tests/test_lm_eval_tasks/test_drift_probe.py``. Importing\n``llm_training`` is not allowed: the mergekit Ray actors resolve the module\nfrom ``--task-search-path`` without the project package on sys.path. The\ndata is read locally, by the path relative to ``__file__`` (the actors\'\nworking directory is the neutral ``$STORAGE_PATH``). The task class follows\n``sealqa_long.py`` and ``scrolls_local.py``.'

from __future__ import annotations

import json
from pathlib import Path

__all__ = [
    "CHECK_TYPES",
    "DriftProbeTask",
    "check_response",
]


# : llm_training/runtime/drift_probe.py

#: The check types with their parameters — the ``check`` field schema (a
#: port from llm_training/runtime/drift_probe.py; the behaviour equivalence
#: is enforced by tests).
CHECK_TYPES: tuple[str, ...] = (
    "min_words",
    "max_words",
    "exact_words",
    "no_commas",
    "json_keys",
    "keyword_min",
    "keyword_max",
    "language_ru",
    "language_en",
)




# The single source of data — the frozen src_data/drift_probe.jsonl (no
# copy: duplicating the storage would make the versions diverge). The path
# is relative to __file__ — the actors' working directory never
# participates.
_PROBE_JSONL = (
    Path(__file__).resolve().parent.parent / "src_data" / "drift_probe.jsonl"
)



# The generation budget = the probe's anchor-side max_new_tokens (256) in
# the training scripts; the longest probe asks for "at least 60 words"
# (~100 tokens).
_MAX_GEN_TOKS = 256

_UNTIL = ["<|im_end|>", "</s>", "<|endoftext|>"]




# --- The ported checks (the logic is unchanged; the behaviour must coincide
# --- with llm_training/runtime/drift_probe.py — the equivalence test) ------


def _words(text: str) -> list[str]:
    'Whitespace tokens carrying ≥1 alphanumeric character.'
    return [t for t in (text or "").split() if any(c.isalnum() for c in t)]


def _strip_json_fences(text: str) -> str:
    'Strips markdown fences around a JSON payload (an observed model\n    behaviour).'
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        if first_nl != -1:
            cleaned = cleaned[first_nl + 1 :]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    return cleaned.strip()


def check_response(check: dict, text: str) -> bool:
    'One deterministic instruction check over a response (a port).\n\n    Unknown check types fail CLOSED (``False``): a record the runtime cannot\n    interpret must never silently pass.'
    ctype = str(check.get("type"))
    if ctype == "min_words":
        return len(_words(text)) >= int(check["n"])
    if ctype == "max_words":
        return len(_words(text)) <= int(check["n"])
    if ctype == "exact_words":
        return len(_words(text)) == int(check["n"])
    if ctype == "no_commas":
        return "," not in (text or "")
    if ctype == "json_keys":
        try:
            payload = json.loads(_strip_json_fences(text))
        except (json.JSONDecodeError, ValueError):
            return False
        if not isinstance(payload, dict):
            return False
        return set(check["keys"]) <= set(payload.keys())
    if ctype == "keyword_min":
        lowered = (text or "").lower()
        return lowered.count(str(check["keyword"]).lower()) >= int(check["n"])
    if ctype == "keyword_max":
        lowered = (text or "").lower()
        return lowered.count(str(check["keyword"]).lower()) <= int(check["n"])
    if ctype == "language_ru":
        letters = [c for c in (text or "") if c.isalpha()]
        if not letters:
            return False
        return sum("а" <= c.lower() <= "я" or c.lower() == "ё" for c in letters) / len(
            letters
        ) >= 0.8
    if ctype == "language_en":
        letters = [c for c in (text or "") if c.isalpha()]
        if not letters:
            return False
        return sum(c.isascii() for c in letters) / len(letters) >= 0.8
    return False




# --- scrolls_local.py) ------------------------------------------------------
# --- The task class (requires lm_eval; without it the class is not created,
# --- the checks stay importable for the equivalence tests — the
# --- scrolls_local.py pattern) ----------------------------------------------

try:
    from lm_eval.api.instance import Instance  # noqa: E402
    from lm_eval.api.metrics import mean  # noqa: E402
    from lm_eval.api.task import ConfigurableTask  # noqa: E402
except ImportError:  
    Instance = mean = ConfigurableTask = None  # type: ignore[assignment]

if ConfigurableTask is not None:

    class DriftProbeTask(ConfigurableTask):
        'The drift probe as a generate_until task: 76 synthetic instructions.'

        VERSION = 0
        DATASET_PATH = "json"

        def __init__(self, config=None):
            
            
            # A minimal config (as in SealQALong): everything else is a class
            # method. The yaml config is deliberately discarded.
            super().__init__(config={"metadata": {"version": self.VERSION}})

        def has_training_docs(self):
            return False

        def has_validation_docs(self):
            return False

        def has_test_docs(self):
            return True

        def download(self, *args, **kwargs):
            
            # _LocalZipDownloadMixin scrolls_local.py.
            # A local jsonl (no network), as a dict with the test split — as
            # in the scrolls_local.py _LocalZipDownloadMixin.
            from datasets import load_dataset

            if not _PROBE_JSONL.is_file():
                raise FileNotFoundError(
                    f"drift probe data not found: {_PROBE_JSONL} "
                    "(produce it with scripts/build_drift_probe.py and keep "
                    "it frozen for the run series)"
                )
            self.dataset = load_dataset(
                "json", data_files={"test": str(_PROBE_JSONL)}
            )

        def test_docs(self):
            return self.dataset["test"]

        def doc_to_text(self, doc):
            
            
            # The probe prompt is self-contained; lm-eval applies the chat
            # template — matching the anchor side (apply_chat_template).
            return doc["prompt"]

        def doc_to_target(self, doc):
            
            # There is no gold string: the checks are parametric (the check
            # field).
            return ""

        def process_results(self, doc, results):
            text = results[0] if results else ""
            return {"probe_acc": float(check_response(doc["check"], text))}

        def aggregation(self):
            return {"probe_acc": mean}

        def higher_is_better(self):
            return {"probe_acc": True}

        def construct_requests(self, doc, ctx, **kwargs):
            # build_all_requests lm-eval
            # chat_template
            # ConfigurableTask.construct_requests
            
            # CMA-ES 2026-09-27
            # fewshot_context (ctx
            
            # lm-eval's build_all_requests ALWAYS passes apply_chat_template
            # and chat_template (even when apply_chat_template=False); the
            # base ConfigurableTask.construct_requests pops them: Instance
            # has no such fields (the TypeError crashed the first CMA-ES
            # population, 2026-09-27). The chat template is applied
            # UPSTREAM — in fewshot_context (ctx arrives already rendered) —
            # so popping the kwargs does not disable it.
            kwargs.pop("apply_chat_template", None)
            kwargs.pop("chat_template", None)
            return Instance(
                request_type="generate_until",
                doc=doc,
                arguments=(ctx, {"until": list(_UNTIL), "max_gen_toks": _MAX_GEN_TOKS}),
                idx=0,
                **kwargs,
            )


def warm_up(limit: int = 3) -> None:
    'A no-GPU, no-lm_eval warm-up: read the jsonl, print samples.\n\n    Called manually before a run (like sealqa.warm_up): a missing data file\n    surfaces before the vLLM init.'
    if not _PROBE_JSONL.is_file():
        raise FileNotFoundError(f"drift probe data not found: {_PROBE_JSONL}")
    with open(_PROBE_JSONL, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    print(f"[drift_probe] records: {len(records)}")
    for rec in records[:limit]:
        print(f"[drift_probe] {rec['id']} [{rec['lang']}] check={rec['check']}")
        print(f"[drift_probe]   prompt: {rec['prompt'][:100]}")


if __name__ == "__main__":
    warm_up()
