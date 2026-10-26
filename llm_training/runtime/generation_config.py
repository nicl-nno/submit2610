"Generation-config hygiene for greedy HF ``.generate`` loops.\n\n\nCheckpoints distilled from the Qwen2.5 line ship a ``generation_config.json``\nwith sampling defaults (``temperature`` / ``top_p`` / ``top_k`` — e.g. 0.7 /\n0.8 / 20). Every entry point of this project generates its validation and\nprobe texts GREEDILY (``do_sample=False``): the drift probe, the judge\nanchors (grpo_train.py / sft_train.py), the per-epoch warm-up validation\n(sft_warmup.py) and the chrF++ validation (cpt_train.py). transformers'\n``GenerationConfig.validate()`` then logs\n\n    The following generation flags are not valid and may be ignored:\n    ['temperature', 'top_p', 'top_k'].\n\non EVERY ``.generate()`` call (76 lines per drift probe, 167 per anchor —\npure log noise; the flags are genuinely ignored under greedy decoding, the\noutputs are unaffected). All the offending checks are guarded by\n``is not None``, so unsetting the greedy-unused sampling flags on the loaded\nmodel's ``generation_config`` silences the warning at the source, keeps the\ncheckpoint's ``eos_token_id`` / ``pad_token_id`` intact (never drop the\nwhole config — generation STOPPING depends on them) and is a no-op for\ncheckpoints that carry no sampling defaults.\n``top_p``\n(``do_sample=False``): ``GenerationConfig.validate()``\n``pad_token_id``"

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Sampling-only flags that ``GenerationConfig.validate()`` flags as "not
#: valid" under greedy decoding (``do_sample=False``). Unset (``None``) —
#: never the eos/pad/bos ids, never the length caps.


# : ``None``; eos/pad/bos
GREEDY_UNUSED_SAMPLING_FLAGS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "typical_p",
    "epsilon_cutoff",
    "eta_cutoff",
)


def unset_greedy_sampling_flags(model: Any) -> list[str]:
    'Unset the greedy-unused sampling flags on ``model.generation_config``.\n\n    ``model.generation_config``.\n\n    Call ONCE after the model is loaded (before any ``.generate()``). Works\n    transparently through peft wrappers (``PeftModel`` forwards the\n    attribute to the base model). Returns the list of flags that were\n    actually unset (``None``-valued / absent flags are skipped).'
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None:
        return []
    unset: list[str] = []
    for flag in GREEDY_UNUSED_SAMPLING_FLAGS:
        if getattr(gen_cfg, flag, None) is not None:
            setattr(gen_cfg, flag, None)
            unset.append(flag)
    if unset:
        logger.info("Generation-config hygiene: unset greedy-unused sampling flags (do_sample=False in probe/anchor/validation loops): %s — silences the per-generate() 'generation flags are not valid' warnings; outputs unaffected.", unset)
    return unset
