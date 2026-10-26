'Loader: ASR summarization triplets → ``asr_summarization``.\n\n\nEach example is reassembled by matching the same numeric suffix ``NN`` across\n``inputNN.txt`` (→ user), ``systemNN.txt`` (→ system, **only for NN 01–13**),\nand ``targetNN.txt`` (→ reference). For NN 14–26 there is no ``systemNN.txt``:\n``inputNN.txt`` already begins with a role/instruction and is used verbatim as\nthe user turn, with an empty system.'

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from llm_training.config import DataConfig
from llm_training.schema import UnifiedExample

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizer

CLASSES = ("asr_summarization",)
SUBDIR = "asr_summarization"
_SUFFIX_RE = re.compile(r"input(\d+)\.txt$")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load(tokenizer: "PreTrainedTokenizer", cfg: DataConfig) -> Iterator[UnifiedExample]:
    """Yield ``asr_summarization`` examples by matching NN across the triplets."""
    base = cfg.data_dir / SUBDIR
    # Discover available NN suffixes via inputNN.txt files.
    
    suffixes: list[tuple[int, str]] = []
    for p in sorted(base.glob("input*.txt")):
        m = _SUFFIX_RE.search(p.name)
        if m:
            suffixes.append((int(m.group(1)), m.group(1).zfill(2)))
    for nn_int, nn in suffixes:
        input_path = base / f"input{nn}.txt"
        target_path = base / f"target{nn}.txt"
        system_path = base / f"system{nn}.txt"
        if not (input_path.exists() and target_path.exists()):
            continue  # incomplete triplet; skip
        user = _read_text(input_path)
        reference = _read_text(target_path)
        if not reference:
            continue
        # NN 01–13 carry a systemNN.txt; NN 14–26 do not (system stays empty).
        
        system = _read_text(system_path) if system_path.exists() else ""
        prompt: list[dict] = []
        if system:
            prompt.append({"role": "system", "content": system})
        prompt.append({"role": "user", "content": user})
        yield UnifiedExample(
            task_class="asr_summarization",
            prompt=prompt,
            reference=reference,
            reference_messages=[{"role": "assistant", "content": reference}],
            source=f"{SUBDIR}/NN{nn}",
            extra={"nn": nn_int, "has_system": bool(system)},
        )
