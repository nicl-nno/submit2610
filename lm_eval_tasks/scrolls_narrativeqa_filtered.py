"PRUNE_MAX_TOKENS,\nlm_eval/tasks/scrolls/task.py\n--include_path lm_eval_tasks\n\nNarrativeQA without left truncation: over-long examples are skipped.\nFull NarrativeQA inputs (avg ~51.7k words ~ 40k tokens) exceed the\n32,768 position window of Meno-Lite: stock lm-eval silently\nleft-truncates them (the document loses its head, the question at the\nend survives). Instead of truncation, over-long examples are SKIPPED via\nlm-eval's built-in SCROLLS filtering mechanism (PRUNE_TOKENIZERS /\nPRUNE_MAX_TOKENS, lm_eval/tasks/scrolls/task.py): the metric is computed\non a deterministic subset shared by every compared model (the tokenizer\nis shared — tokenizer_source: base in the merges). Plug the task in via\nthe --include_path lm_eval_tasks flag, running lm_eval from the repo\ncheckout root. The base class is NarrativeQALocal (scrolls_local.py): the\ndata is loaded via the local tau/scrolls unpacking (datasets 3.2.0\ncompatibility)."


# sys.modules

# lm-eval's yaml loader registers the include-path directory modules in
# sys.modules under their FULL path (not by file name), so a plain sibling
# import is unreliable — the directory is appended to sys.path itself.
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

from scrolls_local import NarrativeQALocal  # noqa: E402

MENO_TOKENIZER = "/path/to/storage/models/meno-lite-0.1"


class NarrativeQAFiltered(NarrativeQALocal):
    'Internal implementation detail.'

    # max_gen_toks 1024.
    
    # Budget: 32768 - the chat template (~150 tokens) - max_gen_toks 1024.
    # PRUNE measures doc_to_text (text + question) WITHOUT the chat template.
    PRUNE_TOKENIZERS = [MENO_TOKENIZER]
    PRUNE_MAX_TOKENS = 30000
    PRUNE_NUM_PROC = 8

    def download(self, *args, **kwargs):
        super().download(*args, **kwargs)
        
        # Diagnostics: how many examples survived the filter, per split.
        sizes = {split: len(ds) for split, ds in self.dataset.items()}
        print(f"[scrolls_narrativeqa_filtered] examples after prune: {sizes}")
