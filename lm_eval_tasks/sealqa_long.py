'LONGSEAL\n\n\nscrolls_narrativeqa_filtered.py):\n\nThe LONGSEAL task (the SealQA long-context variant) for lm-eval 0.4.13.\n\nLONGSEAL is the "needle-in-a-haystack" flavor of SealQA (arXiv:2506.01062):\neach question comes with up to 50 hard negatives, one "gold" document is\nburied among them, and the gold\'s position is random. Here the k=12 variant\nis scored (the ``12_docs`` field — a list of ``{date, text}``): the model\ngets all 12 documents and the question, must find the helpful document and\nanswer. This is a generative task WITH documents (the "w/ search"\nprotocol), the closest to real RAG reading for Meno-Lite.\n\nThe prompt averages ~27.6k tokens; some examples exceed the 32,768 position\nwindow of Meno-Lite: such examples are DROPPED in ``download()`` by the\nMeno tokenizer (a deterministic skip, NOT left truncation — the repo\nconvention and the PRUNE precedent of scrolls_narrativeqa_filtered).\n\nA class task (following scrolls_local.py / scrolls_narrativeqa_filtered.py):\nthe data download + the length filter are overridden, the metric is the\nshared deterministic "relaxed exact match" proxy from sealqa.py. Plug it in\nvia ``--include_path lm_eval_tasks`` running lm_eval from the repo checkout\nroot.'

from __future__ import annotations

import os as _os
import sys as _sys


# sys.modules

# lm-eval's yaml loader registers the include-path directory modules in
# sys.modules under their FULL path (not by file name), so a plain sibling
# import is unreliable — the directory is appended to sys.path itself.
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

from lm_eval.api.instance import Instance  # noqa: E402
from lm_eval.api.metrics import mean  # noqa: E402
from lm_eval.api.task import ConfigurableTask  # noqa: E402

from sealqa import relaxed_contains_score  # noqa: E402



# The shared-window Meno-Lite tokenizer (tokenizer_source: base in the
# merges) — the same hard-coded path as scrolls_narrativeqa_filtered.
_MENO_TOKENIZER = "/path/to/storage/models/meno-lite-0.1"


# The dataset revision pinned for reproducibility (SealQA is dynamic).
_SEALQA_REVISION = "0c298f519c871579c75920189b3858da29fe2c81"

# max_gen_toks 512.

# The doc_to_text budget: 32768 - the chat template (~50 tokens) -
# max_gen_toks 512. The PRUNE filter measures doc_to_text WITHOUT the chat
# template.
_PRUNE_MAX_TOKENS = 32000


class SealQALong(ConfigurableTask):
    'LONGSEAL within the window.'

    VERSION = 0
    DATASET_PATH = "vtllms/sealqa"
    DATASET_NAME = "longseal"

    def __init__(self, config=None):
        
        
        # A minimal config (like _SCROLLSTask): everything else is a class
        # method. The yaml config is deliberately discarded.
        super().__init__(config={"metadata": {"version": self.VERSION}})

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        
        
        
        # 2026-09-28)
        # drift_probe/scrolls_local
        
        
        # download() loads a SINGLE split (split="test" => Dataset, NOT a
        # DatasetDict), so self.dataset["test"] would look up the COLUMN
        # "test" and die with KeyError (the cluster-run crash of
        # 2026-09-28) — the dataset already IS the filtered test split.
        # drift_probe/scrolls_local load a dict (data_files={"test": ...} /
        # no split=), where self.dataset["test"] is legal — the wrong
        # precedent was copied from there.
        return self.dataset

    def download(self, *args, **kwargs):
        from datasets import load_dataset

        self.dataset = load_dataset(
            self.DATASET_PATH,
            self.DATASET_NAME,
            split="test",
            revision=_SEALQA_REVISION,
        )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(_MENO_TOKENIZER)

        def _fits_window(doc):
            
            # A deterministic skip of over-long examples, no left truncation.
            return (
                len(tokenizer(self.doc_to_text(doc)).input_ids)
                <= _PRUNE_MAX_TOKENS
            )

        before = len(self.dataset)
        self.dataset = self.dataset.filter(_fits_window)
        print(
            f"[sealqa_long] examples after prune: {len(self.dataset)} / {before}"
        )

    def doc_to_text(self, doc):
        blocks = []
        for i, d in enumerate(doc["12_docs"], 1):
            date = (d.get("date") or "").strip()
            header = f"Document {i}:" + (f" [{date}]" if date else "")
            blocks.append(f"{header}\n{d['text']}")
        return (
            "Answer the question using only the provided documents.\n\n"
            + "\n\n".join(blocks)
            + f"\n\nQuestion: {doc['question']}"
        )

    def doc_to_target(self, doc):
        return " " + doc["answer"]

    def process_results(self, doc, results):
        prediction = results[0] if results else ""
        return {"contains": relaxed_contains_score(prediction, doc["answer"])}

    def aggregation(self):
        return {"contains": mean}

    def higher_is_better(self):
        return {"contains": True}

    def construct_requests(self, doc, ctx, **kwargs):
        # build_all_requests lm-eval
        # chat_template
        
        
        
        # lm-eval's build_all_requests ALWAYS passes apply_chat_template
        # and chat_template (even when False); the base ConfigurableTask
        # pops them: Instance has no such fields (the same TypeError that
        # crashed the first CMA-ES population on 2026-09-27 in
        # drift_probe). The chat template is applied UPSTREAM — in
        # fewshot_context — popping does not disable it.
        kwargs.pop("apply_chat_template", None)
        kwargs.pop("chat_template", None)
        return Instance(
            request_type="generate_until",
            doc=doc,
            arguments=(
                ctx,
                {
                    "until": ["<|im_end|>", "</s>", "<|endoftext|>"],
                    "max_gen_toks": 512,
                },
            ),
            idx=0,
            **kwargs,
        )


def warm_up(limit: int = 20) -> None:
    'A no-GPU warm-up: load a slice and print one example.'
    from datasets import load_dataset

    ds = load_dataset(
        "vtllms/sealqa", "longseal", split=f"test[:{limit}]",
        revision=_SEALQA_REVISION,
    )
    doc = ds[0]
    print(f"[sealqa_long] test slice: {len(ds)} rows loaded")
    print(f"[sealqa_long] question: {doc['question']}")
    print(f"[sealqa_long] gold answer: {doc['answer']}")
    print(f"[sealqa_long] canary: {doc['canary']}")


if __name__ == "__main__":
    warm_up()
