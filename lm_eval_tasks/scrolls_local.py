"lm-eval 0.4.13\n``zip://<name>/train.jsonl::https://huggingface.co/datasets/tau/scrolls/resolve/main/<name>.zip``,\n``data_files``\n(``hf_hub_download``\n``$HF_HOME/tau_scrolls_extracted/<name>/``\ninclude_path\n\nLocal data loading for the SCROLLS tasks under lm-eval 0.4.13 + datasets 3.2.0.\n\nlm-eval 0.4.13 builds ``zip://<name>/train.jsonl::https://…`` chained\ndata_files URLs for the SCROLLS tasks, and the node's pinned datasets 3.2.0\ncannot resolve such chains — ANY ``zip://…::…`` (a local archive included)\ndies with ``FileNotFoundError`` (the crash of all 8 tau/scrolls consumers in\nthe 2026-09-22 run). Newer datasets resolve the chain (verified on 5.0.1),\nbut upgrading datasets in vllm_env is forbidden: mergekit-evolve needs 3.2.0\nfor the EleutherAI script datasets. The fix: the zip is fetched through the\nproven hub path (``hf_hub_download`` — the HF_HOME cache, file locks) and\nunpacked into ``$HF_HOME/tau_scrolls_extracted/<name>/``, after which\n``load_dataset`` receives plain local jsonl files. The shadow yamls in this\ndirectory override the built-in tasks with the same names (in lm-eval 0.4.13\nthe TaskManager gives include_path priority over the built-in tasks), so\nevery consumer of the name transparently gets the fixed loading. The\nmetrics, prompts and the evaluation protocol are unchanged; on datasets>=5\nthe fix is inert but correct."

import os
import shutil
import zipfile

SCROLLS_SUBSETS = (
    "summ_screen_fd",
    "qasper",
    "qmsum",
    "narrative_qa",
    "gov_report",
    "contract_nli",
    "quality",
)


def _extracted_dir(name: str) -> str:
    '``hf_hub_download``\n\n    The directory with the unpacked ``tau/scrolls/<name>.zip`` (idempotent).\n\n    The ``.done`` marker is the source of truth: if it exists, neither the\n    unpacking nor the hub is touched. Otherwise the zip is fetched via\n    ``hf_hub_download`` (the hub cache: locks, re-download only on change),\n    extracted into a temporary directory and atomically renamed into the\n    final one; the marker guards against the race of parallel benchmark\n    tasks (up to 10 processes share HF_HOME).'
    root = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "tau_scrolls_extracted",
        name,
    )
    if os.path.exists(os.path.join(root, ".done")):
        return os.path.join(root, name)

    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(
        repo_id="tau/scrolls", repo_type="dataset", filename=f"{name}.zip"
    )
    from filelock import FileLock

    os.makedirs(os.path.dirname(root), exist_ok=True)
    with FileLock(root + ".lock"):
        if not os.path.exists(os.path.join(root, ".done")):
            tmp = root + f".tmp.{os.getpid()}"
            shutil.rmtree(tmp, ignore_errors=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
            try:
                os.rename(tmp, root)
            except OSError:
                
                # a concurrent process has won — use its result
                shutil.rmtree(tmp, ignore_errors=True)
            with open(os.path.join(root, ".done"), "w"):
                pass
    return os.path.join(root, name)


try:
    from datasets import load_dataset
    from lm_eval.tasks.scrolls.task import (
        ContractNLI,
        GovReport,
        NarrativeQA,
        QMSum,
        Qasper,
        QuALITY,
        SummScreenFD,
        _drop_duplicates_in_input,
    )
except ImportError:
    
    
    # a machine without lm_eval (local dev/CI): only the unpacking helpers
    # stay importable; the shadow yamls require lm_eval by definition
    pass
else:
    class _LocalZipDownloadMixin:
        'download() through the local unpacking instead of the zip chain.'

        def download(self, *args, **kwargs):
            data_dir = _extracted_dir(self.DATASET_NAME)
            self.dataset = load_dataset(
                "json",
                data_files={
                    split: os.path.join(data_dir, f"{split}.jsonl")
                    for split in ("train", "validation")
                },
            )
            for split in self.dataset:
                self.dataset[split] = _drop_duplicates_in_input(
                    self.dataset[split]
                )
            if self.PRUNE_TOKENIZERS is not None:
                self.prune()

    class GovReportLocal(_LocalZipDownloadMixin, GovReport):
        'the scrolls_govreport shadow.'

    class SummScreenFDLocal(_LocalZipDownloadMixin, SummScreenFD):
        'the scrolls_summscreenfd shadow.'

    class QMSumLocal(_LocalZipDownloadMixin, QMSum):
        'the scrolls_qmsum shadow.'

    class QasperLocal(_LocalZipDownloadMixin, Qasper):
        'the scrolls_qasper shadow.'

    class QuALITYLocal(_LocalZipDownloadMixin, QuALITY):
        'the scrolls_quality shadow.'

    class NarrativeQALocal(_LocalZipDownloadMixin, NarrativeQA):
        'the scrolls_narrativeqa shadow.'

    class ContractNLILocal(_LocalZipDownloadMixin, ContractNLI):
        'the scrolls_contractnli shadow.'


def warm_up() -> None:
    'A no-GPU warm-up: unpack every subset and load the metric.\n\n    Called by bench_slurm.sh before the parallel task fan-out: the shared\n    cache (the hub zip archives + the unpacking + the metric module) is\n    prepared once, and any network/environment failure shows up BEFORE the\n    ten vLLM initializations.'
    for name in SCROLLS_SUBSETS:
        print(f"[scrolls_local] {name}: {_extracted_dir(name)}")
    from evaluate import load as _eval_load
    from lm_eval.tasks.scrolls.task import _download_metric

    _eval_load(_download_metric(), config_name="gov_report")
    print("[scrolls_local] metric: OK")


if __name__ == "__main__":
    warm_up()
