# LLM Training Research Artifact

This anonymous repository contains the training and evaluation code accompanying the submitted paper. It covers continued pre-training, supervised fine-tuning, warm-up, GRPO training, extraction and agent-oriented experiments, checkpoint merging, and benchmark evaluation for a bilingual Russian/English decoder-only language model.

The repository intentionally contains no author names, affiliations, personal accounts, cluster addresses, or private filesystem paths. Russian text remains only where it is part of the research object: prompts, datasets, expected outputs, language-specific rules, and tests.

## Repository layout

- `llm_training/`: reusable datasets, rewards, runtime helpers, and tool environments.
- `scripts/`: corpus construction, calibration, evaluation, and reporting utilities.
- `lm_eval_tasks/`: custom tasks for `lm-evaluation-harness`.
- `src_data/`: frozen task data and compact experiment artifacts included for reproducibility.
- `benchmark_results/`: benchmark outputs referenced by the artifact.
- `tests/`: unit and integration tests.
- Root `*_train.py` files: continued pre-training, SFT, warm-up, and GRPO entry points.
- `run_eval.py` and `run_extraction_eval.py`: evaluation entry points.
- `merge_evolve.yaml`: example evolutionary checkpoint-merging configuration. Its model paths are placeholders and must be adapted locally.

Internal cluster launch scripts, private working notes, manuscript sources, large teacher transcripts, and identity-specific probe artifacts are not part of the anonymous package.

## Installation

Python 3.10 or newer is required.

For development and CPU-only checks:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For the training and full evaluation stack:

```bash
python -m pip install -e ".[train,dev]"
```

The training extra includes GPU-oriented packages such as PyTorch, TRL, and vLLM. Select package versions and CUDA wheels appropriate for the local system. Some optional language rewards also require separately installed spaCy language models; the corresponding code degrades gracefully when they are unavailable.

## Basic checks

```bash
python -m compileall -q .
python -m pytest -m "not network and not gpu"
```

Tests marked `network` download external resources. Tests marked `gpu` require CUDA hardware and the full training stack.

## Main entry points

Each executable exposes its current options through `--help`. Typical discovery commands are:

```bash
python prepare_dataset.py --help
python cpt_tokenize.py --help
python cpt_train.py --help
python sft_train.py --help
python sft_warmup.py --help
python grpo_train.py --help
python grpo_extract_train.py --help
python run_eval.py --help
python run_extraction_eval.py --help
```

Training runs expect the model, tokenizer, dataset, output, and service locations to be supplied by command-line arguments or documented environment variables. Paths in checked-in examples use neutral placeholders and are not expected to exist unchanged.

## Data and reproducibility

The checked-in data includes Russian and English task instances because language-specific behavior is central to the experiments. Dataset licenses or provenance notes are stored next to the relevant files where available. Scripts that obtain external corpora should be reviewed before use because they may require network access, upstream acceptance terms, or local source files.

Randomized pipelines expose explicit seeds. Benchmark outputs are grouped by model and task; machine-specific model-directory components have been normalized to the directory name `model`.

See `ANONYMIZATION_NOTES.md` for the boundaries of the anonymous artifact.

## License

The repository code is released under the MIT License; see `LICENSE`. Included or downloaded datasets retain their own licenses and terms.
