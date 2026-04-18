# Tests

This directory contains the tests included in the package.

## One-shot validation: `validate.sh`

Runs profiling pipeline validation and full model package validation in sequence. Sets `REPO_ROOT` from the script location, so you can run it from the repository root or from `tests/`. Exits on first failure (`set -e`).

**Run:**

```bash
bash tests/validate.sh
```

From repository root, or `./validate.sh` when already in `tests/`. See sections below for dataset requirements and individual script options.

## Downloading datasets

The validation scripts expect data under the repository root at `datasets/`. These datasets are not shipped with the package (they are large). Download them from Hugging Face and place files under `repo/datasets/` as follows:

| Script / run | Expected path under `datasets/` | Hugging Face dataset |
|--------------|----------------------------------|----------------------|
| Profiling Run 1 (MT en→zh) | `opus-100/en-zh/train-00000-of-00001.parquet` | `Helsinki-NLP/opus-100` (config e.g. `en-zh`) |
| Profiling Run 2 (MT en→it) | `opus-100/en-it/train-00000-of-00001.parquet` | `Helsinki-NLP/opus-100` (config e.g. `en-it`) |
| Profiling Run 3 (summarization) | `xsum/xsum_train.jsonl` | `EdinburghNLP/xsum` |
| Full model validation | `wmt24pp/en-zh_CN.jsonl` | Your WMT-style JSONL (source/target columns) |

The same paths (e.g. `wmt24pp/en-zh_CN.jsonl`, `wmt24pp/en-it_IT.jsonl`) are used by the [machine translation benchmarks](../benchmarks/README.md#machine-translation).

Example: from the repo root, use the `datasets` library to download and save (e.g. `load_dataset("Helsinki-NLP/opus-100", "en-zh", split="train")` then save the split to `datasets/opus-100/en-zh/` in the format the script expects). Adjust for xsum and for your MT eval JSONL.

## Smoke tests: `test_vocab_tailor.py`

Package import, version format, `from_pretrained` (with and without LMDB), and `generate()` return value and `gen_metrics` when the tracker is enabled. No dataset required; the first run downloads the model from Hugging Face (`Qwen/Qwen3-1.7B`).

**Run (from repository root):**

```bash
python tests/test_vocab_tailor.py
```

Or: `pytest tests/test_vocab_tailor.py -v`

## Profiling pipeline validation: `validate_profiling_pipeline.py`

Runs `vocab-tailor-build-vocab` for up to three reference configs and compares the generated JSON to the reference files in `vocab_tailor.static_vocab`. Requires `pip install -e ".[profiling]"`.

**Run 1 (MT en→zh) and Run 2 (MT en→it)** use fixed default paths under `datasets/opus-100/` (this package does not ship opus-100; ensure data exists or the build-vocab CLI can load from the Hub). **Run 3 (summarization)** uses `datasets/xsum/xsum_train.jsonl` (included) or Hub.

Use `--run 1` to run only Run 1; `--run 1 3` for Run 1 and 3; omit `--run` to run all three.

**Run (from repository root):**

```bash
PYTHONPATH=src python tests/validate_profiling_pipeline.py [--run 1] [--run 1 3] [--output-dir DIR]
```

## Full model validation: `validate_model_package.py`

Full-stack MT evaluation: load VocabTailor with `from_pretrained`, run on the MT dataset, compute metrics. Requires at least one of `--input_aware` or `--profiling_file`. Omit `--profiling_file` for dynamic-only (use with `--input_aware`). Optional `--compare` to compare against golden metrics in `tests/golden/`.

Default dataset: `datasets/wmt24pp/en-zh_CN.jsonl` (included in the package).

**Run (from repository root):**

```bash
PYTHONPATH=src python tests/validate_model_package.py --input_aware [--compare] [--lmdb_path PATH] [--profiling_file PATH] [options]
```

See `python tests/validate_model_package.py --help` for all options.
