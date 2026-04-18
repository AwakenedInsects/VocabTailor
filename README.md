# VocabTailor

<a href="https://arxiv.org/abs/2508.15229">
  <img alt="arxiv" src="https://img.shields.io/badge/arXiv-2508.15229-b31b1b.svg">
</a>

Input-aware vocabulary tailoring for causal LMs (e.g. Qwen3): reduce memory and compute at inference by using only a subset of the vocabulary per request. Supports optional LMDB offloading and multiple resize strategies (realloc, split_linear, prealloc).

## Setup

From the repository root (where `pyproject.toml` is), create a virtual environment and install:

```bash
pip install -r requirements.txt
pip install -e .
```

For the build-vocab CLI (static vocabulary pipeline), install the profiling extra:

```bash
pip install -e ".[profiling]"
```

If you installed with `pip install -e .`, you can run examples and tests from the repository root without setting `PYTHONPATH`. Without installing, use `PYTHONPATH=src` (see below).

## Requirements

- Python >= 3.9
- Core: `torch>=2.1`, `transformers>=4.54`, `lmdb`, `psutil`, `accelerate`, `safetensors`, `huggingface_hub`, `tqdm`

See [requirements.txt](requirements.txt) and [pyproject.toml](pyproject.toml) for the full list. Optional extras:

- `[profiling]` — `datasets` (for `vocab-tailor-build-vocab` CLI)
- `[dev]` — `datasets`, `evaluate` (for validation scripts)
- `[hub]` — `huggingface_hub` (included in core)
- `[ie]` — `lm-eval` (for information-extraction workflows)

## Project structure

```
VocabTailor/
├── README.md                           # This file
├── LICENSE                             # Apache-2.0
├── pyproject.toml                      # Package metadata, deps, entrypoints
├── requirements.txt                    # Pinned / reference deps
├── docs/                               # Documents
│   ├── RELEASE.md                      # Release checklist
│   └── PUBLISHING.md                   # Step-by-step publishing (GitHub, Hub, PyPI)
├── src/vocab_tailor/                   # Main package
│   ├── __init__.py                     # Public API (VocabTailor, etc.)
│   ├── version.py                      # Package version
│   ├── vocab_tailor.py                 # VocabTailor class, from_pretrained, generate
│   ├── model_utils.py                  # Model loading utilities
│   ├── metrics.py                      # Timing and token metrics
│   ├── baseline.py                     # BaselineGenerator class, from_pretrained, generate
│   ├── lmdb_layers.py                  # LMDB weight provider, build_lmdb_weights
│   ├── split_linear.py                 # Split linear layer (resize strategies)
│   ├── profiling/                      # Static vocabulary pipeline
│   │   ├── cli.py                      # vocab-tailor-build-vocab entrypoint
│   │   ├── filter.py                   # Three-stage filtering logic
│   │   └── unicode_utils.py            # Unicode/category helpers for Stage 2 filtering
│   └── static_vocab/                   # Example task vocabs (bundled; model/task/*.json)
├── benchmarks/                         # Reproducible benchmarks
│   ├── README.md                       # All downstream tasks; run instructions
│   ├── benchmark_lmhead_swap.py        # LM head swap timing (any two static vocabs)
│   ├── machine_translation/            # MT (baseline + VocabTailor)
│   │   ├── mt_harness.py, mt_baseline_eval.py, mt_vocabtailor_eval.py, ...
│   │   ├── mt_eval.sh, mt_vt_eval.sh
│   │   └── ...
│   ├── summarization/                  # Summarization (baseline + VocabTailor)
│   │   ├── summ_harness.py, summ_baseline_eval.py, summ_vocabtailor_eval.py, ...
│   │   ├── summ_eval.sh, summ_vt_eval.sh
│   │   └── ...
│   └── information_extraction/        # IE via lm-eval (baseline + VocabTailor)
│       ├── ie_harness.py, ie_baseline_eval.py, ie_vocabtailor_eval.py
│       ├── ie_eval.sh, ie_vt_eval.sh
│       └── ...
├── tests/                              # Test scripts
│   ├── test_vocab_tailor.py            # Smoke tests (import, from_pretrained)
│   ├── validate.sh                     # Run profiling + model validation (sets REPO_ROOT)
│   ├── validate_profiling_pipeline.py  # Build-vocab validation vs static_vocab
│   ├── validate_model_package.py       # Full-stack MT eval
│   └── README.md                       # How to run tests; dataset download notes
├── examples/                           # Example scripts
│   └── quickstart_mt_qwen3.py          # Minimal MT demo
└── datasets/                           # Optional; used by validation and benchmarks
```

## Quick start

```bash
python examples/quickstart_mt_qwen3.py
```

Or without installing: from the repository root, `PYTHONPATH=src python examples/quickstart_mt_qwen3.py`.

Optional: `--model MODEL` (default: `Qwen/Qwen3-1.7B`), `--device cuda`.

## API

```python
from vocab_tailor import VocabTailor

vt = VocabTailor.from_pretrained(
    "Qwen/Qwen3-1.7B",              # or local path
    device="cuda",
    dtype="bf16",
    lmdb_path=None,                 # or path to .lmdb for offload
    vocab_resize_strategy="prealloc",
    profiling_file=None,            # optional: path to JSON task vocab from vocab-tailor-build-vocab
    enable_metrics_tracker=False,   # set True to record per-run timing in vt.gen_metrics after generate()
)
output_ids = vt.generate(
    input_ids,
    mode="input_aware",
    max_new_tokens=128,
    do_sample=False,
    original_eos_token_id=vt.tokenizer.eos_token_id,
)
# When enable_metrics_tracker=True, per-run metrics (prefill_time, decode_tps, etc.) are in vt.gen_metrics
```

For a **baseline** (no vocabulary tailoring), use `BaselineGenerator` with the same from_pretrained pattern (Hub id or local path, device, dtype, tokenizer_kwargs, and model kwargs):

```python
from vocab_tailor import BaselineGenerator

bg = BaselineGenerator.from_pretrained(
    "Qwen/Qwen3-1.7B",              # or local path
    device="cuda",
    dtype="fp32",
    enable_metrics_tracker=False,   # set True to record per-run timing in bg.gen_metrics after generate()
    tokenizer_kwargs=None,          # e.g. local_files_only=True
)
output_ids = bg.generate(
    input_ids, 
    max_new_tokens=128,
    do_sample=False,
)
# bg.tokenizer, bg.model, bg.device are set. When tracker enabled, bg.gen_metrics holds per-run metrics.
# Both vt.generate() and bg.generate() accept **generate_kwargs (e.g. temperature, top_p) forwarded to model.generate().
```

## Hugging Face Hub

You can use any **Hugging Face model id** with `from_pretrained` (e.g. `"Qwen/Qwen3-1.7B"`). Models are loaded via the Transformers library; the first time you use a Hub id, weights are downloaded and cached.

- **Gated or private models:** Pass your Hugging Face token via `tokenizer_kwargs` and `model_kwargs`, e.g.  
  `VocabTailor.from_pretrained("org/model-name", tokenizer_kwargs={"token": "hf_..."}, token="hf_...")`  
  or set the `HF_TOKEN` environment variable so Transformers can use it automatically.
- **Library on the Hub:** The VocabTailor library repo on the Hugging Face Hub (see [project URLs](pyproject.toml)) hosts the project page and install instructions; install the package with `pip install vocab-tailor` or from source as in Setup above.
- **Task vocabulary:** Optional `profiling_file` in `from_pretrained()`: path to a JSON task vocabulary (from `vocab-tailor-build-vocab`). If provided and the file exists, the LM head is preloaded with those token IDs plus special tokens, so you can apply VocabTailor to any new task and dataset in one call.

## Building the static vocabulary

VocabTailor keeps a **static task-specific vocabulary** (loaded via `--profiling_file`) built by a three-stage pipeline:

1. **Input-aware filtering** — Collect target-side-only token IDs (tokens that appear in outputs but not in inputs) and their frequencies.
2. **Language-specific (Unicode) filtering** — Restrict to tokens whose characters fall in selected Unicode categories (e.g. `english`, `chinese`, `math`).
3. **Tolerance filtering** — Keep the most frequent tokens so that the discarded mass is at most `tolerance × document_count`.

The pipeline is **task-agnostic**: you can build a static vocab for any use case by providing `--model`, `--dataset`, `--input_col`, `--output_col`, `-n` (unicode categories), `--tolerance`, and `--output_dir`. **`--task_name` is optional.** When omitted (no `--task` and no `--task_name`), output is written directly under `--output_dir`; when set (or when using `--task` presets), output is under `--output_dir`/`--task_name`/.

Install the profiling extra and run the CLI (or use the Python API):

```bash
pip install -e ".[profiling]"
vocab-tailor-build-vocab --model Qwen/Qwen3-1.7B --dataset <path-to-data> --input_col source --output_col target -n chinese english --tolerance 0.1 --output_dir ./preprocessed
```

**Optional task presets:** use `--task` to apply default columns and task name. Presets: `machine_translation` (source/target), `summarization` (document/summary), `information_extraction` (context/value), `math` (question/answer), `code_completion` (prompt/completion). Explicit `--dataset`, `--input_col`, `--output_col` override preset defaults.

Output files are written under `--output_dir` (or `--output_dir`/`--task_name`/ when task name is set). Pass that path as `profiling_file` in `VocabTailor.from_pretrained()`.

**Python API:** `from vocab_tailor.profiling import build_static_vocab` — same options via function arguments; see `vocab_tailor/profiling/filter.py` and `cli.py` for parameters.

## Data and models

- Use a Hugging Face model id or put compatible models in a local directory and pass the path.
- The `datasets/` directory in this repository is used by validation tests and benchmarks. For your own runs, use your own data or load from the Hub.

## Benchmarks

Benchmarks live under `benchmarks/`. See [benchmarks/README.md](benchmarks/README.md) for full run instructions.

- **Machine translation** (`benchmarks/machine_translation/`): baseline vs VocabTailor on quality (sacreBLEU, COMET, METEOR), speed/memory, and staged RSS. Optional `--profiling_file` and `--input_aware` for VocabTailor.
- **Summarization** (`benchmarks/summarization/`): baseline vs VocabTailor on ROUGE and speed/memory. Optional `--profiling_file` and `--input_aware` for VocabTailor.
- **Information extraction** (`benchmarks/information_extraction/`): evaluation only via lm-eval (e.g. `squad_completion`). VocabTailor is **dynamic-only** (no `--profiling_file`; input-aware pruning only).

From the repository root:

```bash
PYTHONPATH=src bash benchmarks/machine_translation/mt_eval.sh    # MT baseline
PYTHONPATH=src bash benchmarks/machine_translation/mt_vt_eval.sh # MT VocabTailor
PYTHONPATH=src bash benchmarks/summarization/summ_eval.sh        # Summarization baseline
PYTHONPATH=src bash benchmarks/summarization/summ_vt_eval.sh     # Summarization VocabTailor
PYTHONPATH=src bash benchmarks/information_extraction/ie_eval.sh  # IE baseline
PYTHONPATH=src bash benchmarks/information_extraction/ie_vt_eval.sh # IE VocabTailor
```

Or run individual scripts (e.g. `PYTHONPATH=src python benchmarks/machine_translation/mt_baseline_eval.py [options]`). See [benchmarks/README.md](benchmarks/README.md) for dataset paths and options.

## Tests

- **Smoke tests** — `tests/test_vocab_tailor.py`: package import, version, `from_pretrained` (with and without LMDB), and `generate()` return value and `gen_metrics`.
- **Profiling pipeline validation** — `tests/validate_profiling_pipeline.py`: runs build-vocab for reference configs and compares to `static_vocab/`. Run 1 and 2 require a dataset (e.g. pass `--dataset-run1` / `--dataset-run2` or have opus-100 locally); Run 3 uses `datasets/xsum/` or HF. See `tests/README.md`.
- **Full model validation** — `tests/validate_model_package.py`: full-stack MT evaluation (use `--input_aware` for input-aware mode; `--profiling_file` is optional—omit for dynamic-only). Optional golden compare. Default dataset: `datasets/wmt24pp/en-zh_CN.jsonl`.

From the repository root:

```bash
python tests/test_vocab_tailor.py
```

Or with pytest: `pytest tests/test_vocab_tailor.py -v`

To run both validation scripts in one go: `bash tests/validate.sh` (sets `REPO_ROOT` from script location; can be run from repo root or from `tests/`). To run them individually: `PYTHONPATH=src python tests/validate_profiling_pipeline.py [options]` and `PYTHONPATH=src python tests/validate_model_package.py [options]`. See `tests/README.md` for details.

## Publishing

For release and publishing steps (creating the Hub repo, publishing to PyPI, uploading a dataset), see [docs/RELEASE.md](docs/RELEASE.md).
