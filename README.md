# VocabTailor

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
VocabTailor_Stage1/
├── README.md               # This file
├── LICENSE                 # Apache-2.0
├── pyproject.toml          # Package metadata, deps, entrypoints
├── requirements.txt        # Pinned / reference deps
├── docs/
│   ├── RELEASE.md          # Release checklist
│   └── PUBLISHING.md       # Step-by-step publishing (GitHub, Hub, PyPI)
├── src/vocab_tailor/       # Main package
│   ├── __init__.py         # Public API (VocabTailor, etc.)
│   ├── version.py
│   ├── vocab_tailor.py     # VocabTailor class, from_pretrained, generate
│   ├── model_utils.py      # Model loading utilities
│   ├── metrics.py          # Timing and token metrics
│   ├── baseline.py         # Baseline HF model generation
│   ├── lmdb_layers.py      # LMDB weight provider, build_lmdb_weights
│   ├── split_linear.py     # Split linear layer (resize strategies)
│   ├── profiling/          # Static vocabulary pipeline
│   │   ├── cli.py          # vocab-tailor-build-vocab entrypoint
│   │   ├── filter.py       # Three-stage filtering logic
│   │   └── unicode_utils.py
│   └── static_vocab/       # Example task vocabs (bundled; model/task/*.json)
├── tests/
│   ├── test_vocab_tailor.py           # Smoke tests (import, from_pretrained)
│   ├── validate_profiling_pipeline.py # Build-vocab validation vs static_vocab
│   ├── validate_model_package.py     # Full-stack MT eval
│   └── README.md
├── examples/
│   └── quickstart_mt_qwen3.py        # Minimal MT demo
└── datasets/               # Optional; used by validation scripts
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
    "Qwen/Qwen3-1.7B",  # or local path
    device="cuda",
    dtype="bf16",
    lmdb_path=None,                 # or path to .lmdb for offload
    vocab_resize_strategy="prealloc",
    profiling_file=None,            # optional: path to JSON task vocab from vocab-tailor-build-vocab
)
output_ids, metrics = vt.generate(
    input_ids,
    mode="input_aware",
    max_new_tokens=128,
)
```

## Hugging Face Hub

You can use any **Hugging Face model id** with `from_pretrained` (e.g. `"Qwen/Qwen2.5-0.5B-Instruct"`). Models are loaded via the Transformers library; the first time you use a Hub id, weights are downloaded and cached.

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
- The `datasets/` directory in this repository is used by validation tests only. For your own runs, use your own data or load from the Hub.

## Tests

- **Smoke tests** — `tests/test_vocab_tailor.py`: package import, version, `from_pretrained` (with and without LMDB).
- **Profiling pipeline validation** — `tests/validate_profiling_pipeline.py`: runs build-vocab for reference configs and compares to `static_vocab/`. Run 1 and 2 require a dataset (e.g. pass `--dataset-run1` / `--dataset-run2` or have opus-100 locally); Run 3 uses `datasets/xsum/` or HF. See `tests/README.md`.
- **Full model validation** — `tests/validate_model_package.py`: full-stack MT evaluation (use `--input_aware` for input-aware mode; `--profiling_file` is optional—omit for dynamic-only). Optional golden compare. Default dataset: `datasets/wmt24pp/en-zh_CN.jsonl`.

From the repository root:

```bash
python tests/test_vocab_tailor.py
```

Or with pytest: `pytest tests/test_vocab_tailor.py -v`

For the other validation scripts: `PYTHONPATH=src python tests/validate_profiling_pipeline.py [options]` and `PYTHONPATH=src python tests/validate_model_package.py [options]`. See `tests/README.md` for details.

## Publishing

For release and publishing steps (creating the Hub repo, publishing to PyPI, uploading a dataset), see [docs/RELEASE.md](docs/RELEASE.md).
