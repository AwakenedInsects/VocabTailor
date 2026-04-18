# Benchmarks

This folder contains reproducible benchmarks for downstream tasks, comparing **baseline** (standard Hugging Face generation) and **VocabTailor** (input-aware vocabulary pruning). When adding new tasks, add a subsection below and keep scripts under `benchmarks/<task_name>/`.

**Shell scripts:** The `*_eval.sh` and `*_vt_eval.sh` scripts in each task folder set variables (e.g. `MODEL_NAME`, `PREFIX`, `DATASET`) and invoke the Python benchmark scripts. Results are written under `results/<task>/` by default; add `--output_dir <path>` to the script's `python` invocations to override. When using a local model path (e.g. `MODEL_NAME="/path/to/model"`), add `--local_files_only` to the corresponding `python` command(s).

## LM head swap timing

`benchmark_lmhead_swap.py` measures the time to swap the LM head between **any two** static vocabularies (no model reload). The two vocabs need not be from the same task.

| Argument | Description |
|----------|-------------|
| `--profiling_file1` | Path to the initial static vocab JSON (model is loaded with this). Default: `src/vocab_tailor/static_vocab/qwen3/mt_en_zh/Qwen3_unicode_set_chinese_tol_0.01.json` under repo root. |
| `--profiling_file2` | Path to the second static vocab JSON (swap target). Default: `src/vocab_tailor/static_vocab/qwen3/mt_en_it/Qwen3_unicode_set_italian_tol_0.01.json` under repo root. |
| `--model` | Model (HF id or local path). Default: `Qwen/Qwen3-1.7B`. |
| `--warmup` | Number of warmup swap runs before timing. Default: `1`. |
| `--repeat` | Number of timed runs for mean ± std. Default: `5`. |

**Run (from repository root):**

```bash
PYTHONPATH=src python benchmarks/benchmark_lmhead_swap.py
PYTHONPATH=src python benchmarks/benchmark_lmhead_swap.py --profiling_file1 PATH1 --profiling_file2 PATH2
```

Use `--help` for all options (`--device`, etc.).

## Implemented downstream tasks

### Summarization

Reproducible summarization benchmarks comparing baseline and VocabTailor on quality (ROUGE) and speed/memory. No memory test scripts; use `summ_eval.sh` and `summ_vt_eval.sh` for pipeline runs.

#### Scripts

| Script | Description |
|--------|-------------|
| `summ_harness.py` | Task harness: templates, dataset loading, default paths. Shared helpers (e.g. `validate_vocabtailor_args`, `generate_runs_summary`) live in `benchmark_utils`; scripts import from both. |
| `summ_baseline_eval.py` | Baseline: quality (ROUGE). Uses `BaselineGenerator.from_pretrained`. Default model: `AwakenedInsects/llama-3.2-3b-vocabtailor-sum`. |
| `summ_baseline_benchmark.py` | Baseline: speed & memory via harness. |
| `summ_vocabtailor_eval.py` | VocabTailor: quality (ROUGE). Uses `VocabTailor.from_pretrained`; optional `--profiling_file`, `--input_aware`. Default profiling: `static_vocab/llama3.2/summarization/Llama_unicode_set_english_tol_0.01.json`. |
| `summ_vocabtailor_benchmark.py` | VocabTailor: speed & memory. |
| `summ_eval.sh` | Run baseline pipeline (eval + benchmark). |
| `summ_vt_eval.sh` | Run VocabTailor pipeline (eval + benchmark). |

All scripts live under `benchmarks/summarization/`.

#### Run from repository root

```bash
# Baseline: evaluation and benchmark
PYTHONPATH=src bash benchmarks/summarization/summ_eval.sh

# VocabTailor: evaluation and benchmark
PYTHONPATH=src bash benchmarks/summarization/summ_vt_eval.sh
```

Or run a single script:

```bash
PYTHONPATH=src python benchmarks/summarization/summ_baseline_eval.py --dataset datasets/xsum/xsum_test.jsonl --sample_size 100
PYTHONPATH=src python benchmarks/summarization/summ_vocabtailor_eval.py --dataset datasets/xsum/xsum_test.jsonl --profiling_file src/vocab_tailor/static_vocab/llama3.2/summarization/Llama_unicode_set_english_tol_0.01.json --input_aware
```

Use `--help` on any script for options (`--device`, `--dtype`, `--sample_size`, `--prefix`, `--output_dir`, etc.).

#### Datasets

Default model: `AwakenedInsects/llama-3.2-3b-vocabtailor-sum`. Default dataset: `datasets/xsum/xsum_test.jsonl` under repo root (must exist; or pass `--dataset` with another path). JSON/JSONL must have `document` and `summary` columns.

#### Outputs

- **Eval:** `results/summarization/eval/` — metrics JSON, logs, optional predictions. Override with `--output_dir`.
- **Benchmark:** `results/summarization/benchmark/` — run summary JSON, logs. Override with `--output_dir`.
- Scripts accept `--prefix` for output file basename and `--output_dir` for the results directory.

### Machine translation

Reproducible MT benchmarks comparing baseline and VocabTailor on quality (sacreBLEU, COMET, METEOR), speed/memory, and staged RSS.

#### Scripts

| Script | Description |
|--------|-------------|
| `mt_harness.py` | Task harness: templates, dataset loading, default paths. Shared helpers (e.g. `validate_vocabtailor_args`, `generate_runs_summary`) live in `benchmark_utils`; scripts import from both. |
| `mt_baseline_eval.py` | Baseline: quality (sacreBLEU, COMET, METEOR). Uses `BaselineGenerator.from_pretrained`. |
| `mt_baseline_benchmark.py` | Baseline: speed & memory via harness. |
| `mt_baseline_memory_test.py` | Baseline: staged RSS (no tracker). Loads model/tokenizer/BaselineGenerator in separate steps. |
| `mt_vocabtailor_eval.py` | VocabTailor: quality. Uses `VocabTailor.from_pretrained`; optional `--profiling_file`, `--input_aware`. |
| `mt_vocabtailor_benchmark.py` | VocabTailor: speed & memory. |
| `mt_vocabtailor_memory_test.py` | VocabTailor: staged RSS. Separate load steps for comparable RSS snapshots. |
| `mt_eval.sh` | Run baseline pipeline (eval EN-ZH + EN-IT, benchmark + memory test EN-ZH). |
| `mt_vt_eval.sh` | Run VocabTailor pipeline (same layout). |

All scripts live under `benchmarks/machine_translation/`.

#### Run from repository root

```bash
# Baseline: evaluation (EN-ZH, EN-IT), benchmark and memory test (EN-ZH)
PYTHONPATH=src bash benchmarks/machine_translation/mt_eval.sh

# VocabTailor: same
PYTHONPATH=src bash benchmarks/machine_translation/mt_vt_eval.sh
```

Or run a single script:

```bash
PYTHONPATH=src python benchmarks/machine_translation/mt_baseline_eval.py --dataset datasets/wmt24pp/en-zh_CN.jsonl --source_lang en --target_lang zh
PYTHONPATH=src python benchmarks/machine_translation/mt_vocabtailor_eval.py --dataset datasets/wmt24pp/en-zh_CN.jsonl --profiling_file src/vocab_tailor/static_vocab/qwen3/mt_en_zh/Qwen3_unicode_set_chinese_tol_0.01.json --input_aware
```

Use `--help` on any script for options (`--device`, `--dtype`, `--sample_size`, `--prefix`, `--output_dir`, etc.).

#### Datasets

Benchmarks use the same dataset paths as validation (see `tests/README.md`). Place files under `repo/datasets/`:

| Use | Expected path under `datasets/` |
|-----|----------------------------------|
| MT eval/benchmark (en-zh) | `wmt24pp/en-zh_CN.jsonl` |
| MT eval (en-it) | `wmt24pp/en-it_IT.jsonl` |

JSONL must have `source` and `target` columns. Default: `datasets/wmt24pp/en-zh_CN.jsonl`. If the file is missing, scripts raise `FileNotFoundError` (or use `get_default_dataset_path` which validates the default path).

#### Outputs

- **Eval:** `results/machine_translation/{source_lang}-{target_lang}/eval/` — metrics JSON, logs, optional predictions. Override with `--output_dir`.
- **Benchmark:** `results/machine_translation/{source_lang}-{target_lang}/benchmark/` — run summary JSON, logs. Override with `--output_dir`.
- **Memory test:** `results/machine_translation/{source_lang}-{target_lang}/memory_test/` — RSS breakdown JSON, logs. Override with `--output_dir`.
- Scripts accept `--prefix` for output file basename and `--output_dir` for the results directory.

### Information extraction

Reproducible information extraction benchmarks comparing baseline and VocabTailor on quality via **lm-eval** (task names such as `squad_completion`, `swde`, `fda`). Evaluation only (no benchmark or memory-test scripts). **VocabTailor is dynamic-only for IE:** no static task vocabulary, no `--profiling_file`; input-aware pruning only.

#### Scripts

| Script | Description |
|--------|-------------|
| `ie_harness.py` | LM-eval adapter: VocabTailorLM wraps VocabTailor for `evaluator.evaluate()`. IE is dynamic-only (no profiling_file). Shared helpers (e.g. `infer_short_model_name`) from `benchmark_utils`. |
| `ie_baseline_eval.py` | Baseline: quality via lm-eval (HFLM). Uses `BaselineGenerator.from_pretrained`. |
| `ie_vocabtailor_eval.py` | VocabTailor: quality via lm-eval (VocabTailorLM adapter). Uses `VocabTailor.from_pretrained` with no profiling_file (dynamic-only). |
| `ie_eval.sh` | Run baseline IE evaluation (no sample limit). |
| `ie_vt_eval.sh` | Run VocabTailor IE evaluation: first run without LMDB, second run with `--lmdb_path` (path set in script; no sample limit). |

All scripts live under `benchmarks/information_extraction/`.

#### Run from repository root

```bash
# Baseline: evaluation
PYTHONPATH=src bash benchmarks/information_extraction/ie_eval.sh

# VocabTailor: evaluation (dynamic-only)
PYTHONPATH=src bash benchmarks/information_extraction/ie_vt_eval.sh
```

Or run a single script (add `--limit N` to cap samples per task):

```bash
PYTHONPATH=src python benchmarks/information_extraction/ie_baseline_eval.py --tasks squad_completion
PYTHONPATH=src python benchmarks/information_extraction/ie_vocabtailor_eval.py --tasks squad_completion
```

Use `--help` on any script for options (`--model_name`, `--tasks`, `--limit`, `--lmdb_path`, `--device`, `--dtype`, `--prefix`, `--output_dir`, `--save_predictions`, etc.).

#### Tasks and outputs

- **Tasks:** lm_eval task names (e.g. `squad_completion`, `swde`, `fda`). No dataset path; tasks define data.
- **Outputs:** `results/information_extraction/eval/` — metrics JSON, logs. Override with `--output_dir`. With `--save_predictions`, per-task predictions and metrics are also written to `{base_name}_pred.json` and the main metrics file. Scripts accept `--prefix` and `--output_dir`.
