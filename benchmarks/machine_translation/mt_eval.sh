#!/usr/bin/env bash
# Baseline MT: evaluation (EN-ZH, EN-IT), benchmark and memory test (EN-ZH only).
# Run from repo root: bash benchmarks/machine_translation/mt_eval.sh
# Or from this directory: bash mt_eval.sh (script detects repo root).
# Outputs: results/machine_translation/{en-zh|en-it}/eval|benchmark|memory_test. Add --output_dir to python calls to override.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_NAME1="Qwen/Qwen3-1.7B"
# MODEL_NAME1="${REPO_ROOT}/models/Qwen3-1.7B"
MODEL_NAME2="Qwen/Qwen3-0.6B"
# MODEL_NAME2="${REPO_ROOT}/models/Qwen3-0.6B"

# EN-ZH (evaluation, benchmark, memory test)
DATASET_ZH="${REPO_ROOT}/datasets/wmt24pp/en-zh_CN.jsonl"
PREFIX_ZH="baseline"

# EN-IT (evaluation only)
DATASET_IT="${REPO_ROOT}/datasets/wmt24pp/en-it_IT.jsonl"
PREFIX_IT="baseline"

# Evaluation (quality: sacreBLEU, COMET, METEOR) — EN->ZH and EN->IT
echo "=== Machine Translation Evaluation: EN->ZH (Baseline) ==="
python "$SCRIPT_DIR/mt_baseline_eval.py" \
  --model_name "$MODEL_NAME1" \
  --dataset "$DATASET_ZH" \
  --prefix "$PREFIX_ZH" \
  --source_lang en \
  --target_lang zh \
  --device cuda \
  --dtype fp32 \
  --save_predictions

echo "=== Machine Translation Evaluation: EN->IT (Baseline) ==="
python "$SCRIPT_DIR/mt_baseline_eval.py" \
  --model_name "$MODEL_NAME1" \
  --dataset "$DATASET_IT" \
  --prefix "$PREFIX_IT" \
  --source_lang en \
  --target_lang it \
  --device cuda \
  --dtype fp32 \
  --save_predictions

# Benchmark (speed & memory via harness) — EN->ZH only
echo "=== Machine Translation Benchmark: EN->ZH (Baseline) ==="
python "$SCRIPT_DIR/mt_baseline_benchmark.py" \
  --model_name "$MODEL_NAME2" \
  --dataset "$DATASET_ZH" \
  --prefix "$PREFIX_ZH" \
  --source_lang en \
  --target_lang zh \
  --sample_size 100 \
  --device cuda \
  --dtype bf16

# Memory test (staged RSS breakdown) — EN->ZH only
echo "=== Machine Translation Memory Test: EN->ZH (Baseline) ==="
python "$SCRIPT_DIR/mt_baseline_memory_test.py" \
  --model_name "$MODEL_NAME2" \
  --dataset "$DATASET_ZH" \
  --prefix "$PREFIX_ZH" \
  --source_lang en \
  --target_lang zh \
  --sample_size 100 \
  --device cuda \
  --dtype bf16