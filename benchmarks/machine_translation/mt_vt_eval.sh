#!/usr/bin/env bash
# VocabTailor MT: evaluation (EN-ZH, EN-IT), benchmark and memory test (EN-ZH only).
# Run from repo root: bash benchmarks/machine_translation/mt_vt_eval.sh
# Or from this directory: bash mt_vt_eval.sh (script detects repo root).
# Outputs: results/machine_translation/{en-zh|en-it}/eval|benchmark|memory_test. Add --output_dir to override; for local MODEL_NAME add --local_files_only.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_NAME1="Qwen/Qwen3-1.7B"
LMDB_PATH1="${REPO_ROOT}/datasets/lmdb_store/qwen3-1.7b_weights.lmdb"
MODEL_NAME2="Qwen/Qwen3-0.6B"
LMDB_PATH2="${REPO_ROOT}/datasets/lmdb_store/qwen3-0.6b_weights.lmdb"

# EN-ZH (evaluation, benchmark, memory test)
DATASET_ZH="${REPO_ROOT}/datasets/wmt24pp/en-zh_CN.jsonl"
PROFILING_FILE_ZH="${REPO_ROOT}/src/vocab_tailor/static_vocab/qwen3/mt_en_zh/Qwen3_unicode_set_chinese_tol_0.01.json"
PREFIX_ZH="vt_h_opus_zh_tol0.01"

# EN-IT (evaluation only)
DATASET_IT="${REPO_ROOT}/datasets/wmt24pp/en-it_IT.jsonl"
PROFILING_FILE_IT="${REPO_ROOT}/src/vocab_tailor/static_vocab/qwen3/mt_en_it/Qwen3_unicode_set_italian_tol_0.01.json"
PREFIX_IT="vt_h_opus_it_tol0.01"

# Evaluation (quality: sacreBLEU, COMET, METEOR) — EN->ZH and EN->IT
echo "=== Machine Translation Evaluation: EN->ZH (VocabTailor) ==="
python "$SCRIPT_DIR/mt_vocabtailor_eval.py" \
  --model_name "$MODEL_NAME1" \
  --profiling_file "$PROFILING_FILE_ZH" \
  --dataset "$DATASET_ZH" \
  --prefix "$PREFIX_ZH" \
  --source_lang en \
  --target_lang zh \
  --input_aware \
  --vocab_resize_strategy prealloc \
  --device cuda \
  --dtype fp32 \
  --save_predictions

echo "=== Machine Translation Evaluation: EN->IT (VocabTailor) ==="
python "$SCRIPT_DIR/mt_vocabtailor_eval.py" \
  --model_name "$MODEL_NAME1" \
  --profiling_file "$PROFILING_FILE_IT" \
  --dataset "$DATASET_IT" \
  --prefix "$PREFIX_IT" \
  --source_lang en \
  --target_lang it \
  --input_aware \
  --vocab_resize_strategy prealloc \
  --device cuda \
  --dtype fp32 \
  --save_predictions

# Benchmark (speed & memory via harness) — EN->ZH only
echo "=== Machine Translation Benchmark: EN->ZH (VocabTailor) ==="
python "$SCRIPT_DIR/mt_vocabtailor_benchmark.py" \
  --model_name "$MODEL_NAME2" \
  --profiling_file "$PROFILING_FILE_ZH" \
  --dataset "$DATASET_ZH" \
  --prefix "$PREFIX_ZH" \
  --source_lang en \
  --target_lang zh \
  --input_aware \
  --sample_size 100 \
  --vocab_resize_strategy prealloc \
  --device cuda \
  --dtype bf16

echo "=== Machine Translation Benchmark: EN->ZH (VocabTailor + LMDB + CPU) ==="
python "$SCRIPT_DIR/mt_vocabtailor_benchmark.py" \
  --model_name "$MODEL_NAME2" \
  --lmdb_path "$LMDB_PATH2" \
  --profiling_file "$PROFILING_FILE_ZH" \
  --dataset "$DATASET_ZH" \
  --prefix "$PREFIX_ZH" \
  --source_lang en \
  --target_lang zh \
  --input_aware \
  --sample_size 100 \
  --vocab_resize_strategy prealloc \
  --device cpu \
  --dtype bf16

# Memory test (staged RSS breakdown) — EN->ZH only
echo "=== Machine Translation Memory Test: EN->ZH (VocabTailor) ==="
python "$SCRIPT_DIR/mt_vocabtailor_memory_test.py" \
  --model_name "$MODEL_NAME2" \
  --profiling_file "$PROFILING_FILE_ZH" \
  --dataset "$DATASET_ZH" \
  --prefix "$PREFIX_ZH" \
  --source_lang en \
  --target_lang zh \
  --input_aware \
  --sample_size 100 \
  --vocab_resize_strategy prealloc \
  --device cuda \
  --dtype bf16

echo "=== Machine Translation Memory Test: EN->ZH (VocabTailor + LMDB + CPU) ==="
python "$SCRIPT_DIR/mt_vocabtailor_memory_test.py" \
  --model_name "$MODEL_NAME2" \
  --lmdb_path "$LMDB_PATH2" \
  --profiling_file "$PROFILING_FILE_ZH" \
  --dataset "$DATASET_ZH" \
  --prefix "$PREFIX_ZH" \
  --source_lang en \
  --target_lang zh \
  --input_aware \
  --sample_size 100 \
  --vocab_resize_strategy prealloc \
  --device cpu \
  --dtype bf16