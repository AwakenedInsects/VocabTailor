#!/usr/bin/env bash
# VocabTailor summarization: evaluation and benchmark (no memory test).
# Run from repo root: bash benchmarks/summarization/summ_vt_eval.sh
# Or from this directory: bash summ_vt_eval.sh (script detects repo root).
# Outputs: results/summarization/eval and benchmark. Add --output_dir to python calls to override.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_NAME="AwakenedInsects/llama-3.2-3b-vocabtailor-sum"
# MODEL_NAME="${REPO_ROOT}/models/Llama-3.2-3B"
LMDB_PATH="${REPO_ROOT}/datasets/lmdb_store/llama3.2-3b_weights.lmdb"

DATASET="${REPO_ROOT}/datasets/xsum/xsum_test.jsonl"
PROFILING_FILE="${REPO_ROOT}/src/vocab_tailor/static_vocab/llama3.2/summarization/Llama_unicode_set_english_tol_0.01.json"
PREFIX="vt_h_xsum_en_tol0.01"

# Evaluation (quality: ROUGE)
echo "=== Summarization Evaluation (VocabTailor) ==="
python "$SCRIPT_DIR/summ_vocabtailor_eval.py" \
  --model_name "$MODEL_NAME" \
  --profiling_file "$PROFILING_FILE" \
  --dataset "$DATASET" \
  --prefix "$PREFIX" \
  --input_aware \
  --vocab_resize_strategy prealloc \
  --device cuda \
  --dtype fp32 \
  --save_predictions

# Benchmark (speed & memory via harness)
echo "=== Summarization Benchmark (VocabTailor) ==="
python "$SCRIPT_DIR/summ_vocabtailor_benchmark.py" \
  --model_name "$MODEL_NAME" \
  --profiling_file "$PROFILING_FILE" \
  --dataset "$DATASET" \
  --prefix "$PREFIX" \
  --input_aware \
  --sample_size 100 \
  --vocab_resize_strategy prealloc \
  --device cuda \
  --dtype bf16

echo "=== Summarization Benchmark (VocabTailor + LMDB + CPU) ==="
python "$SCRIPT_DIR/summ_vocabtailor_benchmark.py" \
  --model_name "$MODEL_NAME" \
  --lmdb_path "$LMDB_PATH" \
  --profiling_file "$PROFILING_FILE" \
  --dataset "$DATASET" \
  --prefix "$PREFIX" \
  --input_aware \
  --sample_size 100 \
  --vocab_resize_strategy prealloc \
  --device cpu \
  --dtype bf16
