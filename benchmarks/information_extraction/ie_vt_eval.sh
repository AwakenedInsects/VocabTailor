#!/usr/bin/env bash
# VocabTailor information extraction: evaluation only (lm-eval tasks).
# Dynamic-only: no profiling_file; input-aware pruning only.
# Run from repo root: bash benchmarks/information_extraction/ie_vt_eval.sh
# Or from this directory: bash ie_vt_eval.sh (script detects repo root).
# Outputs: results/information_extraction/eval. Add --output_dir to override; for local MODEL_NAME add --local_files_only.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_NAME="meta-llama/Llama-3.2-1B"
# MODEL_NAME="${REPO_ROOT}/models/Llama-3.2-1B"
LMDB_PATH="${REPO_ROOT}/datasets/lmdb_store/llama3.2-1b_weights.lmdb"
PREFIX="vt_d"
TASKS="squad_completion"

echo "=== Information Extraction Evaluation (VocabTailor) ==="
python "$SCRIPT_DIR/ie_vocabtailor_eval.py" \
  --model_name "$MODEL_NAME" \
  --prefix "$PREFIX" \
  --tasks $TASKS \
  --vocab_resize_strategy prealloc \
  --device cuda \
  --dtype fp32 \
  --save_predictions

echo "=== Information Extraction Evaluation (VocabTailor + LMDB + CPU) ==="
python "$SCRIPT_DIR/ie_vocabtailor_eval.py" \
  --model_name "$MODEL_NAME" \
  --lmdb_path "$LMDB_PATH" \
  --prefix "$PREFIX" \
  --tasks $TASKS \
  --vocab_resize_strategy prealloc \
  --device cpu \
  --dtype fp32 \
  --save_predictions