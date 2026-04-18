#!/usr/bin/env bash
# Run profiling pipeline validation and full model package validation.
# Run from repository root (VocabTailor/) or from tests/; REPO_ROOT is set from script location.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Validate Profiling Pipeline
echo "=== Validate Profiling Pipeline ==="
PYTHONPATH=src python tests/validate_profiling_pipeline.py
# PYTHONPATH=src python tests/validate_profiling_pipeline.py --run 1

# Validate Model Package
MODEL_NAME="Qwen/Qwen3-1.7B"
LMDB_PATH="${REPO_ROOT}/datasets/lmdb_store/qwen3-1.7b_weights.lmdb"

DATASET="${REPO_ROOT}/datasets/wmt24pp/en-zh_CN.jsonl"
PROFILING_FILE="${REPO_ROOT}/src/vocab_tailor/static_vocab/qwen3/mt_en_zh/Qwen3_unicode_set_chinese_tol_0.01.json"
PREFIX="vt_h_zh_tol0.01"


echo "=== Validate Model Package ==="
PYTHONPATH=src python tests/validate_model_package.py \
--model_name "$MODEL_NAME" \
--profiling_file "$PROFILING_FILE" \
--dataset "$DATASET" \
--prefix "$PREFIX" \
--source_lang en \
--target_lang zh \
--input_aware  \
--vocab_resize_strategy prealloc \
--compare

echo "=== Validate Model Package (LMDB + CPU) ==="
PYTHONPATH=src python tests/validate_model_package.py \
--model_name "$MODEL_NAME" \
--lmdb_path "$LMDB_PATH" \
--profiling_file "$PROFILING_FILE" \
--dataset "$DATASET" \
--prefix "$PREFIX" \
--source_lang en \
--target_lang zh \
--input_aware \
--vocab_resize_strategy prealloc \
--compare