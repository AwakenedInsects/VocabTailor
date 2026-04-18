#!/usr/bin/env bash
# Baseline summarization: evaluation and benchmark (no memory test).
# Run from repo root: bash benchmarks/summarization/summ_eval.sh
# Or from this directory: bash summ_eval.sh (script detects repo root).
# Outputs: results/summarization/eval and benchmark. Add --output_dir to python calls to override.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_NAME="AwakenedInsects/llama-3.2-3b-vocabtailor-sum"
# MODEL_NAME="${REPO_ROOT}/models/Llama-3.2-3B"
DATASET="${REPO_ROOT}/datasets/xsum/xsum_test.jsonl"
PREFIX="baseline"

# Evaluation (quality: ROUGE)
echo "=== Summarization Evaluation (Baseline) ==="
python "$SCRIPT_DIR/summ_baseline_eval.py" \
  --model_name "$MODEL_NAME" \
  --dataset "$DATASET" \
  --prefix "$PREFIX" \
  --device cuda \
  --dtype fp32 \
  --save_predictions

# Benchmark (speed & memory via harness)
echo "=== Summarization Benchmark (Baseline) ==="
python "$SCRIPT_DIR/summ_baseline_benchmark.py" \
  --model_name "$MODEL_NAME" \
  --dataset "$DATASET" \
  --prefix "$PREFIX" \
  --sample_size 100 \
  --device cuda \
  --dtype fp32
