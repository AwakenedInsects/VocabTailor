#!/usr/bin/env bash
# Baseline information extraction: evaluation only (lm-eval tasks).
# Run from repo root: bash benchmarks/information_extraction/ie_eval.sh
# Or from this directory: bash ie_eval.sh (script detects repo root).
# Outputs: results/information_extraction/eval. Add --output_dir to override; for local MODEL_NAME add --local_files_only.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_NAME="meta-llama/Llama-3.2-1B"
# MODEL_NAME="${REPO_ROOT}/models/Llama-3.2-1B"
TASKS="squad_completion"
PREFIX="baseline"

echo "=== Information Extraction Evaluation (Baseline) ==="
python "$SCRIPT_DIR/ie_baseline_eval.py" \
  --model_name "$MODEL_NAME" \
  --prefix "$PREFIX" \
  --tasks $TASKS \
  --device cuda \
  --dtype fp32 \
  --save_predictions
