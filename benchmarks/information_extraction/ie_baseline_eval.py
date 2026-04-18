#!/usr/bin/env python3
"""
Information extraction evaluation (baseline) using lm-eval. Uses HFLM and evaluator.evaluate()
with task names (e.g. squad_completion, swde, fda). No dataset path; tasks are defined by lm_eval.

Run from VocabTailor:
  cd VocabTailor && PYTHONPATH=src python benchmarks/information_extraction/ie_baseline_eval.py [options]

Example:
  PYTHONPATH=src python benchmarks/information_extraction/ie_baseline_eval.py --model_name meta-llama/Llama-3.2-1B --tasks squad_completion --limit 50
"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_SRC = os.path.join(_REPO_ROOT, "src")
_BENCHMARKS = os.path.dirname(_SCRIPT_DIR)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _BENCHMARKS not in sys.path:
    sys.path.insert(0, _BENCHMARKS)

import argparse
import logging

import torch
from transformers import logging as tf_logging

try:
    from lm_eval import evaluator, tasks
    from lm_eval.utils import make_table
    from lm_eval.models.huggingface import HFLM
except ImportError as e:
    raise ImportError(
        "lm_eval is required for IE baseline evaluation. Install with: pip install vocab-tailor[ie]"
    ) from e

from vocab_tailor import BaselineGenerator

from ie_harness import save_lmeval_outputs
from benchmark_utils import (
    infer_short_model_name, 
    default_output_file_basename,
)


def main():
    tf_logging.set_verbosity_error()
    parser = argparse.ArgumentParser(description="Baseline information extraction evaluation via lm-eval")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Model path or Hugging Face model id.")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix for output file name.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on.")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="fp32", help="Model dtype.")
    parser.add_argument("--local_files_only", action="store_true", help="Use only local files for Hugging Face model/tokenizer.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results. Default: results/information_extraction/eval under repo root.")
    parser.add_argument("--save_predictions", action="store_true", help="Save extracted predictions to .json")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["squad_completion"],
        help="lm_eval task names (e.g. squad_completion swde fda)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per task (default: all)")
    parser.add_argument("--max_gen_toks", type=int, default=256, help="Max new tokens per request for generate_until")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    # Initialize output file path
    short_model_name = infer_short_model_name(args.model_name)
    base_name = default_output_file_basename(
        short_model_name=short_model_name,
        dtype=args.dtype,
        device=args.device,
        task_name="ie_eval",
        prefix=args.prefix,
        sample_size=args.limit,
    )
    # print(f"Output file basename: {base_name}")

    # prefix = "baseline" if args.prefix is None else args.prefix
    # if args.limit is not None:
    #     prefix += f"_sample{args.limit}"
    
    # base_name = f"{short_model_name}_{args.dtype}_{prefix}_ie_eval"
    
    # if args.device != "cuda":
    #     base_name += f"_{args.device}"
    
    output_dir = f"{_REPO_ROOT}/results/information_extraction/eval" if args.output_dir is None else args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    log_file = f"{output_dir}/logs/{base_name}.log"
    output_file = f"{output_dir}/{base_name}.json"
    pred_file = f"{output_dir}/{base_name}_pred.json"
    # print(f"log file: {log_file}\noutput file: {output_file}\nprediction file: {pred_file}\n")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    logger = logging.getLogger(__name__)

    # Initialize model and tokenizer
    logger.info("Loading model and tokenizer via BaselineGenerator...")
    model_kwargs = {}
    tokenizer_kwargs = {}
    if args.local_files_only:
        model_kwargs["local_files_only"] = True
        tokenizer_kwargs["local_files_only"] = True
    
    bg = BaselineGenerator.from_pretrained(
        args.model_name,
        device=args.device,
        dtype=args.dtype,
        enable_metrics_tracker=False,
        tokenizer_kwargs=tokenizer_kwargs,
        **model_kwargs,
    )
    logger.info("Finish loading BaselineGenerator")

    # Create the LM wrapper
    lm = HFLM(
        pretrained=bg.model,
        tokenizer=bg.tokenizer,
        backend="causal",
        trust_remote_code=args.trust_remote_code,
    )

    # Load tasks
    logger.info(f"Loading tasks: {args.tasks}")
    task_dict = tasks.get_task_dict(args.tasks)

    # Run evaluation
    logger.info("Starting evaluation...")
    results = evaluator.evaluate(
        lm=lm, 
        task_dict=task_dict, 
        limit=args.limit
    )

    logger.info("Evaluation results:\n")
    logger.info(f"{make_table(results)}")
    if args.save_predictions:
        save_lmeval_outputs(results, task_name="squad_completion", pred_path=pred_file, metric_path=output_file)

    logger.info("Done.")


if __name__ == "__main__":
    main()
