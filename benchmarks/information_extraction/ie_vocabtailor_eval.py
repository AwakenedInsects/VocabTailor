#!/usr/bin/env python3
"""
Information extraction evaluation (VocabTailor) using lm-eval. Uses the package VocabTailor
and a VocabTailorLM adapter, then evaluator.evaluate() with the same task names as baseline.
VocabTailor is dynamic-only for IE: no --profiling_file, no static task vocabulary; input-aware pruning only.

Run from VocabTailor:
  cd VocabTailor && PYTHONPATH=src python benchmarks/information_extraction/ie_vocabtailor_eval.py [options]

Example:
  PYTHONPATH=src python benchmarks/information_extraction/ie_vocabtailor_eval.py --model_name meta-llama/Llama-3.2-1B --tasks squad_completion --limit 50
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

from vocab_tailor import VocabTailor

from ie_harness import (
    evaluate_vocabtailor, 
    save_lmeval_outputs,
)
from benchmark_utils import (
    infer_short_model_name,
    default_output_file_basename,
)


def main():
    tf_logging.set_verbosity_error()
    parser = argparse.ArgumentParser(description="VocabTailor information extraction evaluation via lm-eval")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Model path or Hugging Face model id.")
    parser.add_argument("--lmdb_path", type=str, default=None, help="Path to LMDB store for offloading weights (optional).")
    parser.add_argument("--prefix", type = str, default = None, help = "Prefix for output file.")
    parser.add_argument(
        "--vocab_resize_strategy",
        type=str,
        choices=["realloc", "split_linear", "prealloc"],
        default="prealloc",
        help = (
            "Strategy for dynamically resizing the embedding vocabulary:\n"
            "  realloc          : fully reallocate embedding weights when vocab grows\n"
            "  split_linear     : append a separate linear/embedding block (no full reallocation)\n"
            "  prealloc         : pre-allocate extra embedding capacity and grow in-place\n"
        )
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target device to move transformer blocks.")
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
        enable_vocabtailor=True,
        offload_to_lmdb=args.lmdb_path is not None,
        vocabtailor_mode="dynamic",
        vocab_resize_strategy=args.vocab_resize_strategy,
    )
    # print(f"Output file basename: {base_name}")

    # prefix = args.prefix if args.prefix is not None else "vt_d"
    # if args.limit is not None:
    #     prefix += f"_sample{args.limit}"
    
    # base_name = f"{short_model_name}_{args.dtype}_{prefix}_ie_eval"
    
    # if args.device != "cuda":
    #     base_name += f"_{args.device}"

    # if args.vocab_resize_strategy == "split_linear":
    #     base_name += "_splitlinear"
    # elif args.vocab_resize_strategy == "prealloc":
    #     base_name += "_prealloc"
    
    # if args.lmdb_path:
    #     base_name += "_lmdb"
    
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

    # Initialize VocabTailor
    logger.info("Loading VocabTailor...")
    model_kwargs = {}
    tokenizer_kwargs = {}
    if args.local_files_only:
        model_kwargs["local_files_only"] = True
        tokenizer_kwargs["local_files_only"] = True

    vt = VocabTailor.from_pretrained(
        args.model_name,
        device=args.device,
        dtype=args.dtype,
        lmdb_path=args.lmdb_path,
        vocab_resize_strategy=args.vocab_resize_strategy,
        profiling_file=None,
        enable_metrics_tracker=False,
        tokenizer_kwargs=tokenizer_kwargs,
        **model_kwargs,
    )
    logger.info("Finish loading VocabTailor")

    # Run evaluation
    results = evaluate_vocabtailor(vt, args.tasks, limit=args.limit)
    if args.save_predictions:
        save_lmeval_outputs(results, task_name="squad_completion", pred_path=pred_file, metric_path=output_file)

    logger.info("Done.")

if __name__ == "__main__":
    main()
