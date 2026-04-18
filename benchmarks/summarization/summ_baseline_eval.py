#!/usr/bin/env python3
"""
Summarization evaluation (quality): baseline (BaselineGenerator). Uses BaselineGenerator.from_pretrained; computes ROUGE.

Default model: AwakenedInsects/llama-3.2-3b-vocabtailor-sum. Default dataset: datasets/xsum/xsum_test.jsonl under repo root.

Run from VocabTailor:
  cd VocabTailor && PYTHONPATH=src python benchmarks/summarization/summ_baseline_eval.py [options]

Example:
  PYTHONPATH=src python benchmarks/summarization/summ_baseline_eval.py --dataset datasets/xsum/xsum_test.jsonl --sample_size 10
"""
import os
import sys
import json

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import argparse
import logging

import torch
from transformers import logging as tf_logging

try:
    import evaluate
except ImportError:
    evaluate = None

from vocab_tailor import BaselineGenerator

from summ_harness import (
    get_default_dataset_path,
    apply_summarization_template,
    load_summ_dataset,
)
from benchmark_utils import (
    extract_prediction, 
    infer_short_model_name,
    default_output_file_basename,
)


def main():
    tf_logging.set_verbosity_error()
    parser = argparse.ArgumentParser(description="Baseline summarization evaluation: ROUGE.")
    parser.add_argument("--model_name", type=str, default="AwakenedInsects/llama-3.2-3b-vocabtailor-sum", help="Model path or Hugging Face model id.")
    parser.add_argument("--dataset", type=str, default=get_default_dataset_path(_REPO_ROOT), help="Path to JSONL with 'document' and 'summary' columns. Default: datasets/xsum/xsum_test.jsonl under repo root.")
    parser.add_argument("--prefix", type=str, default="", help="Prefix for output file.")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size (default: None = use full dataset).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on.")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16", help="Model dtype.")
    parser.add_argument("--local_files_only", action="store_true", help="Use only local files for Hugging Face model/tokenizer.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results. Default: results/summarization/eval under repo root.")
    parser.add_argument("--save_predictions", action="store_true", help="Save extracted predictions to .pth")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity.")
    args = parser.parse_args()

    if args.dataset and not os.path.isfile(args.dataset):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset}")

    if evaluate is None:
        raise ImportError("evaluate is required for ROUGE: pip install evaluate")

    # Initialize output file path
    short_model_name = infer_short_model_name(args.model_name)
    base_name = default_output_file_basename(
        short_model_name=short_model_name,
        dtype=args.dtype,
        device=args.device,
        task_name="summ_eval",
        prefix=args.prefix,
        sample_size=args.sample_size,
    )
    # print(f"Output file basename: {base_name}")
    
    # prefix = "baseline" if args.prefix is None else args.prefix
    # if args.sample_size is not None:
    #     prefix += f"_sample{args.sample_size}"
    
    # base_name = f"{short_model_name}_{args.dtype}_{prefix}_summ_eval"
    
    # if args.device != "cuda":
    #     base_name += f"_{args.device}"

    output_dir = f"{_REPO_ROOT}/results/summarization/eval" if args.output_dir is None else args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    log_file = f"{output_dir}/logs/{base_name}.log"
    output_file = f"{output_dir}/{base_name}.json"
    pred_file = f"{output_dir}/{base_name}_pred.pth"
    # print(f"log file: {log_file}\noutput file: {output_file}\nprediction file: {pred_file}\n")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    logger = logging.getLogger(__name__)

    # Initialize input dataset
    filtered_dataset = load_summ_dataset(args.dataset, sample_size = args.sample_size, split = "test")
    logger.info(f"Dataset loaded: {len(filtered_dataset)} rows")

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
    tokenizer = bg.tokenizer
    logger.info("Finish loading BaselineGenerator")

    # Create test prompts
    filtered_dataset = filtered_dataset.map(apply_summarization_template, batched=False)
    logger.info("Finish creating prompts...")

    #----------------------
    # Evaluation Test
    #----------------------
    N = len(filtered_dataset)
    logger.info(f"Run evaluation with {N} examples...")

    extracted_pred = []

    for i, message in enumerate(filtered_dataset["chat_messages"]):
        if args.verbose:
            print(f"\n{'=' * 50}\nProcessing Example {i + 1}/{N}\n{'='*50}\n")

        # Input
        if "Qwen3" in args.model_name:
            inputs = tokenizer.apply_chat_template(
                message,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            )
        else:
            inputs = tokenizer.apply_chat_template(
                message,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        
        # Generate
        output_ids = bg.generate(
            inputs, 
            max_new_tokens=2048, 
            do_sample=False
        )

        # Decode & Prediction   
        output_text = tokenizer.batch_decode(output_ids.to("cpu"))[0]
        extracted_text = extract_prediction(output_text, short_model_name)

        # Append results
        extracted_pred.append(extracted_text)

        # Print results
        if args.verbose:
            print(f"Summarization Result\n{'-'*40}")
            print(f"Document: {filtered_dataset['document'][i]}\n")
            print(f"Summary: {extracted_text}\n")

        if i % 100 == 0:
            logger.info(f"Complete examples {i + 1}/{N}")
    logger.info(f"Complete examples {N}/{N}")

    #----------------------
    # Evaluation Metrics
    #----------------------
    rouge = evaluate.load("rouge")
    refs = list(filtered_dataset["summary"])
    results = rouge.compute(predictions=extracted_pred, references=refs)
    logger.info(f"ROUGE: {results}")

    metrics = dict(results)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {output_file}")

    if args.save_predictions:
        torch.save(extracted_pred, pred_file)
        logger.info(f"Predictions saved to {pred_file}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
