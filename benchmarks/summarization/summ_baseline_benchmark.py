#!/usr/bin/env python3
"""
Summarization benchmark: baseline (BaselineGenerator). Speed & memory via BaselineGenerator and shared harness.

Default model: AwakenedInsects/llama-3.2-3b-vocabtailor-sum. Default dataset: datasets/xsum/xsum_test.jsonl under repo root.

Run from VocabTailor:
  cd VocabTailor && PYTHONPATH=src python benchmarks/summarization/summ_baseline_benchmark.py [options]

Example:
  PYTHONPATH=src python benchmarks/summarization/summ_baseline_benchmark.py --dataset datasets/xsum/xsum_test.jsonl --sample_size 10
"""
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import argparse
import logging

import psutil
import torch
from transformers import logging as tf_logging

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
    get_peak_vram_gb,
    generate_runs_summary,
)


def main():
    tf_logging.set_verbosity_error()
    parser = argparse.ArgumentParser(description="Baseline summarization benchmark: speed & memory via BaselineGenerator and shared harness.")
    parser.add_argument("--model_name", type=str, default="AwakenedInsects/llama-3.2-3b-vocabtailor-sum", help="Model path or Hugging Face model id.")
    parser.add_argument("--dataset", type=str, default=get_default_dataset_path(_REPO_ROOT), help="Path to JSONL with 'document' and 'summary' columns. Default: datasets/xsum/xsum_test.jsonl under repo root.")
    parser.add_argument("--prefix", type=str, default="", help="Prefix for output file.")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size (default: None = use full dataset).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on.")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16", help="Model dtype.")
    parser.add_argument("--local_files_only", action="store_true", help="Use only local files for Hugging Face model/tokenizer.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results. Default: results/summarization/benchmark under repo root.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity.")
    args = parser.parse_args()

    if args.dataset and not os.path.isfile(args.dataset):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset}")

    # Initialize output file path
    short_model_name = infer_short_model_name(args.model_name)
    base_name = default_output_file_basename(
        short_model_name=short_model_name,
        dtype=args.dtype,
        device=args.device,
        task_name="summ_benchmark",
        prefix=args.prefix,
        sample_size=args.sample_size,
    )
    # print(f"Output file basename: {base_name}")

    # prefix = "baseline" if args.prefix is None else args.prefix
    # if args.sample_size is not None:
    #     prefix += f"_sample{args.sample_size}"
    
    # base_name = f"{short_model_name}_{args.dtype}_{prefix}_summ_benchmark"
    
    # if args.device != "cuda":
    #     base_name += f"_{args.device}"
    
    output_dir = f"{_REPO_ROOT}/results/summarization/benchmark" if args.output_dir is None else args.output_dir
    os.makedirs(output_dir, exist_ok = True)
    os.makedirs(f"{output_dir}/logs", exist_ok = True)
    log_file = f"{output_dir}/logs/{base_name}.log"
    output_file = f"{output_dir}/{base_name}.json"
    # print(f"log file: {log_file}\noutput file: {output_file}\n")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    logger = logging.getLogger(__name__)

    # Initialize input dataset
    filtered_dataset = load_summ_dataset(args.dataset, sample_size = args.sample_size, split = "test")
    logger.info(f"Dataset loaded: {len(filtered_dataset)} rows")

    # Reset memory usages
    process = psutil.Process(os.getpid())
    init_rss = process.memory_info().rss / (1024**3)

    # Reset CUDA memory stats if available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Initialize BaselineGenerator
    logger.info("Loading BaselineGenerator...")
    model_kwargs = {}
    tokenizer_kwargs = {}
    if args.local_files_only:
        model_kwargs["local_files_only"] = True
        tokenizer_kwargs["local_files_only"] = True

    bg = BaselineGenerator.from_pretrained(
        args.model_name,
        device=args.device,
        dtype=args.dtype,
        enable_metrics_tracker=True,
        tokenizer_kwargs=tokenizer_kwargs,
        **model_kwargs,
    )
    tokenizer = bg.tokenizer
    logger.info("Finish loading BaselineGenerator")

    # Create test prompts
    filtered_dataset = filtered_dataset.map(apply_summarization_template, batched=False)
    logger.info("Finish creating prompts...")

    #----------------------
    # Benchmark Test
    #----------------------
    N = len(filtered_dataset)
    logger.info("Run benchmark with %d examples...", N)
    
    extracted_texts = []
    all_metrics = []
    total_job_time = 0.0

    for i, message in enumerate(filtered_dataset["chat_messages"]):
        if args.verbose:
            print(f"\n{'=' * 50}\nProcessing Example {i + 1}/{N}\n{'=' * 50}\n")
        
        job_start_time = time.perf_counter()

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
            do_sample=False,
        )
        metrics = bg.gen_metrics
        job_time = time.perf_counter() - job_start_time 
        metrics["job_time"] = job_time
        total_job_time += job_time
        
        output_text = tokenizer.batch_decode(output_ids.to("cpu"))[0]
        extracted_text = extract_prediction(output_text, short_model_name)
        extracted_texts.append(extracted_text)
        all_metrics.append(metrics)

        if args.verbose:
            print(f"Summarization Result\n{'-'*40}")
            print(f"Document: {filtered_dataset['document'][i]}\n")
            print(f"Summary: {extracted_text}\n")

            print(f"Metrics\n{'-'*40}")
            print(f"Job time:               {metrics['job_time']:.4f} s")
            print(f"Prefill time:           {metrics['prefill_time']:.4f} s")
            print(f"Prefill tokens:         {metrics['prefill_tokens']:,}")
            print(f"Prefill speed:          {metrics['prefill_tps']:,.4f} tps")
            print(f"Decode time:            {metrics['decode_time']:.4f} s")
            print(f"Decode tokens:          {metrics['decode_tokens']:,}")
            print(f"Decode speed:           {metrics['decode_tps']:,.4f} tps\n")

        if i % 100 == 0:
            logger.info(f"Complete examples {i + 1}/{N}")
    logger.info(f"Complete examples {N}/{N}")

    #----------------------
    # Summary
    #----------------------
    logger.info("Generate metrics summary...")

    # add predictions to the dataset
    filtered_dataset = filtered_dataset.add_column("pred", extracted_texts)
    # update memory usage
    total_ram_gb = process.memory_info().rss / (1024**3) - init_rss
    peak_vram_gb = get_peak_vram_gb()
    # Optional: update tracker and save log for total
    bg.tracker.total_job_time = total_job_time
    bg.tracker.total_ram_gb = total_ram_gb
    bg.tracker.peak_vram_gb = peak_vram_gb
    bg.tracker.save_log(
        log_file_path=log_file,
        enable_vocab_tailor=False,
        offload_to_lmdb=False,
        append_log=True,
    )
    # save results
    mem_usages = {"total_ram_gb": total_ram_gb, "peak_vram_gb": peak_vram_gb}
    _ = generate_runs_summary(
        filtered_dataset, all_metrics, mem_usages, output_file,
        input_col="document", output_col="summary",
    )
    logger.info("Results saved to %s", output_file)
    logger.info("Done.")


if __name__ == "__main__":
    main()
