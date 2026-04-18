#!/usr/bin/env python3
"""
MT benchmark: baseline (BaselineGenerator). Speed & memory via package and shared harness.

Run from VocabTailor:
  cd VocabTailor && PYTHONPATH=src python benchmarks/machine_translation/mt_baseline_benchmark.py [options]

Example:
  PYTHONPATH=src python benchmarks/machine_translation/mt_baseline_benchmark.py --dataset datasets/wmt24pp/en-zh_CN.jsonl --sample_size 5
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

from mt_harness import (
    LANGUAGE_DICT,
    get_default_dataset_path,
    apply_mt_template,
    load_mt_dataset,
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
    parser = argparse.ArgumentParser(description="Baseline machine translation benchmark: speed & memory via BaselineGenerator and shared harness.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B", help="Model path or Hugging Face model id.")
    parser.add_argument("--dataset", type=str, default=get_default_dataset_path(_REPO_ROOT), help="Path to JSONL with 'source' and 'target' columns. Default: datasets/wmt24pp/en-zh_CN.jsonl under repo root.")
    parser.add_argument("--prefix", type = str, default = None, help = "Prefix for output file.")
    parser.add_argument("--source_lang", type=str, default="en", choices=list(LANGUAGE_DICT), help="Source language code (e.g. en, zh).")
    parser.add_argument("--target_lang", type=str, default="zh", choices=list(LANGUAGE_DICT), help="Target language code (e.g. en, zh).")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size for test. (default: None = use full dataset).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on.")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="fp32", help="Model dtype.")
    parser.add_argument("--local_files_only", action="store_true", help="Use only local files for Hugging Face model/tokenizer.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results. Default: results/machine_translation/{args.source_lang}-{args.target_lang}/benchmark under repo root.")
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
        task_name="mt_benchmark",
        prefix=args.prefix,
        sample_size=args.sample_size,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
    )
    # print(f"Output file basename: {base_name}")
    
    # prefix = "baseline" if args.prefix is None else args.prefix
    # if args.sample_size is not None:
    #     prefix += f"_sample{args.sample_size}"

    # base_name = f"{short_model_name}_{args.dtype}_{prefix}_mt_benchmark_{args.source_lang}_{args.target_lang}"
    
    # if args.device != "cuda":
    #     base_name += f"_{args.device}"

    output_dir = f"{_REPO_ROOT}/results/machine_translation/{args.source_lang}-{args.target_lang}/benchmark" if args.output_dir is None else args.output_dir
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
    filtered_dataset = load_mt_dataset(args.dataset, sample_size = args.sample_size)
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
    source_lang = LANGUAGE_DICT[args.source_lang]
    target_lang = LANGUAGE_DICT[args.target_lang]
    filtered_dataset = filtered_dataset.map(
        apply_mt_template,
        fn_kwargs={"source": source_lang, "target": target_lang},
        batched=False,
    )
    logger.info("Finish creating prompts...")

    #----------------------
    # Benchmark Test
    #----------------------
    N = len(filtered_dataset)
    logger.info(f"Run benchmark with {N} examples...")
    
    extracted_texts = []
    all_metrics = []
    total_job_time = 0.0

    for i, message in enumerate(filtered_dataset["chat_messages"]):
        if args.verbose:
            print(f"\n{'=' * 50}\nProcessing Example {i + 1}/{N}\n{'='*50}\n")
        
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
        extracted = extract_prediction(output_text, short_model_name)
        extracted_texts.append(extracted)
        all_metrics.append(metrics)

        if args.verbose:
            print(f"Translation Result\n{'-'*40}")
            print(f"{source_lang}: {filtered_dataset['source'][i]}\n")
            print(f"{target_lang}: {extracted}\n")

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
        input_col="source", output_col="target",
    )
    logger.info(f"Results saved to {output_file}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
