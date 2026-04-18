#!/usr/bin/env python3
"""
MT memory test: baseline (BaselineGenerator). Staged RSS breakdown only; tracker disabled
so measurements are not affected by tracker storage. Use for paper memory figures.
"""
import gc
import json
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
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as tf_logging

from vocab_tailor import BaselineGenerator

from mt_harness import (
    LANGUAGE_DICT,
    get_default_dataset_path,
    apply_mt_template,
    load_mt_dataset,
)
from benchmark_utils import (
    get_torch_dtype, 
    infer_short_model_name,
    default_output_file_basename,
    get_peak_vram_gb, 
)


def main():
    tf_logging.set_verbosity_error()
    parser = argparse.ArgumentParser(description="Baseline machine translation memory test: staged RSS, no tracker")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B", help="Model path or Hugging Face model id.")
    parser.add_argument("--dataset", type=str, default=get_default_dataset_path(_REPO_ROOT), help="Path to JSONL with 'source' and 'target' columns. Default: datasets/wmt24pp/en-zh_CN.jsonl under repo root.")
    parser.add_argument("--prefix", type = str, default = None, help = "Prefix for output file.")
    parser.add_argument("--source_lang", type=str, default="en", choices=list(LANGUAGE_DICT), help="Source language code (e.g. en, zh).")
    parser.add_argument("--target_lang", type=str, default="zh", choices=list(LANGUAGE_DICT), help="Target language code (e.g. en, zh).")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size for test. (default: None = use full dataset).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on.")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="fp32", help="Model dtype.")
    parser.add_argument("--local_files_only", action="store_true", help="Use only local files for Hugging Face model/tokenizer.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results. Default: results/machine_translation/{args.source_lang}-{args.target_lang}/memory_test under repo root.")
    args = parser.parse_args()

    if args.dataset and not os.path.isfile(args.dataset):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset}")

    # Initialize output file path
    short_model_name = infer_short_model_name(args.model_name)
    base_name = default_output_file_basename(
        short_model_name=short_model_name,
        dtype=args.dtype,
        device=args.device,
        task_name="mt_memory_test",
        prefix=args.prefix,
        sample_size=args.sample_size,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
    )
    # print(f"Output file basename: {base_name}")

    # prefix = "baseline" if args.prefix is None else args.prefix
    # if args.sample_size is not None:
    #     prefix += f"_sample{args.sample_size}"
    
    # base_name = f"{short_model_name}_{args.dtype}_{prefix}_mt_memory_test_{args.source_lang}_{args.target_lang}"
    
    # if args.device != "cuda":
    #     base_name += f"_{args.device}"

    output_dir = f"{_REPO_ROOT}/results/machine_translation/{args.source_lang}-{args.target_lang}/memory_test" if args.output_dir is None else args.output_dir
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

    # Create test prompts
    source_lang = LANGUAGE_DICT[args.source_lang]
    target_lang = LANGUAGE_DICT[args.target_lang]
    filtered_dataset = filtered_dataset.map(
        apply_mt_template,
        fn_kwargs={"source": source_lang, "target": target_lang},
        batched=False,
    )
    logger.info("Finish creating prompts...")

    # Reset memory usages
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process(os.getpid())
    rss_init = process.memory_info().rss / (1024**3)
    logger.info(f"Initial RSS: {rss_init:.2f} GB")

    # Initialize model
    model_kwargs = {}
    tokenizer_kwargs = {}
    if args.local_files_only:
        model_kwargs["local_files_only"] = True
        tokenizer_kwargs["local_files_only"] = True
    torch_dtype = get_torch_dtype(args.dtype)

    logger.info("Load model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="cpu",
        torch_dtype=torch_dtype,
        **model_kwargs,
    )
    model.eval()

    rss_after_model = process.memory_info().rss / (1024**3)
    logger.info("Loaded model info:")
    logger.info(f"    model size:     {rss_after_model - rss_init:.2f} GB")
    logger.info(f"    model dtype:    {model.dtype}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tokenizer_kwargs)
    time.sleep(10)  # Let memory settle before measuring tokenizer RSS (match original script)
    
    rss_after_tokenizer = process.memory_info().rss / (1024**3)
    logger.info("Loaded tokenizer info:")
    logger.info(f"    tokenizer size:  {rss_after_tokenizer - rss_after_model:.2f} GB")
    logger.info("Finish loading model and tokenizer")

    # Initialize BaselineGenerator
    logger.info("Initialize BaselineGenerator...")
    bg = BaselineGenerator()
    bg.load_model(
        model=model, 
        device=args.device,
        enable_metrics_tracker=False,
    )

    # FORCE RAM release if model was moved to GPU
    if args.device == "cuda":
        del model
        gc.collect()
        torch.cuda.empty_cache()
    rss_after_setup = process.memory_info().rss / (1024**3)
    logger.info("BaselineGenerator setup info:")
    logger.info(f"    model device:   {args.device}")
    logger.info(f"    setup RSS:      {rss_after_setup - rss_init:.2f} GB")
    logger.info("Finish creating BaselineGenerator")

    #----------------------
    # One-time Warm-up
    #----------------------
    warmup_inputs = tokenizer.apply_chat_template(
        filtered_dataset[0]["chat_messages"],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    _ = bg.generate(warmup_inputs, max_new_tokens=32, do_sample=False)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()                # Wait for warm-up to finish
        torch.cuda.reset_peak_memory_stats()    # Reset so we only track inference peaks
    rss_after_warmup = process.memory_info().rss / (1024**3)

    #----------------------
    # Memory Test
    #----------------------
    N = len(filtered_dataset)
    logger.info(f"Run memory test with {N} examples...")
    peak_rss_observed = rss_after_setup

    for i, message in enumerate(filtered_dataset["chat_messages"]):
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
        _ = bg.generate(
            inputs, 
            max_new_tokens=2048, 
            do_sample=False
        )

        # Track peak RAM spike
        curr_rss = process.memory_info().rss / (1024**3)
        if curr_rss > peak_rss_observed:
            peak_rss_observed = curr_rss
        
        if i % 100 == 0:
            logger.info(f"Complete examples {i + 1} / {N}")
    logger.info(f"Complete examples {N} / {N}")

    #----------------------
    # Summary
    #----------------------
    logger.info("Generate metrics summary...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    rss_final = process.memory_info().rss / (1024**3)
    peak_vram = get_peak_vram_gb()

    # Final Report
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{'#' * 60}\n#{' ' * 20}Memory Usage Report{' ' * 20}#\n{'#' * 60}\n")
        f.write(f"System RAM (RSS):             {rss_final - rss_init:.4f} GB\n")
        f.write(f"GPU VRAM (Est.):              {peak_vram:.4f} GB\n\n")
        f.write(f"{'=' * 60}\n{' ' * 20}CPU RAM Breakdown\n{'=' * 60}\n")
        f.write(f"Model + Tokenizer Load:       {rss_after_tokenizer - rss_init:.4f} GB\n")
        f.write(f"Baseline Load:                {rss_after_setup - rss_after_tokenizer:.4f} GB\n")
        f.write(f"Total Setup:                  {rss_after_setup - rss_init:.4f} GB\n")
        f.write(f"Total Setup2:                 {rss_after_warmup - rss_init:.4f} GB\n")
        f.write(f"rss_after_setup:              {rss_after_setup:.4f} GB\n")
        f.write(f"rss_after_warmup:             {rss_after_warmup:.4f} GB\n")
        f.write(f"Inference Overhead:           {rss_final - rss_after_setup:.4f} GB\n")
        f.write(f"Inference RAM Spike:          {peak_rss_observed:.4f} GB\n\n")
    logger.info(f"Logs saved to {log_file}")

    # Save data
    memory_breakdown = {
        "raw_rss_metrics": {
            "rss_init": rss_init,
            "rss_after_model": rss_after_model,
            "rss_after_tokenizer": rss_after_tokenizer,
            "rss_after_setup": rss_after_setup,
            "rss_after_warmup": rss_after_warmup,
            "peak_rss_observed": peak_rss_observed,
            "rss_final": rss_final,
        },
        "run_stats": {
            "total_rss": rss_final - rss_init,
            "rss_model_load": rss_after_model - rss_init,
            "rss_tokenizer": rss_after_tokenizer - rss_after_model,
            "rss_bg_load": rss_after_setup - rss_after_tokenizer,
            "rss_setup": rss_after_setup - rss_init,
            "rss_inference_overhead": rss_final - rss_after_setup,
            "peak_rss_observed": peak_rss_observed,
            "peak_vram": peak_vram,
        },
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(memory_breakdown, f, indent=4)
    logger.info(f"Results saved to {output_file}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
