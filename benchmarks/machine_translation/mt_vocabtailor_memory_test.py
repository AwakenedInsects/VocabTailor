#!/usr/bin/env python3
"""
MT memory test: VocabTailor. Staged RSS breakdown only; tracker disabled so measurements
are not affected by tracker storage. Use for paper memory figures.

Unlike mt_vocabtailor_eval.py and mt_vocabtailor_benchmark.py, this script does NOT use
VocabTailor.from_pretrained. It loads model, tokenizer, and VocabTailor in separate steps
so we can record RSS after each stage (model, tokenizer, VT setup, LM head init, etc.).
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

from vocab_tailor import VocabTailor
from vocab_tailor.lmdb_layers import build_lmdb_weights
from vocab_tailor.model_utils import load_model_backbone

from mt_harness import (
    LANGUAGE_DICT,
    get_default_dataset_path,
    apply_mt_template,
    load_mt_dataset,
)
from benchmark_utils import (
    validate_vocabtailor_args,
    get_torch_dtype,
    infer_short_model_name,
    infer_vocabtailor_mode,
    default_output_file_basename,
    get_peak_vram_gb,
)


def main():
    tf_logging.set_verbosity_error()
    parser = argparse.ArgumentParser(description="VocabTailor machine translation memory test: staged RSS, no tracker")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B", help="Model path or Hugging Face model id.")
    parser.add_argument("--lmdb_path", type=str, default=None, help = "Path to LMDB store for offloading weights (optional).")
    parser.add_argument("--profiling_file", type=str, default=None, help="Path to static vocab JSON (optional). If omitted, no static vocab; use with --input_aware for dynamic-only.")
    parser.add_argument("--dataset", type=str, default=get_default_dataset_path(_REPO_ROOT), help="Path to JSONL with 'source' and 'target' columns. Default: datasets/wmt24pp/en-zh_CN.jsonl under repo root.")
    parser.add_argument("--prefix", type = str, default = None, help = "Prefix for output file.")
    parser.add_argument("--source_lang", type=str, default="en", choices=list(LANGUAGE_DICT), help="Source language code (e.g. en, zh).")
    parser.add_argument("--target_lang", type=str, default="zh", choices=list(LANGUAGE_DICT), help="Target language code (e.g. en, zh).")
    parser.add_argument("--input_aware", action = 'store_true', help = "Enable input-aware pruning.")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size for test. (default: None = use full dataset).")
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
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results. Default: results/machine_translation/{args.source_lang}-{args.target_lang}/memory_test under repo root.")
    args = parser.parse_args()

    validate_vocabtailor_args(args.dataset, args.profiling_file, args.input_aware)
    dataset_path = args.dataset
    profiling_path = args.profiling_file

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
        enable_vocabtailor=True,
        offload_to_lmdb=args.lmdb_path is not None,
        vocabtailor_mode=infer_vocabtailor_mode(args.input_aware, args.profiling_file is not None),
        vocab_resize_strategy=args.vocab_resize_strategy,
    )
    # print(f"Output file basename: {base_name}")

    # if args.prefix is None:
    #     prefix = "vt"
    #     if args.input_aware and profiling_path is not None:
    #         prefix += "_h" # hybrid
    #     elif args.input_aware:
    #         prefix += "_d" # dynamic-only
    #     else:
    #         prefix += "_s" # static-only
    # else:
    #     prefix = args.prefix
    
    # if args.sample_size is not None:
    #     prefix += f"_sample{args.sample_size}"

    # # Initialize output file path
    # base_name = f"{short_model_name}_{args.dtype}_{prefix}_mt_memory_test_{args.source_lang}_{args.target_lang}"
    
    # if args.device != "cuda":
    #     base_name += f"_{args.device}"
    
    # if args.vocab_resize_strategy == "split_linear":
    #     base_name += "_splitlinear"
    # elif args.vocab_resize_strategy == "prealloc":
    #     base_name += "_prealloc"
    
    # if args.lmdb_path:
    #     base_name += "_lmdb"

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
    filtered_dataset = load_mt_dataset(dataset_path, sample_size=args.sample_size)
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

    # Initialize profiling file
    preload_ids = []
    if profiling_path is not None and os.path.isfile(profiling_path):
        with open(profiling_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            preload_ids = list(data.values())
        elif isinstance(data, list):
            preload_ids = [int(x) for x in data]
        else:
            raise ValueError("profiling_file JSON must be a dict (token->id) or list of token ids.")

    # Reset memory usages
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process(os.getpid())
    rss_init = process.memory_info().rss / (1024 ** 3)
    logger.info(f"Initial RSS: {rss_init:.2f} GB")

    # Initialize model
    model_kwargs = {}
    tokenizer_kwargs = {}
    if args.local_files_only:
        model_kwargs["local_files_only"] = True
        tokenizer_kwargs["local_files_only"] = True
    lmdb_path = args.lmdb_path
    torch_dtype = get_torch_dtype(args.dtype)

    logger.info("Load model and tokenizer...")
    if lmdb_path:
        # 1. Check LMDB, build a new one if not exist
        logger.info("Checking LMDB...")
        build_lmdb_weights(args.model_name, lmdb_path)

        # 2. Load model without embedding and lm head
        logger.info("Load only transformer backbone...")
        model = load_model_backbone(
            args.model_name,
            device="cpu", # match the default settings in non-lmdb loading
            dtype=torch_dtype,
            include_embeddings=False,
            include_lm_head=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="cpu",
            torch_dtype=torch_dtype,
            **model_kwargs,
        )
    model.eval()

    rss_after_model = process.memory_info().rss / (1024**3)
    logger.info("Loaded model info:")
    logger.info(f"    model size:     {process.memory_info().rss / (1024 ** 3) - rss_init:.2f} GB")
    logger.info(f"    model dtype:    {model.dtype}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tokenizer_kwargs)
    original_eos_token_id = tokenizer.eos_token_id
    time.sleep(10)  # Let memory settle before measuring tokenizer RSS (match original script)
    
    rss_after_tokenizer = process.memory_info().rss / (1024**3)
    logger.info("Loaded tokenizer info:")
    logger.info(f"    tokenizer size:  {rss_after_tokenizer - rss_after_model:.2f} GB")
    logger.info("Finish loading model and tokenizer")

    # Initialize VocabTailor
    mode = "input_aware" if args.input_aware else "default"
    logger.info("Initialize VocabTailor...")
    vt = VocabTailor()
    vt.load_model(
        model=model, 
        device=args.device, 
        lmdb_path=lmdb_path, 
        vocab_resize_strategy=args.vocab_resize_strategy,
        enable_metrics_tracker=False,
    )

    logger.info("VocabTailor model info:")
    logger.info(f"    {'name':<20} {'dtype':<20} {'device':<10}")
    logger.info("    " + "-"*60)
    if vt.offload_to_lmdb:
        logger.info(f"    {'embedding':<20} {str(vt.provider.fetch_dtype):<20} {str(vt.model.model.embed_tokens.device):<10}")
    else:
        logger.info(f"    {'embedding':<20} {str(vt.model.model.embed_tokens.full_embedding.weight.dtype):<20} {str(vt.model.model.embed_tokens.device):<10}")
    logger.info(f"    {'lm head':<20} {str(vt.model.lm_head.weight_dtype):<20} {str(vt.model.lm_head.device):<10}")
    logger.info(f"    {'transformer body':<20} {str(model.dtype):<20} {str(args.device):<10}\n")
    
    cpu_weights_size = sum(p.numel() * p.element_size() for p in vt.model.model.embed_tokens.full_embedding.parameters()) / (1024 ** 3)
    logger.info(f"    CPU weights size: {cpu_weights_size} GB")

    # FORCE RAM release if model was moved to GPU
    if args.device == "cuda":
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    rss_after_setup = process.memory_info().rss / (1024**3)
    logger.info(f"    setup RSS:      {rss_after_setup - rss_init:.2f} GB")
    logger.info("Finish creating VocabTailor")

    # Initialize LM head with common tokens and task-specific tokens
    special_token_ids = tokenizer.all_special_ids
    init_ids = torch.tensor(list(set(preload_ids + special_token_ids)))
    
    vt.update_lm_head(init_ids, temp=False)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    rss_after_head_init = process.memory_info().rss / (1024 ** 3)
    logger.info(f"LM head init RSS: {rss_after_head_init - rss_after_setup:.2f} GB")
    logger.info("Finish initializing lm head")

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
    _ = vt.generate(
        warmup_inputs,
        mode=mode,
        max_new_tokens=32,
        do_sample=False,
        original_eos_token_id=original_eos_token_id,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()               # Wait for warm-up to finish
        torch.cuda.reset_peak_memory_stats()   # Reset so we only track inference peaks
    rss_after_warmup = process.memory_info().rss / (1024**3)

    #----------------------
    # Memory Test
    #----------------------
    N = len(filtered_dataset)
    logger.info(f"Run memory test with {N} examples...")
    peak_rss_observed = rss_after_head_init

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
        _ = vt.generate(
            inputs,
            mode=mode,
            max_new_tokens=2048,
            do_sample=False,
            original_eos_token_id=original_eos_token_id,
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
        f.write(f"VocabTailor Load:             {rss_after_setup - rss_after_tokenizer:.4f} GB\n")
        f.write(f"LM Head Init:                 {rss_after_head_init - rss_after_setup:.4f} GB\n")
        f.write(f"Total Setup:                  {rss_after_head_init - rss_init:.4f} GB\n")
        f.write(f"Total Setup2:                 {rss_after_warmup - rss_init:.4f} GB\n")
        f.write(f"rss_after_head_init:          {rss_after_head_init:.4f} GB\n")
        f.write(f"rss_after_warmup:             {rss_after_warmup:.4f} GB\n")
        f.write(f"Inference Overhead:           {rss_final - rss_after_head_init:.4f} GB\n")
        f.write(f"Inference RAM Spike:          {peak_rss_observed:.4f} GB\n\n")
    logger.info(f"Logs saved to {log_file}")

    # Save data
    memory_breakdown = {
        "raw_rss_metrics": {
            "rss_init": rss_init,
            "rss_after_model": rss_after_model,
            "rss_after_tokenizer": rss_after_tokenizer,
            "rss_after_setup": rss_after_setup,
            "rss_after_head_init": rss_after_head_init,
            "peak_rss_observed": peak_rss_observed,
            "rss_final": rss_final,
        },
        "run_stats": {
            "total_rss": rss_final - rss_init,
            "rss_model_load": rss_after_model - rss_init,
            "rss_tokenizer": rss_after_tokenizer - rss_after_model,
            "rss_vt_load": rss_after_setup - rss_after_tokenizer,
            "rss_head_init": rss_after_head_init - rss_after_setup,
            "rss_setup": rss_after_head_init - rss_init,
            "rss_inference_overhead": rss_final - rss_after_head_init,
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