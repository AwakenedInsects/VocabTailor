"""
Shared utilities for benchmarks: validation, dtype, prediction extraction, model name,
stats, VRAM, run summary, VocabTailor mode inference, and default output file basename.

Provides: validate_vocabtailor_args, get_torch_dtype, extract_prediction, infer_short_model_name,
infer_vocabtailor_mode, default_output_file_basename, compute_stats, get_peak_vram_gb,
generate_runs_summary.

Used by: mt_harness, summ_harness (and thus by all mt_* and summ_* scripts), and ie_* scripts.
"""
import json
import os
import re
from typing import Any, Optional

import numpy as np
import torch


def validate_vocabtailor_args(
    dataset_path: Optional[str],
    profiling_path: Optional[str],
    input_aware: bool,
) -> None:
    """Raise FileNotFoundError or ValueError if paths or input_aware/profiling are invalid.

    Args:
        dataset_path (str, optional): Path to dataset JSONL; if set, must exist.
        profiling_path (str, optional): Path to static vocab JSON; if set, must exist.
        input_aware (bool): Whether input-aware mode is enabled.

    Raises:
        FileNotFoundError: If dataset_path or profiling_path is set but file missing.
        ValueError: If both profiling_path is None and input_aware is False.
    """
    if dataset_path and not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if profiling_path and not os.path.isfile(profiling_path):
        raise FileNotFoundError(f"Profiling file not found: {profiling_path}")
    if profiling_path is None and not input_aware:
        raise ValueError("VocabTailor requires at least one of --input_aware or --profiling_file not None.")


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Map CLI dtype string to torch.dtype.

    Args:
        dtype_str (str): One of 'bf16', 'fp16', 'fp32'.

    Returns:
        torch.dtype: Corresponding torch dtype.
    """
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if dtype_str not in dtype_map:
        raise ValueError(f"Invalid dtype: {dtype_str}")
    return dtype_map[dtype_str]


def extract_prediction(text: str, short_model_name: str) -> str:
    """Extract assistant reply from model output using model-specific delimiters.

    Args:
        text (str): Full decoded model output.
        short_model_name (str): One of 'Qwen3', 'Qwen', 'Llama', or 'unknown'.

    Returns:
        str: Extracted reply text, or empty string if no match.
    """
    if short_model_name == "Qwen3":
        match = re.search(
            r"<\|im_start\|>assistant\n<think>\n\n</think>\n\n(.*?)\<\|im_end\|>",
            text,
            re.DOTALL,
        )
    elif short_model_name == "Qwen":
        match = re.search(r"<\|im_start\|>assistant\n(.*?)\<\|im_end\|>", text, re.DOTALL)
    elif short_model_name == "Llama":
        match = re.search(
            r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)\<\|eot_id\|>",
            text,
            re.DOTALL,
        )
    else:
        match = None
    return match.group(1).strip() if match else ""


def infer_short_model_name(model_name: str) -> str:
    """Map HF model id or path to short name (Qwen3, Qwen, Llama, etc.).

    Args:
        model_name (str): Hugging Face model id or local path.

    Returns:
        str: 'Qwen3', 'Qwen', 'Llama', or 'unknown'.
    """
    if "Qwen3" in model_name or "qwen3" in model_name:
        return "Qwen3"
    if "Qwen" in model_name or "qwen" in model_name:
        return "Qwen"
    if "Llama" in model_name or "llama" in model_name:
        return "Llama"
    return "unknown"


def infer_vocabtailor_mode(input_aware: bool, enable_profiling: bool) -> str:
    """
    Infer VocabTailor mode from input_aware and whether a profiling file is provided.

    Args:
        input_aware (bool): Whether input-aware mode is enabled.
        enable_profiling (bool): True when a profiling file is provided (e.g. profiling_file is not None).

    Returns:
        str: 'hybrid' (input_aware + profiling), 'dynamic' (input_aware only), or 'static' (profiling only).
    """
    if not input_aware and not enable_profiling:
        raise ValueError("VocabTailor requires at least enable input_aware or profiling.")
    if input_aware and enable_profiling:
        return "hybrid"
    if input_aware:
        return "dynamic"
    return "static"


def default_output_file_basename(
    short_model_name: str,
    dtype: str,
    device: str,
    task_name: str,
    prefix: str = None,
    sample_size: int = None,
    source_lang: str = None,
    target_lang: str = None,
    enable_vocabtailor: bool = False,
    offload_to_lmdb: bool = False,
    vocabtailor_mode: str = None,
    vocab_resize_strategy: str = None,
) -> str:
    """
    Generate default output file basename for benchmark results.

    Args:
        short_model_name (str): Short model name (e.g. Qwen3, Qwen, Llama).
        dtype (str): Data type used to load model weights (bfloat16, float16, or float32).
        device (str): Device to run the model on.
        task_name (str): Name of the task and test type (e.g. mt, mt_eval, mt_benchmark, mt_memory_test, summarization).
        prefix (str, optional): Prefix for output file; when None and enable_vocabtailor is False, "baseline" is used.
        sample_size (int): Sample size for test.
        source_lang (str): Source language code (e.g. en, zh). Only used for machine translation tasks.
        target_lang (str): Target language code (e.g. en, zh). Only used for machine translation tasks.
        enable_vocabtailor (bool): Whether VocabTailor is enabled.
        offload_to_lmdb (bool): Whether to offload embedding and lm head weights to LMDB.
        vocabtailor_mode (str): Mode of VocabTailor (e.g. hybrid, dynamic, static).
        vocab_resize_strategy (str): Strategy for dynamically resizing the embedding vocabulary (e.g. split_linear, prealloc).

    Returns:
        str: Default output file basename.
    """
    if enable_vocabtailor and (vocabtailor_mode is None or vocab_resize_strategy is None):
        raise ValueError("VocabTailor requires both vocabtailor_mode and vocab_resize_strategy to be set.")
    
    if prefix is None:
        if enable_vocabtailor:
            prefix = "vt"
            if vocabtailor_mode == "hybrid":
                prefix += "_h"
            elif vocabtailor_mode == "dynamic":
                prefix += "_d"
            elif vocabtailor_mode == "static":
                prefix += "_s"
        else:
            prefix = "baseline"
    
    if sample_size is not None:
        prefix += f"_sample{sample_size}"
    
    base_name = f"{short_model_name}_{dtype}_{prefix}_{task_name}"
    if source_lang and target_lang:
        base_name += f"_{source_lang}_{target_lang}"

    if device != "cuda":
        base_name += f"_{device}"
    
    if enable_vocabtailor:
        if vocab_resize_strategy == "split_linear":
            base_name += "_splitlinear"
        elif vocab_resize_strategy == "prealloc":
            base_name += "_prealloc"
        
        if offload_to_lmdb:
            base_name += "_lmdb"
    
    return base_name


def compute_stats(data: list) -> dict:
    """Compute total, mean, and std of a list of numbers.

    Args:
        data (list): List of numeric values.

    Returns:
        dict: Keys 'total', 'avg', 'std'.
    """
    return {
        "total": float(np.sum(data)),
        "avg": float(np.mean(data)),
        "std": float(np.std(data)),
    }


def get_peak_vram_gb() -> float:
    """Return peak VRAM allocated in GB (CUDA or MPS), or 0.0 if unavailable."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / (1024**3)
    return 0.0


def generate_runs_summary(
    dataset: Any,
    all_metrics: list[dict],
    mem_usages: dict,
    output_file: str,
    input_col: str = "source",
    output_col: str = "target",
) -> dict:
    """Aggregate per-run metrics, run stats, and dataset columns; write JSON to output_file.

    Args:
        dataset (Any): Dataset with input_col, output_col, and 'pred' columns or attributes.
        all_metrics (list[dict]): List of metric dicts (same keys each).
        mem_usages (dict): Must have 'total_ram_gb' and 'peak_vram_gb'.
        output_file (str): Path to write JSON.
        input_col (str): Name of input/reference column (e.g. 'source', 'document').
        output_col (str): Name of reference output column (e.g. 'target', 'summary').

    Returns:
        dict: Aggregated result with dataset, raw_metrics, run_stats; or {} if all_metrics empty.
    """
    if not all_metrics:
        return {}
    raw_metrics = {k: [m[k] for m in all_metrics] for k in all_metrics[0]}
    run_stats = {k: compute_stats(v) for k, v in raw_metrics.items()}
    run_stats["sample_size"] = len(all_metrics)
    run_stats["total_ram_gb"] = mem_usages["total_ram_gb"]
    run_stats["peak_vram_gb"] = mem_usages["peak_vram_gb"]

    result = {
        "dataset": {
            input_col: dataset[input_col],
            output_col: dataset[output_col],
            "pred": dataset["pred"],
        },
        "raw_metrics": raw_metrics,
        "run_stats": run_stats,
    }
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    print(f"\nResults saved to '{output_file}'")
    return result
