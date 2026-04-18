#!/usr/bin/env python3
"""
Benchmark LM head swap time between any two static vocabularies (no model reload).

Loads VocabTailor with profiling_file1, then swaps to profiling_file2 via update_lm_head
and reports the switching time. The two vocabs need not be from the same task.

Run (from repository root, with VocabTailor as cwd or PYTHONPATH=src):

  cd VocabTailor && PYTHONPATH=src python benchmarks/benchmark_lmhead_swap.py
  PYTHONPATH=src python benchmarks/benchmark_lmhead_swap.py --profiling_file1 PATH --profiling_file2 PATH
"""
import argparse
import json
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch

from vocab_tailor import VocabTailor

DEFAULT_PROFILING_FILE1 = "qwen3/mt_en_zh/Qwen3_unicode_set_chinese_tol_0.01.json"
DEFAULT_PROFILING_FILE2 = "qwen3/mt_en_it/Qwen3_unicode_set_italian_tol_0.01.json"


def _get_default_profiling_path(subpath: str) -> str:
    """Default path under repo src/vocab_tailor/static_vocab/."""
    return os.path.join(_REPO_ROOT, "src", "vocab_tailor", "static_vocab", *subpath.split("/"))


def _resolve_profiling_file(path_or_none, default_subpath: str) -> str:
    """Return absolute path. If path_or_none is set, use it; else use repo default."""
    if path_or_none is not None:
        return os.path.abspath(path_or_none)
    return _get_default_profiling_path(default_subpath)


def _load_vocab_ids(profiling_path, tokenizer):
    """Load token IDs (and merge with special IDs) from a profiling JSON file."""
    with open(profiling_path, encoding="utf-8") as f:
        preload_vocab = json.load(f)
    preload_ids = list(preload_vocab.values())
    special_ids = tokenizer.all_special_ids
    return torch.tensor(list(set(preload_ids + special_ids)))


def main():
    parser = argparse.ArgumentParser(description="Benchmark LM head swap time between any two static vocabularies.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B", help="Model (HF id or local path)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--profiling_file1",
        type=str,
        default=None,
        help=f"Initial static vocab (JSON). Default: repo src/vocab_tailor/static_vocab/{DEFAULT_PROFILING_FILE1}",
    )
    parser.add_argument(
        "--profiling_file2",
        type=str,
        default=None,
        help=f"Second static vocab to swap to (JSON). Default: repo src/vocab_tailor/static_vocab/{DEFAULT_PROFILING_FILE2}",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup swap runs before timing")
    parser.add_argument("--repeat", type=int, default=5, help="Number of timed swap runs for mean ± std")
    args = parser.parse_args()

    path1 = _resolve_profiling_file(args.profiling_file1, DEFAULT_PROFILING_FILE1)
    path2 = _resolve_profiling_file(args.profiling_file2, DEFAULT_PROFILING_FILE2)

    for label, p in [("profiling_file1", path1), ("profiling_file2", path2)]:
        if not os.path.isfile(p):
            print(f"Error: {label} not found: {p}")
            sys.exit(1)

    print(f"Loading model: {args.model} with initial vocab: {path1}")
    vt = VocabTailor.from_pretrained(
        args.model,
        device=args.device,
        dtype="bf16" if args.device == "cuda" else "fp32",
        vocab_resize_strategy="prealloc",
        profiling_file=path1,
        enable_metrics_tracker=False,
    )
    tokenizer = vt.tokenizer
    init_ids = _load_vocab_ids(path2, tokenizer)

    for _ in range(args.warmup):
        vt.reset()
        vt.update_lm_head(init_ids, temp=False)

    times = []
    for _ in range(args.repeat):
        st = time.perf_counter()
        vt.reset()
        vt.update_lm_head(init_ids, temp=False)
        et = time.perf_counter()
        times.append(et - st)

    mean_t = sum(times) / len(times)
    variance = sum((t - mean_t) ** 2 for t in times) / len(times)
    std_t = variance ** 0.5
    print(f"vocab1: {path1}")
    print(f"vocab2: {path2}")
    print(f"Swap time (vocab1 -> vocab2): {mean_t:.4f} ± {std_t:.4f} s (n={args.repeat})")
    print("Done.")


if __name__ == "__main__":
    main()
