"""
Shared harness for MT benchmark: templates, dataset loading, default paths.
Task-specific only; for shared helpers (e.g. validate_vocabtailor_args, generate_runs_summary)
scripts import from benchmark_utils.

Used by: mt_baseline_eval, mt_baseline_benchmark, mt_baseline_memory_test;
         mt_vocabtailor_eval, mt_vocabtailor_benchmark, mt_vocabtailor_memory_test.
"""
import json
import os
import sys
from typing import Any, Optional

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

_BENCHMARKS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BENCHMARKS not in sys.path:
    sys.path.insert(0, _BENCHMARKS)

LANGUAGE_DICT = {
    "ar": "Arabic",
    "es": "Spanish",
    "en": "English",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "zh": "Chinese",
}


def get_default_dataset_path(repo_root: str) -> str:
    """Default path to MT JSONL under repo datasets (e.g. wmt24pp/en-zh_CN.jsonl)."""
    path = os.path.join(repo_root, "datasets", "wmt24pp", "en-zh_CN.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return path

def get_default_profiling_path(repo_root: str) -> str:
    """Default path to package static_vocab for MT en-zh (Qwen3, tol 0.01)."""
    try:
        import vocab_tailor
        pkg_dir = os.path.dirname(vocab_tailor.__file__)
    except ImportError:
        pkg_dir = os.path.join(repo_root, "src", "vocab_tailor")
    return os.path.join(
        pkg_dir,
        "static_vocab",
        "qwen3",
        "mt_en_zh",
        "Qwen3_unicode_set_chinese_tol_0.01.json",
    )


def apply_mt_template(data: dict, source: str, target: str) -> dict:
    """Build chat messages for MT: system + user with source/target labels.

    Args:
        data (dict): Example with 'source' key.
        source (str): Source language label (e.g. "English").
        target (str): Target language label (e.g. "Chinese").

    Returns:
        dict: Single key 'chat_messages' with list of message dicts.
    """
    messages = [
        {"role": "system", "content": "You are a translator"},
        {"role": "user", "content": f"{source}: {data['source']}\n{target}:"},
    ]
    return {"chat_messages": messages}


def load_mt_dataset(dataset_path: str, sample_size: Optional[int] = None) -> Any:
    """Load JSONL with source/target; filter is_bad_source; optionally take sample_size.

    Args:
        dataset_path (str): Path to JSONL file.
        sample_size (int, optional): If set, take only this many rows. Defaults to None (full dataset).

    Returns:
        Any: Hugging Face Dataset (train split) with filtered rows.
    """
    if load_dataset is None:
        raise ImportError("datasets is required: pip install datasets")
    origin = load_dataset("json", data_files=dataset_path)
    filtered = origin["train"].filter(
        lambda ex: not ex.get("is_bad_source", False)
    )
    if sample_size is not None:
        filtered = filtered.take(sample_size)
    return filtered