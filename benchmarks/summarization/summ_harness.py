"""
Shared harness for summarization benchmark: templates, dataset loading, default paths.
Task-specific only; for shared helpers (e.g. validate_vocabtailor_args, generate_runs_summary)
scripts import from benchmark_utils.

Used by: summ_baseline_eval, summ_baseline_benchmark; summ_vocabtailor_eval, summ_vocabtailor_benchmark.
"""
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


def get_default_dataset_path(repo_root: str) -> str:
    """Default path to summarization data under repo datasets (e.g. xsum/xsum_test.jsonl)."""
    path = os.path.join(repo_root, "datasets", "xsum", "xsum_test.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return path


def get_default_profiling_path(repo_root: str) -> str:
    """Default path to package static_vocab for summarization (Llama, tol 0.01)."""
    try:
        import vocab_tailor
        pkg_dir = os.path.dirname(vocab_tailor.__file__)
    except ImportError:
        pkg_dir = os.path.join(repo_root, "src", "vocab_tailor")
    return os.path.join(
        pkg_dir,
        "static_vocab",
        "llama3.2",
        "summarization",
        "Llama_unicode_set_english_tol_0.01.json",
    )


def apply_summarization_template(data: dict) -> dict:
    """Build chat messages for one document (system + user with document)."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant good at summarization. Please provide one sentence summarization for the given document.",
        },
        {"role": "user", "content": f"Document:\n{data['document']}\n"},
    ]
    return {"chat_messages": messages}


def load_summ_dataset(
    dataset_path: str,
    sample_size: Optional[int] = None,
    split: str = "test",
) -> Any:
    """Load summarization dataset: HF name (e.g. EdinburghNLP/xsum) or local JSON/JSONL path.

    Args:
        dataset_path: HuggingFace dataset name or path to JSON/JSONL with document/summary columns.
        sample_size: If set, take only this many rows.
        split: Split name for HF datasets (e.g. test).

    Returns:
        Hugging Face Dataset with document and summary columns.
    """
    if load_dataset is None:
        raise ImportError("datasets is required: pip install datasets")
    if os.path.isfile(dataset_path):
        origin = load_dataset("json", data_files=dataset_path)
        filtered = origin["train"]
    else:
        filtered = load_dataset(dataset_path, split=split)
    if sample_size is not None:
        filtered = filtered.take(sample_size)
    return filtered


