#!/usr/bin/env python3
"""
CLI to build static task vocabulary (profiling JSON) for VocabTailor.
Requires: pip install vocab-tailor[profiling] or pip install datasets.

Optional task presets: use --task to set default input_col/output_col (and optionally
dataset path via VOCABTAILOR_DATA_ROOT). Explicit --dataset/--input_col/--output_col
override preset defaults.
"""
import argparse
import os
import sys
from typing import Any, Optional, Union

# Optional task presets: default columns and optional dataset path template (relative to
# VOCABTAILOR_DATA_ROOT). Template may use {source_lang}, {target_lang} for MT.
TASK_PRESETS = {
    "machine_translation": {
        "input_col": "source",
        "output_col": "target",
        "task_name": "machine_translation",
        "dataset_template": None,
    },
    "summarization": {
        "input_col": "document",
        "output_col": "summary",
        "task_name": "summarization",
        "dataset_template": None,
    },
    "information_extraction": {
        "input_col": "context",
        "output_col": "value",
        "task_name": "information_extraction",
        "dataset_template": None,
    },
    "math": {
        "input_col": "question",
        "output_col": "answer",
        "task_name": "math",
        "dataset_template": None,
    },
    "code_completion": {
        "input_col": "prompt",
        "output_col": "completion",
        "task_name": "code_completion",
        "dataset_template": None,
    },
}


def _infer_model_type(model_path: str) -> str:
    """Infer model type string (e.g. Llama, Qwen3) from model path or Hub id."""
    m = model_path.lower()
    if "llama" in m:
        return "Llama"
    if "qwen3" in m or "qwen" in m:
        return "Qwen3"
    if "deepseek" in m:
        return "DeepSeek"
    if "rho" in m:
        return "Rho"
    return "unknown"


def _load_dataset(
    dataset: str,
    split: Optional[str],
    data_files: Optional[Union[dict, list, str]],
    cache_dir: Optional[str],
    input_col: str,
    output_col: str,
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
) -> Any:
    """Load Hugging Face dataset from name, path, or data_files; map translation columns if needed."""
    from datasets import load_dataset as hf_load
    def _format_for_path(path: str) -> str:
        """Return 'parquet' or 'json' from file path extension."""
        path_lower = path.lower()
        if path_lower.endswith(".parquet"):
            return "parquet"
        return "json"
    if data_files is not None:
        if isinstance(data_files, str):
            data_files = {"train": data_files}
            fmt = _format_for_path(data_files["train"])
        elif isinstance(data_files, list):
            data_files = {"train": data_files}
            first = data_files["train"][0] if data_files["train"] else ""
            fmt = _format_for_path(first) if isinstance(first, str) else "json"
        else:
            fmt = "json"
        ds = hf_load(fmt, data_files=data_files, split=split or "train", cache_dir=cache_dir)
    else:
        if dataset and os.path.isfile(dataset):
            fmt = _format_for_path(dataset)
            ds = hf_load(fmt, data_files=dataset, split=split or "train", cache_dir=cache_dir)
        else:
            ds = hf_load(dataset, split=split or "train", cache_dir=cache_dir)
    # If dataset has only "translation" (e.g. opus-100 parquet), map to source/target when lang codes given
    if (
        "translation" in ds.column_names
        and (input_col not in ds.column_names or output_col not in ds.column_names)
        and source_lang
        and target_lang
    ):
        def _add_source_target(example):
            tr = example["translation"]
            example["source"] = tr.get(source_lang, "")
            example["target"] = tr.get(target_lang, "")
            return example
        ds = ds.map(_add_source_target)
    if input_col not in ds.column_names or output_col not in ds.column_names:
        raise ValueError(
            f"Dataset must have columns '{input_col}' and '{output_col}'. "
            f"Available: {ds.column_names}"
        )
    return ds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build static task vocabulary (JSON) for VocabTailor via three-stage filtering."
    )
    parser.add_argument("--model", type=str, required=True, help="Tokenizer path or HuggingFace model id")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name or path (JSON/JSONL); optional if --task preset has a dataset_template and VOCABTAILOR_DATA_ROOT is set")
    parser.add_argument("--task", type=str, default=None, choices=list(TASK_PRESETS), help="Optional task preset: sets default input_col, output_col, task_name; may resolve --dataset from VOCABTAILOR_DATA_ROOT when preset has dataset_template")
    parser.add_argument("--source_lang", type=str, default=None, help="Source language code (e.g. en); used with --task machine_translation for preset dataset path")
    parser.add_argument("--target_lang", type=str, default=None, help="Target language code (e.g. zh); used with --task machine_translation for preset dataset path")
    parser.add_argument("--input_col", type=str, default=None, help="Input column name (overrides preset when --task is set)")
    parser.add_argument("--output_col", type=str, default=None, help="Output column name (overrides preset when --task is set)")
    parser.add_argument("--task_name", type=str, default=None, help="Task name for output dir (overrides preset when --task is set)")
    parser.add_argument("--tolerance", nargs="+", type=float, default=[0.1], help="Tolerance value(s), e.g. 0.1 or 0.06 0.1")
    parser.add_argument("-n", "--unicode_filter_categories", nargs="+", default=["english", "chinese"], help="Unicode categories, e.g. english chinese math")
    parser.add_argument("--output_dir", type=str, default="./preprocessed", help="Output directory for JSON files")
    parser.add_argument("--ablation", type=str, default=None, choices=["wo_input_aware", "wo_unicode"], help="Ablation: skip input-aware or unicode stage")
    parser.add_argument("--data_files", type=str, nargs="+", default=None, help="Explicit data file(s) for load_dataset")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache dir for HuggingFace datasets")
    parser.add_argument("--split", type=str, default=None, help="Dataset split (default: train)")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset identifier for branch logic (e.g. opus-100, kde4)")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code for tokenizer")
    args = parser.parse_args()

    # Apply optional task preset: defaults for input_col, output_col, task_name; optionally resolve dataset path
    if args.task is not None:
        preset = TASK_PRESETS[args.task]
        if args.input_col is None:
            args.input_col = preset["input_col"]
        if args.output_col is None:
            args.output_col = preset["output_col"]
        if args.task_name is None:
            args.task_name = preset["task_name"]
        if args.dataset is None and preset.get("dataset_template"):
            data_root = os.environ.get("VOCABTAILOR_DATA_ROOT", "")
            if not data_root:
                data_root = os.getcwd()
            template = preset["dataset_template"]
            if "{source_lang}" in template or "{target_lang}" in template:
                src = args.source_lang or "en"
                tgt = args.target_lang or "zh"
                template = template.format(source_lang=src, target_lang=tgt)
            args.dataset = os.path.join(data_root, template) if data_root else template

    if args.dataset is None:
        parser.error("--dataset is required (or use --task with a preset that has dataset_template and set VOCABTAILOR_DATA_ROOT)")
    if args.input_col is None:
        args.input_col = "source"
    if args.output_col is None:
        args.output_col = "target"

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers is required: pip install transformers", file=sys.stderr)
        sys.exit(1)
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets is required for the build-vocab CLI: pip install datasets (or pip install vocab-tailor[profiling])", file=sys.stderr)
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model_type = _infer_model_type(args.model)

    data_files = None
    if args.data_files:
        data_files = args.data_files if len(args.data_files) > 1 else args.data_files[0]
    elif args.dataset and os.path.isfile(args.dataset):
        data_files = args.dataset

    dataset = _load_dataset(
        args.dataset,
        args.split,
        data_files,
        args.cache_dir,
        args.input_col,
        args.output_col,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
    )

    from .filter import build_static_vocab

    tol = args.tolerance if len(args.tolerance) > 1 else args.tolerance[0]
    build_static_vocab(
        tokenizer=tokenizer,
        dataset=dataset,
        input_colname=args.input_col,
        output_colname=args.output_col,
        unicode_filter_categories=args.unicode_filter_categories,
        task_name=args.task_name,
        model_type=model_type,
        output_dir=args.output_dir,
        tolerance=tol,
        dataset_name=args.dataset_name,
        ablation=args.ablation,
        verbose=True,
    )
    out_dir = args.output_dir if not args.task_name else os.path.join(args.output_dir, args.task_name)
    print(f"Static vocabulary written under {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
