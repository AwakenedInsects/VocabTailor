#!/usr/bin/env python3
"""
Validate the full VocabTailor stack: load model with from_pretrained(..., profiling_file=...),
run MT evaluation, compare metrics to golden.

Omit --profiling_file for no static vocab (use with --input_aware for dynamic-only).
Run from VocabTailor repo root with PYTHONPATH=src:
  PYTHONPATH=src python tests/validate_model_package.py [options]

Golden metrics: tests/golden/qwen3-1.7b_fp32_vt_zh_tol0.01_mt_eval_en_zh.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Any, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch
from transformers import logging as tf_logging

try:
    import evaluate
except ImportError:
    evaluate = None
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from vocab_tailor import VocabTailor

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

def apply_mt_template(data: dict, source: str, target: str) -> dict:
    """Build chat messages for MT: system + user with source/target labels."""
    messages = [
        {"role": "system", "content": "You are a translator"},
        {"role": "user", "content": f"{source}: {data['source']}\n{target}:"},
    ]
    return {"chat_messages": messages}


def extract_prediction(text: str, short_model_name: str) -> str:
    """Extract assistant reply from model output using model-specific delimiters."""
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


def load_mt_dataset(dataset_path: str, sample_size: Optional[int] = None) -> Any:
    """Load JSONL with source/target; filter is_bad_source; optionally take sample_size."""
    if load_dataset is None:
        raise ImportError("datasets is required: pip install datasets")
    origin = load_dataset("json", data_files=dataset_path)
    filtered = origin["train"].filter(
        lambda ex: not ex.get("is_bad_source", False)
    )
    if sample_size is not None:
        filtered = filtered.take(sample_size)
    return filtered


def infer_short_model_name(model_name: str) -> str:
    """Map HF model id or path to short name (Qwen3, Qwen, Llama, etc.)."""
    if "Qwen3" in model_name or "qwen3" in model_name:
        return "Qwen3"
    if "Qwen" in model_name or "qwen" in model_name:
        return "Qwen"
    if "Llama" in model_name or "llama" in model_name:
        return "Llama"
    return "unknown"


def _default_dataset_path() -> str:
    return os.path.join(_REPO_ROOT, "datasets", "wmt24pp", "en-zh_CN.jsonl")


def _default_profiling_path() -> str:
    """Package static_vocab for MT en-zh (Qwen3, tol 0.01)."""
    try:
        import vocab_tailor
        pkg_dir = os.path.dirname(vocab_tailor.__file__)
    except ImportError:
        pkg_dir = os.path.join(_REPO_ROOT, "src", "vocab_tailor")
    return os.path.join(
        pkg_dir,
        "static_vocab",
        "qwen3",
        "mt_en_zh",
        "Qwen3_unicode_set_chinese_tol_0.01.json",
    )


def _golden_metrics_path() -> str:
    return os.path.join(_SCRIPT_DIR, "golden", "qwen3-1.7b_fp32_vt_zh_tol0.01_mt_eval_en_zh.json")


def main() -> int:
    tf_logging.set_verbosity_error()
    parser = argparse.ArgumentParser(description="Validate full VocabTailor stack: from_pretrained + profiling_file, MT eval, compare to golden.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B", help="Model path or Hugging Face model id.")
    parser.add_argument("--lmdb_path", type=str, default=None, help = "Path to LMDB store for offloading weights (optional).")
    parser.add_argument("--profiling_file", type=str, default=None, help="Path to static vocab JSON (optional). If omitted, no static vocab; use with --input_aware for dynamic-only.")
    parser.add_argument("--dataset", type=str, default=_default_dataset_path(), help="Path to JSONL with 'source' and 'target' columns. Default: datasets/wmt24pp/en-zh_CN.jsonl under repo root.")
    parser.add_argument("--prefix", type = str, default = "vt_h_zh_tol0.01", help = "Prefix for output file.")
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
    parser.add_argument("--compare", action="store_true", help="Compare generated metrics with golden metrics.")
    args = parser.parse_args()

    dataset_path = args.dataset
    if dataset_path and not os.path.isfile(dataset_path):
        print(f"Dataset file not found: {dataset_path}", file=sys.stderr)
        return 1

    profiling_path = args.profiling_file
    if profiling_path and not os.path.isfile(profiling_path):
        print(f"Profiling file not found: {profiling_path}", file=sys.stderr)
        return 1

    if profiling_path is None and not args.input_aware:
        print("VocabTailor requires at least one of --input_aware or --profiling_file not None.", file=sys.stderr)
        return 1

    golden_path = _golden_metrics_path()
    if args.compare and not os.path.isfile(golden_path):
        print(f"Golden metrics file not found: {golden_path}", file=sys.stderr)
        return 1

    if evaluate is None:
        print("evaluate is required: pip install evaluate", file=sys.stderr)
        return 1

    # Initialize output file path
    short_name = infer_short_model_name(args.model_name)

    if args.prefix is None:
        prefix = "vt"
        if args.input_aware and profiling_path is not None:
            prefix += "_h" # hybrid
        elif args.input_aware:
            prefix += "_d" # dynamic-only
        else:
            prefix += "_s" # static-only

        if args.sample_size is not None:
            prefix += f"_sample{args.sample_size}"
    else:
        prefix = args.prefix

    base_name = f"{short_name}_{args.dtype}_{prefix}_mt_eval_{args.source_lang}_{args.target_lang}"
    
    if args.device != "cuda":
        base_name += f"_{args.device}"
    
    if args.vocab_resize_strategy == "split_linear":
        base_name += "_splitlinear"
    elif args.vocab_resize_strategy == "prealloc":
        base_name += "_prealloc"
    
    if args.lmdb_path:
        base_name += "_lmdb"

    output_dir = f"{_REPO_ROOT}/results/machine_translation/{args.source_lang}-{args.target_lang}"
    os.makedirs(output_dir, exist_ok = True)
    os.makedirs(f"{output_dir}/logs", exist_ok = True)
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

    filtered_dataset = load_mt_dataset(dataset_path, sample_size=args.sample_size)
    logger.info("Dataset loaded: %d rows", len(filtered_dataset))

    logger.info("Loading VocabTailor...")
    model_kwargs = {}
    tokenizer_kwargs = {}
    if args.local_files_only:
        model_kwargs["local_files_only"] = True
        tokenizer_kwargs["local_files_only"] = True

    vt = VocabTailor.from_pretrained(
        args.model_name,
        device=args.device,
        dtype=args.dtype,
        lmdb_path=args.lmdb_path,
        vocab_resize_strategy=args.vocab_resize_strategy,
        tokenizer_kwargs=tokenizer_kwargs,
        profiling_file=profiling_path,
        **model_kwargs,
    )
    vt.tracker = None
    tokenizer = vt.tokenizer
    original_eos_token_id = tokenizer.eos_token_id
    logger.info("Finish loading VocabTailor")

    logger.info("VocabTailor model info:")
    logger.info(f"{'name':<20} {'dtype':<20} {'device':<10}")
    logger.info("-"*60)
    if vt.offload_to_lmdb:
        logger.info(f"{'embedding':<20} {str(vt.provider.fetch_dtype):<20} {str(vt.model.model.embed_tokens.device):<10}")
    else:
        logger.info(f"{'embedding':<20} {str(vt.model.model.embed_tokens.full_embedding.weight.dtype):<20} {str(vt.model.model.embed_tokens.device):<10}")
    logger.info(f"{'lm head':<20} {str(vt.model.lm_head.weight_dtype):<20} {str(vt.model.lm_head.device):<10}")
    logger.info(f"{'transformer body':<20} {str(vt.model.dtype):<20} {str(args.device):<10}")

    source_label = LANGUAGE_DICT[args.source_lang]
    target_label = LANGUAGE_DICT[args.target_lang]
    filtered_dataset = filtered_dataset.map(
        apply_mt_template,
        fn_kwargs={"source": source_label, "target": target_label},
        batched=False,
    )

    #----------------------
    # Evaluation Test
    #----------------------    
    N = len(filtered_dataset)
    logger.info(f"Run evaluation with {N} examples...")

    mode = "input_aware" if args.input_aware else "default"
    extracted_pred = []
    
    for i, message in enumerate(filtered_dataset["chat_messages"]):
        print(f"\n{'=' * 50}\nProcessing Example {i + 1}/{N}\n{'='*50}\n")

        # Input
        inputs = tokenizer.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )

        # Generate
        outputs, _ = vt.generate(
            inputs,
            mode=mode,
            max_new_tokens=2048,
            do_sample=False,
            original_eos_token_id=original_eos_token_id,
        )

        # Decode & Prediction
        output_text = tokenizer.batch_decode(outputs.cpu())[0]
        extracted_text = extract_prediction(output_text, short_name)

        # Append results
        extracted_pred.append(extracted_text)

        # Print results
        print(f"Translation Result\n{'-'*40}")
        print(f"{source_label}: {filtered_dataset['source'][i]}\n")
        print(f"{target_label}: {extracted_text}\n")

        if i % 100 == 0:
            logger.info(f"Complete examples {i + 1}/{N}")
    logger.info(f"Complete examples {N}/{N}")

    #----------------------
    # Evaluation Metrics
    #----------------------
    sacrebleu = evaluate.load("sacrebleu")
    meteor = evaluate.load("meteor")
    comet = evaluate.load("comet")

    refs = list(filtered_dataset["target"])
    sacrebleu_score = sacrebleu.compute(predictions=extracted_pred, references=refs)["score"]
    comet_score = comet.compute(
        predictions=extracted_pred,
        references=refs,
        sources=list(filtered_dataset["source"]),
    )["mean_score"]
    meteor_score = meteor.compute(predictions=extracted_pred, references=refs)["meteor"]

    metrics = {
        "sacreBLEU": sacrebleu_score,
        "COMET": comet_score,
        "METEOR": meteor_score,
    }

    logger.info(f"sacreBLEU: {sacrebleu_score}")
    logger.info(f"COMET: {comet_score}")
    logger.info(f"METEOR: {meteor_score}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {output_file}")

    torch.save(extracted_pred, pred_file)
    logger.info(f"Predictions saved to {pred_file}")

    #----------------------
    # Compare with Golden Metrics
    #----------------------
    if args.compare:
        with open(golden_path, encoding="utf-8") as f:
            golden = json.load(f)

        tol = 0.05
        ok = True
        for key in ("sacreBLEU", "COMET", "METEOR"):
            g = golden.get(key)
            m = metrics.get(key)
            if g is None:
                logger.warning(f"Golden missing key {key}")
                ok = False
                continue
            if float(m) < float(g) and abs(float(m) - float(g)) > tol:
                logger.error("%s: computed %.6f vs golden %.6f (tolerance %.0e)", key, m, g, tol)
                ok = False
            else:
                logger.info("%s within tolerance: %.6f vs %.6f", key, m, g)

        if ok:
            logger.info("Validation passed: metrics outperform or match golden within tolerance.")
            return 0
        logger.error("Validation failed: metrics differ from golden.")
        return 1
    else:
        logger.info("Validation skipped: --compare is not set.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
