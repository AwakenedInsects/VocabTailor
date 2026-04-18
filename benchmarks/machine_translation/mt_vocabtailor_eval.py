#!/usr/bin/env python3
"""
MT evaluation (quality): VocabTailor. Uses VocabTailor.from_pretrained; computes sacreBLEU, COMET, METEOR.

Run from VocabTailor:
  cd VocabTailor && PYTHONPATH=src python benchmarks/machine_translation/mt_vocabtailor_eval.py [options]

Example:
  PYTHONPATH=src python benchmarks/machine_translation/mt_vocabtailor_eval.py --dataset datasets/wmt24pp/en-zh_CN.jsonl --profiling_file <path-to-vocab.json> --input_aware
"""
import os
import sys
import json

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import argparse
import logging

import torch
from transformers import logging as tf_logging

try:
    import evaluate
except ImportError:
    evaluate = None

from vocab_tailor import VocabTailor

from mt_harness import (
    LANGUAGE_DICT,
    get_default_dataset_path,
    apply_mt_template,
    load_mt_dataset,
)
from benchmark_utils import (
    validate_vocabtailor_args,
    extract_prediction,
    infer_short_model_name,
    infer_vocabtailor_mode,
    default_output_file_basename,
)


def main():
    tf_logging.set_verbosity_error()
    parser = argparse.ArgumentParser(description="VocabTailor machine translation evaluation: from_pretrained + profiling_file, compute sacreBLEU/COMET/METEOR.")
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
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results. Default: results/machine_translation/{args.source_lang}-{args.target_lang}/eval under repo root.")
    parser.add_argument("--save_predictions", action="store_true", help="Save extracted predictions to .pth")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity.")
    args = parser.parse_args()

    validate_vocabtailor_args(args.dataset, args.profiling_file, args.input_aware)

    if evaluate is None:
        raise ImportError("evaluate is required: pip install evaluate")

    # Initialize output file path
    short_model_name = infer_short_model_name(args.model_name)
    base_name = default_output_file_basename(
        short_model_name=short_model_name,
        dtype=args.dtype,
        device=args.device,
        task_name="mt_eval",
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
    #     if args.input_aware and args.profiling_file is not None:
    #         prefix += "_h" # hybrid
    #     elif args.input_aware:
    #         prefix += "_d" # dynamic-only
    #     else:
    #         prefix += "_s" # static-only
    # else:
    #     prefix = args.prefix
    
    # if args.sample_size is not None:
    #     prefix += f"_sample{args.sample_size}"

    # base_name = f"{short_model_name}_{args.dtype}_{prefix}_mt_eval_{args.source_lang}_{args.target_lang}"
    
    # if args.device != "cuda":
    #     base_name += f"_{args.device}"
    
    # if args.vocab_resize_strategy == "split_linear":
    #     base_name += "_splitlinear"
    # elif args.vocab_resize_strategy == "prealloc":
    #     base_name += "_prealloc"
    
    # if args.lmdb_path:
    #     base_name += "_lmdb"

    output_dir = f"{_REPO_ROOT}/results/machine_translation/{args.source_lang}-{args.target_lang}/eval" if args.output_dir is None else args.output_dir
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

    # Initialize input dataset
    filtered_dataset = load_mt_dataset(args.dataset, sample_size=args.sample_size)
    logger.info(f"Dataset loaded: {len(filtered_dataset)} rows")

    # Initialize VocabTailor
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
        profiling_file=args.profiling_file,
        enable_metrics_tracker=False,
        tokenizer_kwargs=tokenizer_kwargs,
        **model_kwargs,
    )
    tokenizer = vt.tokenizer
    original_eos_token_id = tokenizer.eos_token_id
    logger.info("Finish loading VocabTailor")

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
    # Evaluation Test
    #----------------------    
    N = len(filtered_dataset)
    logger.info(f"Run evaluation with {N} examples...")

    mode = "input_aware" if args.input_aware else "default"
    extracted_pred = []
    
    for i, message in enumerate(filtered_dataset["chat_messages"]):
        if args.verbose:
            print(f"\n{'=' * 50}\nProcessing Example {i + 1}/{N}\n{'='*50}\n")

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
        output_ids = vt.generate(
            inputs,
            mode=mode,
            max_new_tokens=2048,
            do_sample=False,
            original_eos_token_id=original_eos_token_id,
        )

        # Decode & Prediction
        output_text = tokenizer.batch_decode(output_ids.to("cpu"))[0]
        extracted_text = extract_prediction(output_text, short_model_name)

        # Append results
        extracted_pred.append(extracted_text)

        # Print results
        if args.verbose:
            print(f"Translation Result\n{'-'*40}")
            print(f"{source_lang}: {filtered_dataset['source'][i]}\n")
            print(f"{target_lang}: {extracted_text}\n")

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

    if args.save_predictions:
        torch.save(extracted_pred, pred_file)
        logger.info(f"Predictions saved to {pred_file}")
    
    logger.info("Done.")


if __name__ == "__main__":
    main()
