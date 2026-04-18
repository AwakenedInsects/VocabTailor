#!/usr/bin/env python3
"""
Math evaluation (baseline) via LightEval. Uses BaselineGenerator.from_pretrained and LightEval
Pipeline with task lighteval|math_500|0|0. No dataset path; task defines the data.

Run from VocabTailor:
  cd VocabTailor && PYTHONPATH=src python benchmarks/math/math_baseline_eval.py [options]

Example:
  PYTHONPATH=src python benchmarks/math/math_baseline_eval.py --model_name microsoft/rho-math-1b-interpreter-v0.1
"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_SRC = os.path.join(_REPO_ROOT, "src")
_BENCHMARKS = os.path.dirname(_SCRIPT_DIR)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _BENCHMARKS not in sys.path:
    sys.path.insert(0, _BENCHMARKS)

import argparse
import logging

import torch
from transformers import logging as tf_logging
from accelerate import Accelerator

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.model_input import GenerationParameters
from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager

from vocab_tailor import BaselineGenerator
from benchmark_utils import infer_short_model_name


def main():
    tf_logging.set_verbosity_error()
    parser = argparse.ArgumentParser(description="Baseline math evaluation via LightEval")
    parser.add_argument("--model_name", type=str, default="microsoft/rho-math-1b-interpreter-v0.1", help="Model path or Hugging Face model id.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on.")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="fp32", help="Model dtype.")
    parser.add_argument("--local_files_only", action="store_true", help="Use only local files for Hugging Face model/tokenizer.")
    parser.add_argument("--save_predictions", action="store_true", help="Save extracted predictions to .pth")
    args = parser.parse_args()

    # Initialize output file path
    short_name = infer_short_model_name(args.model_name)

    base_name = f"{short_name}_{args.dtype}_math_eval"

    if args.device != "cuda":
        base_name += f"_{args.device}"

    output_dir = f"{_REPO_ROOT}/results/math/eval"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    log_file = f"{output_dir}/logs/{base_name}.log"
    output_file = f"{output_dir}/{base_name}.json"
    pred_file = f"{output_dir}/{base_name}_pred.pth"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    logger = logging.getLogger(__name__)

    # Initialize model and tokenizer
    logger.info("Loading model and tokenizer via BaselineGenerator...")
    model_kwargs = {}
    tokenizer_kwargs = {}
    if args.local_files_only:
        model_kwargs["local_files_only"] = True
        tokenizer_kwargs["local_files_only"] = True

    bg = BaselineGenerator.from_pretrained(
        args.model_name,
        device=args.device,
        dtype=args.dtype,
        tokenizer_kwargs=tokenizer_kwargs,
        **model_kwargs,
    )
    bg.tracker = None  # close the tracker
    logger.info("Finish loading BaselineGenerator")

    # Build LightEval config and model (baseline = standard TransformersModel)
    config = TransformersModelConfig(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        use_chat_template=True,
        generation_parameters=GenerationParameters(max_new_tokens=2048),
    )
    accelerator = Accelerator()
    le_model = TransformersModel.from_model(
        bg.model,
        config=config,
        accelerator=accelerator,
    )

    # Set up evaluation tracking
    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        save_details=True
    )

    # Configure the pipeline
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.CUSTOM,
    )

    # Create and run the pipeline
    pipeline = Pipeline(
        tasks="lighteval|math_500|0|0",
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model=le_model,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()

    logger.info("Done.")


if __name__ == "__main__":
    main()
