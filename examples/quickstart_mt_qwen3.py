#!/usr/bin/env python3
"""
Minimal example: load VocabTailor and run one EN->ZH generation with the chat template.

Run (from VocabTailor):
  cd VocabTailor && PYTHONPATH=src python examples/quickstart_mt_qwen3.py [--model MODEL]

Defaults: model Qwen/Qwen3-1.7B; profiling_file = static_vocab/qwen3/mt_en_zh/Qwen3_unicode_set_chinese_tol_0.01.json (from package).
"""
import os
import sys

# Ensure package is importable when run from repo root or from VocabTailor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import torch

import vocab_tailor
from vocab_tailor import VocabTailor


def _default_profiling_file():
    """Default: example EN->ZH static vocab shipped with the package."""
    return os.path.join(
        os.path.dirname(vocab_tailor.__file__),
        "static_vocab",
        "qwen3",
        "mt_en_zh",
        "Qwen3_unicode_set_chinese_tol_0.01.json",
    )


def main():
    parser = argparse.ArgumentParser(description="Quickstart: VocabTailor + Chat template for Machine Translation")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Model name or path (HF id or local directory)",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    default_profiling = _default_profiling_file()
    if not os.path.isfile(default_profiling):
        default_profiling = None

    print(f"Loading VocabTailor: {args.model} (device={args.device})")
    vt = VocabTailor.from_pretrained(
        args.model,
        device=args.device,
        dtype="bf16",
        lmdb_path=None,
        vocab_resize_strategy="prealloc",
        profiling_file=default_profiling,
    )
    tokenizer = vt.tokenizer

    # Prompt
    messages = [
        {"role": "system", "content": "You are a translator"},
        {"role": "user", "content": f"Translate to Chinese: In Germany, where workers and bosses run many companies jointly, a big strike is unusual. A wave of big strikes is almost unheard of. Right now the country of \"co-determination\" is simultaneously facing an eight-day \"action week\" by irate farmers, who blocked roads with tractors, a three-day strike of railway workers and, to top it off, a looming strike of doctors, who already closed surgeries between Christmas and New Year's Day. This Mistgabelmop (pitchfork mob), as some have taken to calling it, will test Germany's harmonious labour relations in the year to come."}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    
    outputs, metrics = vt.generate(
        inputs,
        mode="input_aware",
        max_new_tokens=2048,
        do_sample=False,
        original_eos_token_id=tokenizer.eos_token_id,
    )

    output_text = tokenizer.batch_decode(outputs.cpu())[0]

    print(f"\n{'-'*50}\nDecoded output:")
    print(output_text)
    print(f"{'-'*50}\n")
    
    if metrics:
        print("Metrics (prefill_tps, decode_tps):", metrics.get("prefill_tps"), metrics.get("decode_tps"))

    print("\nDone.")


if __name__ == "__main__":
    main()
