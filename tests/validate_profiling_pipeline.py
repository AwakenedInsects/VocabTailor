#!/usr/bin/env python3
"""
Validate the profiling pipeline by regenerating static vocabularies and comparing
to the reference files shipped in vocab_tailor.static_vocab.

This script is for local/CI validation only. The released package does not
depend on any of the dataset or model paths used here; users run
vocab-tailor-build-vocab with their own --model and --dataset.

Usage (from repository root):
  PYTHONPATH=src python tests/validate_profiling_pipeline.py [options]

  --run 1          Run only Run 1 (MT en→zh).
  --run 1 3        Run only Run 1 and Run 3.
  (omit --run)     Run all three (default).

Default dataset paths (relative to repo root; not overridable via CLI):
  Run 1 (MT en→zh):      datasets/opus-100/en-zh/train-00000-of-00001.parquet
  Run 2 (MT en→it):      datasets/opus-100/en-it/train-00000-of-00001.parquet
  Run 3 (summarization): datasets/xsum/xsum_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Optional


def _static_vocab_dir():
    """Return path to vocab_tailor/static_vocab (package-relative)."""
    try:
        import vocab_tailor
        return os.path.join(os.path.dirname(vocab_tailor.__file__), "static_vocab")
    except ImportError:
        # Run from repo without install: assume VocabTailor is cwd and src is on PYTHONPATH
        return os.path.join(os.path.dirname(__file__), "..", "src", "vocab_tailor", "static_vocab")


def _ids_from_json(path: str) -> set:
    """Load a profiling JSON (dict token->id or list of ids) and return the set of token IDs."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return set(data.values())
    if isinstance(data, list):
        return set(int(x) for x in data)
    raise ValueError(f"JSON at {path!r} must be a dict or list of ids, got {type(data)}")


def vocab_ids_match(generated_path: str, reference_path: str) -> tuple[bool, str]:
    """
    Compare two profiling JSON files by token ID set.
    Returns (True, "") if sets are equal, else (False, error_message).
    """
    if not os.path.isfile(generated_path):
        return False, f"Generated file missing: {generated_path}"
    if not os.path.isfile(reference_path):
        return False, f"Reference file missing: {reference_path}"
    try:
        gen_ids = _ids_from_json(generated_path)
        ref_ids = _ids_from_json(reference_path)
    except (json.JSONDecodeError, ValueError) as e:
        return False, str(e)
    if gen_ids != ref_ids:
        return False, (
            f"ID set mismatch: generated {len(gen_ids)} ids, reference {len(ref_ids)} ids; "
            f"symdiff size {len(gen_ids ^ ref_ids)}"
        )
    return True, ""


def run_build_vocab(args: list[str], cwd: Optional[str] = None) -> tuple[int, str]:
    """Run vocab-tailor-build-vocab with given args; stream output to terminal and return (returncode, full_output)."""
    cmd = [sys.executable, "-m", "vocab_tailor.profiling.cli"] + args
    cwd_resolved = cwd or os.getcwd()
    env = os.environ.copy()
    src = os.path.join(cwd_resolved, "src")
    if "PYTHONPATH" not in env and os.path.isdir(src):
        env["PYTHONPATH"] = src
    proc = subprocess.Popen(
        cmd,
        cwd=cwd_resolved,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    out_lines: list[str] = []
    if proc.stdout:
        for line in proc.stdout:
            print(line, end="")
            out_lines.append(line)
    return proc.wait(), "".join(out_lines)


# Default dataset paths (relative to repo root). Run 1/2 use opus-100; Run 3 uses xsum.
DEFAULT_DATASET_RUN1 = "datasets/opus-100/en-zh/train-00000-of-00001.parquet"
DEFAULT_DATASET_RUN2 = "datasets/opus-100/en-it/train-00000-of-00001.parquet"
DEFAULT_DATASET_RUN3 = "datasets/xsum/xsum_train.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate profiling pipeline: run build-vocab for up to 3 reference configs and compare to static_vocab."
    )
    parser.add_argument(
        "--run",
        type=int,
        nargs="*",
        default=None,
        metavar="N",
        help="Run only these (e.g. --run 1 or --run 1 3). Default: run all three.",
    )
    parser.add_argument("--output-dir", type=str, default="./results/preprocessed", help="Output dir for build-vocab")
    args = parser.parse_args()

    runs_to_run = args.run if args.run is not None else [1, 2, 3]
    invalid = [n for n in runs_to_run if n not in (1, 2, 3)]
    if invalid:
        print(f"Invalid --run values: {invalid}. Use 1, 2, and/or 3.", file=sys.stderr)
        return 2

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    static = _static_vocab_dir()
    if not os.path.isdir(static):
        print(f"static_vocab not found at {static}", file=sys.stderr)
        return 2

    failed = []
    output_dir = args.output_dir
    # Ensure output_dir is absolute when we pass to CLI and when we compare paths
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(repo_root, output_dir)

    # Run 1: MT en→zh
    if 1 in runs_to_run:
        print(f"Run 1: MT en→zh\nCreate static vocabulary for Chinese...")
        dataset_run1 = os.path.join(repo_root, DEFAULT_DATASET_RUN1) if not os.path.isabs(DEFAULT_DATASET_RUN1) else DEFAULT_DATASET_RUN1
        rc, out = run_build_vocab([
            "--model", "Qwen/Qwen3-1.7B",
            "--task", "machine_translation", "--source_lang", "en", "--target_lang", "zh",
            "--dataset", dataset_run1, "--dataset_name", "opus-100",
            "-n", "chinese", "--tolerance", "0.01", "--output_dir", output_dir,
        ], cwd=repo_root)
        if rc != 0:
            failed.append(("Run 1", f"build-vocab exited {rc}: {out[:500]}"))
        else:
            gen = os.path.join(output_dir, "machine_translation", "Qwen3_unicode_set_chinese_tol_0.01.json")
            ref = os.path.join(static, "qwen3", "mt_en_zh", "Qwen3_unicode_set_chinese_tol_0.01.json")
            ok, msg = vocab_ids_match(gen, ref)
            if not ok:
                failed.append(("Run 1", msg))
            else:
                print("  Run 1: OK")

    # Run 2: MT en→it
    if 2 in runs_to_run:
        print(f"Run 2: MT en→it\nCreate static vocabulary for Italian...")
        dataset_run2 = os.path.join(repo_root, DEFAULT_DATASET_RUN2) if not os.path.isabs(DEFAULT_DATASET_RUN2) else DEFAULT_DATASET_RUN2
        rc, out = run_build_vocab([
            "--model", "Qwen/Qwen3-1.7B",
            "--task", "machine_translation", "--source_lang", "en", "--target_lang", "it",
            "--dataset", dataset_run2, "--dataset_name", "opus-100",
            "-n", "italian", "--tolerance", "0.01", "--output_dir", output_dir,
        ], cwd=repo_root)
        if rc != 0:
            failed.append(("Run 2", f"build-vocab exited {rc}: {out[:500]}"))
        else:
            gen = os.path.join(output_dir, "machine_translation", "Qwen3_unicode_set_italian_tol_0.01.json")
            ref = os.path.join(static, "qwen3", "mt_en_it", "Qwen3_unicode_set_italian_tol_0.01.json")
            ok, msg = vocab_ids_match(gen, ref)
            if not ok:
                failed.append(("Run 2", msg))
            else:
                print("  Run 2: OK")

    # Run 3: summarization
    if 3 in runs_to_run:
        print(f"Run 3: summarization\nCreate static vocabulary for English...")
        dataset_run3 = os.path.join(repo_root, DEFAULT_DATASET_RUN3) if not os.path.isabs(DEFAULT_DATASET_RUN3) else DEFAULT_DATASET_RUN3
        rc, out = run_build_vocab([
            "--model", "AwakenedInsects/llama-3.2-3b-vocabtailor-sum",
            "--task", "summarization",
            "--dataset", dataset_run3,
            "-n", "english", "--tolerance", "0.01", "--output_dir", output_dir,
        ], cwd=repo_root)
        if rc != 0:
            failed.append(("Run 3", f"build-vocab exited {rc}: {out[:500]}"))
        else:
            gen = os.path.join(output_dir, "summarization", "Llama_unicode_set_english_tol_0.01.json")
            ref = os.path.join(static, "llama3.2", "summarization", "Llama_unicode_set_english_tol_0.01.json")
            ok, msg = vocab_ids_match(gen, ref)
            if not ok:
                failed.append(("Run 3", msg))
            else:
                print("  Run 3: OK")

    if failed:
        for name, err in failed:
            print(f"FAILED {name}: {err}", file=sys.stderr)
        return 1
    print("All runs passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
