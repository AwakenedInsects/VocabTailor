"""Smoke tests for the vocab_tailor package."""
import sys
import os
import torch

# Allow importing vocab_tailor when run from repo root or VocabTailor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_import_api():
    """Package and public API are importable."""
    from vocab_tailor import BaselineGenerator, VocabTailor, __version__

    assert VocabTailor is not None
    assert BaselineGenerator is not None
    assert __version__ is not None
    assert isinstance(__version__, str)
    assert hasattr(VocabTailor, "from_pretrained")
    assert hasattr(VocabTailor, "generate")
    assert hasattr(BaselineGenerator, "load_model")
    assert hasattr(BaselineGenerator, "generate")


def test_version_format():
    """Version is semver-like."""
    from vocab_tailor import __version__

    parts = __version__.split(".")
    assert len(parts) >= 2
    for p in parts[:2]:
        assert p.isdigit(), f"Expected numeric version part, got {p}"


def _from_pretrained_common():
    """Shared model_id and profiling_file for from_pretrained tests."""
    import vocab_tailor
    model_id = "Qwen/Qwen3-1.7B"
    profiling_file = os.path.join(
        os.path.dirname(vocab_tailor.__file__),
        "static_vocab",
        "qwen3",
        "mt_en_zh",
        "Qwen3_unicode_set_chinese_tol_0.01.json",
    )
    return model_id, profiling_file


def test_from_pretrained():
    """VocabTailor.from_pretrained loads a HF model with profiling_file (no LMDB); LM head gets task vocab."""
    from vocab_tailor import VocabTailor

    model_id, profiling_file = _from_pretrained_common()
    vt = VocabTailor.from_pretrained(
        model_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bf16",
        lmdb_path=None,
        vocab_resize_strategy="prealloc",
        profiling_file=profiling_file,
    )
    assert vt is not None
    assert vt.tokenizer is not None
    assert hasattr(vt, "generate")
    assert hasattr(vt, "model")
    assert vt.model.lm_head.current_inds is not None


def test_from_pretrained_with_lmdb():
    """VocabTailor.from_pretrained with LMDB path: backbone + LMDB provider + profiling_file."""
    from vocab_tailor import VocabTailor

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_id, profiling_file = _from_pretrained_common()
    lmdb_path = os.path.join(repo_root, "datasets", "lmdb_store", "qwen3-1.7b_weights.lmdb")

    vt = VocabTailor.from_pretrained(
        model_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bf16",
        lmdb_path=lmdb_path,
        vocab_resize_strategy="prealloc",
        profiling_file=profiling_file,
    )
    assert vt is not None
    assert vt.tokenizer is not None
    assert hasattr(vt, "generate")
    assert hasattr(vt, "model")
    assert vt.model.lm_head.current_inds is not None


if __name__ == "__main__":
    test_import_api()
    test_version_format()
    test_from_pretrained()
    test_from_pretrained_with_lmdb()
    print("Smoke tests OK.")
