"""
Profiling pipeline: build static task vocabulary via three-stage filtering.
(1) Input-aware filtering; (2) Language-specific Unicode filtering; (3) Tolerance filtering.
Use build_static_vocab() from Python or the CLI entrypoint to produce JSON consumed by --profiling_file.
"""
from .filter import VocabTailorFilter, build_static_vocab

__all__ = ["VocabTailorFilter", "build_static_vocab"]
