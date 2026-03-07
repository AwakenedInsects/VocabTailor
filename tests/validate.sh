PYTHONPATH=src python tests/validate_profiling_pipeline.py
# PYTHONPATH=src python tests/validate_profiling_pipeline.py --run 1
PYTHONPATH=src python tests/validate_model_package.py --input_aware --compare
PYTHONPATH=src python tests/validate_model_package.py --input_aware --lmdb_path datasets/lmdb_store/qwen3-1.7b_weights.lmdb --compare