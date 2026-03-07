"""
Three-stage static vocabulary filter for VocabTailor profiling.
(1) Input-aware filtering; (2) Language-specific Unicode filtering; (3) Tolerance filtering.
"""
import itertools
import json
from collections import Counter
from typing import Any, Optional

from . import unicode_utils


def _sentencepiece_prune(input_vocab: dict, unicode_vocab: dict, tokenizer: Any) -> dict:
    """Prunes vocabulary for SentencePiece-style tokenization."""
    pruned = {}
    for token, tid in input_vocab.items():
        decoded = tokenizer.decode([tid])
        if all(c in unicode_vocab for c in decoded):
            pruned[token] = tid
    return pruned


def _calculate_document_frequency(documents: list, min_freq: int) -> dict:
    """Document frequency of token ids across documents (each doc = list of ids)."""
    df = Counter()
    for doc in documents:
        df.update(set(doc))
    if min_freq > 1:
        return {k: v for k, v in df.items() if v >= min_freq}
    return dict(df)


class VocabTailorFilter:
    """
    Builds static task vocabulary via three stages: input-aware -> unicode -> tolerance.
    Dataset must support dataset[input_colname] and dataset[output_colname] (e.g. HF Dataset, pandas).
    """

    def __init__(
        self,
        tokenizer: Any,
        dataset: Any,
        input_colname: str,
        output_colname: str,
        unicode_filter_categories: list[str],
        task_name: Optional[str],
        model_type: str,
        output_dir: str,
        dataset_name: Optional[str] = None,
    ) -> None:
        """
        Args:
            tokenizer (Any): Hugging Face tokenizer.
            dataset (Any): Dataset with columns input_colname and output_colname (e.g. HF Dataset).
            input_colname (str): Name of the input text column.
            output_colname (str): Name of the output text column.
            unicode_filter_categories (list[str]): Language/script names for Stage 2 (e.g. ["chinese"], ["english"]).
            task_name (str, optional): Task label for output subdir and branch logic ("machine_translation", "math", etc.).
                If None or empty, output is written directly under output_dir and pipeline uses input-aware + BPE.
            model_type (str): Model type string for output filename (e.g. "Qwen3", "Llama").
            output_dir (str): Root directory for writing JSON.
            dataset_name (str, optional): Dataset identifier. For MT, "opus-100" or "kde4" selects input-aware Stage 1.
        """
        self.tokenizer = tokenizer
        self.tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
        self.input_colname = input_colname
        self.output_colname = output_colname
        self.dataset = dataset
        self.unicode_filter_categories = unicode_filter_categories
        self.task_name = task_name  # None or "" => write under output_dir only; "machine_translation"/"math" => branch logic
        self.model_type = model_type
        self.output_dir = output_dir
        self.dataset_name = dataset_name or ""

    def generate_static_vocab(
        self,
        tol: float | list[float] = 0,
        ablation: Optional[str] = None,
        verbose: bool = True,
    ) -> dict[str, int] | list[dict[str, int]]:
        """
        Run the three-stage pipeline. Returns the final vocab dict (or list of dicts if tol is a list).
        When ablation is set, only that branch is run (wo_input_aware, wo_unicode).

        Args:
            tol (float or list[float]): Tolerance(s) for Stage 3. If list, one output per value.
            ablation (str, optional): If set, run only that branch: "wo_input_aware" or "wo_unicode".
            verbose (bool, optional): Whether to print progress. Defaults to True.

        Returns:
            dict[str, int] or list[dict[str, int]]: Final token->id vocab, or list of such dicts if tol is a list.
        """
        if ablation == "wo_input_aware":
            vocab, unique_ids_freq = self._corpus_filtering()
        elif ablation == "wo_unicode":
            vocab, unique_ids_freq = self._input_aware_filtering(verbose)
        else:
            use_corpus = (
                self.task_name == "machine_translation"
                and self.dataset_name not in ("kde4", "opus-100")
            )
            if use_corpus:
                vocab, unique_ids_freq = self._corpus_filtering()
            else:
                vocab, unique_ids_freq = self._input_aware_filtering(verbose)

            if self.task_name == "math":
                vocab, unique_ids_freq = self._unicode_filtering_sp(vocab, unique_ids_freq, verbose)
            else:
                vocab, unique_ids_freq = self._unicode_filtering(vocab, unique_ids_freq, verbose)

        document_size = len(self.dataset)
        if isinstance(tol, list):
            result_list = []
            for t in tol:
                t_f = float(t) if isinstance(t, str) else t
                if t_f > 0:
                    v = self._tolerance_filtering(vocab, unique_ids_freq, document_size, t_f, verbose)
                else:
                    v = vocab
                self._save_vocab(v, t_f, ablation)
                result_list.append(v)
            return result_list
        else:
            if tol > 0:
                vocab = self._tolerance_filtering(vocab, unique_ids_freq, document_size, tol, verbose)
            self._save_vocab(vocab, tol, ablation)
            return vocab

    def _save_vocab(self, vocab: dict, tol: float, ablation: Optional[str] = None) -> None:
        """Write vocab JSON under output_dir, with optional task_name subdir. Filename: {model_type}_unicode_set_{categories}_tol_{tol}.json"""
        import os
        unicode_set = "_".join(self.unicode_filter_categories)
        out_dir = self.output_dir if not self.task_name else os.path.join(self.output_dir, self.task_name)
        os.makedirs(out_dir, exist_ok=True)
        if ablation is None:
            path = os.path.join(out_dir, f"{self.model_type}_unicode_set_{unicode_set}_tol_{tol}.json")
        else:
            path = os.path.join(out_dir, f"{self.model_type}_unicode_set_{unicode_set}_tol_{tol}_{ablation}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)

    def _corpus_filtering(self) -> tuple[dict, dict]:
        """Stage 1 (corpus mode): all tokens in output column + chat template."""
        output = self.tokenizer(
            list(self.dataset[self.output_colname]),
            padding=False,
        )
        input_ids = output["input_ids"]
        unique_ids_freq = _calculate_document_frequency(input_ids, 1)
        flat_ids = list(itertools.chain.from_iterable(input_ids))
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            chat_ids = self.tokenizer(self.tokenizer.chat_template)["input_ids"]
            chat_ids = [i for i in chat_ids if i < 128000]
            flat_ids = flat_ids + chat_ids
        id_set = set(flat_ids)
        vocab = self.tokenizer_json["model"]["vocab"]
        filtered = {t: i for t, i in vocab.items() if i in id_set}
        return filtered, unique_ids_freq

    def _input_aware_filtering(self, verbose: bool = True) -> tuple[dict, Counter]:
        """Stage 1 (input-aware): target-only token ids and their frequencies."""
        vocab = self.tokenizer_json["model"]["vocab"]
        vocab_map = {i: token for token, i in vocab.items()}
        unique_ids_freq = Counter()
        percentages = []
        inputs_col = self.dataset[self.input_colname]
        outputs_col = self.dataset[self.output_colname]
        for source, target in zip(inputs_col, outputs_col):
            source_ids = self.tokenizer.encode(source, add_special_tokens=False)
            target_ids = self.tokenizer.encode(target, add_special_tokens=False)
            if len(target_ids) == 0:
                continue
            s_set = set(source_ids)
            t_set = set(target_ids)
            common = s_set & t_set
            unique = t_set - s_set
            percentages.append(len(common) / len(t_set))
            unique = [idx for idx in unique if idx < len(vocab)]
            unique_ids_freq.update(unique)
        if verbose and percentages:
            pct = 100 * sum(percentages) / len(percentages)
            print(f"Input-related tokens percentage: {pct:.2f}%")
        return {vocab_map[i]: i for i in unique_ids_freq}, unique_ids_freq

    def _unicode_filtering(
        self, unfiltered_vocab: dict, unique_ids_freq: Counter, verbose: bool = True
    ) -> tuple[dict, dict]:
        """Stage 2: restrict to tokens in unicode categories (BPE)."""
        orig_vocab = self.tokenizer_json["model"]["vocab"]
        orig_merges = self.tokenizer_json["model"].get("merges", [])
        unicode_vocab = unicode_utils.generate_unicode_based_tokens(
            user_input=self.unicode_filter_categories,
            orig_vocab=orig_vocab,
            orig_merges=orig_merges,
            tokenizer=self.tokenizer,
            extreme_compress=False,
            save_to_json=False,
            verbose=verbose,
        )
        if verbose:
            print(f"language: num of tokens before unicode filtering: {len(unfiltered_vocab)}")
        filtered = unicode_utils.find_intersection(unfiltered_vocab, unicode_vocab)
        if verbose:
            print(f"language: num of tokens after unicode filtering: {len(filtered)}")
        freq_filtered = {
            tid: cnt for tid, cnt in unique_ids_freq.items() if tid in filtered.values()
        }
        return filtered, freq_filtered

    def _unicode_filtering_sp(
        self, unfiltered_vocab: dict, unique_ids_freq: Counter, verbose: bool = True
    ) -> tuple[dict, dict]:
        """Stage 2 for math/SentencePiece: prune by unicode character set."""
        unicode_pts = unicode_utils.get_unicode_code_points_dict_from_user_inputs(
            user_input=self.unicode_filter_categories, extreme_compress=False
        )
        unicode_vocab = unicode_utils.convert_to_unicode_vocab_dict(unicode_pts, verbose=verbose)
        if verbose:
            print(f"language: num of tokens before unicode filtering: {len(unfiltered_vocab)}")
        filtered = _sentencepiece_prune(unfiltered_vocab, unicode_vocab, self.tokenizer)
        if verbose:
            print(f"language: num of tokens after unicode filtering: {len(filtered)}")
        freq_filtered = {
            tid: cnt for tid, cnt in unique_ids_freq.items() if tid in filtered.values()
        }
        return filtered, freq_filtered

    def _tolerance_filtering(
        self,
        unfiltered_vocab: dict,
        unique_ids_freq: Counter | dict,
        document_num: int,
        tol: float,
        verbose: bool = True,
    ) -> dict:
        """Stage 3: keep top tokens by frequency so discarded mass <= document_num * tol."""
        if isinstance(unique_ids_freq, dict):
            unique_ids_freq = Counter(unique_ids_freq)
        if tol <= 0:
            if verbose:
                print("tol <= 0, filtering skipped")
            return unfiltered_vocab
        if verbose:
            print(f"num of tokens before tolerance filtering: {len(unfiltered_vocab)}")
        sorted_freqs = sorted(unique_ids_freq.values())
        kept_len = len(unfiltered_vocab)
        for i in range(1, len(sorted_freqs) + 1):
            if sum(sorted_freqs[:i]) > document_num * tol:
                kept_len = kept_len - i + 1
                break
        vocab = self.tokenizer_json["model"]["vocab"]
        vocab_map = {i: token for token, i in vocab.items()}
        if verbose:
            print(f"num of tokens after tolerance filtering: {kept_len}")
        return {vocab_map[i]: i for i, _ in unique_ids_freq.most_common(kept_len)}


def build_static_vocab(
    tokenizer: Any,
    dataset: Any,
    input_colname: str,
    output_colname: str,
    unicode_filter_categories: list[str],
    model_type: str,
    output_dir: str,
    task_name: Optional[str] = None,
    tolerance: float | list[float] = 0.1,
    dataset_name: Optional[str] = None,
    ablation: Optional[str] = None,
    verbose: bool = True,
) -> dict[str, int] | list[dict[str, int]]:
    """
    Build static task vocabulary (token -> id) via three-stage filtering and save JSON.

    Pipeline: (1) Input-aware or corpus filtering -> (2) Unicode filtering -> (3) Tolerance pruning.
    Output written to ``{output_dir}/{task_name}/`` if task_name is set, else ``{output_dir}/``.
    Filename: ``{model_type}_unicode_set_{categories}_tol_{tolerance}.json``.

    Args:
        tokenizer (Any): Hugging Face tokenizer.
        dataset (Any): Dataset with columns input_colname and output_colname.
        input_colname (str): Input text column name (e.g. "source", "document").
        output_colname (str): Output text column name (e.g. "target", "summary").
        unicode_filter_categories (list[str]): Language/script names for Stage 2 (e.g. ["chinese"], ["english"]).
        model_type (str): Model type for filename (e.g. "Qwen3", "Llama").
        output_dir (str): Root directory for JSON output (required).
        task_name (str, optional): Task label for output subdir and branch logic. If None or empty, output goes under output_dir
            and pipeline uses input-aware Stage 1 + BPE Stage 2. Use "machine_translation" or "math" for task-specific branching.
        tolerance (float or list[float], optional): Stage 3 tolerance; default 0.1. List => one JSON per value.
        dataset_name (str, optional): Dataset identifier. For task_name=="machine_translation", "opus-100" or "kde4" selects
            input-aware Stage 1 (target-only tokens); otherwise corpus Stage 1 is used.
        ablation (str, optional): "wo_input_aware" or "wo_unicode" to skip a stage.
        verbose (bool, optional): Print stage stats. Defaults to True.

    Returns:
        dict[str, int] or list[dict[str, int]]: Final vocab(s); also written to disk.
    """
    vtf = VocabTailorFilter(
        tokenizer=tokenizer,
        dataset=dataset,
        input_colname=input_colname,
        output_colname=output_colname,
        unicode_filter_categories=unicode_filter_categories,
        task_name=task_name,
        model_type=model_type,
        output_dir=output_dir,
        dataset_name=dataset_name,
    )
    return vtf.generate_static_vocab(tol=tolerance, ablation=ablation, verbose=verbose)


__all__ = ["VocabTailorFilter", "build_static_vocab"]
