"""
Baseline generator for any causal LM: standard Hugging Face generation with the same
timing/metrics interface as VocabTailor for fair comparison.

Load via BaselineGenerator.from_pretrained(model_name_or_path, device=..., dtype=...)
for a Hub model id or local path; or instantiate and call load_model(model, device)
with an already-loaded model. The instance exposes .model, .tokenizer, and .device.
"""
from __future__ import annotations

import time
from typing import Optional, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, RepetitionPenaltyLogitsProcessor

from .metrics import MetricsTracker, TimingStreamer, _safe_div


class BaselineGenerator:
    """
    Wrapper around any Hugging Face causal LM that runs standard generation
    and records prefill/decode timing in a MetricsTracker, matching the
    semantics used by VocabTailor for fair comparison.

    Load from a Hub model id or local path with from_pretrained(); or
    instantiate and call load_model(model, device) with an already-loaded model.

    Attributes:
        .model, .tokenizer (set by from_pretrained), .device.
        tracker: None unless enable_metrics_tracker=True in from_pretrained() or
            load_model(). No cleanup needed when enable_metrics_tracker=False.
        gen_metrics: After generate(), holds per-run metrics (prefill_time, decode_tps, etc.)
            when tracker is enabled; None otherwise.
    """

    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self.device: Optional[str] = None
        self.tracker: Optional[MetricsTracker] = None

    @staticmethod
    def _resolve_dtype(dtype: Optional[str | torch.dtype]) -> Optional[torch.dtype]:
        if dtype is None:
            return None
        if isinstance(dtype, torch.dtype):
            return dtype
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        return mapping.get(dtype, None)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        device: str = "cuda",
        dtype: Optional[str | torch.dtype] = "bf16",
        enable_metrics_tracker: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **model_kwargs: Any,
    ) -> "BaselineGenerator":
        """
        Load a causal LM and tokenizer from a local path or Hugging Face Hub model id.

        Args:
            model_name_or_path (str): Hugging Face Hub model id (e.g. "Qwen/Qwen3-1.7B")
                or local path to the model directory.
            device (str): Target device for the model (e.g. 'cuda', 'cpu').
            dtype (str or torch.dtype, optional): Torch dtype or string alias ('bf16', 'fp16', 'fp32').
                If None, the model config default is used.
            enable_metrics_tracker (bool): Whether to enable metrics tracker. Defaults to False.
            tokenizer_kwargs (dict, optional): Extra kwargs for tokenizer loading (e.g. local_files_only=, token=).
            **model_kwargs: Extra kwargs for AutoModelForCausalLM.from_pretrained
                (e.g. local_files_only=, trust_remote_code=, token=).

        Returns:
            BaselineGenerator: Instance with .model, .tokenizer, and .device set, ready for generate().
        """
        torch_dtype = cls._resolve_dtype(dtype)
        if dtype is not None and torch_dtype is None:
            raise ValueError(f"dtype must be one of 'bf16', 'bfloat16', 'fp16', 'float16', 'fp32', 'float32' or a torch.dtype, got {dtype!r}")

        # Initialize model
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="cpu",
            torch_dtype=torch_dtype,
            **model_kwargs,
        )
        model.eval()

        # Initialize tokenizer
        tok_kwargs = tokenizer_kwargs or {}
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tok_kwargs)

        # Initialize BaselineGenerator
        bg = cls()
        bg.tokenizer = tokenizer
        bg.load_model(
            model = model, 
            device = device,
            enable_metrics_tracker = enable_metrics_tracker,
        )
        return bg

    def load_model(
        self, 
        model: Any, 
        device: str = "cuda",
        enable_metrics_tracker: bool = False,
    ) -> None:
        """
        Load and move model to the given device.

        For loading from a Hub id or local path with tokenizer, use from_pretrained() instead.

        Args:
            model (Any): Hugging Face causal LM (e.g. AutoModelForCausalLM instance).
            device (str): Target device (e.g. 'cuda', 'cpu'). Defaults to 'cuda'.
            enable_metrics_tracker (bool): Whether to enable metrics tracker. Defaults to False.
        """
        self.model = model
        self.device = device
        self.tracker = MetricsTracker() if enable_metrics_tracker else None
        
        print(f"Moving model to {self.device}...")
        self.model.to(self.device)

    def generate(
        self,
        inputs_ids: torch.Tensor,
        *,
        do_sample: bool = False,
        max_new_tokens: int = 128,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        """
        Run standard HF generation.

        Args:
            inputs_ids (torch.Tensor): Input token ids, shape (batch_size, seq_len).
            do_sample (bool, optional): Whether to sample. Defaults to False.
            max_new_tokens (int, optional): Maximum new tokens. Defaults to 128.
            **generate_kwargs: Forwarded to model.generate() (e.g. temperature, top_p).

        Returns:
            torch.Tensor: Generated token ids in original vocabulary space (prompt + generated).
            Per-run metrics (prefill_time, decode_tps, etc.) are stored in self.gen_metrics
            when the tracker is enabled; self.gen_metrics is None otherwise.
        """
        # 1. Setup Generation
        rep_pen_processor = RepetitionPenaltyLogitsProcessor(
            penalty=1.1,
            prompt_ignore_length=inputs_ids.shape[-1],
        )

        # 2. Setup Timer
        timing_streamer: Optional[TimingStreamer] = None
        if self.tracker:
            input_tokens_count = inputs_ids.numel()
            timing_streamer = TimingStreamer()

        # 3. Generate
        output_ids = self.model.generate(
            inputs=inputs_ids.to(self.device),
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            logits_processor=[rep_pen_processor],
            streamer=timing_streamer,
            **generate_kwargs,
        )

        # 4. Calculate tracker metrics
        self.gen_metrics = None
        if self.tracker and timing_streamer is not None:
            gen_end = time.perf_counter()
            if timing_streamer.first_token_time is None:
                timing_streamer.first_token_time = gen_end

            prefill_dt = timing_streamer.first_token_time - timing_streamer.start_time
            decode_dt = gen_end - timing_streamer.first_token_time
            output_tokens_count = output_ids[0][len(inputs_ids[0]) :].numel()

            # 5. Update tracker
            self.tracker.total_prefill_time += prefill_dt
            self.tracker.total_decode_time += decode_dt
            self.tracker.total_prefill_tokens += input_tokens_count
            self.tracker.total_decode_tokens += output_tokens_count

            self.gen_metrics = {
                "prefill_time": prefill_dt,
                "decode_time": decode_dt,
                "prefill_tokens": input_tokens_count,
                "decode_tokens": output_tokens_count,
                "prefill_tps": _safe_div(input_tokens_count, prefill_dt),
                "decode_tps": _safe_div(output_tokens_count, decode_dt),
            }

        return output_ids
