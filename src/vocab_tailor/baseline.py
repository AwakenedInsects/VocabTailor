"""
Baseline generator for any causal LM: standard Hugging Face generation with the same
timing/metrics interface as VocabTailor for fair comparison.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import RepetitionPenaltyLogitsProcessor

from .metrics import MetricsTracker, TimingStreamer, _safe_div


class BaselineGenerator:
    """
    Wrapper around any Hugging Face causal LM that runs standard generation
    and records prefill/decode timing in a MetricsTracker, matching the
    semantics used by VocabTailor for fair comparison.
    """

    def __init__(self) -> None:
        self.model: Any = None
        self.device: str = "cuda"
        self.tracker = MetricsTracker()

    def load_model(self, model: Any, device: str = "cuda") -> None:
        """Load and move model to the given device.

        Args:
            model (Any): Hugging Face causal LM (e.g. AutoModelForCausalLM instance).
            device (str): Target device (e.g. 'cuda', 'cpu'). Defaults to 'cuda'.
        """
        self.model = model
        self.device = device
        print(f"Moving model to {self.device}...")
        self.model.to(self.device)

    def generate(
        self,
        inputs_ids: torch.Tensor,
        do_sample: bool = False,
        max_new_tokens: int = 128,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """Run standard HF generation and return (output_ids, metrics_dict).

        Args:
            inputs_ids (torch.Tensor): Input token ids, shape (batch_size, seq_len).
            do_sample (bool, optional): Whether to sample. Defaults to False.
            max_new_tokens (int, optional): Maximum new tokens. Defaults to 128.
            *args (Any): Passed to model.generate().
            **kwargs (Any): Passed to model.generate().

        Returns:
            Tuple[torch.Tensor, Optional[Dict[str, float]]]: (output_ids, metrics_dict or None).
        """
        # 1. Setup Generation
        rep_pen_processor = RepetitionPenaltyLogitsProcessor(
            penalty=1.1,
            prompt_ignore_length=inputs_ids.shape[-1],
        )

        # 2. Setup Timer
        timing_streamer: Optional[TimingStreamer] = None
        if self.tracker:
            timing_streamer = TimingStreamer()

        # 3. Generate
        output_ids = self.model.generate(
            inputs=inputs_ids.to(self.device),
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            logits_processor=[rep_pen_processor],
            streamer=timing_streamer,
            *args,
            **kwargs,
        )

        # 4. Calculate tracker metrics
        metrics: Optional[Dict[str, float]] = None
        if self.tracker and timing_streamer is not None:
            gen_end = time.perf_counter()
            if timing_streamer.first_token_time is None:
                timing_streamer.first_token_time = gen_end

            prefill_dt = timing_streamer.first_token_time - timing_streamer.start_time
            decode_dt = gen_end - timing_streamer.first_token_time

            input_tokens_count = inputs_ids.numel()
            output_tokens_count = output_ids[0][len(inputs_ids[0]) :].numel()

            # 5. Update tracker
            self.tracker.total_prefill_time += prefill_dt
            self.tracker.total_decode_time += decode_dt
            self.tracker.total_prefill_tokens += input_tokens_count
            self.tracker.total_decode_tokens += output_tokens_count

            metrics = {
                "prefill_time": prefill_dt,
                "decode_time": decode_dt,
                "prefill_tokens": input_tokens_count,
                "decode_tokens": output_tokens_count,
                "prefill_tps": _safe_div(input_tokens_count, prefill_dt),
                "decode_tps": _safe_div(output_tokens_count, decode_dt),
            }

        return output_ids, metrics
