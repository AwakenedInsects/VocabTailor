import time
import torch
from transformers.generation.streamers import BaseStreamer


def _safe_div(num: float, den: float) -> float:
    """Safe division with zero check.

    Args:
        num (float): Numerator.
        den (float): Denominator.

    Returns:
        float: num / den if den > 0, else 0.0.
    """
    return num / den if den > 0 else 0.0


class MetricsTracker:
    """
    A shared object to store performance metrics across different modules.
    """

    def __init__(self):
        self.total_lmdb_emb_time = 0.0
        self.total_lmdb_emb_calls = 0
        self.total_lmdb_emb_tokens = 0
        self.lmdb_emb_tps = 0.0
        self.lmdb_emb_latency = 0.0

        self.total_lmdb_head_time = 0.0
        self.total_lmdb_head_calls = 0
        self.total_lmdb_head_tokens = 0
        self.lmdb_head_tps = 0.0
        self.lmdb_head_latency = 0.0

        self.total_dynamic_loading_time = 0.0
        self.total_prefill_time = 0.0
        self.total_decode_time = 0.0
        self.total_prefill_tokens = 0
        self.total_decode_tokens = 0
        self.prefill_tps = 0.0
        self.decode_tps = 0.0

        self.vocab_extend_time = 0.0
        self.vocab_peak_vram_gb = 0.0

        self.total_job_time = 0.0
        self.total_ram_gb = 0.0
        self.peak_vram_gb = 0.0

    def reset(self) -> None:
        self.__init__()

    def get_peak_vram_gb(self) -> float:
        """Get max memory spike in GB."""
        peak_vram_gb = 0.0
        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            peak_vram_gb = torch.mps.current_allocated_memory() / (1024 ** 3)
        return peak_vram_gb

    def _update_vocab_extension_metrics(self, dt: float, mem_gb: float) -> None:
        """Update the metrics related to vocab tailor."""
        self.vocab_extend_time += dt
        if mem_gb > self.vocab_peak_vram_gb:
            self.vocab_peak_vram_gb = mem_gb

    def _update_tps_latency(self) -> None:
        """Update tps and latency metrics."""
        pairs = [
            ["tps", "tokens", "time"],
            ["latency", "time", "calls"],
        ]

        for prefix in ["lmdb_emb", "lmdb_head", "prefill", "decode"]:
            for attr, num_name, den_name in pairs:
                if prefix in ["prefill", "decode"] and attr == "latency":
                    continue

                num = getattr(self, f"total_{prefix}_{num_name}")
                den = getattr(self, f"total_{prefix}_{den_name}")
                setattr(self, f"{prefix}_{attr}", _safe_div(num, den))

    def _all_metrics(self) -> dict:
        """Return all metrics in a dict."""
        self._update_tps_latency()

        return {
            "total_job_time": self.total_job_time,
            "total_ram_gb": self.total_ram_gb,
            "peak_vram_gb": self.peak_vram_gb,
            "total_lmdb_emb_time": self.total_lmdb_emb_time,
            "total_lmdb_emb_calls": self.total_lmdb_emb_calls,
            "total_lmdb_emb_tokens": self.total_lmdb_emb_tokens,
            "lmdb_emb_speed": self.lmdb_emb_tps,
            "lmdb_emb_latency": self.lmdb_emb_latency,
            "total_lmdb_head_time": self.total_lmdb_head_time,
            "total_lmdb_head_calls": self.total_lmdb_head_calls,
            "total_lmdb_head_tokens": self.total_lmdb_head_tokens,
            "lmdb_head_speed": self.lmdb_head_tps,
            "lmdb_head_latency": self.lmdb_head_latency,
            "total_dynamic_loading_time": self.total_dynamic_loading_time,
            "total_prefill_time": self.total_prefill_time,
            "total_prefill_tokens_count": self.total_prefill_tokens,
            "prefill_speed": self.prefill_tps,
            "total_decode_time": self.total_decode_time,
            "total_decode_tokens_count": self.total_decode_tokens,
            "decode_speed": self.decode_tps,
            "vocab_extend_time": self.vocab_extend_time,
            "vocab_peak_vram_gb": self.vocab_peak_vram_gb,
        }

    def save_log(
        self,
        log_file_path: str,
        enable_vocab_tailor: bool = True,
        offload_to_lmdb: bool = True,
        append_log: bool = False,
    ) -> None:
        """Save tracking metrics to a log file.

        Args:
            log_file_path (str): Path to the output log file.
            enable_vocab_tailor (bool, optional): Whether to include vocab-tailor breakdown. Defaults to True.
            offload_to_lmdb (bool, optional): Whether to include LMDB operation stats. Defaults to True.
            append_log (bool, optional): If True, append to file; otherwise overwrite. Defaults to False.
        """
        self._update_tps_latency()

        mode = "a" if append_log else "w"
        with open(log_file_path, mode) as f:
            f.write(f"{'#' * 60}\n#{' ' * 20}Performance Report{' ' * 20}#\n{'#' * 60}\n")

            f.write(f"{'=' * 60}\n{' ' * 20}Total Speed and Memory\n{'=' * 60}\n")
            f.write(f"Total Job Time:               {self.total_job_time:.4f} s\n")
            f.write(f"System RAM (RSS):             {self.total_ram_gb:.4f} GB\n")
            f.write(f"GPU VRAM (Est.):              {self.peak_vram_gb:.4f} GB\n")
            f.write("\n")

            if offload_to_lmdb:
                f.write(f"{'=' * 60}\n{' ' * 20}LMDB Operation\n{'=' * 60}\n")
                f.write(f"Total LMDBEmb. Time:          {self.total_lmdb_emb_time:.4f} s\n")
                f.write(f"Total LMDBEmb. Calls:         {self.total_lmdb_emb_calls:,}\n")
                f.write(f"Total LMDBEmb. Tokens:        {self.total_lmdb_emb_tokens:,}\n")
                f.write(f"LMDBEmb. Speed (tokens/s):    {self.lmdb_emb_tps:.4f} tps\n")
                f.write(f"LMDBEmb. Latency (ms/call):   {self.lmdb_emb_latency * 1000:.4f} ms\n")
                f.write("\n")
                f.write(f"Total LMDBHead Time:          {self.total_lmdb_head_time:.4f} s\n")
                f.write(f"Total LMDBHead Calls:         {self.total_lmdb_head_calls:,}\n")
                f.write(f"Total LMDBHead Tokens:        {self.total_lmdb_head_tokens:,}\n")
                f.write(f"LMDBHead Speed (tokens/s):    {self.lmdb_head_tps:.4f} tps\n")
                f.write(f"LMDBHead Latency (ms/call):   {self.lmdb_head_latency * 1000:.4f} ms\n")
                f.write("\n")

            f.write(f"{'=' * 60}\n{' ' * 20}Granular Breakdown\n{'=' * 60}\n")
            if enable_vocab_tailor:
                f.write(f"Total Dynamic Loading Time:   {self.total_dynamic_loading_time:.4f} s\n")
                f.write("\n")

            f.write(f"Total Prefill Time:           {self.total_prefill_time:.4f} s\n")
            f.write(f"Prefill Tokens:               {self.total_prefill_tokens:,}\n")
            f.write(f"Prefill Speed (tokens/s):     {self.prefill_tps:,.4f} tps\n")
            f.write("\n")
            f.write(f"Total Decode Time:            {self.total_decode_time:.4f} s\n")
            f.write(f"Decode Tokens:                {self.total_decode_tokens:,}\n")
            f.write(f"Decode Speed (tokens/s):      {self.decode_tps:,.4f} tps\n")
            f.write("\n")

            if enable_vocab_tailor:
                f.write(f"{'=' * 60}\n{' ' * 20}Dynamic Vocab Extension\n{'=' * 60}\n")
                f.write(f"Total Update Time:            {self.vocab_extend_time:.4f} s\n")
                f.write(f"Peak Mem Overhead:            {self.vocab_peak_vram_gb:.4f} GB\n")
                f.write("\n")


class TimingStreamer(BaseStreamer):
    """
    A Streamer hook to detect when the 'Prefill' phase ends and the 'Decode' phase begins.
    """

    def __init__(self):
        self.start_time = time.perf_counter()
        self.first_token_time = None

    def put(self, value) -> None:
        """Called when a new token is generated."""
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()

    def end(self) -> None:
        """Required by BaseStreamer. Called when generation ends."""
        pass


__all__ = ["MetricsTracker", "TimingStreamer", "_safe_div"]

