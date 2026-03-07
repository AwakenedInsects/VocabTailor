import os
import struct
import time
from typing import Optional

import lmdb
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from tqdm import tqdm

from .metrics import MetricsTracker


def _tensor_to_bytes(t: torch.Tensor, dtype: torch.dtype) -> bytes:
    """
    Safely convert a tensor to raw bytes with the given dtype. Ensures correct bfloat16 storage.
    """
    return (
        t.to(dtype)
        .contiguous()
        .cpu()
        .numpy()      # uint16 bit pattern for bf16
        .tobytes()
    )


def build_lmdb_weights(model_path: str, lmdb_path: str, dtype: torch.dtype = torch.float16, force_create: bool = False) -> None:
    """
    Build LMDB for input embedding and optionally LM head (if not weight_shared).

    Keys:
        w_{i}: input embedding for token i
        u_{i}: lm head weight for token i (only if not tied)
        b_{i}: lm head bias for token i (if exists)

    Args:
        model_path (str): Local file path to model.
        lmdb_path (str): Path to create the .lmdb file.
        dtype (torch.dtype): Target dtype for saved weights.
        force_create (bool): Force to to create a new lmdb file (replace if exists).
    """
    if not force_create and os.path.exists(lmdb_path):
        print(f"LMDB already exists at '{lmdb_path}'. Skip building.")
        return
    
    lmdb_dir = os.path.dirname(lmdb_path)
    if lmdb_dir:
        os.makedirs(lmdb_dir, exist_ok=True)

    assert dtype in (torch.bfloat16, torch.float16, torch.float32)

    # Load model temporarily
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype, # save fp16 to save memory
        device_map="cpu",
        local_files_only=True,
    )

    emb = model.get_input_embeddings().weight.detach()
    lm_head = model.lm_head.weight.detach()
    bias = model.lm_head.bias.detach() if model.lm_head.bias is not None else None

    weight_shared = emb.data_ptr() == lm_head.data_ptr()
    vocab_size, hidden_dim = emb.shape

    # Calculate map_size
    element_size = {
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.float32: 4,
    }[dtype]
    total_bytes = vocab_size * hidden_dim * element_size * (1 if weight_shared else 2)
    env = lmdb.open(lmdb_path, map_size=total_bytes * 5)

    with env.begin(write=True) as txn:
        # metadata
        txn.put(b"__meta_hidden_dim__", struct.pack("i", hidden_dim))
        txn.put(b"__meta_vocab_size__", struct.pack("i", vocab_size))
        txn.put(b"__meta_weight_shared__", b"1" if weight_shared else b"0")
        txn.put(b"__meta_has_bias__", b"1" if bias is not None else b"0")
        txn.put(b"__meta_dtype__", str(dtype).encode("utf-8"))

        for i in tqdm(range(vocab_size), desc="Writing weights to LMDB..."):
            txn.put(f"w_{i}".encode(), _tensor_to_bytes(emb[i], dtype))
            if not weight_shared:
                txn.put(f"u_{i}".encode(), _tensor_to_bytes(lm_head[i], dtype))
            if bias is not None:
                txn.put(f"b_{i}".encode(), _tensor_to_bytes(bias[i], dtype))

    env.close()
    del model
    print(f"Finish building LMDB at '{lmdb_path}'")


class LMDBWeightProvider:
    """Helper to fetch weight slices for either Embedding or LM Head from disk-backed LMDB file to CPU."""

    def __init__(
        self,
        lmdb_path: str,
        tracker: Optional["MetricsTracker"] = None,
        fetch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Open LMDB and read metadata.

        Args:
            lmdb_path (str): Path to the .lmdb file.
            tracker (MetricsTracker, optional): Optional metrics tracker for timing.
            fetch_dtype (torch.dtype, optional): Dtype for returned tensors; defaults to LMDB saved dtype.
        """
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"LMDB path not found: {lmdb_path}")

        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.tracker = tracker
        self.fetch_dtype = fetch_dtype

        with self.env.begin(write=False) as txn:
            self.hidden_dim = struct.unpack("i", txn.get(b"__meta_hidden_dim__"))[0]
            self.vocab_size = struct.unpack("i", txn.get(b"__meta_vocab_size__"))[0]
            self.weight_shared = txn.get(b"__meta_weight_shared__") == b"1"
            self.has_bias = txn.get(b"__meta_has_bias__") == b"1"
            # LMDB saved dtype
            dtype_str = txn.get(b"__meta_dtype__").decode("utf-8")
            torch_dtype_map = {
                "torch.bfloat16": torch.bfloat16,
                "torch.float16": torch.float16,
                "torch.float32": torch.float32,
            }
            np_dtype_map = {
                "torch.bfloat16": np.uint16,
                "torch.float16": np.float16,
                "torch.float32": np.float32,
            }
            self.saved_torch_dtype = torch_dtype_map.get(dtype_str)
            if self.saved_torch_dtype is None:
                raise ValueError(f"Unknown dtype in LMDB: {dtype_str}")
            self.saved_np_dtype = np_dtype_map.get(dtype_str)

        # update fetch_dtype with LMDB saved_dtype if not specified
        if self.fetch_dtype is None:
            self.fetch_dtype = self.saved_torch_dtype

    def _update_metrics(self, category: str, dt: float, num_tokens: int) -> None:
        """
        Update LMDB-related metrics internally.
        - lmdb_emb/head_time: wall time spent inside LMDB
        - lmdb_emb/head_calls: number of LMDB batch fetches
        - lmdb_emb/head_tokens: number of rows fetched

        Args:
            category (str): 'emb' for Input Embeddings, 'head' for LM Head.
            dt (float): Wall time spent inside LMDB.
            num_tokens (int): Number of rows fetched.
        """
        if not self.tracker:
            return

        self.tracker.__dict__[f"total_lmdb_{category}_time"] += dt
        self.tracker.__dict__[f"total_lmdb_{category}_calls"] += 1
        self.tracker.__dict__[f"total_lmdb_{category}_tokens"] += num_tokens

    def _numpy_to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """Convert to final tensor."""
        t = torch.from_numpy(arr)
        if self.saved_torch_dtype == torch.bfloat16:
            t = t.view(torch.bfloat16)
        return t.to(dtype=self.fetch_dtype)

    def _fetch_weight_matrix(self, indices: torch.Tensor, prefix: str, category: str) -> torch.Tensor:
        """
        Fetch weight matrix of shape [K, hidden_dim].

        Args:
            indices (torch.Tensor): Tensor of token ids, shape (K,) or flattened.
            prefix (str): 'w_' for weights in tied case, 'u_' for non-tied case.
            category (str): 'emb' for Input Embeddings, 'head' for LM Head.
        """
        start_time = time.perf_counter()

        flat_ids = indices.view(-1).cpu().numpy()
        num_tokens = len(flat_ids)

        out = np.zeros((num_tokens, self.hidden_dim), dtype=self.saved_np_dtype)

        with self.env.begin(write=False) as txn:
            get_val = txn.get
            for i, idx in enumerate(flat_ids):
                key = f"{prefix}{idx}".encode()
                val_bytes = get_val(key)
                if val_bytes:
                    out[i] = np.frombuffer(val_bytes, dtype=self.saved_np_dtype)

        dt = time.perf_counter() - start_time
        self._update_metrics(category, dt, num_tokens)

        return self._numpy_to_tensor(out)

    def _fetch_bias_vector(self, indices: torch.Tensor, category: str) -> torch.Tensor:
        """
        Fetch bias vector of shape [K].

        Args:
            indices (torch.Tensor): Tensor of token ids (shape [K] or flattened).
            category (str): 'emb' for Input Embeddings, 'head' for LM Head.

        Returns:
            torch.Tensor: Bias vector of shape [K] in fetch_dtype.
        """
        start_time = time.perf_counter()

        flat_ids = indices.view(-1).cpu().numpy().flatten()
        num_tokens = len(flat_ids)

        out = np.zeros(num_tokens, dtype=self.saved_np_dtype)

        with self.env.begin(write=False) as txn:
            get_val = txn.get
            for i, idx in enumerate(flat_ids):
                key = f"b_{idx}".encode()
                val_bytes = get_val(key)
                if val_bytes:
                    out[i] = np.frombuffer(val_bytes, dtype=self.saved_np_dtype)[0]

        dt = time.perf_counter() - start_time
        self._update_metrics(category, dt, num_tokens)

        return self._numpy_to_tensor(out)

    def fetch_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        """Fetch input embedding matrix of shape [K, hidden_dim]."""
        return self._fetch_weight_matrix(indices, prefix="w_", category="emb")

    def fetch_head_weights(self, indices: torch.Tensor) -> torch.Tensor:
        """Fetch LM head weight matrix of shape [K, hidden_dim]."""
        prefix = "w_" if self.weight_shared else "u_"
        return self._fetch_weight_matrix(indices, prefix=prefix, category="head")

    def fetch_head_biases(self, indices: torch.Tensor) -> torch.Tensor:
        """Fetch LM head bias vector of shape [K] or None."""
        if not self.has_bias:
            return None
        return self._fetch_bias_vector(indices, category="head")

    def close(self) -> None:
        """Close the LMDB environment."""
        if hasattr(self, "env"):
            self.env.close()
            print("LMDB Environment closed.")


class LMDBEmbedding(nn.Module):
    """
    LMDB-backed Input Embedding.
    """

    def __init__(self, provider: LMDBWeightProvider):
        super().__init__()
        self.provider = provider
        self.hidden_dim = provider.hidden_dim
        self.device = "cpu"

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat_ids = input_ids.view(-1).cpu()
        emb = self.provider.fetch_embeddings(flat_ids)
        return emb.view(*input_ids.shape, self.hidden_dim)


class LMDBHead(nn.Module):
    """
    LMDB-backed LM Head supporting full vocab or candidate-only projection.
    """

    def __init__(self, provider: LMDBWeightProvider):
        super().__init__()
        self.provider = provider
        self.hidden_dim = provider.hidden_dim
        self.vocab_size = provider.vocab_size
        self.has_bias = provider.has_bias

    def forward(self, hidden_states: torch.Tensor, token_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, H = hidden_states.shape
        hidden_flat = hidden_states.view(-1, H)

        if token_ids is None:
            token_ids = torch.arange(self.vocab_size, device="cpu")
        else:
            token_ids = token_ids.view(-1).to("cpu")

        W = self.provider.fetch_head_weights(token_ids)
        W = W.to(device=hidden_states.device, non_blocking=True)

        if hidden_flat.dtype != W.dtype:
            hidden_flat = hidden_flat.to(W.dtype)

        logits = hidden_flat @ W.T

        if self.has_bias:
            b = self.provider.fetch_head_biases(token_ids)
            if b is not None:
                logits += b.to(device=hidden_states.device, dtype=logits.dtype, non_blocking=True)

        return logits.view(B, T, -1)


__all__ = ["build_lmdb_weights", "LMDBWeightProvider", "LMDBEmbedding", "LMDBHead"]

