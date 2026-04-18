from __future__ import annotations

import copy
import json
import logging
import os
import time
from typing import Optional, Dict, Any

import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    RepetitionPenaltyLogitsProcessor,
    logging as tf_logging,
)

from .split_linear import SplitLinear
from .lmdb_layers import build_lmdb_weights, LMDBWeightProvider, LMDBEmbedding
from .metrics import MetricsTracker, TimingStreamer, _safe_div
from .model_utils import check_weight_sharing, load_model_backbone


os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
tf_logging.set_verbosity_error()


def _bytes_to_gb(num_bytes: int) -> float:
    return float(num_bytes) / (1024**3)


class OffloadEmbedding(nn.Module):
    """
    Embedding layer that can either:
    - mirror in-memory weights from a standard embedding layer, or
    - lazily fetch rows from an LMDB-backed provider.
    """

    def __init__(self, embedding: Optional[nn.Embedding], provider: Optional[LMDBWeightProvider] = None, weight_share: bool = True) -> None:
        """Initialize offload embedding from in-memory embedding or LMDB provider.

        Args:
            embedding (nn.Embedding, optional): In-memory embedding; used when provider is None.
            provider (LMDBWeightProvider, optional): LMDB provider for lazy weight fetch. Defaults to None.
            weight_share (bool): Whether to share weights with LM head. Defaults to True.
        """
        super().__init__()

        if provider is not None:
            self.full_embedding = LMDBEmbedding(provider)
        else:
            self.full_embedding = copy.deepcopy(embedding)

        self.embedding: Optional[nn.Embedding] = None
        self.device = "cpu"
        self.weight_share = weight_share
        self._linked_lm_head: Optional["LockedLMHead"] = None

    @property
    def current_inds(self) -> Optional[torch.Tensor]:
        return None if self._linked_lm_head is None else self._linked_lm_head.current_inds

    @current_inds.setter
    def current_inds(self, value: torch.Tensor) -> None:
        if self._linked_lm_head is not None:
            self._linked_lm_head.current_inds = value

    def link_to_lm_head(self, lm_head_obj: "LockedLMHead") -> None:
        self._linked_lm_head = lm_head_obj

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        if self.weight_share:
            return self.embedding(x.to(self.device)).to(device)
        else:
            mapped_id = self.current_inds[x.to(self.device)]
            return self.full_embedding(mapped_id).to(device)

    def update_weights(self, new_weights: torch.Tensor, device: torch.device | str) -> None:
        self.embedding = nn.Embedding(
            num_embeddings=new_weights.data.shape[0],
            embedding_dim=new_weights.data.shape[1],
        )

        if isinstance(new_weights, nn.Parameter):
            self.embedding.weight = new_weights
        else:
            self.embedding.weight = nn.Parameter(new_weights)

        self.device = device


class LockedLMHead(nn.Module):
    """
    A dynamically resizable LM head that only materializes the subset of vocabulary
    needed for the current request. Supports:
      - naive reallocation,
      - SplitLinear composition,
      - pre-allocated buffer strategy.
    """

    def __init__(
        self,
        device: str,
        full_lm_head: Optional[nn.Linear] = None,
        provider: Optional[LMDBWeightProvider] = None,
        embedding: Optional[OffloadEmbedding] = None,
        weight_share: bool = True,
        tracker: Optional[MetricsTracker] = None,
        default_buffer_size: int = 128,
        vocab_resize_strategy: str = "prealloc",
    ) -> None:
        """Initialize locked LM head from full head or LMDB provider.

        Args:
            device (str): Target device for the active head (e.g. 'cuda', 'cpu').
            full_lm_head (nn.Linear, optional): Full in-memory LM head; used when provider is None.
            provider (LMDBWeightProvider, optional): LMDB provider for lazy weight fetch.
            embedding (OffloadEmbedding, optional): For weight tying with input embeddings.
            weight_share (bool): Whether to share weights with embedding. Defaults to True.
            tracker (MetricsTracker, optional): Optional metrics tracker.
            default_buffer_size (int): Buffer size for prealloc strategy. Defaults to 128.
            vocab_resize_strategy (str): 'realloc' | 'split_linear' | 'prealloc'. Defaults to "prealloc".
        """
        super().__init__()

        if full_lm_head is None and provider is None:
            raise ValueError("LockedLMHead requires either 'full_lm_head' (nn.Linear) or 'provider' (LMDBWeightProvider).")

        if provider is not None:
            print("Offloading lm head to LMDB...")

        self.device = device
        self.full_lm_head = full_lm_head
        self.provider = provider
        self.embedding = embedding
        self.weight_share = weight_share
        self.tracker = tracker if provider is None else provider.tracker
        self.buffer_size = default_buffer_size
        self.vocab_resize_strategy = vocab_resize_strategy

        if self.provider is not None:
            self.hidden_dim = self.provider.hidden_dim
            self.has_bias = self.provider.has_bias
            self.weight_dtype = self.provider.fetch_dtype
        else:
            self.hidden_dim = self.full_lm_head.weight.data.shape[1]
            self.has_bias = self.full_lm_head.bias is not None
            self.weight_dtype = self.full_lm_head.weight.dtype

        self._current_inds: Optional[torch.Tensor] = None
        self.current_head: Optional[nn.Module] = None
        self.temp_ind: Optional[int] = None

    @property
    def current_inds(self) -> Optional[torch.Tensor]:
        return self._current_inds

    @current_inds.setter
    def current_inds(self, value: torch.Tensor) -> None:
        self._current_inds = value

    def update_inds(self, inds: torch.Tensor, temp: bool) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()

        if self.current_inds is None:
            self.current_inds = torch.unique(inds)
            if temp:
                self.temp_ind = 0
        else:
            unique_inds = torch.unique(inds)
            new_inds = unique_inds[~torch.isin(unique_inds, self.current_inds)]
            if temp:
                self.temp_ind = len(self.current_inds)
            if not new_inds.numel() == 0:
                self.current_inds = torch.concat([self.current_inds, new_inds])
            else:
                if self.tracker:
                    self.tracker._update_vocab_extension_metrics(time.perf_counter() - start_time, 0.0)
                return

        if self.current_head is None:
            self.create_new_head(self.current_inds)
        else:
            self.extend_head(new_inds)

        if self.tracker:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - start_time
            vocab_ext_peak_vram = self.tracker.get_peak_vram_gb()
            self.tracker._update_vocab_extension_metrics(dt, vocab_ext_peak_vram)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.current_head(x)

    # --- helpers to read weights/biases ---

    def _get_head_weights(self, inds: torch.Tensor) -> torch.Tensor:
        if self.provider:
            return self.provider.fetch_head_weights(inds).to(self.device)
        elif self.full_lm_head:
            return self.full_lm_head.weight.data[inds].to(self.device)
        else:
            raise RuntimeError("No weights source available")
    
    def _get_head_biases(self, inds: torch.Tensor) -> torch.Tensor:
        if self.provider:
            return self.provider.fetch_head_biases(inds).to(self.device)
        elif self.full_lm_head:
            return self.full_lm_head.bias.data[inds].to(self.device)
        else:
             raise RuntimeError("No weights source available")

    # --- strategy dispatch ---

    def create_new_head(self, inds: torch.Tensor) -> None:
        if self.vocab_resize_strategy == "realloc":
            self._create_new_head(inds)
        elif self.vocab_resize_strategy == "split_linear":
            self._create_new_head_splitlinear(inds)
        elif self.vocab_resize_strategy == "prealloc":
            self._create_new_head_buffer(inds)
        else:
            raise ValueError(f"Invalid strategy: {self.vocab_resize_strategy}")
    
    def extend_head(self, new_inds: torch.Tensor) -> None:
        if self.vocab_resize_strategy == "realloc":
            self._extend_head(new_inds)
        elif self.vocab_resize_strategy == "split_linear":
            self._extend_head_splitlinear(new_inds)
        elif self.vocab_resize_strategy == "prealloc":
            self._extend_head_buffer(new_inds)
        else:
            raise ValueError(f"Invalid strategy: {self.vocab_resize_strategy}")
    
    def reset_head(self) -> None:
        if self.vocab_resize_strategy == "realloc":
            self._reset_head()
        elif self.vocab_resize_strategy == "split_linear":
            self._reset_head_splitlinear()
        elif self.vocab_resize_strategy == "prealloc":
            self._reset_head_buffer()
        else:
            raise ValueError(f"Invalid strategy: {self.vocab_resize_strategy}")

    # ================= Re-allocation (Naive implementation) =================

    def _create_new_head(self, inds: torch.Tensor) -> None:
        self.current_head = nn.Linear(
            in_features=self.hidden_dim,
            out_features=inds.shape[0],
            bias=self.has_bias,
            dtype=self.weight_dtype,
            device=self.device
        )
        
        with torch.no_grad():
            self.current_head.weight.copy_(self._get_head_weights(inds))
            if self.has_bias:
                self.current_head.bias.copy_(self._get_head_biases(inds))
        
        if self.weight_share and self.embedding is not None:
            self.embedding.update_weights(self.current_head.weight, device=self.device)

    def _extend_head(self, new_inds: torch.Tensor) -> None:
        """Naive implementation: Re-allocate and copy all weights."""
        combined_weight = torch.concat([
            self.current_head.weight.data,
            self._get_head_weights(new_inds)
        ], dim=0)
        
        if self.has_bias:
            combined_bias = torch.concat([
                self.current_head.bias.data,
                self._get_head_biases(new_inds)
            ], dim=0)
        
        self.current_head = nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.current_inds.shape[0],
            bias=self.has_bias,
            dtype=self.weight_dtype,
            device=self.device
        )
        
        with torch.no_grad():
            self.current_head.weight.copy_(combined_weight)
            if self.has_bias:
                self.current_head.bias.copy_(combined_bias)
        
        if self.weight_share and self.embedding is not None:
            self.embedding.update_weights(self.current_head.weight, self.device)     

    def _reset_head(self) -> None:
        if self.temp_ind is None:
            return

        if self.temp_ind == 0:
            self.current_inds = None
            self.current_head = None
        else:
            self.current_inds = self.current_inds[:self.temp_ind]
            saved_weight = self.current_head.weight.data[:self.temp_ind,:]
            if self.has_bias:
                saved_bias = self.current_head.bias.data[:self.temp_ind,:]
            
            self.current_head = nn.Linear(
                in_features=self.hidden_dim,
                out_features=saved_weight.shape[0],
                bias=self.has_bias,
                dtype=self.weight_dtype,
                device=self.device
            )
            with torch.no_grad():
                self.current_head.weight.copy_(saved_weight)
                if self.has_bias:
                    self.current_head.bias.copy_(saved_bias)
            
            if self.weight_share and self.embedding is not None:
                self.embedding.update_weights(self.current_head.weight, self.device)

    # ================= SplitLinear strategy =================

    def _create_new_head_splitlinear(self, inds: torch.Tensor) -> None:
        """Initializes the head as a SplitLinear container with the first chunk."""
        first_head = nn.Linear(
            in_features=self.hidden_dim,
            out_features=inds.shape[0],
            bias=self.has_bias,
            dtype=self.weight_dtype,
            device=self.device
        )

        with torch.no_grad():
            first_head.weight.copy_(self._get_head_weights(inds))
            if self.has_bias:
                first_head.bias.copy_(self._get_head_biases(inds))
        
        # Initialize as SplitLinear
        self.current_head = SplitLinear([first_head])
        
        if self.weight_share and self.embedding is not None:
            self.embedding.update_weights(self.current_head.weight, device=self.device)

    def _extend_head_splitlinear(self, new_inds: torch.Tensor) -> None:
        """
        Extends the current LM head to include new vocabulary indices using a split-architecture approach.

        Args:
            new_inds (torch.Tensor): A 1D tensor containing the unique indices from the full vocabulary that need to be added to the current head.
        """
        # 1. Create a distinct Linear layer for the new indices only
        new_head_part = nn.Linear(
            in_features=self.hidden_dim,
            out_features=new_inds.shape[0],
            bias=self.has_bias,
            dtype=self.weight_dtype,
            device=self.device
        )
        
        # 2. Copy the specific weights/biases for these new indices
        with torch.no_grad():
            new_head_part.weight.copy_(self._get_head_weights(new_inds))
            if self.has_bias:
                new_head_part.bias.copy_(self._get_head_biases(new_inds))
        
        # 3. Update self.current_head        
        # If we are already using the split architecture, just append the new part
        if isinstance(self.current_head, SplitLinear):
            self.current_head.heads.append(new_head_part)
        # If it's a standard nn.Linear, wrap it into a SplitLinear.
        else:
            self.current_head = SplitLinear([self.current_head, new_head_part])
        
        # 4. Handle weight sharing compatibility
        if self.weight_share and self.embedding is not None:
            # The SplitLinear.weight property dynamically concatenates weights on access
            self.embedding.update_weights(self.current_head.weight, self.device)
    
    def _reset_head_splitlinear(self) -> None:
        """Removes the temporary heads added during the forward pass."""
        if self.temp_ind is None:
            return 

        if self.temp_ind == 0:
            self.current_inds = None
            self.current_head = None
        else:
            # Revert indices
            self.current_inds = self.current_inds[:self.temp_ind]

            # Revert heads
            if isinstance(self.current_head, SplitLinear):
                # Truncate and get the actual new size
                _ = self.current_head.truncate_to_inplace(self.temp_ind)
            
                # If only 1 head remains, unwrap it back to a standard Linear layer 
                if len(self.current_head.heads) == 1:
                    self.current_head = self.current_head.heads[0]
            
            # Update shared embedding weights
            if self.weight_share and self.embedding is not None:
                self.embedding.update_weights(self.current_head.weight, self.device)

    # ================= PreAlloc strategy =================

    def _create_new_head_buffer(self, inds: torch.Tensor) -> None:
        self.current_head = nn.Linear(
            in_features=self.hidden_dim,
            out_features=inds.shape[0] + self.buffer_size,
            bias=self.has_bias,
            dtype=self.weight_dtype,
            device=self.device
        )

        nn.init.zeros_(self.current_head.weight.data)
        if self.has_bias:
            nn.init.zeros_(self.current_head.bias.data)
        
        self.temp_ind = len(inds)
        with torch.no_grad():
            self.current_head.weight.data[:self.temp_ind] = self._get_head_weights(inds)
            if self.has_bias:
                self.current_head.bias.data[:self.temp_ind] = self._get_head_biases(inds)
        
        if self.weight_share and self.embedding is not None:
            self.embedding.update_weights(self.current_head.weight, device=self.device)
    
    def _extend_head_buffer(self, new_inds: torch.Tensor) -> None:
        dynamic_length = len(new_inds)
        
        # if new_inds exceeds buffer length, 
        # 1. delete old lm_head 
        # 2. create new linear head (n * buffer_size that fits the new_inds)
        # 3. copy static weight, delete old lm head
        # 4. assign dynamic weight
        if dynamic_length > self.buffer_size:
            new_buffer_size = self.buffer_size * (dynamic_length // self.buffer_size + 1)
            self.current_head = nn.Linear(
                in_features=self.hidden_dim,
                out_features=self.temp_ind + new_buffer_size,
                bias=self.has_bias,
                dtype=self.weight_dtype,
                device=self.device
            )

            with torch.no_grad():
                self.current_head.weight.data[:self.temp_ind] = self._get_head_weights(self.current_inds[:self.temp_ind])
                self.current_head.weight.data[self.temp_ind:(self.temp_ind+dynamic_length),:] = self._get_head_weights(new_inds)
                if self.has_bias:
                    self.current_head.bias.data[:self.temp_ind] = self._get_head_biases(self.current_inds[:self.temp_ind])
                    self.current_head.bias.data[self.temp_ind:(self.temp_ind+dynamic_length),:] = self._get_head_biases(new_inds)
            
            if self.weight_share and self.embedding is not None:
                self.embedding.update_weights(self.current_head.weight, device=self.device)
        else:
            with torch.no_grad():
                self.current_head.weight.data[self.temp_ind:(self.temp_ind+dynamic_length),:] = self._get_head_weights(new_inds)
                if self.has_bias:
                    self.current_head.bias.data[self.temp_ind:(self.temp_ind+dynamic_length),:] = self._get_head_biases(new_inds)
    
    def _reset_head_buffer(self) -> None:
        if self.temp_ind is None:
            return
        
        if self.temp_ind == 0:
            self.current_inds = None
            self.current_head = None
        else:
            self.current_inds = self.current_inds[:self.temp_ind]
            if len(self.current_inds) != self.temp_ind + self.buffer_size:
                self._create_new_head_buffer(self.current_inds)
            else:
                self.current_head[self.temp_ind:] = 0
            
            if self.weight_share and self.embedding is not None:
                self.embedding.update_weights(self.current_head.weight, self.device)


class VocabTailor:
    """
    VocabTailor wrapper for Llama-family causal LMs (input-aware vocabulary pruning).

    Supports models that use ``model.model.embed_tokens`` and ``model.lm_head``,
    including Qwen, Llama, Mistral, and DeepSeek. Qwen3 is the primary tested
    model; this class is a thin layer over OffloadEmbedding + LockedLMHead for
    use from downstream code and examples.

    Attributes:
        tracker: None by default; set when enable_metrics_tracker=True in
            from_pretrained() or load_model(). No need to set vt.tracker = None
            when using enable_metrics_tracker=False.
        gen_metrics: After generate(), holds per-run metrics (e.g. prefill_time,
            decode_tps) when tracker is enabled; None when tracker is disabled.
    """

    def __init__(self):
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer = None
        self.device: Optional[str] = None
        self.tracker: Optional[MetricsTracker] = None
        self.provider: Optional[LMDBWeightProvider] = None
        self.offload_to_lmdb: bool = False
        self.dtype = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

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
        lmdb_path: Optional[str] = None,
        vocab_resize_strategy: str = "prealloc",
        profiling_file: Optional[str] = None,
        enable_metrics_tracker: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **model_kwargs: Any,
    ) -> "VocabTailor":
        """
        Load a Llama-family model and wrap it with VocabTailor.

        Supported architectures: Qwen, Llama, Mistral, DeepSeek (any model with
        ``model.model.embed_tokens`` and ``model.lm_head``). Qwen3 is the primary
        tested model.

        Args:
            model_name_or_path (str): Hugging Face Hub model id (e.g. "Qwen/Qwen3-1.7B")
                or local path to the model directory.
            device (str): Target device for transformer blocks.
            dtype (str or torch.dtype, optional): Torch dtype or string alias.
            lmdb_path (str, optional): Path to LMDB weights (if using LMDB offload).
            vocab_resize_strategy (str): 'realloc' | 'split_linear' | 'prealloc'.
            profiling_file (str, optional): Path to a JSON task vocabulary (from vocab-tailor-build-vocab).
                If provided and the file exists, the LM head is initialized with these token IDs
                plus special tokens. JSON must be a dict (token string -> id) or a list of token ids.
                If the path does not exist, a warning is logged and no update is performed.
            enable_metrics_tracker (bool): Whether to enable metrics tracker. Defaults to False.
                When False, tracker remains None and no cleanup is needed.
            tokenizer_kwargs (dict, optional): Extra kwargs for tokenizer loading (e.g. token=, trust_remote_code=).
            **model_kwargs: Extra kwargs forwarded to AutoModelForCausalLM.from_pretrained
                (e.g. token=, use_auth_token=, trust_remote_code= for Hub or gated models).

        Returns:
            VocabTailor: Initialized wrapper ready for generate().
        """
        torch_dtype = cls._resolve_dtype(dtype)
        if dtype is not None and torch_dtype is None:
            raise ValueError(f"dtype must be one of 'bf16', 'bfloat16', 'fp16', 'float16', 'fp32', 'float32' or a torch.dtype, got {dtype!r}")

        # Initialize model
        if lmdb_path:
            # 1. Check LMDB, build a new one if not exist
            print("Checking LMDB...")
            build_lmdb_weights(model_name_or_path, lmdb_path)

            # Headless backbone: all heavy layers except embeddings/lm_head
            model = load_model_backbone(
                model_path=model_name_or_path,
                device="cpu",
                dtype=torch_dtype,
                include_embeddings=False,
                include_lm_head=False,
            )
        else:
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

        # Initialize VocabTailor
        vt = cls()
        vt.tokenizer = tokenizer
        vt.load_model(
            model = model, 
            device=device, 
            lmdb_path=lmdb_path, 
            vocab_resize_strategy=vocab_resize_strategy,
            enable_metrics_tracker=enable_metrics_tracker,
        )

        # Initialize and update LM head: with static task-specific vocab from profiling_file if given, else special tokens only.
        special_ids = vt.tokenizer.all_special_ids
        init_ids = torch.tensor(special_ids, dtype=torch.long, device="cpu")
        if profiling_file is not None:
            if not os.path.isfile(profiling_file):
                logging.getLogger(__name__).warning(f"profiling_file {profiling_file} does not exist; using special token ids only for LM head.")
            else:
                with open(profiling_file, 'r', encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    ids = list(data.values())
                elif isinstance(data, list):
                    ids = [int(x) for x in data]
                else:
                    raise ValueError(
                        "profiling_file JSON must be a dict (token->id) or list of token ids."
                    )
                init_ids = torch.tensor(
                    list(set(ids + special_ids)),
                    dtype=torch.long,
                    device="cpu",
                )
        else:
            logging.getLogger(__name__).warning("No profiling file provided. Use special token ids from the original tokenizer.")
        
        vt.update_lm_head(init_ids, temp=False)
        return vt

    # ------------------------------------------------------------------
    # Core wiring
    # ------------------------------------------------------------------

    def load_model(
        self,
        model: AutoModelForCausalLM,
        device: str = "cuda",
        lmdb_path: Optional[str] = None,
        vocab_resize_strategy: str = "prealloc",
        enable_metrics_tracker: bool = False,
    ) -> None:
        """
        Hybrid-loads a model by moving transformer blocks to GPU and offloading embeddings/heads to LMDB or CPU.

        Args:
            model (AutoModelForCausalLM): The Hugging Face model instance. Can be a full model or a 'headless' shell loaded via `init_empty_weights`.
            device (str): The target device for transformer blocks (e.g., 'cuda', 'cpu'). Defaults to 'cuda'.
            lmdb_path (str, optional): Path to the LMDB database containing full vocabulary weights.
                If provided, the model will use LMDB-backed dynamic offloading.
                If None, it wraps existing CPU weights.
            vocab_resize_strategy (str): Strategy for handling the local sub-vocabulary buffer.
                Values include "realloc", "split_linear", "prealloc". Defaults to "prealloc".
            enable_metrics_tracker (bool): Whether to enable metrics tracker. Defaults to False.
                When False, tracker remains None.
        """
        self.model = model
        self.device = device
        self.provider = None
        self.offload_to_lmdb = lmdb_path is not None
        self.dtype = model.dtype
        self.tracker = MetricsTracker() if enable_metrics_tracker else None

        # Disable HF weight tying before swapping modules
        self.model.config.tie_word_embeddings = False
        
        # Move Transformer Blocks to target device
        print(f"Move transformer blocks to {device}...")

        if lmdb_path:
            # Move transformer blocks to device (embeddings + lm_head stay meta/CPU)
            for name, module in self.model.named_children():
                if name == "lm_head":
                    continue

                if name == "model":
                    for sub_name, sub_module in module.named_children():
                        if sub_name == "embed_tokens":
                            continue
                        self._move_to_device(sub_module, device)
                else:
                    self._move_to_device(module, device)

            # Initialize LMDB provider
            self.provider = LMDBWeightProvider(lmdb_path, self.tracker, fetch_dtype=self.dtype)
            self.weight_share = self.provider.weight_shared

            # Initialize offloaded embedding
            offloaded_embedding = OffloadEmbedding(embedding=None, provider=self.provider, weight_share=self.weight_share)

            # Initialize locked LM head
            locked_lm_head = LockedLMHead(
                device=device,
                full_lm_head=None,
                provider=self.provider,
                embedding=offloaded_embedding,
                weight_share=self.weight_share,
                vocab_resize_strategy=vocab_resize_strategy,
            )
        else:
            # Regular mode: transformer on device, embeddings/head on CPU
            for name, module in self.model.named_modules():
                if name != "lm_head" and name != "model.embed_tokens":
                    module.to(device)
                else:
                    module.to("cpu")

            self.weight_share = check_weight_sharing(model)

            # Initialize offloaded embedding
            offloaded_embedding = OffloadEmbedding(
                embedding=self.model.get_input_embeddings(),
                provider=None,
                weight_share=self.weight_share,
            )

            # Initialize locked LM head
            locked_lm_head = LockedLMHead(
                device=device,
                full_lm_head=self.model.get_output_embeddings(),
                provider=None,
                embedding=offloaded_embedding,
                weight_share=self.weight_share,
                tracker=self.tracker,
                vocab_resize_strategy=vocab_resize_strategy,
            )

        # Connect layers
        offloaded_embedding.link_to_lm_head(locked_lm_head)

        # Inject into model
        self.model.set_input_embeddings(offloaded_embedding)
        self.model.set_output_embeddings(locked_lm_head)
        self.model.model.embed_tokens = offloaded_embedding
        self.model.lm_head = locked_lm_head

    def _move_to_device(self, module: nn.Module, device: str) -> None:
        is_meta = any(p.device.type == "meta" for p in module.parameters())
        if is_meta:
            module.to_empty(device=device)
        else:
            module.to(device)

    def generate(
        self,
        inputs_ids: torch.Tensor,
        *,
        mode: str = "input_aware",
        do_sample: bool = False,
        max_new_tokens: int = 128,
        original_eos_token_id: Optional[int] = None,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        """
        Run generation with optional input-aware pruning.

        Args:
            inputs_ids (torch.Tensor or BatchEncoding or dict): Token ids of shape (batch_size, seq_len), or a BatchEncoding/dict
                with an "input_ids" key (e.g. from tokenizer.apply_chat_template(..., return_tensors="pt")).
                If not a tensor, input_ids is extracted automatically.
            mode (str, optional): 'input_aware' to apply input-aware pruning. Defaults to "input_aware".
            do_sample (bool, optional): Whether to sample. Defaults to False.
            max_new_tokens (int, optional): Maximum new tokens to generate. Defaults to 128.
            original_eos_token_id (int, optional): EOS token id in original vocab for stopping.
            **generate_kwargs: Forwarded to model.generate() (e.g. temperature, top_p).

        Returns:
            torch.Tensor: Generated token ids in original vocabulary space (prompt + generated).
            Per-run metrics (prefill_time, decode_tps, etc.) are stored in self.gen_metrics
            when the tracker is enabled; self.gen_metrics is None otherwise.
        """
        # 1. Setup Generation
        self.gen_metrics, LMDB_METRICS = None, None
        if self.tracker:
            self.gen_metrics, LMDB_METRICS = {}, []
            if self.offload_to_lmdb:
                LMDB_METRICS = [
                    f"lmdb_{prefix}_{name}"
                    for name in ["time", "calls", "tokens"]
                    for prefix in ["emb", "head"]
                ]
                self.gen_metrics = {k: self.tracker.__dict__[f"total_{k}"] for k in LMDB_METRICS}

        inputs_embeds = self.model.model.embed_tokens.full_embedding(inputs_ids).detach()
        new_eos_token_id = (self.model.lm_head.current_inds == original_eos_token_id).nonzero()
        rep_pen_processor = RepetitionPenaltyLogitsProcessor(
            penalty=1.1,
            prompt_ignore_length=inputs_ids.shape[-1],
        )

        # 2. Setup Timer
        timing_streamer: Optional[TimingStreamer] = None
        if self.tracker:
            input_tokens_count = inputs_ids.numel()
            timing_streamer = TimingStreamer()

        # 3. Input-aware Pruning
        if mode == "input_aware":
            self.input_aware_pruning(inputs_ids)

        if self.tracker:
            end_dynamic_loading = time.perf_counter()

        # 4. Generation
        output_ids = self.model.generate(
            inputs=inputs_ids.to(self.device),
            inputs_embeds=inputs_embeds.to(self.device),
            do_sample=do_sample,
            eos_token_id=new_eos_token_id,
            max_new_tokens=max_new_tokens,
            logits_processor=[rep_pen_processor],
            streamer=timing_streamer,
            **generate_kwargs,
        )

        # 5. Calculate tracker metrics
        if self.tracker and timing_streamer is not None:
            gen_end = time.perf_counter()
            if timing_streamer.first_token_time is None:
                timing_streamer.first_token_time = gen_end

            prefill_dt = timing_streamer.first_token_time - timing_streamer.start_time
            decode_dt = gen_end - timing_streamer.first_token_time
            dynamic_loading_dt = end_dynamic_loading - timing_streamer.start_time

            if self.offload_to_lmdb:
                for k in LMDB_METRICS:
                    curr_val = self.tracker.__dict__[f"total_{k}"]
                    self.gen_metrics[k] = curr_val - self.gen_metrics[k]

            output_tokens_count = output_ids[0][len(inputs_ids[0]) :].numel()

        # 6. Map generated indices back to original vocab and concat with prompt (for decoding)
        if self.model.lm_head.current_inds is not None:
            result = torch.cat((inputs_ids, self.model.lm_head.current_inds[output_ids[0][len(inputs_ids[0]):].to('cpu')].view(1,-1)), dim=1)
        else:
            result = output_ids

        if mode == "input_aware":
            self.model.lm_head.reset_head()

        # 7. Update tracker
        if self.tracker:
            self.tracker.total_prefill_time += prefill_dt
            self.tracker.total_decode_time += decode_dt
            self.tracker.total_prefill_tokens += input_tokens_count
            self.tracker.total_decode_tokens += output_tokens_count
            self.tracker.total_dynamic_loading_time += dynamic_loading_dt

            if self.offload_to_lmdb:
                for prefix in ["lmdb_emb", "lmdb_head"]:
                    self.gen_metrics[f"{prefix}_tps"] = _safe_div(self.gen_metrics[f"{prefix}_tokens"], self.gen_metrics[f"{prefix}_time"])
                    self.gen_metrics[f"{prefix}_latency"] = _safe_div(self.gen_metrics[f"{prefix}_time"], self.gen_metrics[f"{prefix}_calls"])

            self.gen_metrics.update(
                {
                    "dynamic_loading_time": dynamic_loading_dt,
                    "prefill_time": prefill_dt,
                    "decode_time": decode_dt,
                    "prefill_tokens": input_tokens_count,
                    "decode_tokens": output_tokens_count,
                    "prefill_tps": _safe_div(input_tokens_count, prefill_dt),
                    "decode_tps": _safe_div(output_tokens_count, decode_dt),
                }
            )

        return result

    def input_aware_pruning(self, inputs_ids: torch.Tensor) -> None:
        sorted_inputs, _ = torch.sort(torch.unique(inputs_ids))
        self.update_lm_head(sorted_inputs, temp=True)

    def update_lm_head(self, inds: torch.Tensor, temp: bool = False) -> None:
        self.model.lm_head.update_inds(inds, temp)

    def reset(self) -> None:
        self.model.lm_head.current_inds = None
        self.model.lm_head.current_head = None
        self.model.temp_ind = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["VocabTailor", "OffloadEmbedding", "LockedLMHead"]

