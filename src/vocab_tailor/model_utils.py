import glob
import os
from typing import Optional

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM


def check_weight_sharing(model: AutoModelForCausalLM) -> bool:
    """Check if the model uses weight tying between input embeddings and the LM head."""
    # 1. Safely get input embeddings
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        print("Could not find input embeddings.")
        return False
    
    embedding_weights = input_embeddings.weight

    # 2. Find the output head
    # Many HF models use model.lm_head, but some use model.output
    output_layer = getattr(model, 'lm_head', None) or getattr(model, 'output', None)

    if output_layer is None:
        # Some models (like GPT-2) store it under the base model or specific names
        # Check if the weight is tied via the config attribute as a fallback
        is_tied_config = getattr(model.config, "tie_word_embeddings", False)
        print(f"Could not find 'lm_head' or 'output' attribute. Config 'tie_word_embeddings': {is_tied_config}")
        return is_tied_config
    
    output_weights = output_layer.weight

    # 3. Check memory identity
    if embedding_weights is output_weights:
        print("Weight sharing is ENABLED between the input and output embedding layers.")
        return True
    else:
        print("Weight sharing is NOT enabled between the input and output embedding layers.")
        return False


def check_model_weights_dtype_and_device(model: AutoModelForCausalLM) -> None:
    """Helper to check weights dtype, device, and shape in different model components."""
    print(f"{'name':<50} {'dtype':<20} {'device':<10} {'shape':<30}")
    print('-'*120)
    for name, module in model.named_modules():
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            p = module.weight
            print(f"{name:<50} {str(p.dtype):<20} {str(p.device):<10} {str(p.shape):<30}")
        
        if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor):
            p = module.bias
            print(f"{name:<50} {str(p.dtype):<20} {str(p.device):<10} {str(p.shape):<30}")


def load_model_backbone(
    model_path: str,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
    include_embeddings: bool = False,
    include_lm_head: bool = False,
) -> AutoModelForCausalLM:
    """
    Load a HF model (local or Hub), optionally skipping embedding and lm_head weights to save memory.

    Args:
        model_path (str): Local directory containing model weights, or Hugging Face Hub model ID (e.g. "meta-llama/Llama-3.2-3B").
        device (str): Target device for transformer weights.
        dtype (torch.dtype, optional): Target dtype for loaded weights. If None, config default is used.
        include_embeddings (bool): If False, skip input embedding weights.
        include_lm_head (bool): If False, skip lm_head/output head weights.
    
    Returns:
        AutoModelForCausalLM: Model with transformer weights loaded, embeddings & lm_head absent.
    """
    # 1. Load config only (support Hub ID or local path)
    if os.path.isdir(model_path):
        local_path = model_path
    else:
        local_path = snapshot_download(model_path)

    # 1. Load config only (from Hub or local)
    print(f"Loading config from {model_path}...")
    config = AutoConfig.from_pretrained(model_path)
    config.torch_dtype = dtype

    # 2. Initialize model on meta (NO weights allocated, also 0 RAM)
    print("Initializing empty model shell...")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)

    # 3. Locate local checkpoint files (handle sharded or single files)
    #    We check for safetensors first, then pytorch_model.bin
    files = glob.glob(os.path.join(local_path, "*.safetensors"))
    use_safetensors = True

    if not files:
        files = glob.glob(os.path.join(local_path, "pytorch_model.bin*"))
        use_safetensors = False

    if not files:
        raise ValueError("No checkpoint files (.safetensors or .bin) found in the path.")

    print(f"Found {len(files)} checkpoint file(s). Loading weights...")

    # 4. Specify skipped layers and load the rest
    layer_map = {
        "embeddings": ["embed_tokens", "wte", "wpe", "word_embeddings"],
        "lm_head": ["lm_head", "output_projection", "embed_out"],
    }
    skipped_layers: list[str] = []
    if not include_embeddings:
        skipped_layers += layer_map["embeddings"]
    if not include_lm_head:
        skipped_layers += layer_map["lm_head"]

    for file in sorted(files):
        # Load the raw state dictionary from disk
        if use_safetensors:
            state_dict = load_file(file)
        else:
            state_dict = torch.load(file, map_location="cpu")

        # Filter and load parameters
        for param_name, param_tensor in state_dict.items():
            if any(skip in param_name for skip in skipped_layers):
                continue

            if dtype is not None:
                param_tensor = param_tensor.to(dtype=dtype)

            set_module_tensor_to_device(
                model,
                param_name,
                device=device,
                value=param_tensor,
            )
        
        # cleanup to save RAM before loading next file
        del state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.tie_weights()
    model.eval()
    return model


__all__ = ["check_weight_sharing", "load_model_backbone"]

