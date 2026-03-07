from typing import Optional

import torch
from torch import nn


class SplitLinear(nn.Module):
    """
    Composite Linear layer that concatenates outputs from multiple Linear heads.
    """

    def __init__(self, heads: list[nn.Module]):
        super().__init__()
        self.heads = nn.ModuleList(heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(x) for head in self.heads], dim=-1)

    def truncate_to(self, target_out_features: int) -> int:
        current_size = 0
        keep_heads = []

        for head in self.heads:
            if current_size + head.out_features <= target_out_features:
                keep_heads.append(head)
                current_size += head.out_features
            else:
                break

        if current_size != target_out_features:
            print(
                "[SplitLinear Warning] Truncation mismatch. "
                f"Target: {target_out_features}, Result: {current_size}. "
                "Dropping partial head is not supported."
            )

        self.heads = nn.ModuleList(keep_heads)
        return current_size

    def truncate_to_inplace(self, target_out_features: int) -> int:
        current_size = 0
        keep_count = 0

        for head in self.heads:
            if current_size + head.out_features <= target_out_features:
                current_size += head.out_features
                keep_count += 1
            else:
                break

        if current_size != target_out_features:
            print(
                "[SplitLinear Warning] Truncation mismatch. "
                f"Target: {target_out_features}, Result: {current_size}. "
                "Dropping partial head is not supported."
            )

        if keep_count < len(self.heads):
            del self.heads[keep_count:]

        return current_size

    @property
    def weight(self) -> torch.Tensor:
        return torch.cat([h.weight for h in self.heads], dim=0)

    @property
    def bias(self) -> Optional[torch.Tensor]:
        if len(self.heads) == 0 or self.heads[0].bias is None:
            return None
        return torch.cat([h.bias for h in self.heads], dim=0)

    @property
    def out_features(self) -> int:
        return sum(h.out_features for h in self.heads)


__all__ = ["SplitLinear"]

