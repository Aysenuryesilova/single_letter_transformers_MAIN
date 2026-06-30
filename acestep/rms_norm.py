"""RMSNorm — the normalization Qwen3 uses everywhere instead of LayerNorm.

RMSNorm rescales a vector by its root-mean-square (no mean subtraction, no bias).
The math is done in float32 for stability, then cast back to the input dtype,
exactly like Qwen3RMSNorm in HuggingFace transformers.
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # learnable per-feature scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()                                   # compute in float32
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)              # scale and cast back
