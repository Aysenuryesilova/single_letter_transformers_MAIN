"""Causal self-attention with the three Qwen3 twists:

  1. QK-Norm  : RMSNorm applied to each head's query and key vectors.
  2. RoPE     : rotary position embedding applied to query and key.
  3. GQA      : fewer key/value heads than query heads (saves memory).

The actual attention computation uses PyTorch's scaled_dot_product_attention
with is_causal=True, so we never have to build a mask by hand.
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import ModelConfig
from rms_norm import RMSNorm
from rotary import apply_rotary


def repeat_kv(x: torch.Tensor, n_repeat: int) -> torch.Tensor:
    """Expand KV heads so each query head has a matching KV head (GQA).

    x: [B, num_kv_heads, T, head_dim] -> [B, num_kv_heads * n_repeat, T, head_dim]
    """
    if n_repeat == 1:
        return x
    return x.repeat_interleave(n_repeat, dim=1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.n_repeat = cfg.num_heads // cfg.num_kv_heads

        # Projections (no bias, as in Qwen3).
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)

        # QK-Norm: normalize each head's query/key vector (length head_dim).
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Project, then split into heads: [B, T, n_heads, head_dim] -> [B, n_heads, T, head_dim]
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK-Norm, then RoPE (order matters: Qwen3 normalizes before rotating).
        q = apply_rotary(self.q_norm(q), cos, sin)
        k = apply_rotary(self.k_norm(k), cos, sin)

        # GQA: grow KV heads to match the number of query heads.
        k = repeat_kv(k, self.n_repeat)
        v = repeat_kv(v, self.n_repeat)

        # Causal attention. scaling defaults to 1/sqrt(head_dim).
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Merge heads back together and project out.
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)
