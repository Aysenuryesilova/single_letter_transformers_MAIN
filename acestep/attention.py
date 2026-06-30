"""Attention for ACE-Step's two transformers.

Two flavours, both built from the same Qwen3 recipe (GQA + QK-Norm + RoPE):

  * Attention      : self-attention. ``is_causal=True`` for the planner LM
                     (a token may only see the past), ``is_causal=False`` for
                     the DiT (every latent frame may see every other frame —
                     a song's timing needs to look both ways).
  * CrossAttention : the latent *asks questions of* the conditioning. Queries
                     come from the DiT latent; keys/values come from the
                     conditioning sequence (tag embedding + FSQ source latent).
                     No RoPE and no causal mask here — the conditioning is a
                     small unordered context, not a time axis.
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import AceConfig
from rms_norm import RMSNorm
from rotary import apply_rotary


def repeat_kv(x: torch.Tensor, n_repeat: int) -> torch.Tensor:
    """Expand KV heads so each query head has a matching KV head (GQA)."""
    if n_repeat == 1:
        return x
    return x.repeat_interleave(n_repeat, dim=1)


class Attention(nn.Module):
    """Self-attention with QK-Norm + RoPE + GQA (Qwen3 style)."""

    def __init__(self, cfg: AceConfig, is_causal: bool = True):
        super().__init__()
        self.is_causal = is_causal
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.n_repeat = cfg.num_heads // cfg.num_kv_heads

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)

        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK-Norm then RoPE (Qwen3 order).
        q = apply_rotary(self.q_norm(q), cos, sin)
        k = apply_rotary(self.k_norm(k), cos, sin)

        k = repeat_kv(k, self.n_repeat)
        v = repeat_kv(v, self.n_repeat)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)

        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)


class CrossAttention(nn.Module):
    """Latent queries attend to a conditioning context (no RoPE, no mask)."""

    def __init__(self, cfg: AceConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim

        # Queries from the latent x; keys/values from the conditioning context.
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)

        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape          # T  = latent frames (queries)
        S = context.shape[1]       # S  = conditioning length (keys/values)

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)
