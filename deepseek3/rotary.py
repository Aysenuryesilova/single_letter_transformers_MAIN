"""Rotary Position Embeddings (RoPE).

RoPE encodes position by rotating pairs of features by an angle that grows with
position. There is nothing to learn, so these are just two small functions:

  * precompute_cos_sin -> build the cos/sin rotation tables once
  * apply_rotary       -> rotate a query or key tensor using those tables
"""

import torch


def precompute_cos_sin(head_dim: int, max_seq_len: int, theta: float):
    """Return cos, sin tables of shape [max_seq_len, head_dim]."""
    # One frequency per feature pair: 1 / theta^(2i/head_dim) for i = 0, 2, 4, ...
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()        # [max_seq_len]
    angles = torch.outer(positions, inv_freq)            # [max_seq_len, head_dim/2]
    angles = torch.cat([angles, angles], dim=-1)         # [max_seq_len, head_dim]
    return angles.cos(), angles.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Turn [x1, x2] (split in half) into [-x2, x1]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape [B, n_heads, T, head_dim].

    cos/sin are [T, head_dim]; we add head dims so they broadcast.
    """
    cos = cos[None, None, :, :]   # [1, 1, T, head_dim]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin
