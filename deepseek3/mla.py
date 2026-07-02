"""MLA — Multi-head Latent Attention, DeepSeek's compressed-KV attention.

Ordinary attention projects the hidden state straight into per-head K and V,
and at inference you must cache all of them: [T, num_heads * (k_dim + v_dim)]
numbers per layer. MLA inserts a bottleneck:

    x --W_down--> kv_latent (tiny, kv_lora_rank wide) --W_up--> all heads' K and V

Only the tiny latent would ever need caching, yet every head still gets its own
K/V (unlike GQA, which makes heads share). Queries get the same low-rank
treatment (down, norm, up) purely to save parameters.

One wrinkle: RoPE rotates K, but a rotation cannot pass through the W_up matrix
(it would have to be re-applied for every cached token). DeepSeek's fix is
"decoupled RoPE": the compressed dims carry NO position ("nope"), and a few
extra dims — computed directly from x, one copy shared by all heads — carry the
rotation. Query and key are the concatenation [nope | rope].

Simplified here: at inference DeepSeek never materializes K/V — W_up is
"absorbed" into the query/output projections and attention runs against the
latent itself. That trick changes no math, so we keep the readable form.
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import ModelConfig
from rms_norm import RMSNorm
from rotary import apply_rotary


class MLA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        self.v_head_dim = cfg.v_head_dim
        # Scale by the FULL q/k dim (nope + rope), as in DeepSeek.
        self.scale = self.qk_head_dim ** -0.5

        # Queries: compress, normalize, re-expand to all heads' [nope | rope].
        self.q_down = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps)
        self.q_up = nn.Linear(cfg.q_lora_rank, cfg.num_heads * self.qk_head_dim, bias=False)

        # Keys/values: compress to the KV latent (the would-be cache),
        # normalize, re-expand to all heads' [k_nope | v].
        self.kv_down = nn.Linear(cfg.hidden_size, cfg.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(cfg.kv_lora_rank, cfg.rms_norm_eps)
        self.kv_up = nn.Linear(cfg.kv_lora_rank,
                               cfg.num_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
                               bias=False)

        # Decoupled RoPE key: straight from x, ONE copy shared by every head.
        self.k_rope_proj = nn.Linear(cfg.hidden_size, cfg.qk_rope_head_dim, bias=False)

        self.o_proj = nn.Linear(cfg.num_heads * cfg.v_head_dim, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H = self.num_heads

        # Queries -> [B, H, T, qk_head_dim], split into no-position / rope parts.
        q = self.q_up(self.q_norm(self.q_down(x))).view(B, T, H, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_rope = apply_rotary(q_rope, cos, sin)

        # KV latent -> every head's k_nope and v.
        kv = self.kv_up(self.kv_norm(self.kv_down(x)))
        kv = kv.view(B, T, H, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Shared RoPE key: [B, 1, T, rope_dim], broadcast over heads.
        k_rope = self.k_rope_proj(x).view(B, T, 1, self.qk_rope_head_dim).transpose(1, 2)
        k_rope = apply_rotary(k_rope, cos, sin).expand(B, H, T, self.qk_rope_head_dim)

        # Assemble [nope | rope] and attend causally.
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)

        out = out.transpose(1, 2).reshape(B, T, H * self.v_head_dim)
        return self.o_proj(out)
