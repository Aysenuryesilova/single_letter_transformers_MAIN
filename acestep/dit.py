"""The DiT renderer — the coral "diffusion" region.

A small Diffusion Transformer that works on the 25Hz latent. It does not predict
the latent directly; it predicts the flow-matching *velocity* that drags noise
toward the target latent (see flow.py). Each block does three things:

    x = x + self_attention(x)        # latent frames talk to each other (time coherence)
    x = x + cross_attention(x, cond) # the latent reads the conditioning (tag + 5Hz skeleton)
    x = x + mlp(x)

The conditioning context is the tag embedding concatenated with the FSQ "source
latent" (projected to tokens) — i.e. the blue bridge's two arms joined. The
diffusion timestep is folded in as an additive embedding, the simplest way to
tell the network "how noisy is the input right now".
"""

import torch
from torch import nn

from config import AceConfig
from rms_norm import RMSNorm
from attention import Attention, CrossAttention
from mlp import MLP
from rotary import precompute_cos_sin

# --- Real ACE-Step v1.5, for comparison ------------------------------------
# The real DiT is a ~2B-parameter hybrid-attention Transformer: *odd* layers use
# Sliding-Window Attention, *even* layers use Global Grouped-Query Attention, and
# caption conditioning comes from Qwen3-0.6B embeddings via cross-attention. A
# patchify layer first halves the 25Hz latent to 12.5Hz for throughput. (The XL
# model is ~4B.) Here (toy): a few full self-attention + cross-attention blocks,
# no sliding window and no patchify; the latent stays at the toy "25Hz".
# ---------------------------------------------------------------------------


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal embedding of a scalar timestep t in [0, 1]. -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=t.device).float() / half * 4.0)
    angles = t[:, None].float() * freqs[None, :] * 10.0
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


class DiTBlock(nn.Module):
    def __init__(self, cfg: AceConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg, is_causal=False)        # bidirectional over time
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.cross_attn = CrossAttention(cfg)
        self.norm3 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, context, cos, sin):
        x = x + self.self_attn(self.norm1(x), cos, sin)
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.mlp(self.norm3(x))
        return x


class DiT(nn.Module):
    def __init__(self, cfg: AceConfig):
        super().__init__()
        self.cfg = cfg

        self.in_proj = nn.Linear(cfg.latent_dim, cfg.hidden_size)    # latent frame -> token
        self.cond_proj = nn.Linear(cfg.latent_dim, cfg.hidden_size)  # source-latent frame -> token
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size), nn.SiLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
        )
        self.blocks = nn.ModuleList([DiTBlock(cfg) for _ in range(cfg.dit_layers)])
        self.norm_out = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.latent_dim)   # token -> velocity frame

        cos, sin = precompute_cos_sin(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, noisy: torch.Tensor, t: torch.Tensor,
                text_embed: torch.Tensor, source_latent: torch.Tensor) -> torch.Tensor:
        """noisy [B, d, 16], t [B], text_embed [B, 1, h], source_latent [B, d, 16]
        -> predicted velocity [B, d, 16]."""
        B, d, T = noisy.shape
        cos, sin = self.cos[:T], self.sin[:T]

        x = self.in_proj(noisy.transpose(1, 2))                      # [B, T, h]
        x = x + self.time_mlp(timestep_embedding(t, self.cfg.hidden_size))[:, None, :]

        # Conditioning context = tag token  ++  the 5Hz skeleton as tokens.
        cond_tokens = self.cond_proj(source_latent.transpose(1, 2))  # [B, T, h]
        context = torch.cat([text_embed, cond_tokens], dim=1)        # [B, 1+T, h]

        for block in self.blocks:
            x = block(x, context, cos, sin)

        v = self.out_proj(self.norm_out(x))                          # [B, T, d]
        return v.transpose(1, 2)                                     # [B, d, T]
