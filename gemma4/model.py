"""The full tiny Gemma-style language model.

Gemma-specific pieces wired together here:
  * input embeddings scaled by sqrt(hidden_size)
  * two RoPE tables (small base for local layers, large base for global)
  * two attention masks (sliding-window for local, plain causal for global)
  * a layer is global on every `global_every`-th layer, otherwise local
  * Per-Layer Embeddings: a small extra embedding added after each layer
  * final RMSNorm + tied lm_head
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from config import ModelConfig
from rms_norm import RMSNorm
from block import TransformerBlock
from rotary import precompute_cos_sin


def build_mask(T: int, window: int | None, device=None) -> torch.Tensor:
    """Additive float mask [T, T]: 0.0 where attention is allowed, -inf where blocked.

    window=None -> causal only.  window=w -> causal AND within the last w tokens.
    """
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    allowed = j <= i                       # causal: can't look at the future
    if window is not None:
        allowed = allowed & (i - j < window)   # sliding window: only recent tokens
    mask = torch.zeros(T, T, device=device)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask


class TinyGemma(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.embed_scale = math.sqrt(cfg.hidden_size)   # Gemma scales embeddings up

        # Per-Layer Embeddings: one extra hidden-size vector per (token, layer).
        self.per_layer_embeddings = nn.Embedding(cfg.vocab_size, cfg.num_layers * cfg.hidden_size)

        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.layer_is_global = [(i + 1) % cfg.global_every == 0 for i in range(cfg.num_layers)]

        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight    # weight tying

        # Two RoPE tables: local (small base) and global (large base).
        cos_l, sin_l = precompute_cos_sin(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta_local)
        cos_g, sin_g = precompute_cos_sin(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta_global)
        self.register_buffer("cos_local", cos_l, persistent=False)
        self.register_buffer("sin_local", sin_l, persistent=False)
        self.register_buffer("cos_global", cos_g, persistent=False)
        self.register_buffer("sin_global", sin_g, persistent=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape

        x = self.embed_tokens(idx) * self.embed_scale            # [B, T, H]
        ple = self.per_layer_embeddings(idx).view(B, T, self.cfg.num_layers, self.cfg.hidden_size)

        # Build the two masks once for this sequence length.
        local_mask = build_mask(T, self.cfg.sliding_window, x.device)
        global_mask = build_mask(T, None, x.device)

        for i, layer in enumerate(self.layers):
            if self.layer_is_global[i]:
                cos, sin, mask = self.cos_global[:T], self.sin_global[:T], global_mask
            else:
                cos, sin, mask = self.cos_local[:T], self.sin_local[:T], local_mask
            x = layer(x, cos, sin, mask)
            x = x + ple[:, :, i, :]                              # add per-layer embedding

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = None,
                 eos_id: int = None) -> torch.Tensor:
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if eos_id is not None:
                next_id[finished] = eos_id                       # keep finished rows on eos
                finished = finished | (next_id.squeeze(1) == eos_id)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and bool(finished.all()):
                break                                            # everyone hit eos
        return idx
