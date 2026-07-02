"""A DeepSeek-V3 transformer block: pre-norm, MLA mixer, dense-or-MoE MLP.

Same pre-norm skeleton as Qwen3 —

    x = x + mla(norm(x))
    x = x + ffn(norm(x))

— but the token mixer is MLA, and the feed-forward is a big dense MLP in the
first `first_dense_layers` layers and a sparse MoE everywhere else (DeepSeek
keeps the early layers dense because they are the hardest to load-balance).
"""

import torch
from torch import nn

from config import ModelConfig
from rms_norm import RMSNorm
from mla import MLA
from mlp import MLP
from moe import MoE


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = MLA(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        # The only per-layer difference: dense MLP early, MoE after.
        if layer_idx < cfg.first_dense_layers:
            self.mlp = MLP(cfg.hidden_size, cfg.intermediate_size)
        else:
            self.mlp = MoE(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
