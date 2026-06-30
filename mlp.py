"""SwiGLU feed-forward network, the MLP block used by Qwen3.

Instead of a single hidden layer, SwiGLU uses two parallel projections:
a "gate" (passed through SiLU) and an "up". Their elementwise product is
projected back down. This gating tends to learn better than a plain MLP.
"""

from torch import nn
import torch.nn.functional as F

from config import ModelConfig


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
