"""SwiGLU feed-forward network — used three ways in this folder.

The dense first layer uses one big MLP; each routed expert in the MoE layers is
a small MLP; and the always-on shared expert is another small MLP. Same module,
different `intermediate_size`.
"""

from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
