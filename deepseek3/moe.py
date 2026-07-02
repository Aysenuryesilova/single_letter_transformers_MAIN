"""DeepSeek-V3-style Mixture of Experts with aux-loss-free load balancing.

Instead of one MLP per block, `n_routed_experts` small MLPs. For each token a
router picks the `top_k` best-matching experts, and only those run — so the
layer holds many parameters but each token only pays for a few ("sparse").
A small *shared* expert always runs, catching whatever is common to all tokens.

Routing, exactly the V3 recipe:

  1. scores  = sigmoid(router(x))          — affinity of the token to each expert
  2. pick top_k by  scores + expert_bias   — bias used for SELECTION only
  3. gate weights = the raw scores of the chosen experts, normalized to sum to 1
     (the bias never touches the output — it only steers who gets picked)

The `expert_bias` is the aux-loss-free balancer: after each training step, the
bias of every OVERloaded expert is nudged down and every UNDERloaded expert up
(by `bias_update_speed`). Busy experts become slightly less attractive to the
router, so load evens out without an auxiliary loss term fighting the LM loss.

Simplified here: real DeepSeek-V3 has 256 routed experts with node-limited
routing across GPUs, plus a tiny sequence-wise balance loss; the expert loop
below is plain Python instead of a fused kernel.
"""

import torch
from torch import nn

from config import ModelConfig
from mlp import MLP


class MoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.top_k = cfg.top_k
        self.n_experts = cfg.n_routed_experts
        self.bias_update_speed = cfg.bias_update_speed

        self.router = nn.Linear(cfg.hidden_size, cfg.n_routed_experts, bias=False)
        self.experts = nn.ModuleList(
            [MLP(cfg.hidden_size, cfg.moe_intermediate_size) for _ in range(cfg.n_routed_experts)]
        )
        # n_shared_experts always-on experts, fused into one wider MLP.
        self.shared_expert = MLP(cfg.hidden_size,
                                 cfg.n_shared_experts * cfg.moe_intermediate_size)

        # The load-balancing bias. A buffer, not a parameter: gradients never
        # touch it — it is updated by the sign rule below.
        self.register_buffer("expert_bias", torch.zeros(cfg.n_routed_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        flat = x.reshape(B * T, H)                       # route token by token

        scores = torch.sigmoid(self.router(flat))        # [N, E] affinities in (0,1)
        _, top_idx = (scores + self.expert_bias).topk(self.top_k, dim=-1)   # selection
        weights = scores.gather(-1, top_idx)             # gate with UNbiased scores
        weights = weights / weights.sum(-1, keepdim=True)  # normalize over the chosen k

        out = self.shared_expert(flat)                   # the always-on expert
        for e, expert in enumerate(self.experts):
            token, slot = (top_idx == e).nonzero(as_tuple=True)  # who chose expert e
            if token.numel():
                out[token] += weights[token, slot, None] * expert(flat[token])

        if self.training:
            self._update_bias(top_idx)
        return out.reshape(B, T, H)

    @torch.no_grad()
    def _update_bias(self, top_idx: torch.Tensor):
        """Aux-loss-free balancing: nudge busy experts down, idle experts up."""
        load = torch.bincount(top_idx.flatten(), minlength=self.n_experts).float()
        self.load = load / load.sum()                    # fraction of slots each expert got
        target = 1.0 / self.n_experts                    # perfectly even share
        self.expert_bias += self.bias_update_speed * torch.sign(target - self.load)
