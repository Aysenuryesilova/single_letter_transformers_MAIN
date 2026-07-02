"""The full tiny DeepSeek-V3-style language model.

Flow:  token ids -> embeddings -> N blocks (MLA + dense/MoE) -> final RMSNorm
       -> tied linear head -> logits over the vocabulary.

Identical skeleton to the Qwen3 folder; only the parts inside the blocks
differ. Note the RoPE tables here are only `qk_rope_head_dim` wide — position
lives solely in MLA's small decoupled rope dims.
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import ModelConfig
from rms_norm import RMSNorm
from block import TransformerBlock
from rotary import precompute_cos_sin


class TinyDeepSeek(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg, layer_idx=i) for i in range(cfg.num_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight    # weight tying

        # RoPE tables only span the decoupled rope dims (see mla.py).
        cos, sin = precompute_cos_sin(cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        cos, sin = self.cos[:T], self.sin[:T]

        x = self.embed_tokens(idx)
        for layer in self.layers:
            x = layer(x, cos, sin)
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
