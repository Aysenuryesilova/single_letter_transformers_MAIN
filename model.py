"""The full tiny Qwen3-style language model.

Flow:  token ids -> embeddings -> N transformer blocks -> final RMSNorm
       -> linear head -> logits over the vocabulary.

The output head shares its weights with the input embedding ("weight tying"),
a standard trick that saves parameters and usually helps small models.
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import ModelConfig
from rms_norm import RMSNorm
from block import TransformerBlock
from rotary import precompute_cos_sin


class TinyQwen(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # Weight tying: the head reuses the embedding matrix.
        self.lm_head.weight = self.embed_tokens.weight

        # RoPE tables are fixed, so precompute them once and store as buffers
        # (buffers move with .to(device) but are not trained).
        cos, sin = precompute_cos_sin(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """idx: [B, T] of token ids. Returns (logits, loss).

        loss is None unless targets ([B, T]) are provided.
        """
        B, T = idx.shape
        cos, sin = self.cos[:T], self.sin[:T]   # slice RoPE tables to current length

        x = self.embed_tokens(idx)              # [B, T, hidden]
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)                # [B, T, vocab]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),  # [B*T, vocab]
                targets.reshape(-1),                  # [B*T]
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """Autoregressively extend idx [B, T] by max_new_tokens tokens."""
        for _ in range(max_new_tokens):
            # Never feed more than max_seq_len tokens of context.
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # logits for the next token

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # sample one token
            idx = torch.cat([idx, next_id], dim=1)
        return idx
