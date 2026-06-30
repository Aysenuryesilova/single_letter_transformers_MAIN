"""The 5Hz-lm planner — the purple "Composer Agent".

This is a Qwen3-style causal language model (the very same TinyQwen recipe used
in ../qwen3), but its vocabulary is not letters-of-a-name: it is

    [ letter tags ]  ++  [ 64 audio-code tokens ]
      ids 0..n_letters-1     ids n_letters..n_letters+63

Given one tag token it autoregressively writes the song's coarse 5Hz blueprint —
``code_len`` audio-code tokens — answering "what to play" before any audio is
rendered. It never has to produce a continuous signal; that is the DiT's job.
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import AceConfig
from rms_norm import RMSNorm
from block import TransformerBlock
from rotary import precompute_cos_sin

# --- Real ACE-Step v1.5, for comparison ------------------------------------
# The real 5Hz-lm is Qwen3-based (0.6B / 1.7B / 4B). It first *reasons* inside a
# <think> block of YAML metadata, then emits one <|audio_code_N|> token per
# 200ms (5/sec -> 1200 tokens for a 240s song), N indexing a ~64k codebook:
#
#     <think>
#     bpm: 187
#     keyscale: D major
#     timesignature: 4
#     language: ja
#     duration: 344
#     caption: <expanded tags / description>
#     </think>
#     <|audio_code_5434|><|audio_code_20161|><|audio_code_7418|> ...
#         ... <|audio_code_35639|><|audio_code_35847|><|audio_code_15174|>
#
# Here (toy): no <think> YAML, the "caption" is a single letter, the codebook is
# 64 (not ~64k), and we emit code_len=4 plain integers (not 1200 tokens).
# ---------------------------------------------------------------------------


def make_batch(tag_ids: torch.Tensor, codes: torch.Tensor, n_letters: int):
    """Build (input, target) token sequences for next-token training.

    For a tag with codes [c0,c1,c2,c3] (code_len=4):
        input  = [tag,        c0+off, c1+off, c2+off]
        target = [c0+off, c1+off, c2+off, c3+off]
    where off = n_letters shifts code values into the audio-code id range.
    """
    code_tokens = codes + n_letters                      # [B, code_len]
    inp = torch.cat([tag_ids[:, None], code_tokens[:, :-1]], dim=1)
    return inp, code_tokens                              # both [B, code_len]


class Planner(nn.Module):
    def __init__(self, cfg: AceConfig):
        super().__init__()
        self.cfg = cfg
        self.offset = cfg.n_letters                      # where audio-code ids start

        self.embed_tokens = nn.Embedding(cfg.planner_vocab, cfg.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.planner_vocab, bias=False)
        self.lm_head.weight = self.embed_tokens.weight   # weight tying

        cos, sin = precompute_cos_sin(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        cos, sin = self.cos[:T], self.sin[:T]

        x = self.embed_tokens(idx)
        for layer in self.layers:
            x = layer(x, cos, sin)
        logits = self.lm_head(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, tag_ids: torch.Tensor) -> torch.Tensor:
        """tag_ids [B] -> 5Hz blueprint codes [B, code_len] (values in 0..num_codes-1)."""
        idx = tag_ids[:, None]                           # [B, 1]
        for _ in range(self.cfg.code_len):
            logits, _ = self(idx[:, -self.cfg.max_seq_len:])
            logits = logits[:, -1, :]
            logits[:, :self.offset] = float("-inf")      # only audio-code tokens are legal
            next_token = logits.argmax(dim=-1, keepdim=True)   # data is deterministic -> greedy
            idx = torch.cat([idx, next_token], dim=1)
        return idx[:, 1:] - self.offset                  # drop tag, shift back to code values
