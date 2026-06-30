"""Tag encoder — the blue left arm of the conditioning bridge.

In ACE-Step a Qwen3-Embedding text encoder turns the caption/lyrics into vectors
that feed the DiT's cross-attention. Our caption is a single letter, so the
"encoder" is just an embedding table that lifts the tag id into one
conditioning token of width hidden_size.
"""

import torch
from torch import nn

from config import AceConfig

# --- Real ACE-Step v1.5, for comparison ------------------------------------
# The real conditioning encoder is Qwen3-Embedding-0.6B: it turns the caption
# (free-text tags / description) into embeddings consumed by the DiT's cross-
# attention. Here (toy): one nn.Embedding row per single-letter tag.
# ---------------------------------------------------------------------------


class TextEncoder(nn.Module):
    def __init__(self, cfg: AceConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.n_letters, cfg.hidden_size)

    def forward(self, tag_ids: torch.Tensor) -> torch.Tensor:
        """tag_ids [B] -> conditioning [B, 1, hidden_size] (a length-1 context)."""
        return self.embed(tag_ids).unsqueeze(1)
