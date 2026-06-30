"""FSQ bridge — the blue right arm that joins the planner and the DiT.

The planner LM speaks in *discrete tokens*; the DiT renders a *continuous*
latent. FSQ (Finite Scalar Quantization) is the translator between them, and it
also drops the resolution from 25Hz to 5Hz:

    latent  [B, 8, 16]  --pool x4--> [B, 4, 8*4]  --proj--> [B, 4, 2]  (2 scalar dims)
                                         quantize each dim to its levels (8, 8)
    -> one integer code per 5Hz frame: codes [B, 4]   (codebook = 8*8 = 64)

and the inverse turns codes back into a continuous "source latent" that seeds
the DiT. Quantization uses the straight-through trick (round on the forward
pass, identity gradient on the backward pass) so the projections still train.

We index each scalar dim into {0 .. level-1} via a sigmoid (no fiddly even/odd
centering), then pack the per-dim integers into one code with mixed-radix
arithmetic — exactly how a multi-dimensional code becomes a single codebook id.
"""

import torch
from torch import nn

from config import AceConfig

# --- Real ACE-Step v1.5, for comparison ------------------------------------
# The real FSQ tokenizer uses *attention pooling* to compress 25Hz VAE latents
# into 5Hz discrete codes over a codebook of ~64k (hence ids like
# <|audio_code_35639|>). The codebook is so large that a real failure mode is
# the LM emitting an id beyond its range (ace-step/ACE-Step-1.5 issue #92).
# Here (toy): levels=(8, 8) -> a 64-entry codebook, code ids 0..63, and the
# 25Hz -> 5Hz pooling (x4) is a plain reshape instead of attention pooling.
# ---------------------------------------------------------------------------


class FSQBridge(nn.Module):
    def __init__(self, cfg: AceConfig):
        super().__init__()
        self.cfg = cfg
        self.dim = len(cfg.fsq_levels)              # scalar dims per frame (e.g. 2)
        self.pool = cfg.latent_len // cfg.code_len  # 25Hz -> 5Hz pooling factor (4)

        # levels per dim and the mixed-radix basis to pack/unpack a single code id.
        levels = torch.tensor(cfg.fsq_levels)                  # [dim]
        basis = torch.cumprod(torch.tensor([1] + list(cfg.fsq_levels[:-1])), 0)
        self.register_buffer("levels", levels, persistent=False)
        self.register_buffer("basis", basis, persistent=False)

        # 25Hz latent window  <->  the few FSQ scalars for that 5Hz frame.
        self.proj_in = nn.Linear(cfg.latent_dim * self.pool, self.dim)
        self.proj_out = nn.Linear(self.dim, cfg.latent_dim * self.pool)

    # ---- reshapes between 25Hz latent and 5Hz frames ----------------------
    def _pool(self, latent: torch.Tensor) -> torch.Tensor:
        """[B, d, 16] -> [B, code_len, pool*d]: group latent time into 5Hz frames."""
        B, d, T = latent.shape
        return latent.transpose(1, 2).reshape(B, self.cfg.code_len, self.pool * d)

    def _unpool(self, x: torch.Tensor) -> torch.Tensor:
        """[B, code_len, pool*d] -> [B, d, 16]: scatter 5Hz frames back over time."""
        B = x.shape[0]
        x = x.reshape(B, self.cfg.latent_len, self.cfg.latent_dim)
        return x.transpose(1, 2)

    # ---- the quantizer ----------------------------------------------------
    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Map reals to per-dim integers in {0..level-1}, straight-through."""
        qf = torch.sigmoid(z) * (self.levels - 1)              # [..., dim] in [0, L-1]
        return qf + (torch.round(qf) - qf).detach()            # round, but pass gradient

    def _pack(self, q: torch.Tensor) -> torch.Tensor:
        """Per-dim integers [..., dim] -> one code id [...]  (mixed radix)."""
        return (q.long() * self.basis).sum(-1)

    def _unpack(self, codes: torch.Tensor) -> torch.Tensor:
        """One code id [...] -> per-dim integers [..., dim]."""
        return (codes.unsqueeze(-1) // self.basis) % self.levels

    @staticmethod
    def _centered(q: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        """Integer codes -> continuous values in [-1, 1] for reconstruction."""
        return q / (levels - 1) * 2 - 1

    # ---- public API -------------------------------------------------------
    def encode(self, latent: torch.Tensor) -> torch.Tensor:
        """latent [B, d, 16] -> discrete codes [B, code_len] (the 5Hz blueprint)."""
        q = self.quantize(self.proj_in(self._pool(latent)))
        return self._pack(q)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes [B, code_len] -> source latent [B, d, 16] (seeds the DiT)."""
        q = self._unpack(codes)
        x = self.proj_out(self._centered(q.float(), self.levels))
        return self._unpool(x)

    def forward(self, latent: torch.Tensor):
        """Round-trip used in training. Returns (source_latent, codes).

        Uses the straight-through quantized values so gradients reach both
        projections; the integer codes come along for the planner's targets.
        """
        q = self.quantize(self.proj_in(self._pool(latent)))    # [B, code_len, dim]
        codes = self._pack(q)
        source = self._unpool(self.proj_out(self._centered(q, self.levels)))
        return source, codes
