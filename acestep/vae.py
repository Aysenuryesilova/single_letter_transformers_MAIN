"""A tiny Oobleck-style VAE — the green "decode" region.

ACE-Step's AutoencoderOobleck compresses 48kHz audio into a 25Hz continuous
latent and decodes it back. Ours does the same job at toy scale with two strided
1-D convolutions each way:

    encode:  waveform [B, 1, 64]  --x4 down-->  latent [B, 8, 16]   (the "25Hz")
    decode:  latent   [B, 8, 16]  --x4 up----> waveform [B, 1, 64]

A kernel of 4 with stride 2 and padding 1 halves (encoder) or doubles (decoder)
the length exactly, so two layers give the x4 ratio. Trained with plain
reconstruction MSE; we keep it a deterministic autoencoder (no KL term) so the
pipeline stays easy to follow.
"""

import torch
from torch import nn

from config import AceConfig

# --- Real ACE-Step v1.5, for comparison ------------------------------------
# The real AutoencoderOobleck is a pure waveform-domain 1D VAE: it compresses
# 48kHz *stereo* audio into a 64-dimensional latent at 25Hz — a 1920x temporal
# squeeze (48000 -> 25 frames/sec). So a 240s song is:
#     waveform [2, 11_520_000]  <->  latent [64, 6000]
# Here (toy): mono, an 8-dim latent, a x4 temporal squeeze (64 -> 16):
#     waveform [1, 64]          <->  latent [8, 16]
# ---------------------------------------------------------------------------


class AutoencoderOobleckTiny(nn.Module):
    def __init__(self, cfg: AceConfig):
        super().__init__()
        d, h = cfg.latent_dim, 16        # h = intermediate channel width

        self.encoder = nn.Sequential(
            nn.Conv1d(1, h, kernel_size=4, stride=2, padding=1),   # 64 -> 32
            nn.SiLU(),
            nn.Conv1d(h, d, kernel_size=4, stride=2, padding=1),   # 32 -> 16
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(d, h, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.SiLU(),
            nn.ConvTranspose1d(h, 1, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.Tanh(),                                                     # keep audio in [-1, 1]
        )

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.encoder(waveform)              # [B, 1, 64] -> [B, d, 16]

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)                # [B, d, 16] -> [B, 1, 64]

    def forward(self, waveform: torch.Tensor):
        latent = self.encode(waveform)
        return self.decode(latent), latent         # (reconstruction, latent)
