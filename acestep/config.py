"""One config for the whole tiny ACE-Step pipeline.

ACE-Step is four models wired in series, but they all share the same little
transformer recipe (the planner LM and the DiT) and the same resolution
hierarchy, so everything fits in one dataclass. Defaults are deliberately tiny
so all four stages train on a CPU in a few seconds.

The resolution hierarchy (ACE-Step's 48kHz -> 25Hz -> 5Hz, shrunk to toy size):

    waveform_len = 64   samples   ("48kHz" audio, the green region's output)
        |  VAE encoder, x4 down
    latent_len   = 16   frames    ("25Hz" latent, what the DiT renders)
        |  FSQ pooling, x4 down
    code_len     =  4   codes     ("5Hz" blueprint, what the planner writes)

Real ACE-Step v1.5, for comparison (a 240-second song):
    waveform   [2, 11_520_000]   48kHz stereo
        |  AutoencoderOobleck, 1920x temporal squeeze, 64-dim latent
    latent     [64, 6000]        25Hz, 64 dims/frame
        |  FSQ attention pooling, x5 down, codebook ~64k
    codes      [1200]            5Hz, ids 0..~64000  (1200 = 240s x 5Hz)
"""

from dataclasses import dataclass, field
from math import prod


@dataclass
class AceConfig:
    # ---- resolution hierarchy --------------------------------------------
    waveform_len: int = 64          # samples in one toy "audio" clip (the WAV)
    latent_dim: int = 8             # channels of the VAE / DiT latent
    latent_len: int = 16            # latent frames  (= waveform_len // 4, the "25Hz")
    code_len: int = 4               # 5Hz code frames (= latent_len // 4, the "5Hz")
    fsq_levels: tuple = (8, 8)      # FSQ quantization levels per dim -> codebook = 8*8 = 64

    # ---- shared transformer recipe (used by planner LM *and* DiT) --------
    hidden_size: int = 32           # model / embedding dimension
    num_layers: int = 2             # planner LM transformer blocks
    dit_layers: int = 4             # DiT blocks (the renderer needs a little more depth)
    num_heads: int = 4              # query heads
    num_kv_heads: int = 2           # key/value heads (GQA)
    head_dim: int = 8               # dimension per head
    intermediate_size: int = 64     # SwiGLU hidden dim
    max_seq_len: int = 16           # longest sequence (>= latent_len and planner length)
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

    # ---- planner LM vocabulary -------------------------------------------
    # Full vocab = [letter tags] + [audio-code tokens]. n_letters is filled in
    # from the data at train time (like ModelConfig(vocab_size=...) elsewhere).
    n_letters: int = 0

    # ---- diffusion -------------------------------------------------------
    num_inference_steps: int = 16   # Euler steps when sampling (ACE-Step distils to ~8)

    @property
    def num_codes(self) -> int:
        """Size of the FSQ codebook = product of the per-dim levels."""
        return prod(self.fsq_levels)

    @property
    def planner_vocab(self) -> int:
        """Planner token ids: letters first, then the audio codes after them."""
        return self.n_letters + self.num_codes
