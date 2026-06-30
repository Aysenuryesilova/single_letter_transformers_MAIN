"""The toy "audio": every Turkish letter is a tag that maps to one short tone.

This is the continuous-signal analogue of the other folders' name corpus.
Names are discrete (you can't diffuse over characters), so instead each of the
29 Turkish letters becomes a *caption* whose "song" is a tiny waveform:

    letter index i  ->  a sine of (i+1) cycles, with a small decay envelope.

Because each letter has a distinct number of cycles, its waveform has a distinct
dominant frequency — which is exactly what the end-to-end test checks for.

The whole dataset is deterministic (one tone per letter), so the planner can
genuinely learn the letter -> 5Hz-code mapping, while the DiT still has to learn
to *denoise* its way to the right latent.
"""

import torch

from config import AceConfig

# --- Real ACE-Step v1.5 input, for comparison ------------------------------
# A real prompt is a caption (free text, e.g. "lofi hip hop, mellow piano"),
# lyrics with structure tags ([Intro], [Verse], [Chorus], [Instrumental], ...),
# and meta (bpm 60-180, key, duration 2-4min). Empty lyrics -> instrumental;
# ~2-3 words/sec keeps vocals natural. Here (toy): the entire "caption + lyrics"
# is a single Turkish letter, and its "song" is one deterministic tone.
# ---------------------------------------------------------------------------

# 29 Turkish lowercase letters — the tag/caption vocabulary.
ALPHABET = list("abcçdefgğhıijklmnoöprsştuüvyz")
N_LETTERS = len(ALPHABET)

stoi = {ch: i for i, ch in enumerate(ALPHABET)}   # letter -> tag id
itos = {i: ch for i, ch in enumerate(ALPHABET)}   # tag id -> letter


def letter_tone(letter_idx: int, cfg: AceConfig) -> torch.Tensor:
    """The waveform for one letter: [waveform_len], peak-normalized to [-1, 1]."""
    cycles = letter_idx + 1                                    # 1..N_LETTERS distinct pitches
    t = torch.arange(cfg.waveform_len).float() / cfg.waveform_len
    env = torch.exp(-3.0 * t)                                  # gentle decay = a bit of "timbre"
    phase = letter_idx * 0.21                                  # small per-letter phase offset
    wave = env * torch.sin(2 * torch.pi * cycles * t + phase)
    return wave / wave.abs().max()                             # normalize peak to 1


def all_tones(cfg: AceConfig) -> torch.Tensor:
    """Every letter's tone stacked: [N_LETTERS, 1, waveform_len]."""
    tones = [letter_tone(i, cfg) for i in range(N_LETTERS)]
    return torch.stack(tones).unsqueeze(1)                     # add the 1-channel axis


def get_batch(cfg: AceConfig, batch_size: int):
    """Sample random letters. Returns (tag_ids [B], waveforms [B, 1, waveform_len])."""
    tones = all_tones(cfg)                                     # [N_LETTERS, 1, L]
    tag_ids = torch.randint(N_LETTERS, (batch_size,))
    return tag_ids, tones[tag_ids]
