"""Generate a waveform from a letter (or several) with the trained pipeline.

Run:  python generate.py            # letter "a"
      python generate.py e          # letter "e"
      python generate.py merhaba    # one tone per letter, concatenated

It prints the shape at every region boundary, checks that each generated tone
has the right dominant frequency (letter index + 1 cycles), and writes the
result to out.wav using only the stdlib `wave` module.
"""

import struct
import sys
import wave

import torch

from data import ALPHABET, stoi
from vae import AutoencoderOobleckTiny
from fsq import FSQBridge
from text_encoder import TextEncoder
from planner import Planner
from dit import DiT
from pipeline import AceStepPipeline

CHECKPOINT = "acestep.pt"
SAMPLE_RATE = 8000          # toy playback rate for the .wav


def load() -> AceStepPipeline:
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    vae = AutoencoderOobleckTiny(cfg); vae.load_state_dict(ckpt["vae"])
    fsq = FSQBridge(cfg); fsq.load_state_dict(ckpt["fsq"])
    text_encoder = TextEncoder(cfg); text_encoder.load_state_dict(ckpt["text_encoder"])
    planner = Planner(cfg); planner.load_state_dict(ckpt["planner"])
    dit = DiT(cfg); dit.load_state_dict(ckpt["dit"])
    return AceStepPipeline(cfg, vae, fsq, text_encoder, planner, dit, ckpt["latent_scale"])


def dominant_cycles(wave1d: torch.Tensor) -> int:
    """The strongest non-DC frequency = number of cycles across the clip."""
    spectrum = torch.fft.rfft(wave1d).abs()
    return spectrum[1:].argmax().item() + 1            # skip bin 0 (DC), shift back


def write_wav(path: str, wave1d: torch.Tensor, sample_rate: int):
    pcm = (wave1d.clamp(-1, 1) * 32767).short().tolist()
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(b"".join(struct.pack("<h", s) for s in pcm))


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "a"
    letters = [c for c in text.lower() if c in stoi]
    if not letters:
        print(f"no usable letters in {text!r}; alphabet is: {''.join(ALPHABET)}")
        return

    pipe = load()
    tags = torch.tensor([stoi[c] for c in letters])

    print("region shapes (one batch through the whole pipeline):")
    waves = pipe.generate(tags, verbose=True)          # [B, 1, 64]

    print("\nletter  expected  got  ok")
    segments = []
    for i, c in enumerate(letters):
        w = waves[i, 0]
        got, expected = dominant_cycles(w), stoi[c] + 1
        ok = "ok" if got == expected else "MISS"
        print(f"   {c}       {expected:>3}    {got:>3}  {ok}")
        segments.append(w)

    write_wav("out.wav", torch.cat(segments), SAMPLE_RATE)
    print(f"\nwrote out.wav  ({len(letters)} tone(s) @ {SAMPLE_RATE} Hz)")


if __name__ == "__main__":
    main()
