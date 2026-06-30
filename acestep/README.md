# acestep — ACE-Step v1.5 (the two-brain music pipeline)

The odd one out. The other three folders are single language models that emit
discrete letters. ACE-Step is **four models in series** that turn a tag into a
*continuous* signal, so the toy task changes: each **Turkish letter is a tag**
whose "song" is a tiny waveform (a sine of `letter_index + 1` cycles). The whole
point is the **two-brain split** — a low-resolution autoregressive **planner**
decides *what to play*, and a high-resolution **diffusion renderer** decides
*how it sounds*.

## The four regions (read the files in this order)

| Region            | File(s)                          | Job                                                        |
| ----------------- | -------------------------------- | ---------------------------------------------------------- |
| 🟪 **plan**       | `planner.py`                     | 5Hz-lm: tag → coarse **5Hz code** blueprint (reuses TinyQwen) |
| 🟦 **bridge**     | `fsq.py`, `text_encoder.py`      | FSQ: discrete codes ↔ continuous "source latent"; tag → cross-attn conditioning |
| 🟧 **render**     | `dit.py`, `flow.py`              | DiT denoises noise → **25Hz latent** in a few flow-matching steps |
| 🟩 **decode**     | `vae.py`                         | tiny Oobleck VAE: 25Hz latent → **waveform** (the WAV)     |

`pipeline.py` wires all four together (print the shapes!), `data.py` makes the
toy tones, `config.py` holds every shape, `train.py` trains the four stages in
order, `generate.py` writes `out.wav`.

Every file also has a **`# --- Real ACE-Step v1.5, for comparison ---`** comment
block right next to the toy code, quoting the real model's actual numbers and
example output — the real `<think>` YAML + `<|audio_code_5434|>` token stream,
the real ~64k FSQ codebook, the real 48kHz-stereo/64-dim/25Hz VAE shapes, the
real 2B-param hybrid-attention DiT, the real 50→8-step distillation. Read a toy
function next to its real-world counterpart to see exactly what got shrunk.

## The resolution hierarchy (the heart of it)

ACE-Step's `48kHz → 25Hz → 5Hz`, shrunk to toy ratios:

```
tag (1 letter)                              real: caption + lyrics + bpm/key/duration
  │  planner (autoregressive)
5Hz codes      [4]                          real: [1200]   (240s song, codebook ~64k)
  │  fsq.decode  (5Hz → 25Hz, discrete → continuous)
source latent  [8, 16]                      real: [64, 6000]
  │  dit + flow  (×16 Euler steps, cross-attends to tag + skeleton)
clean latent   [8, 16]                      real: [64, 6000]   (~8 distilled steps)
  │  vae.decode  (25Hz → waveform)
waveform       [1, 64]                      real: [2, 11_520_000]   (48kHz stereo)
```

## Signature ideas (kept), simplifications (made)

**Kept:** two-brain split (AR planner + diffusion renderer); discrete↔continuous
**FSQ** bridge with a mixed-radix codebook and straight-through rounding;
**flow-matching** training (`x_t = (1−t)·noise + t·latent`, predict the velocity)
with few-step Euler sampling; a DiT with **self-attention** (time coherence) +
**cross-attention** (conditioning); a conv **VAE** decode. The planner and DiT
reuse the exact Qwen3 blocks (`rms_norm`, `rotary`, `attention`, `mlp`, `block`).

**Simplified:** mono 64-sample "audio", not 48kHz stereo; a 64-entry FSQ codebook,
not ~64k; one letter as the whole caption/lyrics; no LoRA, no intrinsic-RL
alignment, no chain-of-thought planning; tones are deterministic, so the planner
*memorizes* tag → codes (alignment "emerges" only in spirit). Fidelity is traded
for being able to read every step.

## Run it

```bash
python3 train.py        # trains all 4 stages (~80s on CPU); prints a loss per stage
python3 generate.py a   # one letter → out.wav, with a per-region shape trace
python3 generate.py merhaba   # one tone per letter, concatenated
```

`generate.py` also runs the end-to-end **correctness check**: each generated
tone's dominant FFT frequency should equal `letter_index + 1` — i.e. planner +
FSQ + DiT + VAE really did reconstruct the right pitch, not noise. A trained run
hits 29/29 letters.
