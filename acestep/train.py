"""Train the tiny ACE-Step pipeline, one region at a time.

Run:  python train.py

The four models are trained in series, exactly the order their outputs feed each
other — you cannot make 5Hz code targets before the VAE+FSQ exist, and the DiT
needs the VAE's latents to denoise toward:

    1. VAE       reconstruct the waveform                 (green)
    2. FSQ       round-trip the 25Hz latent through codes (blue bridge)
    3. Planner   tag -> its 5Hz code sequence             (purple)
    4. DiT       flow-match noise -> latent               (coral)

Everything is torch-only and tiny, so all four stages finish in a few seconds.
"""

import torch

from config import AceConfig
from data import get_batch, all_tones, N_LETTERS
from vae import AutoencoderOobleckTiny
from fsq import FSQBridge
from text_encoder import TextEncoder
from planner import Planner, make_batch
from dit import DiT
from flow import FlowMatchScheduler

BATCH_SIZE = 64
LEARNING_RATE = 3e-3
VAE_STEPS = 800
FSQ_STEPS = 800
PLANNER_STEPS = 800
DIT_STEPS = 3000
SEED = 1337

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

cfg = AceConfig(n_letters=N_LETTERS)
vae = AutoencoderOobleckTiny(cfg).to(device)
fsq = FSQBridge(cfg).to(device)
text_encoder = TextEncoder(cfg).to(device)
planner = Planner(cfg).to(device)
dit = DiT(cfg).to(device)
flow = FlowMatchScheduler(cfg.num_inference_steps)

n_params = sum(p.numel() for m in (vae, fsq, text_encoder, planner, dit)
               for p in m.parameters())
print(f"device={device}  letters={N_LETTERS}  codebook={cfg.num_codes}  parameters={n_params:,}\n")
mse = torch.nn.MSELoss()


# ---------------------------------------------------------------------------
# Stage 1 — VAE: learn to reconstruct the toy waveforms.
# ---------------------------------------------------------------------------
opt = torch.optim.AdamW(vae.parameters(), lr=LEARNING_RATE)
for step in range(1, VAE_STEPS + 1):
    _, waveform = get_batch(cfg, BATCH_SIZE)
    waveform = waveform.to(device)
    recon, _ = vae(waveform)
    loss = mse(recon, waveform)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 400 == 0 or step == 1:
        print(f"[1/4 vae]     step {step:5d}  recon mse {loss.item():.5f}")

# All letter latents in one shot; their std normalizes the latent for the DiT.
vae.eval()
with torch.no_grad():
    tones = all_tones(cfg).to(device)            # [N, 1, 64]
    latents = vae.encode(tones)                  # [N, d, 16]
latent_scale = latents.std().item()
print(f"              latent_scale = {latent_scale:.4f}\n")


# ---------------------------------------------------------------------------
# Stage 2 — FSQ: round-trip the latent through discrete 5Hz codes.
# ---------------------------------------------------------------------------
opt = torch.optim.AdamW(fsq.parameters(), lr=LEARNING_RATE)
for step in range(1, FSQ_STEPS + 1):
    _, waveform = get_batch(cfg, BATCH_SIZE)
    with torch.no_grad():
        latent = vae.encode(waveform.to(device))
    source, _ = fsq(latent)
    loss = mse(source, latent)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 400 == 0 or step == 1:
        print(f"[2/4 fsq]     step {step:5d}  roundtrip mse {loss.item():.5f}")

# Each letter's 5Hz code sequence — the planner's training targets.
fsq.eval()
with torch.no_grad():
    codes_per_letter = fsq.encode(latents)       # [N, code_len]
unique = len(set(map(tuple, codes_per_letter.tolist())))
print(f"              {unique}/{N_LETTERS} letters got a distinct code sequence\n")


# ---------------------------------------------------------------------------
# Stage 3 — Planner: a tiny LM that writes tag -> 5Hz codes.
# ---------------------------------------------------------------------------
opt = torch.optim.AdamW(planner.parameters(), lr=LEARNING_RATE)
for step in range(1, PLANNER_STEPS + 1):
    tags = torch.randint(N_LETTERS, (BATCH_SIZE,), device=device)
    inp, tgt = make_batch(tags, codes_per_letter[tags], N_LETTERS)
    _, loss = planner(inp, tgt)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0 or step == 1:
        print(f"[3/4 planner] step {step:5d}  code ce {loss.item():.5f}")

planner.eval()
with torch.no_grad():
    pred = planner.generate(torch.arange(N_LETTERS, device=device))
acc = (pred == codes_per_letter).float().mean().item()
print(f"              planner code accuracy = {acc * 100:.1f}%\n")


# ---------------------------------------------------------------------------
# Stage 4 — DiT: flow-match noise -> latent, conditioned on tag + 5Hz skeleton.
# ---------------------------------------------------------------------------
opt = torch.optim.AdamW(list(dit.parameters()) + list(text_encoder.parameters()),
                        lr=LEARNING_RATE)
# Cosine-decay the LR to ~0 so the velocity field settles instead of jittering.
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=DIT_STEPS)
for step in range(1, DIT_STEPS + 1):
    tags, waveform = get_batch(cfg, BATCH_SIZE)
    tags, waveform = tags.to(device), waveform.to(device)
    with torch.no_grad():
        latent = vae.encode(waveform)
        source = fsq.decode(fsq.encode(latent)) / latent_scale   # conditioning skeleton
    target = latent / latent_scale
    x_t, t, velocity = flow.add_noise(target)
    text_embed = text_encoder(tags)
    pred = dit(x_t, t, text_embed, source)
    loss = mse(pred, velocity)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 500 == 0 or step == 1:
        print(f"[4/4 dit]     step {step:5d}  velocity mse {loss.item():.5f}")


# ---------------------------------------------------------------------------
# Save one checkpoint with all four models.
# ---------------------------------------------------------------------------
torch.save({
    "cfg": cfg,
    "latent_scale": latent_scale,
    "vae": vae.state_dict(),
    "fsq": fsq.state_dict(),
    "text_encoder": text_encoder.state_dict(),
    "planner": planner.state_dict(),
    "dit": dit.state_dict(),
}, "acestep.pt")
print("\nsaved checkpoint to acestep.pt")
