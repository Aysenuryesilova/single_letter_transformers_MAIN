"""The whole pipeline, end to end — read this file to follow each step.

It bundles the four trained models and runs one tag through all four colored
regions, optionally printing the shape (and a couple of real numbers) at every
boundary so you can watch the signal grow from a single letter into a waveform:

    tag  --planner-->  5Hz codes  --fsq-->  source latent
         --text enc-->  conditioning
         --dit + flow-->  25Hz latent  --vae-->  waveform
"""

import torch

from flow import FlowMatchScheduler


class AceStepPipeline:
    def __init__(self, cfg, vae, fsq, text_encoder, planner, dit, latent_scale):
        self.cfg = cfg
        self.vae = vae.eval()
        self.fsq = fsq.eval()
        self.text_encoder = text_encoder.eval()
        self.planner = planner.eval()
        self.dit = dit.eval()
        self.flow = FlowMatchScheduler(cfg.num_inference_steps)
        self.latent_scale = latent_scale          # VAE latents are normalized for the DiT

    @torch.no_grad()
    def generate(self, tag_ids: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        cfg = self.cfg
        B = tag_ids.shape[0]

        def show(name, t):
            if verbose:
                print(f"  {name:<16} {tuple(t.shape)}")

        show("tag ids", tag_ids)

        # --- Real ACE-Step v1.5 shape trace (240s song), for comparison -----
        #   tag/caption  -> planner  -> 5Hz codes      [1200]
        #                -> fsq      -> source latent  [64, 6000]      (5Hz -> 25Hz)
        #                -> dit+flow -> clean latent   [64, 6000]      (~8 steps)
        #                -> vae      -> waveform       [2, 11_520_000] (48kHz stereo)
        # The toy trace below is the same five steps at code_len=4 / latent=[8,16].

        # --- purple: the planner writes the 5Hz blueprint -------------------
        codes = self.planner.generate(tag_ids)                     # [B, code_len]
        show("5Hz codes", codes)

        # --- blue: FSQ -> source latent, tag -> conditioning ----------------
        source = self.fsq.decode(codes) / self.latent_scale        # [B, d, 16]
        text_embed = self.text_encoder(tag_ids)                    # [B, 1, h]
        show("source latent", source)
        show("text embed", text_embed)

        # --- coral: the DiT renders the 25Hz latent in N flow steps ---------
        shape = (B, cfg.latent_dim, cfg.latent_len)
        latent = self.flow.sample(self.dit, text_embed, source, shape,
                                  device=tag_ids.device)           # [B, d, 16]
        latent = latent * self.latent_scale
        show("clean latent", latent)

        # --- green: the VAE decodes to a waveform ---------------------------
        waveform = self.vae.decode(latent)                         # [B, 1, 64]
        show("waveform", waveform)

        if verbose:
            print(f"  codes[0]        {codes[0].tolist()}")
            print(f"  waveform[0,0,:6]{[round(v, 3) for v in waveform[0, 0, :6].tolist()]}")
        return waveform
