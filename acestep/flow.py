"""FlowMatchScheduler — the rule the DiT is trained and sampled under (coral).

Flow matching draws a straight line between pure noise (t=0) and a real latent
(t=1):

    x_t = (1 - t) * noise + t * latent          # a point on that line
    v   = latent - noise                        # the (constant) velocity along it

Training: pick a random t, build x_t, and ask the DiT to predict v. That's it —
no noise schedules, no variance tables.

Sampling: start at pure noise and walk the line forward with a few Euler steps,
following the DiT's predicted velocity. ``num_inference_steps`` is ACE-Step's
distilled "x8 steps".
"""

import torch

# --- Real ACE-Step v1.5, for comparison ------------------------------------
# Same flow-matching objective, but distillation cuts inference from 50 steps to
# ~8 (with dynamic shift sampling from {1, 2, 3}) — a ~200x speedup that renders
# a 240s track in ~1s on an A100. Here (toy): plain Euler, num_inference_steps=16.
# ---------------------------------------------------------------------------


class FlowMatchScheduler:
    def __init__(self, num_inference_steps: int = 8):
        self.num_inference_steps = num_inference_steps

    def add_noise(self, latent: torch.Tensor):
        """Make one training example. Returns (x_t, t, target_velocity)."""
        B = latent.shape[0]
        noise = torch.randn_like(latent)
        t = torch.rand(B, device=latent.device)                 # one t per item, in [0, 1]
        t_ = t[:, None, None]                                    # broadcast over [d, T]
        x_t = (1 - t_) * noise + t_ * latent
        velocity = latent - noise
        return x_t, t, velocity

    @torch.no_grad()
    def sample(self, dit, text_embed, source_latent, shape, device="cpu"):
        """Integrate noise -> latent in num_inference_steps Euler steps."""
        x = torch.randn(shape, device=device)                   # start at t = 0 (pure noise)
        dt = 1.0 / self.num_inference_steps
        for i in range(self.num_inference_steps):
            t = torch.full((shape[0],), i * dt, device=device)  # current position on the line
            v = dit(x, t, text_embed, source_latent)
            x = x + v * dt                                       # step forward toward the latent
        return x
