"""Turns noise into a latent by integrating the model's predicted velocity from t=0 to t=1.

The training objective (``imggen.objectives.flow_matching``) fits the model to predict the
velocity ``v = x_1 - x_0`` along the straight path ``x_t = (1-t) x_0 + t x_1``. Sampling reverses
that: start from ``x_0 ~ N(0, I)`` and integrate ``dx/dt = v_hat(x, t, text)`` forward to ``t=1``.
Euler's method — the simplest fixed-step integrator, ``x += v_hat * dt`` — is what's implemented
here; a higher-order solver (e.g. Heun, DPM-Solver) would need fewer steps for the same accuracy
but isn't built yet (see the module docstring's follow-up note).
"""

from __future__ import annotations

import torch

from imggen.guidance import broadcast_null_embedding, combine_predictions
from imggen.registry import SAMPLER


@torch.no_grad()
def euler_sample(
    model: torch.nn.Module,
    shape: tuple[int, int, int, int],
    text_emb: torch.Tensor,
    text_mask: torch.Tensor | None = None,
    *,
    steps: int = 30,
    guidance_scale: float = 1.0,
    null_emb: torch.Tensor | None = None,
    null_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
    return_trajectory: bool = False,
) -> torch.Tensor | list[torch.Tensor]:
    """``shape`` is ``(batch, channels, height, width)`` in latent space (i.e. what the
    autoencoder's ``encode`` produces, not pixel space). ``guidance_scale > 1`` requires
    ``null_emb``/``null_mask`` (from ``imggen.data.shards.load_null_embedding``) and doubles the
    per-step cost (one forward pass each for the conditional and unconditional prediction).
    ``return_trajectory=True`` returns every intermediate ``x_t`` (for inspection/debugging)
    instead of just the final sample.
    """
    if guidance_scale != 1.0 and (null_emb is None or null_mask is None):
        raise ValueError("guidance_scale != 1 requires null_emb and null_mask")
    device = device or next(model.parameters()).device
    b = shape[0]

    x = torch.randn(shape, generator=generator, device=device)
    trajectory = [x.clone()] if return_trajectory else None
    dt = 1.0 / steps

    text_emb = text_emb.to(device)
    text_mask = text_mask.to(device) if text_mask is not None else None
    if guidance_scale != 1.0:
        u_emb, u_mask = broadcast_null_embedding(null_emb, null_mask, b)
        u_emb, u_mask = u_emb.to(device), u_mask.to(device)

    for i in range(steps):
        t = torch.full((b,), i * dt, device=device)
        v_cond = model(x, t, text_emb, text_mask)
        if guidance_scale != 1.0:
            v_uncond = model(x, t, u_emb, u_mask)
            v = combine_predictions(v_cond, v_uncond, guidance_scale)
        else:
            v = v_cond
        x = x + v * dt
        if return_trajectory:
            trajectory.append(x.clone())

    return trajectory if return_trajectory else x


SAMPLER.register("euler")(euler_sample)
