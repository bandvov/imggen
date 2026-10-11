"""Conditional flow matching / rectified flow objective.

Straight-line interpolation between noise and data: x_0 ~ N(0, I), x_1 = latent, t ~ some
distribution on (0, 1), x_t = (1-t) x_0 + t x_1. The model predicts the velocity v = x_1 - x_0
(the direction of the straight path); the loss is a plain MSE between predicted and true
velocity. At inference, integrating dx/dt = v_hat(x, t) from t=0 to t=1 (Euler or a higher-order
solver — not implemented yet, see ADR 0007) turns noise into a sample.

Two timestep distributions: ``uniform`` (simplest, the default) and ``logit_normal`` (SD3-style:
sample a normal in logit space and squash through sigmoid, which concentrates more timesteps at
the middle of the schedule where the paper reports the loss is hardest to fit). Which one helps
here is an open question for the roadmap's step-10 ablations, not something this step decides.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from imggen.registry import OBJECTIVE


@OBJECTIVE.register("flow_matching")
class FlowMatchingObjective:
    def __init__(
        self,
        t_dist: str = "uniform",
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
        min_t: float = 1e-5,
    ) -> None:
        if t_dist not in ("uniform", "logit_normal"):
            raise ValueError(f"unknown t_dist {t_dist!r} (expected 'uniform' or 'logit_normal')")
        self.t_dist = t_dist
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std
        self.min_t = min_t  # keeps t away from the endpoints, where the path is degenerate

    def sample_t(self, batch_size: int, device, dtype, generator=None) -> torch.Tensor:
        if self.t_dist == "uniform":
            t = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        else:
            z = torch.randn(batch_size, device=device, dtype=dtype, generator=generator)
            t = torch.sigmoid(z * self.logit_normal_std + self.logit_normal_mean)
        return t.clamp(self.min_t, 1.0 - self.min_t)

    def loss(
        self,
        model: nn.Module,
        latent: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Returns (scalar loss, detached logging stats). ``latent``/``text_emb``/``text_mask``
        are exactly what ``imggen.data.loader.build_dataloader`` yields (mask: True = real
        token)."""
        b = latent.shape[0]
        noise = torch.randn(
            latent.shape, device=latent.device, dtype=latent.dtype, generator=generator
        )
        t = self.sample_t(b, latent.device, latent.dtype, generator=generator)
        t_ = t.view(b, *([1] * (latent.ndim - 1)))
        x_t = (1 - t_) * noise + t_ * latent
        target = latent - noise

        pred = model(x_t, t, text_emb, text_mask)
        loss = F.mse_loss(pred, target)
        with torch.no_grad():
            logs = {
                "loss": loss.item(),
                "t_mean": t.mean().item(),
                "target_std": target.std().item(),
                "pred_std": pred.detach().std().item(),
            }
        return loss, logs
