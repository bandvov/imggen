"""A trimmed-down training loop for hyperparameter sweeps and ablations: same building blocks as
``imggen.train.run_training`` (backbone, objective, muP-aware param groups, LR schedule), but no
run directory, no checkpointing, and it returns the loss history in memory instead of logging to
a file. Meant for many short runs (tens to low hundreds of steps each), not real training.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from imggen.backbones import build_backbone
from imggen.config import ExperimentConfig
from imggen.mup import MuP
from imggen.objectives import build_objective
from imggen.train import build_param_groups, cycle
from imggen.utils.schedule import build_lr_lambda


@dataclass
class SweepResult:
    losses: list[float] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def final_loss(self) -> float:
        return self.losses[-1]

    def tail_avg_loss(self, n: int = 10) -> float:
        tail = self.losses[-n:]
        return sum(tail) / len(tail)


def run_sweep_point(
    cfg: ExperimentConfig,
    steps: int,
    data_iter,
    *,
    seed: int = 0,
    device: torch.device | None = None,
) -> SweepResult:
    """Trains ``cfg`` for ``steps`` steps, pulling batches from ``data_iter`` (an infinite
    iterator yielding the dict shape ``build_dataloader`` produces — pass
    ``cycle(build_dataloader(...))`` once and reuse it across sweep points to keep every point
    seeing the same data stream, or a fresh one per point for independence; the caller decides
    which comparison it wants)."""
    device = device or torch.device("cpu")
    torch.manual_seed(seed)
    model = build_backbone(cfg.backbone).to(device)
    objective = build_objective(cfg.objective)
    mup = MuP(cfg.backbone.mup.base_width) if cfg.backbone.mup.enabled else None
    opt = torch.optim.AdamW(
        build_param_groups(model, cfg.train.lr, cfg.train.weight_decay, mup),
        betas=(cfg.train.adam_beta1, cfg.train.adam_beta2),
        eps=cfg.train.adam_eps,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt,
        build_lr_lambda(
            cfg.train.warmup_steps, steps + 1, cfg.train.lr_schedule, cfg.train.min_lr_ratio
        ),
    )

    result = SweepResult()
    t0 = time.time()
    for _ in range(steps):
        batch = next(data_iter)
        opt.zero_grad(set_to_none=True)
        loss, _ = objective.loss(
            model,
            batch["latent"].to(device),
            batch["text_emb"].to(device),
            batch["mask"].to(device),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
        opt.step()
        scheduler.step()
        result.losses.append(loss.item())
    result.seconds = time.time() - t0
    return result


def fixed_synthetic_batches(cfg: ExperimentConfig, n_batches: int, batch_size: int, seed: int = 0):
    """An in-memory, dataloader-shaped iterator of fixed random batches — for sweeps that don't
    need (or want the setup cost of) a real precomputed cache, e.g. the width-transfer check,
    which only cares about optimization dynamics, not what the model actually learns."""
    g = torch.Generator().manual_seed(seed)
    b = cfg.backbone
    batches = [
        {
            "latent": torch.randn(batch_size, b.in_channels, 8, 8, generator=g) * 0.5,
            "text_emb": torch.randn(batch_size, 6, b.text_dim, generator=g),
            "mask": torch.ones(batch_size, 6, dtype=torch.bool),
        }
        for _ in range(n_batches)
    ]
    return cycle(batches)
