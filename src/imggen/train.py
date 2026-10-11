"""Single-process training loop: flow matching, gradient accumulation, bf16/fp16 autocast, EMA,
warmup+decay LR, checkpoint/resume. Multi-GPU (FSDP/tensor parallel) is step 8 — see ADR 0007
for why this loop already writes checkpoints in a format that step won't need to change.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from imggen.backbones import build_backbone
from imggen.config import ExperimentConfig
from imggen.data.loader import build_dataloader
from imggen.data.shards import has_null_embedding, load_null_embedding
from imggen.guidance import apply_caption_dropout
from imggen.layers.moe import collect_aux_loss, moe_metrics, step_router_bias
from imggen.mup import MuP
from imggen.objectives import build_objective
from imggen.utils.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from imggen.utils.ema import EMA
from imggen.utils.schedule import build_lr_lambda

log = logging.getLogger("imggen.train")


def select_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_param_groups(
    model: nn.Module, lr: float, weight_decay: float, mup: MuP | None = None
) -> list[dict]:
    """muP's role-based groups when enabled; otherwise the common "no decay on biases/norms"
    split (any parameter with fewer than 2 dims is a bias, norm scale, or similar 1-D tensor)."""
    if mup is not None:
        return mup.param_groups(model, lr=lr, weight_decay=weight_decay)
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return [
        {"params": decay, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay, "lr": lr, "weight_decay": 0.0},
    ]


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return torch.autocast(device.type, enabled=False)
    dtype = torch.bfloat16
    if precision == "fp16":
        if device.type == "cpu":
            log.warning("fp16 autocast is not supported on CPU; using bf16 instead this run")
        else:
            dtype = torch.float16
    return torch.autocast(device.type, dtype=dtype)


def cycle(loader):
    """Iterate ``loader`` forever, restarting it each time it's exhausted.

    Raises ``ValueError`` if a full pass yields nothing — the classic cause is a ``DataLoader``
    with ``drop_last=True`` and ``batch_size`` larger than the dataset, which produces zero
    batches. Without this check the loop below would spin forever without yielding, hanging the
    caller silently instead of failing (found when a CLI smoke test with a 4-sample cache and the
    default ``batch_size=32`` never returned)."""
    while True:
        yielded_any = False
        for item in loader:
            yielded_any = True
            yield item
        if not yielded_any:
            raise ValueError(
                "data loader produced no batches; is batch_size larger than the dataset "
                "(with drop_last=True)?"
            )


@dataclass
class TrainState:
    step: int = 0
    last_loss: float | None = None


def run_training(
    cfg: ExperimentConfig, run_dir: str | Path, *, device: torch.device | None = None
) -> TrainState:
    device = device or select_device()
    run_dir = Path(run_dir)
    model: nn.Module = build_backbone(cfg.backbone).to(device)
    objective = build_objective(cfg.objective)
    mup = MuP(cfg.backbone.mup.base_width) if cfg.backbone.mup.enabled else None

    # Parallelize BEFORE building the optimizer: FSDP2 (`fully_shard`) and tensor-parallel
    # (`parallelize_module`) replace parameter *objects* with DTensor-wrapped ones in place, so
    # an optimizer built from the pre-parallelize parameters would hold stale references. Both
    # keep parameter *names* identical to the unwrapped model (no "module." prefix, unlike DDP),
    # so muP's name-based role classification (imggen.mup.classify) still works unchanged on the
    # parallelized model. DDP is the opposite case: it preserves parameter identity but adds a
    # "module." prefix at its own named_parameters() — for that one case, use `.module` (the
    # inner, unwrapped reference DDP itself keeps) to build param groups against clean names.
    if cfg.parallel.strategy != "none" or cfg.parallel.tp_size > 1:
        from imggen.parallel import parallelize

        model = parallelize(model, cfg.parallel, device=device)

    param_source = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    opt = torch.optim.AdamW(
        build_param_groups(param_source, cfg.train.lr, cfg.train.weight_decay, mup),
        betas=(cfg.train.adam_beta1, cfg.train.adam_beta2),
        eps=cfg.train.adam_eps,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt,
        build_lr_lambda(
            cfg.train.warmup_steps, cfg.train.steps, cfg.train.lr_schedule, cfg.train.min_lr_ratio
        ),
    )
    ema = EMA(model, decay=cfg.train.ema_decay)
    scaler = torch.amp.GradScaler(
        device.type, enabled=(cfg.train.precision == "fp16" and device.type == "cuda")
    )

    ckpt_dir = run_dir / "checkpoints"
    state = TrainState()
    latest = find_latest_checkpoint(ckpt_dir)
    if latest is not None:
        state.step = load_checkpoint(
            latest, model=model, optimizer=opt, scheduler=scheduler, ema=ema
        )
        log.info("resumed from %s at step %d", latest, state.step)

    if state.step >= cfg.train.steps:
        log.info("already at or past cfg.train.steps (%d); nothing to do", cfg.train.steps)
        return state

    loader = build_dataloader(cfg.data, batch_size=cfg.train.batch_size, backbone=cfg.backbone)
    data_iter = cycle(loader)

    null_emb = null_mask = None
    if cfg.train.cfg_dropout > 0:
        if not has_null_embedding(cfg.data.root):
            raise FileNotFoundError(
                f"train.cfg_dropout={cfg.train.cfg_dropout} but {cfg.data.root} has no "
                "null_embedding.pt; rerun precompute.py (it now saves one automatically) or set "
                "train.cfg_dropout=0 to train without classifier-free guidance"
            )
        null_emb, null_mask = load_null_embedding(cfg.data.root)
        null_emb, null_mask = null_emb.to(device), null_mask.to(device)

    model.train()
    t0 = time.time()
    for step in range(state.step, cfg.train.steps):
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(cfg.train.grad_accum):
            batch = next(data_iter)
            latent = batch["latent"].to(device)
            text_emb = batch["text_emb"].to(device)
            mask = batch["mask"].to(device)
            if cfg.train.cfg_dropout > 0:
                text_emb, mask = apply_caption_dropout(
                    text_emb, mask, null_emb, null_mask, cfg.train.cfg_dropout
                )
            with autocast_context(device, cfg.train.precision):
                loss, _ = objective.loss(model, latent, text_emb, mask)
                loss = loss + collect_aux_loss(model)
            scaler.scale(loss / cfg.train.grad_accum).backward()
            total_loss += loss.item() / cfg.train.grad_accum

        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
        scaler.step(opt)
        scaler.update()
        scheduler.step()
        ema.update(model, step)
        step_router_bias(model)

        state.step, state.last_loss = step + 1, total_loss
        if state.step % cfg.train.log_every == 0 or state.step == cfg.train.steps:
            elapsed = max(time.time() - t0, 1e-9)
            log.info(
                "step %d/%d loss=%.4f grad_norm=%.3f lr=%.2e (%.1f steps/s)",
                state.step,
                cfg.train.steps,
                total_loss,
                grad_norm,
                scheduler.get_last_lr()[0],
                cfg.train.log_every / elapsed,
            )
            for k, v in moe_metrics(model).items():
                log.info("  %s = %.3f", k, v)
            t0 = time.time()
        if state.step % cfg.train.ckpt_every == 0 or state.step == cfg.train.steps:
            save_checkpoint(
                ckpt_dir / f"step_{state.step:08d}",
                model=model,
                optimizer=opt,
                scheduler=scheduler,
                ema=ema,
                step=state.step,
            )
    return state
