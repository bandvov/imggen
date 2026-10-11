"""Checkpointing via ``torch.distributed.checkpoint`` (DCP), through the officially recommended
``get_state_dict``/``set_state_dict`` helpers (``torch.distributed.checkpoint.state_dict``).

Why those helpers and not a bare ``optimizer.state_dict()``: a freshly constructed optimizer has
no per-parameter state until its first ``step()``, so an in-place DCP load has nothing to fill in
for it — ``get_state_dict`` pre-populates that structure (verified directly: a never-stepped
optimizer's ``get_state_dict`` result already has real per-parameter entries). The same helpers
also handle FSDP2's DTensor-sharded parameters and optimizer state transparently, which is what
makes a checkpoint saved under one world size resumable under another (verified with a real
2-rank FSDP run whose checkpoint was then loaded single-process, and the reverse; see ADR 0008).
Model, optimizer and EMA all go through DCP for this reason. The LR scheduler and step counter
are plain Python state (not parameter-shaped), so they stay on ``torch.save``/``torch.load``.

This supersedes ADR 0007's split (optimizer/scheduler/EMA all on plain ``torch.save``): that
design hadn't yet been tested against ``get_state_dict``/``set_state_dict``, which turns out to
solve the fresh-optimizer problem more simply, and for the distributed case besides.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from imggen.utils.ema import EMA

log = logging.getLogger("imggen.checkpoint")

_STEP_DIR = re.compile(r"^step_(\d{8})$")


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: EMA,
    step: int,
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model_sd, optim_sd = get_state_dict(model, optimizer)
    dcp.save(
        {"model": model_sd, "optimizer": optim_sd, "ema": ema.state_dict()}, checkpoint_id=str(path)
    )
    torch.save({"scheduler": scheduler.state_dict(), "step": step}, path / "extra.pt")
    log.info("saved checkpoint at %s (step %d)", path, step)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: EMA,
    map_location: str | torch.device = "cpu",
) -> int:
    """Restores all four objects in place and returns the step the checkpoint was saved at
    (i.e. the step training should resume *from*: the next step to run)."""
    path = Path(path)
    model_sd, optim_sd = get_state_dict(model, optimizer)
    state_dict = {"model": model_sd, "optimizer": optim_sd, "ema": ema.state_dict()}
    dcp.load(state_dict, checkpoint_id=str(path))
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state_dict["model"],
        optim_state_dict=state_dict["optimizer"],
    )
    ema.load_state_dict(state_dict["ema"])
    extra = torch.load(path / "extra.pt", map_location=map_location, weights_only=False)
    scheduler.load_state_dict(extra["scheduler"])
    return extra["step"]


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """The highest-numbered ``step_XXXXXXXX`` directory under ``checkpoint_dir``, or None if it
    doesn't exist yet or holds nothing recognisable."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return None
    candidates = [p for p in checkpoint_dir.iterdir() if p.is_dir() and _STEP_DIR.match(p.name)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(_STEP_DIR.match(p.name).group(1)))
