"""Linear warmup, then a decay shape, expressed as the multiplier ``torch.optim.lr_scheduler.
LambdaLR`` expects (LR at step i = base_lr * lambda(i)).
"""

from __future__ import annotations

import math
from collections.abc import Callable


def build_lr_lambda(
    warmup_steps: int, total_steps: int, schedule: str = "cosine", min_lr_ratio: float = 0.0
) -> Callable[[int], float]:
    if schedule not in ("constant", "cosine", "linear"):
        raise ValueError(f"unknown lr_schedule {schedule!r}")
    if warmup_steps >= total_steps:
        raise ValueError(f"warmup_steps ({warmup_steps}) must be < total_steps ({total_steps})")
    decay_steps = max(total_steps - warmup_steps, 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps  # step 0 -> 1/warmup_steps, never exactly 0
        if schedule == "constant":
            return 1.0
        progress = min((step - warmup_steps) / decay_steps, 1.0)
        if schedule == "linear":
            factor = 1.0 - progress
        else:  # cosine
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * factor

    return lr_lambda
