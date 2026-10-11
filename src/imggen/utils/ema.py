"""Exponential moving average of a model's floating-point state (parameters and buffers).

Non-float buffers (none currently in this project, but e.g. an int counter) are skipped: an EMA
of a step counter or a boolean flag is meaningless.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: bool = True) -> None:
        self.decay = decay
        self.warmup = warmup
        self.shadow: OrderedDict[str, torch.Tensor] = OrderedDict(
            (k, v.detach().clone().float())
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        )

    def _effective_decay(self, step: int) -> float:
        if not self.warmup:
            return self.decay
        # standard EMA warmup: ramps from 0 up to `decay` over the first few hundred steps,
        # so early (large, noisy) updates don't get frozen into the average at full strength.
        return min(self.decay, (1.0 + step) / (10.0 + step))

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        decay = self._effective_decay(step)
        sd = model.state_dict()
        for name, shadow in self.shadow.items():
            shadow.mul_(decay).add_(sd[name].detach().float(), alpha=1 - decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self.shadow)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        for name, tensor in state_dict.items():
            self.shadow[name].copy_(tensor)

    def copy_to(self, model: nn.Module) -> None:
        """Overwrite ``model``'s weights with the EMA average, in place (e.g. before sampling)."""
        sd = model.state_dict()
        for name, shadow in self.shadow.items():
            sd[name].copy_(shadow.to(sd[name].dtype))

    class _Swap:
        def __init__(self, ema: EMA, model: nn.Module) -> None:
            self.ema, self.model = ema, model

        def __enter__(self) -> nn.Module:
            sd = self.model.state_dict()
            self._backup = {name: sd[name].detach().clone() for name in self.ema.shadow}
            self.ema.copy_to(self.model)
            return self.model

        def __exit__(self, *exc) -> None:
            sd = self.model.state_dict()
            for name, tensor in self._backup.items():
                sd[name].copy_(tensor)

    def average_parameters(self, model: nn.Module) -> EMA._Swap:
        """Context manager: swap in the EMA weights, restore the training weights on exit.

        with ema.average_parameters(model):
            sample(model, ...)
        """
        return EMA._Swap(self, model)
