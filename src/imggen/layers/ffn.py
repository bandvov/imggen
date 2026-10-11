"""Feed-forward blocks. Interface: ``ffn(x, hw) -> x`` with x: (B, N, width), hw: (grid_h, grid_w).

``hw`` is unused by the dense FFN; it is passed so spatial variants (depthwise-conv "Mix-FFN")
can share one call signature.

Construction: ``FFN.build(name, width=w, hidden=r, moe=cfg.moe)``. ``hidden`` is the dense
FFN hidden size; the ``moe`` config is only read by the MoE variant. An FFN may define
``init_weights()``, which the backbone calls after its own initialisation.
"""

from __future__ import annotations

import torch.nn as nn

from imggen.registry import FFN


@FFN.register("dense")
class DenseFFN(nn.Module):
    def __init__(self, width: int, hidden: int, moe=None) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, width)

    def forward(self, x, hw=None):
        return self.fc2(self.act(self.fc1(x)))
