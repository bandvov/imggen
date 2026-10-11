"""Backbones registered in imggen.registry.BACKBONE (auto-discovered)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

    from imggen.config import BackboneConfig


def build_backbone(cfg: BackboneConfig) -> nn.Module:
    """Build the backbone named by ``cfg.name`` (imports the implementations on first use)."""
    from imggen.registry import BACKBONE, discover

    discover(("layers", "backbones"))
    return BACKBONE.build(cfg.name, cfg)
