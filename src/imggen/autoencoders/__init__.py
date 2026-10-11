"""Autoencoders registered in imggen.registry.AUTOENCODER (auto-discovered)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Autoencoder

if TYPE_CHECKING:
    from imggen.config import ComponentConfig

__all__ = ["Autoencoder", "build_autoencoder"]


def build_autoencoder(cfg: ComponentConfig) -> Autoencoder:
    """``cfg.args`` may include ``weights_path`` (loaded via ``load_state_dict`` after
    construction, then popped so it never reaches the registered class's constructor) — a
    generic hook any autoencoder implementation can use, not just
    ``imggen.autoencoders.toy.ToyAutoencoder``, whose weights start random and are otherwise
    never trained (ADR 0006)."""
    from imggen.registry import AUTOENCODER, discover

    discover(("autoencoders",))
    args = dict(cfg.args)
    weights_path = args.pop("weights_path", None)
    model = AUTOENCODER.build(cfg.name, **args)
    if weights_path is not None:
        import torch

        model.load_state_dict(torch.load(weights_path, weights_only=True, map_location="cpu"))
    return model
