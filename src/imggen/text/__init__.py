"""Text encoders registered in imggen.registry.TEXT_ENCODER (auto-discovered)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import TextEncoder

if TYPE_CHECKING:
    from imggen.config import ComponentConfig

__all__ = ["TextEncoder", "build_text_encoder"]


def build_text_encoder(cfg: ComponentConfig) -> TextEncoder:
    from imggen.registry import TEXT_ENCODER, discover

    discover(("text",))
    return cfg.build(TEXT_ENCODER)
