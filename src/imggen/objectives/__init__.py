"""Objectives registered in imggen.registry.OBJECTIVE (auto-discovered)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from imggen.config import ComponentConfig


def build_objective(cfg: ComponentConfig) -> Any:
    from imggen.registry import OBJECTIVE, discover

    discover(("objectives",))
    return cfg.build(OBJECTIVE)
