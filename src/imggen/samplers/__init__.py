"""Samplers registered in imggen.registry.SAMPLER (auto-discovered)."""

from __future__ import annotations


def build_sampler(name: str = "euler"):
    from imggen.registry import SAMPLER, discover

    discover(("samplers",))
    return SAMPLER.get(name)
