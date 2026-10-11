"""Named backbone sizes. A rung is only (width, depth); everything else is derived.

Rules the ladder follows:
- head_dim is fixed (64), so heads = width // 64 and width scales by adding heads (muP-friendly).
- width grows faster than depth: muP hyperparameter transfer is reliable across width, while
  depth transfer needs extra care, so adjacent rungs often share depth.
- Sizes beyond the named rungs need no code: pass explicit ``width``/``depth``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SizeSpec:
    name: str
    width: int
    depth: int


SIZES: dict[str, SizeSpec] = {
    s.name: s
    for s in (
        SizeSpec("nano", 256, 8),
        SizeSpec("proxy", 384, 12),  # the muP tuning proxy
        SizeSpec("s", 512, 12),
        SizeSpec("b", 768, 16),
        SizeSpec("l", 1024, 20),
        SizeSpec("xl", 1280, 24),  # ~0.7B: the v1 target size
        SizeSpec("2xl", 1536, 28),
        SizeSpec("3xl", 2048, 32),
        SizeSpec("4xl", 3072, 40),
    )
}


def get_size(name: object) -> SizeSpec:
    if not isinstance(name, str):
        raise ValueError(f"size must be a string, got {name!r}")
    try:
        return SIZES[name.lower()]
    except KeyError:
        raise ValueError(f"unknown size {name!r} (available: {list(SIZES)})") from None
