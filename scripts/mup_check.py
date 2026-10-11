"""Coarse muP diagnostic: print a block's self-attention output magnitude, at init, across a
width sweep, with muP on vs off. Flatter (muP) vs drifting (standard) is the expected shape.

This is a quick manual sanity check, not the rigorous coordinate/transfer check (roadmap step 9),
which needs several seeds, several layers, real training steps, and a pass/fail tolerance.

Usage: python scripts/mup_check.py [--base-width 384] [--widths 128 256 384 768 1536]
"""

from __future__ import annotations

import argparse

from imggen.config import BackboneConfig
from imggen.mup import activation_rms_by_width


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-width", type=int, default=384)
    ap.add_argument("--widths", type=int, nargs="+", default=[128, 256, 384, 768, 1536, 3072])
    ap.add_argument("--depth", type=int, default=4)
    args = ap.parse_args(argv)

    base = BackboneConfig(
        width=args.base_width,
        depth=args.depth,
        mup={"enabled": True, "base_width": args.base_width},
    )
    print(f"block-0 attention output |mean| at init, base_width={args.base_width}\n")
    print(f"{'width':>7}{'muP':>12}{'standard':>12}")
    mup_out = activation_rms_by_width(base, args.widths)
    std_out = activation_rms_by_width(base.replace(mup={"enabled": False}), args.widths)
    for w in args.widths:
        print(f"{w:>7}{mup_out[w]:>12.4f}{std_out[w]:>12.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
