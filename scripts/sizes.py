"""Print the size ladder: shape, heads, provisional param count, max tensor-parallel degree.

Usage: python scripts/sizes.py [--head-dim 64] [--mlp-ratio 4.0] [--ffn dense|moe]
"""

from __future__ import annotations

import argparse

from imggen.config import SIZES, BackboneConfig, estimate_params, format_params


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--mlp-ratio", type=float, default=4.0)
    ap.add_argument("--ffn", choices=["dense", "moe"], default="dense")
    args = ap.parse_args(argv)

    print(
        f"{'size':<7}{'width':>7}{'depth':>7}{'heads':>7}{'ffn':>7}"
        f"{'params':>9}{'active':>9}{'max TP':>8}"
    )
    for name in SIZES:
        cfg = BackboneConfig(
            size=name, head_dim=args.head_dim, mlp_ratio=args.mlp_ratio, ffn=args.ffn
        )
        max_tp = cfg.num_heads & -cfg.num_heads  # largest power of two dividing the head count
        est = estimate_params(cfg)
        total, active = format_params(est["total"]), format_params(est["active"])
        print(
            f"{name:<7}{cfg.width:>7}{cfg.depth:>7}{cfg.num_heads:>7}"
            f"{cfg.ffn_hidden:>7}{total:>9}{active:>9}{max_tp:>8}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
