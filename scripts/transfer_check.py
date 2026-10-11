"""The muP transfer check: sweep a learning rate grid at the proxy width and at larger widths,
under muP and under standard parametrization, and see whether the *argmin* LR stays in the same
place as width grows. This is the actual promise ADR 0005 introduced muP for and ADR 0009's
coordinate check explicitly deferred ("tests activation flatness... not the actual promise").

Uses fixed synthetic batches (imggen.sweep.fixed_synthetic_batches), not a real cache: this check
is purely about optimization dynamics (where does the loss-minimizing LR sit), not about what the
model learns, so there's nothing to gain from real data here and a lot to gain from keeping every
width/seed/LR combination cheap and self-contained.

Usage: python scripts/transfer_check.py [--widths 192 384 768] [--steps 40] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

from imggen.config import BackboneConfig, ComponentConfig, ExperimentConfig, TrainConfig
from imggen.sweep import fixed_synthetic_batches, run_sweep_point
from imggen.utils import setup_logging

log = logging.getLogger("imggen.transfer_check")

DEFAULT_LR_GRID = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]


def make_cfg(
    width: int, base_width: int, use_mup: bool, lr: float, warmup_steps: int
) -> ExperimentConfig:
    backbone = BackboneConfig(
        width=width, depth=4, head_dim=32, text_dim=128, in_channels=16, mlp_ratio=4.0,
        mup={"enabled": use_mup, "base_width": base_width},
    )  # fmt: skip
    train = TrainConfig(steps=1000, lr=lr, warmup_steps=warmup_steps, batch_size=8)
    return ExperimentConfig(
        backbone=backbone, train=train,
        autoencoder=ComponentConfig(name="toy"), text_encoder=ComponentConfig(name="hash"),
        objective=ComponentConfig(name="flow_matching"),
    )  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--widths", type=int, nargs="+", default=[192, 384, 768])
    ap.add_argument("--base-width", type=int, default=None, help="defaults to widths[0]")
    ap.add_argument("--lr-grid", type=float, nargs="+", default=DEFAULT_LR_GRID)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument(
        "--condition", choices=["mup", "standard", "both"], default="both",
        help="run only the muP condition, only standard, or both",
    )  # fmt: skip
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    base_width = args.base_width or args.widths[0]
    conditions = {"mup": [True], "standard": [False], "both": [True, False]}[args.condition]

    setup_logging()
    rows: list[dict] = []
    for use_mup in conditions:
        for width in args.widths:
            for lr in args.lr_grid:
                losses = []
                for seed in args.seeds:
                    cfg = make_cfg(
                        width, base_width, use_mup, lr, warmup_steps=min(3, args.steps - 1)
                    )
                    data_iter = fixed_synthetic_batches(cfg, n_batches=4, batch_size=8, seed=seed)
                    t0 = time.time()
                    result = run_sweep_point(cfg, steps=args.steps, data_iter=data_iter, seed=seed)
                    losses.append(result.tail_avg_loss())
                row = {
                    "mup": use_mup, "width": width, "lr": lr,
                    "mean_tail_loss": statistics.mean(losses),
                    "seconds": round(time.time() - t0, 2),
                }  # fmt: skip
                rows.append(row)
                log.info("%s", row)

    # For each (mup, width), find the LR with the lowest mean tail loss (only meaningful once
    # every width/condition combination has been run — chunked invocations skip this here and
    # the results get merged and analyzed afterward instead).
    best: dict[tuple[bool, int], float] = {}
    for use_mup in conditions:
        for width in args.widths:
            candidates = [r for r in rows if r["mup"] == use_mup and r["width"] == width]
            if candidates:
                best[(use_mup, width)] = min(candidates, key=lambda r: r["mean_tail_loss"])["lr"]

    log.info("best LR per (mup, width):")
    for (use_mup, width), lr in best.items():
        log.info("  mup=%s width=%d -> best_lr=%s", use_mup, width, lr)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"rows": rows, "best_lr": {str(k): v for k, v in best.items()}}, indent=2)
        )
        log.info("wrote results to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
