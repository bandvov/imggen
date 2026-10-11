"""Coordinate-wise hyperparameter sweep on the proxy config: lr, weight_decay, warmup_steps,
batch_size, and the flow-matching objective's timestep distribution. One knob varies at a time
from a fixed baseline (not a full grid — the original plan's "sweep learning rate, weight decay,
warmup, batch size... on the ~50M proxy" is naturally coordinate-wise at this stage; a full
cross-product is what the later, larger-scale tuning pass would do).

Uses a real precomputed cache (build one with precompute.py first) rather than synthetic data,
so results say something about the actual data pipeline, not just optimization mechanics.

Usage: python scripts/sweep.py configs/experiment/proxy_baseline.yaml data.root=/tmp/cache
       python scripts/sweep.py configs/experiment/proxy_baseline.yaml data.root=/tmp/cache \
           --steps 50 --out docs/results/step10_sweep.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from imggen.config import dump_config, load_config
from imggen.data.loader import build_dataloader
from imggen.sweep import run_sweep_point
from imggen.train import cycle
from imggen.utils import setup_logging

log = logging.getLogger("imggen.sweep")

# (dotted override path, [candidate values]); baseline is the config's own value for anything
# not listed here, so each knob is varied one at a time against the same fixed baseline.
KNOBS: dict[str, list] = {
    "train.lr": [3e-4, 1e-3, 3e-3, 1e-2],
    "train.weight_decay": [0.0, 0.01, 0.1],
    "train.warmup_steps": [1, 5, 20],
    "train.batch_size": [4, 8, 16],
    "objective.args.t_dist": ["uniform", "logit_normal"],
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*", help="dotted overrides applied to every sweep point")
    ap.add_argument("--steps", type=int, default=50, help="training steps per sweep point")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write results as JSON to this path")
    ap.add_argument(
        "--only", default=None, help="comma-separated knob name(s) to sweep, e.g. train.lr"
    )
    args = ap.parse_args(argv)

    setup_logging()
    base_cfg = load_config(args.config, args.overrides)
    log.info("baseline config:\n%s", dump_config(base_cfg))

    knobs = KNOBS
    if args.only:
        wanted = set(args.only.split(","))
        knobs = {k: v for k, v in KNOBS.items() if k in wanted}

    results: list[dict] = []
    for knob, values in knobs.items():
        for value in values:
            cfg = load_config(args.config, [*args.overrides, f"{knob}={value}"])
            loader = build_dataloader(
                cfg.data, batch_size=cfg.train.batch_size, backbone=cfg.backbone
            )
            data_iter = cycle(loader)
            t0 = time.time()
            result = run_sweep_point(cfg, steps=args.steps, data_iter=data_iter, seed=args.seed)
            row = {
                "knob": knob,
                "value": value,
                "final_loss": result.final_loss,
                "tail_avg_loss": result.tail_avg_loss(),
                "seconds": round(time.time() - t0, 2),
            }
            results.append(row)
            log.info("%s", row)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        log.info("wrote %d results to %s", len(results), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
