"""First ablation matrix: attention kind (softmax / linear / hybrid) and FFN kind (dense / MoE
at ~equal active parameters, per ADR 0004's expert-size formula). Trains each variant on a real
precomputed cache and records loss and per-step wall-clock.

Latency/VRAM/throughput, as the original plan asked for, need a GPU this sandbox doesn't have —
what's reported here is CPU wall-clock per step, which is not a proxy for GPU latency (different
bottlenecks entirely) and is labeled as such in the output, not silently substituted for it.

Usage: python scripts/ablations.py configs/experiment/proxy_baseline.yaml data.root=/tmp/cache
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from imggen.config import dump_config, load_config
from imggen.data.loader import build_dataloader
from imggen.sweep import run_sweep_point
from imggen.train import cycle
from imggen.utils import setup_logging

log = logging.getLogger("imggen.ablations")

VARIANTS: dict[str, list[str]] = {
    "attn": [
        "backbone.attn=softmax",
        "backbone.attn=linear",
        "backbone.attn=softmax|backbone.attn_pattern=[linear,linear,softmax,linear]",
    ],
    "ffn": [
        "backbone.ffn=dense",
        'backbone.ffn=moe|backbone.moe={"num_experts": 8, "top_k": 2, "shared_experts": 1}',
    ],
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    setup_logging()
    base_cfg = load_config(args.config, args.overrides)
    log.info("baseline config:\n%s", dump_config(base_cfg))

    results: list[dict] = []
    for group, variants in VARIANTS.items():
        for variant in variants:
            overrides = [*args.overrides, *variant.split("|")]
            cfg = load_config(args.config, overrides)
            loader = build_dataloader(
                cfg.data, batch_size=cfg.train.batch_size, backbone=cfg.backbone
            )
            data_iter = cycle(loader)
            result = run_sweep_point(cfg, steps=args.steps, data_iter=data_iter, seed=args.seed)
            params = _param_counts(cfg)
            row = {
                "group": group,
                "variant": variant,
                "final_loss": result.final_loss,
                "tail_avg_loss": result.tail_avg_loss(),
                "cpu_seconds_per_step": round(result.seconds / args.steps, 4),
                "total_params": params["total"],
                "active_params": params["active"],
            }
            results.append(row)
            log.info("%s", row)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        log.info("wrote %d results to %s", len(results), args.out)
    return 0


def _param_counts(cfg) -> dict[str, int]:
    from imggen.config import estimate_params

    return estimate_params(cfg.backbone)


if __name__ == "__main__":
    raise SystemExit(main())
