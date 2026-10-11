"""Latency, throughput, and VRAM benchmark for a backbone config — the measurement the original
plan asked every ablation to carry, and that no CPU-only sandbox can produce (see ADR 0012).

Reports, per step (forward + backward, after a warmup period, timed with
``torch.cuda.synchronize()`` around each step so the numbers reflect actual device completion,
not just kernel-launch return):
  - latency: median wall-clock seconds per step
  - throughput: batch_size * world_size / latency, in samples/sec
  - peak VRAM: ``torch.cuda.max_memory_allocated()``, reset before each variant

On a CPU-only machine (this sandbox), VRAM is reported as null and latency/throughput are CPU
numbers — printed and saved, but explicitly labeled, never silently presented as GPU numbers.

Usage (single process):
    python scripts/benchmark.py --width 1280 --depth 24 --attn softmax --batch-size 8
    python scripts/benchmark.py --size xl --ffn moe --out docs/results/benchmark_xl_moe.json

Usage (multi-GPU, once available):
    torchrun --standalone --nproc-per-node=N scripts/benchmark.py --size xl \
        --parallel-strategy fsdp --tp-size 2
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

import torch

from imggen.backbones import build_backbone
from imggen.config import BackboneConfig, estimate_params, format_params
from imggen.objectives.flow_matching import FlowMatchingObjective
from imggen.parallel import init_distributed, is_distributed_run, parallelize
from imggen.utils import setup_logging

log = logging.getLogger("imggen.benchmark")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_config(
    cfg: BackboneConfig,
    *,
    batch_size: int,
    resolution: int,
    device: torch.device,
    warmup_steps: int = 5,
    measure_steps: int = 20,
    world_size: int = 1,
    model: torch.nn.Module | None = None,
) -> dict:
    """Pass ``model`` already built (and, for a distributed run, already parallelized — build
    the optimizer only after that, per ADR 0008: FSDP/TP replace parameter objects, so an
    optimizer built first would hold stale references) to benchmark that exact model. Otherwise
    builds and benchmarks a fresh, unparallelized one directly from ``cfg`` — this is the
    single-process path every non-distributed call uses."""
    if model is None:
        model = build_backbone(cfg).to(device)
    objective = FlowMatchingObjective()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    g = torch.Generator(device="cpu").manual_seed(0)
    latent = torch.randn(batch_size, cfg.in_channels, resolution, resolution, generator=g).to(
        device
    )
    text = torch.randn(batch_size, 8, cfg.text_dim, generator=g).to(device)
    mask = torch.ones(batch_size, 8, dtype=torch.bool, device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        loss, _ = objective.loss(model, latent, text, mask)
        loss.backward()
        opt.step()

    for _ in range(warmup_steps):
        step()
    _sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latencies = []
    for _ in range(measure_steps):
        _sync(device)
        t0 = time.perf_counter()
        step()
        _sync(device)
        latencies.append(time.perf_counter() - t0)

    latency = statistics.median(latencies)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None
    params = estimate_params(cfg)
    return {
        "device": device.type,
        "latency_s_per_step": round(latency, 4),
        "throughput_samples_per_s": round(batch_size * world_size / latency, 2),
        "peak_vram_mb": round(peak_vram_mb, 1) if peak_vram_mb is not None else None,
        "total_params": params["total"],
        "total_params_fmt": format_params(params["total"]),
        "active_params": params["active"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", default=None, help="a named rung from imggen.config.sizes.SIZES")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--attn", default="softmax", choices=["softmax", "linear"])
    ap.add_argument("--ffn", default="dense", choices=["dense", "moe"])
    ap.add_argument("--in-channels", type=int, default=32)
    ap.add_argument("--text-dim", type=int, default=2048)
    ap.add_argument("--resolution", type=int, default=32, help="latent H=W, in tokens")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--measure-steps", type=int, default=20)
    ap.add_argument(
        "--parallel-strategy", default="none", choices=["none", "ddp", "fsdp"],
        help="requires launching under torchrun",
    )  # fmt: skip
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    setup_logging()
    if args.size:
        backbone = BackboneConfig(
            size=args.size, head_dim=args.head_dim, attn=args.attn, ffn=args.ffn,
            in_channels=args.in_channels, text_dim=args.text_dim,
        )  # fmt: skip
    else:
        if args.width is None or args.depth is None:
            ap.error("pass --size, or both --width and --depth")
        backbone = BackboneConfig(
            width=args.width, depth=args.depth, head_dim=args.head_dim, attn=args.attn,
            ffn=args.ffn, in_channels=args.in_channels, text_dim=args.text_dim,
        )  # fmt: skip

    world_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if is_distributed_run():
        init_distributed()
        import torch.distributed as dist

        world_size = dist.get_world_size()
        from imggen.config import ParallelConfig

        pcfg = ParallelConfig(strategy=args.parallel_strategy, tp_size=args.tp_size)
        model = build_backbone(backbone).to(device)
        if pcfg.strategy != "none" or pcfg.tp_size > 1:
            model = parallelize(model, pcfg, device=device)  # benchmarked below, sharded

    if device.type == "cpu":
        log.warning(
            "no CUDA device available — latency/throughput below are CPU numbers, not a proxy "
            "for GPU behavior, and peak_vram_mb is null (see ADR 0012)"
        )

    result = benchmark_config(
        backbone, batch_size=args.batch_size, resolution=args.resolution, device=device,
        warmup_steps=args.warmup_steps, measure_steps=args.measure_steps, world_size=world_size,
        model=model,
    )  # fmt: skip
    result["config"] = {
        "width": backbone.width, "depth": backbone.depth, "attn": backbone.attn,
        "ffn": backbone.ffn, "batch_size": args.batch_size, "resolution": args.resolution,
        "world_size": world_size,
    }  # fmt: skip
    log.info("%s", result)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        log.info("wrote results to %s", args.out)

    if torch.distributed.is_initialized():
        # Without this, a torchrun launch can hang at process-group teardown (found by actually
        # launching under torchrun, not by inspection — every other multi-process entry point in
        # this project either never initializes a process group it owns, or a test harness tears
        # one down explicitly; this script is the first to both initialize one itself and need
        # to exit cleanly on its own).
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
