"""Entry point: resolve config, seed, create (or reuse) a run dir, train.

Usage: python scripts/train.py configs/experiment/proxy_baseline.yaml [key.path=value ...]
       python scripts/train.py configs/experiment/proxy_baseline.yaml \
           --run-dir runs/proxy_baseline/<ts>_<hash>  # resumes that run's checkpoint, if any
"""

from __future__ import annotations

import argparse
import logging

import torch.distributed as dist

from imggen.config import dump_config, load_config
from imggen.parallel import init_distributed, is_distributed_run
from imggen.registry import discover
from imggen.train import run_training
from imggen.utils import create_run_dir, seed_everything, setup_logging

log = logging.getLogger("imggen.train")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. backbone.width=512")
    ap.add_argument("--print-config", action="store_true", help="print resolved config and exit")
    ap.add_argument(
        "--run-dir", default=None, help="reuse an existing run dir (resumes its checkpoint)"
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    if args.print_config:
        print(dump_config(cfg))
        return 0

    # Under a distributed launcher (torchrun sets WORLD_SIZE), join the process group *before*
    # anything touches torch.distributed — parallelize() inside run_training needs one to
    # already exist (previously nothing here called this at all: a real torchrun launch would
    # have crashed with "Default process group has not been initialized" the moment it ran; see
    # ADR 0012). Harmless/no-op for the single-process case this is normally tested under.
    if is_distributed_run():
        init_distributed()

    seed_everything(cfg.run.seed, cfg.run.deterministic)

    # Only rank 0 picks (or is given) the run dir and creates it — every other rank must use the
    # *same* directory (checkpointing is a collective: every rank calls save_checkpoint
    # together), so the path is broadcast rather than each rank computing its own. Without this,
    # every rank would independently call create_run_dir() and either land on different
    # directories or race to create the same timestamped one.
    is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
    run_dir = (args.run_dir or create_run_dir(cfg)) if is_rank0 else None
    if dist.is_initialized():
        holder = [run_dir]
        dist.broadcast_object_list(holder, src=0)
        run_dir = holder[0]

    setup_logging(run_dir=run_dir)
    discover()
    log.info("run dir: %s", run_dir)
    run_training(cfg, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
