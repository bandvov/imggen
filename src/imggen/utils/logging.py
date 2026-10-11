"""Rank-aware logging: rank 0 logs at the requested level, other ranks only warnings."""

from __future__ import annotations

import logging
import os
from pathlib import Path

_TAG = "_imggen_handler"


def setup_logging(level: str = "INFO", run_dir: str | Path | None = None) -> logging.Logger:
    rank = int(os.environ.get("RANK", "0"))
    logger = logging.getLogger("imggen")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for h in list(logger.handlers):
        if getattr(h, _TAG, False):
            logger.removeHandler(h)
            h.close()

    fmt = logging.Formatter(f"%(asctime)s [rank{rank}] %(levelname)s %(name)s: %(message)s")
    eff_level = level if rank == 0 else "WARNING"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if run_dir is not None:
        handlers.append(logging.FileHandler(Path(run_dir) / f"log.rank{rank}.txt"))
    for h in handlers:
        h.setLevel(eff_level)
        h.setFormatter(fmt)
        setattr(h, _TAG, True)
        logger.addHandler(h)
    return logger
