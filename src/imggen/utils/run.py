"""Run directories: every run records its resolved config, git state and environment."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from imggen.config import ExperimentConfig, config_hash, dump_config


def git_info(cwd: str | Path | None = None) -> dict:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def _env_info() -> dict:
    def ver(pkg: str) -> str | None:
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": ver("torch"),
        "imggen": ver("imggen"),
    }


def create_run_dir(
    cfg: ExperimentConfig, *, timestamp: str | None = None, git_cwd: str | Path | None = None
) -> Path:
    """Create ``<output_dir>/<name>/<timestamp>_<confighash>/`` and write metadata into it."""
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.run.output_dir) / cfg.run.name / f"{ts}_{config_hash(cfg)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.resolved.yaml").write_text(dump_config(cfg))
    (run_dir / "git.json").write_text(json.dumps(git_info(git_cwd), indent=2))
    (run_dir / "env.json").write_text(json.dumps(_env_info(), indent=2))
    return run_dir
