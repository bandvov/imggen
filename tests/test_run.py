import json
import logging
import shutil
import subprocess

import pytest
import yaml

from imggen.config import config_hash, load_config
from imggen.utils import create_run_dir, git_info, setup_logging


def _cfg(experiment_yaml, tmp_path):
    return load_config(experiment_yaml, [f"run.output_dir={tmp_path / 'runs'}"])


def test_create_run_dir_writes_metadata(experiment_yaml, tmp_path):
    cfg = _cfg(experiment_yaml, tmp_path)
    d = create_run_dir(cfg, timestamp="20260101-000000", git_cwd=tmp_path)
    assert d.name == f"20260101-000000_{config_hash(cfg)}"
    assert d.parent.name == "proxy_baseline"
    resolved = yaml.safe_load((d / "config.resolved.yaml").read_text())
    assert resolved["run"]["name"] == "proxy_baseline"
    assert set(json.loads((d / "git.json").read_text())) == {"commit", "dirty"}
    assert "python" in json.loads((d / "env.json").read_text())


def test_run_dir_collision_fails(experiment_yaml, tmp_path):
    cfg = _cfg(experiment_yaml, tmp_path)
    create_run_dir(cfg, timestamp="t")
    with pytest.raises(FileExistsError):
        create_run_dir(cfg, timestamp="t")


def test_git_info_outside_repo(tmp_path):
    info = git_info(tmp_path)
    assert info == {"commit": None, "dirty": None}


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_info_in_repo_reports_dirty(tmp_path):
    def git(*a):
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", *a],
            cwd=tmp_path, check=True, capture_output=True,
        )  # fmt: skip

    git("init")
    (tmp_path / "f.txt").write_text("x")
    git("add", "f.txt")
    git("commit", "-m", "init")
    clean = git_info(tmp_path)
    assert len(clean["commit"]) == 40 and clean["dirty"] is False
    (tmp_path / "f.txt").write_text("y")
    assert git_info(tmp_path)["dirty"] is True


def test_setup_logging_idempotent_and_writes_file(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    setup_logging(run_dir=tmp_path)
    logger = setup_logging(run_dir=tmp_path)  # second call must not stack handlers
    assert len(logger.handlers) == 2
    logging.getLogger("imggen.test").info("hello")
    for h in logger.handlers:
        h.flush()
    assert "hello" in (tmp_path / "log.rank0.txt").read_text()


def test_nonzero_rank_only_warns(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "3")
    logger = setup_logging(run_dir=tmp_path)
    logging.getLogger("imggen.test").info("quiet")
    logging.getLogger("imggen.test").warning("loud")
    for h in logger.handlers:
        h.flush()
    text = (tmp_path / "log.rank3.txt").read_text()
    assert "loud" in text and "quiet" not in text
    setup_logging()  # restore rank-0 style handlers for other tests
