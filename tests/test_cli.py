import subprocess
import sys

import pytest
import yaml


def _run(repo_root, script, *args):
    return subprocess.run(
        [sys.executable, str(repo_root / "scripts" / script), *args],
        capture_output=True, text=True,
    )  # fmt: skip


def _precompute(repo_root, experiment_yaml, cache, num_samples=12):
    r = _run(
        repo_root, "precompute.py", str(experiment_yaml), f"data.root={cache}",
        "backbone.size=nano", "--num-samples", str(num_samples), "--batch-size", "4",
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    return r


def _train_args(experiment_yaml, cache, out, **overrides):
    base = {
        "run.output_dir": str(out), "data.root": str(cache), "data.num_workers": 0,
        "backbone.size": "nano", "train.steps": 4, "train.batch_size": 4,
        "train.ckpt_every": 4, "train.warmup_steps": 1,
    }  # fmt: skip
    base.update(overrides)
    return [str(experiment_yaml), *(f"{k}={v}" for k, v in base.items())]


def test_train_script_creates_run_dir_and_checkpoint(repo_root, experiment_yaml, tmp_path):
    cache, out = tmp_path / "cache", tmp_path / "runs"
    _precompute(repo_root, experiment_yaml, cache)
    r = _run(repo_root, "train.py", *_train_args(experiment_yaml, cache, out))
    assert r.returncode == 0, r.stderr
    (run_dir,) = list((out / "proxy_baseline").iterdir())
    cfg = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    assert cfg["backbone"]["size"] == "nano"
    assert (run_dir / "log.rank0.txt").exists()
    assert (run_dir / "checkpoints" / "step_00000004").is_dir()


def test_train_script_resumes_with_run_dir_flag(repo_root, experiment_yaml, tmp_path):
    cache, out = tmp_path / "cache", tmp_path / "runs"
    _precompute(repo_root, experiment_yaml, cache)
    r1 = _run(
        repo_root, "train.py", *_train_args(experiment_yaml, cache, out, **{"train.steps": 4})
    )
    assert r1.returncode == 0, r1.stderr
    (run_dir,) = list((out / "proxy_baseline").iterdir())

    r2 = _run(
        repo_root, "train.py", str(experiment_yaml),
        f"data.root={cache}", "data.num_workers=0", "backbone.size=nano",
        "train.steps=8", "train.batch_size=4", "train.ckpt_every=8", "train.warmup_steps=1",
        "--run-dir", str(run_dir),
    )  # fmt: skip
    assert r2.returncode == 0, r2.stderr
    assert "resumed from" in r2.stderr and "step 4" in r2.stderr
    ckpts = sorted(p.name for p in (run_dir / "checkpoints").iterdir())
    assert ckpts == ["step_00000004", "step_00000008"]


def test_print_config_does_not_create_run(repo_root, experiment_yaml, tmp_path):
    out = tmp_path / "runs"
    r = _run(repo_root, "train.py", str(experiment_yaml), f"run.output_dir={out}", "--print-config")
    assert r.returncode == 0, r.stderr
    assert "proxy_baseline" in r.stdout and not out.exists()


def test_typo_in_override_fails_nonzero(repo_root, experiment_yaml, tmp_path):
    r = _run(
        repo_root, "train.py", str(experiment_yaml), f"run.output_dir={tmp_path}", "train.lrr=1"
    )
    assert r.returncode != 0 and "lrr" in r.stderr


def test_train_toy_autoencoder_script_runs_and_saves_weights(repo_root, tmp_path):
    out = tmp_path / "ae.pt"
    r = _run(
        repo_root, "train_toy_autoencoder.py", "--out", str(out), "--steps", "10",
        "--batch-size", "4", "--resolution", "32", "--latent-channels", "8",
        "--num-downsamples", "2",
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert out.exists()
    import torch

    sd = torch.load(out, weights_only=True)
    assert "encoder.0.weight" in sd


def test_sample_script_runs_end_to_end(repo_root, experiment_yaml, tmp_path):
    cache, out = tmp_path / "cache", tmp_path / "runs"
    _precompute(repo_root, experiment_yaml, cache, num_samples=8)
    train_args = _train_args(experiment_yaml, cache, out)
    r = _run(repo_root, "train.py", *train_args)
    assert r.returncode == 0, r.stderr
    (run_dir,) = list((out / "proxy_baseline").iterdir())

    sample_out = tmp_path / "sample.pt"
    r = _run(
        repo_root, "sample.py", str(run_dir / "checkpoints" / "step_00000004"),
        str(experiment_yaml),
        f"data.root={cache}", "backbone.size=nano",
        "--caption", "a red circle", "--steps", "5", "--out", str(sample_out),
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    import torch

    data = torch.load(sample_out, weights_only=False)
    assert data["captions"] == ["a red circle"]
    assert data["images"].shape[0] == 1 and data["images"].shape[1] == 3


# ---------------------------------------------------------------- real torchrun launch (ADR 0012)


def test_train_script_under_real_torchrun_with_fsdp(repo_root, experiment_yaml, tmp_path):
    """Regression for ADR 0012: nothing previously called init_distributed(), so a real torchrun
    launch of this exact entry point would crash with "Default process group has not been
    initialized" the moment parallelize() ran. Uses the actual torchrun binary, not a simulation."""
    import shutil

    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not available in this environment")

    cache, out = tmp_path / "cache", tmp_path / "runs"
    _precompute(repo_root, experiment_yaml, cache, num_samples=16)
    args = _train_args(
        experiment_yaml, cache, out, **{"train.steps": 4, "train.ckpt_every": 4,
                                         "parallel.strategy": "fsdp"},
    )  # fmt: skip
    train_py = str(repo_root / "scripts" / "train.py")
    r = subprocess.run(
        ["torchrun", "--standalone", "--nproc-per-node=2", train_py, *args],
        capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert "view tensor" not in r.stderr  # ADR 0012's other regression, checked here too
    (run_dir,) = list((out / "proxy_baseline").iterdir())
    assert (run_dir / "checkpoints" / "step_00000004").is_dir()
    # both ranks must have agreed on the same run dir (broadcast, not independently created)
    assert len(list((out / "proxy_baseline").iterdir())) == 1
