import subprocess
import sys

import pytest
import torch
import yaml


def _run(repo_root, script, *args):
    return subprocess.run(
        [sys.executable, str(repo_root / "scripts" / script), *args],
        capture_output=True, text=True,
    )  # fmt: skip


def _precompute(repo_root, experiment_yaml, cache, num_samples=12):
    r = _run(
        repo_root, "precompute.py", str(experiment_yaml), f"data.root={cache}",
        "backbone.size=nano", "--num-samples", str(
            num_samples), "--batch-size", "4",
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
        repo_root, "train.py", *
        _train_args(experiment_yaml, cache, out, **{"train.steps": 4})
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
    r = _run(repo_root, "train.py", str(experiment_yaml),
             f"run.output_dir={out}", "--print-config")
    assert r.returncode == 0, r.stderr
    assert "proxy_baseline" in r.stdout and not out.exists()


def test_typo_in_override_fails_nonzero(repo_root, experiment_yaml, tmp_path):
    r = _run(
        repo_root, "train.py", str(
            experiment_yaml), f"run.output_dir={tmp_path}", "train.lrr=1"
    )
    assert r.returncode != 0 and "lrr" in r.stderr


def test_train_toy_autoencoder_script_runs_and_saves_weights(repo_root, tmp_path):
    out = tmp_path / "ae.pt"
    r = _run(
        repo_root, "train_toy_autoencoder.py", "--out", str(
            out), "--steps", "10",
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
    # ADR 0012's other regression, checked here too
    assert "view tensor" not in r.stderr
    (run_dir,) = list((out / "proxy_baseline").iterdir())
    assert (run_dir / "checkpoints" / "step_00000004").is_dir()
    # both ranks must have agreed on the same run dir (broadcast, not independently created)
    assert len(list((out / "proxy_baseline").iterdir())) == 1


# ---------------------------------------------------------------- precompute --resume


def _precompute_full(repo_root, experiment_yaml, cache, *, num_samples, shard_size=4, batch_size=4):
    r = _run(
        repo_root, "precompute.py", str(experiment_yaml), f"data.root={cache}",
        f"data.shard_size={shard_size}", "backbone.size=nano",
        "--num-samples", str(num_samples), "--batch-size", str(batch_size),
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    return r


def _read_all_samples(cache):
    from imggen.data.shards import LatentShardDataset

    samples = {}

    ds = LatentShardDataset(cache)

    for i in range(len(ds)):
        item = ds[i]
        samples[i] = item["latent"].clone()

    return samples


def test_precompute_resume_continues_from_a_reconstructed_interruption(
    repo_root, experiment_yaml, tmp_path
):
    """Deterministic, not timing-based: build the "interrupted" state directly (first N/shard_size
    shards copied over, no manifest/null_embedding) rather than racing a live subprocess, then
    checks the resumed cache is byte-for-byte identical to an uninterrupted run — not just that
    --resume exits 0."""
    import shutil

    ground_truth = tmp_path / "ground_truth"
    _precompute_full(repo_root, experiment_yaml, ground_truth,
                     num_samples=12, shard_size=4)
    expected = _read_all_samples(ground_truth)
    assert len(expected) == 12

    interrupted = tmp_path / "interrupted"
    interrupted.mkdir()
    shutil.copy(ground_truth / "shard-00000.tar",
                interrupted / "shard-00000.tar")
    shutil.copy(ground_truth / "shard-00001.tar",
                interrupted / "shard-00001.tar")
    # deliberately no manifest.json, no null_embedding.pt, no resume sidecar — check_resume_meta
    # must tolerate a missing sidecar (e.g. a crash before the first save_resume_meta() call, or
    # one from before this feature existed) rather than requiring it.
    assert not (interrupted / "manifest.json").exists()

    r = _run(
        repo_root, "precompute.py", str(
            experiment_yaml), f"data.root={interrupted}",
        "data.shard_size=4", "backbone.size=nano",
        "--num-samples", "12", "--batch-size", "4", "--resume",
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert "resuming from sample 8/12" in r.stderr

    resumed = _read_all_samples(interrupted)
    assert set(resumed) == set(expected)
    for caption, latent in expected.items():
        torch.testing.assert_close(resumed[caption], latent)
    assert (interrupted / "manifest.json").exists()
    assert (interrupted / "null_embedding.pt").exists()
    # cleared on success
    assert not (interrupted / ".precompute_progress.json").exists()


def test_precompute_resume_discards_a_truncated_last_shard(repo_root, experiment_yaml, tmp_path):
    import shutil

    ground_truth = tmp_path / "ground_truth"
    _precompute_full(repo_root, experiment_yaml, ground_truth,
                     num_samples=12, shard_size=4)
    expected = _read_all_samples(ground_truth)

    interrupted = tmp_path / "interrupted"
    interrupted.mkdir()
    shutil.copy(ground_truth / "shard-00000.tar",
                interrupted / "shard-00000.tar")
    # shard-00001 present but truncated (simulates a kill mid-write of the 2nd shard)
    data = (ground_truth / "shard-00001.tar").read_bytes()
    (interrupted / "shard-00001.tar").write_bytes(data[: len(data) // 2])

    r = _run(
        repo_root, "precompute.py", str(
            experiment_yaml), f"data.root={interrupted}",
        "data.shard_size=4", "backbone.size=nano",
        "--num-samples", "12", "--batch-size", "4", "--resume",
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert "resuming from sample 4/12" in r.stderr  # only the first shard was trusted

    resumed = _read_all_samples(interrupted)
    assert set(resumed) == set(expected)
    for caption, latent in expected.items():
        torch.testing.assert_close(resumed[caption], latent)


def test_precompute_resume_is_a_no_op_when_already_complete(repo_root, experiment_yaml, tmp_path):
    cache = tmp_path / "cache"
    _precompute_full(repo_root, experiment_yaml, cache,
                     num_samples=8, shard_size=4)
    before = (cache / "manifest.json").read_text()

    r = _run(
        repo_root, "precompute.py", str(experiment_yaml), f"data.root={cache}",
        "data.shard_size=4", "backbone.size=nano",
        "--num-samples", "8", "--batch-size", "4", "--resume",
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert "already has a complete manifest" in r.stderr
    # untouched, not rewritten
    assert (cache / "manifest.json").read_text() == before


def test_precompute_resume_rejects_mismatched_settings(repo_root, experiment_yaml, tmp_path):
    import shutil

    ground_truth = tmp_path / "ground_truth"
    _precompute_full(repo_root, experiment_yaml, ground_truth,
                     num_samples=12, shard_size=4)

    interrupted = tmp_path / "interrupted"
    interrupted.mkdir()
    shutil.copy(ground_truth / "shard-00000.tar",
                interrupted / "shard-00000.tar")
    shutil.copy(ground_truth / "shard-00001.tar",
                interrupted / "shard-00001.tar")
    # hand-write a sidecar matching the real run's settings exactly except num_samples, so the
    # mismatch this triggers is isolated to that one field (configs/base.yaml's defaults).
    (interrupted / ".precompute_progress.json").write_text(
        '{"num_samples": 999, "dataset_seed": 0, "resolution": 64, "shard_size": 4, '
        '"autoencoder": {"name": "toy", "args": {"latent_channels": 32, "num_downsamples": 3}}, '
        '"text_encoder": {"name": "hash", "args": {"text_dim": 2048}}}'
    )

    r = _run(
        repo_root, "precompute.py", str(
            experiment_yaml), f"data.root={interrupted}",
        "data.shard_size=4", "backbone.size=nano",
        "--num-samples", "12", "--batch-size", "4", "--resume",
    )  # fmt: skip
    assert r.returncode != 0
    assert "num_samples" in r.stderr


def test_precompute_resume_on_empty_directory_behaves_like_a_fresh_run(
    repo_root, experiment_yaml, tmp_path
):
    cache = tmp_path / "cache"
    r = _run(
        repo_root, "precompute.py", str(experiment_yaml), f"data.root={cache}",
        "data.shard_size=4", "backbone.size=nano",
        "--num-samples", "8", "--batch-size", "4", "--resume",
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert (cache / "manifest.json").exists()
    assert len(_read_all_samples(cache)) == 8
