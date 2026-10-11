import torch

from imggen.config import BackboneConfig, ComponentConfig, ExperimentConfig, TrainConfig
from imggen.sweep import SweepResult, fixed_synthetic_batches, run_sweep_point


def tiny_cfg(**train_kw) -> ExperimentConfig:
    backbone = BackboneConfig(
        width=32, depth=1, head_dim=16, text_dim=16, in_channels=4, mlp_ratio=2.0
    )
    train = TrainConfig(steps=1000, warmup_steps=1, batch_size=4, **train_kw)
    return ExperimentConfig(
        backbone=backbone, train=train,
        autoencoder=ComponentConfig(name="toy"), text_encoder=ComponentConfig(name="hash"),
        objective=ComponentConfig(name="flow_matching"),
    )  # fmt: skip


def test_sweep_result_final_and_tail_avg():
    r = SweepResult(losses=[5.0, 4.0, 3.0, 2.0, 1.0])
    assert r.final_loss == 1.0
    assert r.tail_avg_loss(3) == (3.0 + 2.0 + 1.0) / 3
    assert r.tail_avg_loss(100) == sum(r.losses) / len(r.losses)  # clamps to available history


def test_fixed_synthetic_batches_shapes_and_determinism():
    cfg = tiny_cfg()
    it = fixed_synthetic_batches(cfg, n_batches=2, batch_size=4, seed=0)
    b1 = next(it)
    assert b1["latent"].shape == (4, 4, 8, 8)
    assert b1["text_emb"].shape == (4, 6, 16)
    assert b1["mask"].shape == (4, 6) and b1["mask"].all()

    it_a = fixed_synthetic_batches(cfg, n_batches=1, batch_size=4, seed=1)
    it_b = fixed_synthetic_batches(cfg, n_batches=1, batch_size=4, seed=1)
    torch.testing.assert_close(next(it_a)["latent"], next(it_b)["latent"])


def test_fixed_synthetic_batches_cycles():
    cfg = tiny_cfg()
    it = fixed_synthetic_batches(cfg, n_batches=2, batch_size=4, seed=0)
    first_two = [next(it)["latent"] for _ in range(2)]
    next_two = [next(it)["latent"] for _ in range(2)]
    torch.testing.assert_close(first_two[0], next_two[0])
    torch.testing.assert_close(first_two[1], next_two[1])


def test_run_sweep_point_trains_and_reduces_loss():
    cfg = tiny_cfg(lr=3e-3)
    torch.manual_seed(0)
    data_iter = fixed_synthetic_batches(cfg, n_batches=1, batch_size=4, seed=0)
    result = run_sweep_point(cfg, steps=60, data_iter=data_iter, seed=0)
    assert len(result.losses) == 60
    assert result.tail_avg_loss(10) < result.losses[0]
    assert result.seconds > 0


def test_run_sweep_point_is_deterministic_given_a_seed():
    cfg = tiny_cfg(lr=3e-3)
    results = []
    for _ in range(2):
        torch.manual_seed(0)
        data_iter = fixed_synthetic_batches(cfg, n_batches=1, batch_size=4, seed=0)
        results.append(run_sweep_point(cfg, steps=10, data_iter=data_iter, seed=0).losses)
    assert results[0] == results[1]


def test_run_sweep_point_respects_mup_param_groups():
    cfg = tiny_cfg(lr=3e-3)
    cfg = cfg.model_copy(
        update={"backbone": cfg.backbone.replace(mup={"enabled": True, "base_width": 32})}
    )
    data_iter = fixed_synthetic_batches(cfg, n_batches=1, batch_size=4, seed=0)
    result = run_sweep_point(cfg, steps=10, data_iter=data_iter, seed=0)
    assert len(result.losses) == 10 and all(
        torch.isfinite(torch.tensor(loss)) for loss in result.losses
    )


# ---------------------------------------------------------------- CLI smoke tests


def _run_script(repo_root, script, *args):
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, str(repo_root / "scripts" / script), *args], capture_output=True, text=True
    )


def test_sweep_script_runs_one_knob_and_writes_json(repo_root, experiment_yaml, tmp_path):
    import json

    from imggen.data.shards import Manifest, ShardWriter

    cache = tmp_path / "cache"
    writer = ShardWriter(cache, shard_size=4)
    g = torch.Generator().manual_seed(0)
    for i in range(4):
        writer.add(
            f"{i:04d}", torch.randn(4, 4, 4, generator=g), torch.randn(3, 16, generator=g),
            torch.ones(3, dtype=torch.bool), f"c{i}",
        )  # fmt: skip
    writer.close()
    Manifest(
        num_samples=4, num_shards=1, shard_size=4, latent_channels=4, spatial_compression=8,
        text_dim=16, resolution=32,
    ).save(cache)  # fmt: skip

    out = tmp_path / "results.json"
    r = _run_script(
        repo_root, "sweep.py", str(experiment_yaml),
        f"data.root={cache}", "data.num_workers=0",
        "backbone.width=32", "backbone.depth=1", "backbone.head_dim=16",
        "backbone.text_dim=16", "backbone.in_channels=4", "backbone.mlp_ratio=2.0",
        "train.warmup_steps=1", "train.batch_size=2",
        "--only", "train.lr", "--steps", "3", "--out", str(out),
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    rows = json.loads(out.read_text())
    assert len(rows) == 4 and all(row["knob"] == "train.lr" for row in rows)


def test_transfer_check_script_runs_and_writes_json(repo_root, tmp_path):
    import json

    out = tmp_path / "tc.json"
    r = _run_script(
        repo_root, "transfer_check.py", "--widths", "32", "64", "--lr-grid", "1e-3", "3e-3",
        "--steps", "3", "--seeds", "0", "--out", str(out),
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    data = json.loads(out.read_text())
    assert len(data["rows"]) == 2 * 2 * 2  # 2 conditions x 2 widths x 2 lrs
    assert len(data["best_lr"]) == 4


def test_ablations_script_runs_and_writes_json(repo_root, experiment_yaml, tmp_path):
    import json

    from imggen.data.shards import Manifest, ShardWriter

    cache = tmp_path / "cache"
    writer = ShardWriter(cache, shard_size=4)
    g = torch.Generator().manual_seed(0)
    for i in range(4):
        writer.add(
            f"{i:04d}", torch.randn(4, 4, 4, generator=g), torch.randn(3, 16, generator=g),
            torch.ones(3, dtype=torch.bool), f"c{i}",
        )  # fmt: skip
    writer.close()
    Manifest(
        num_samples=4, num_shards=1, shard_size=4, latent_channels=4, spatial_compression=8,
        text_dim=16, resolution=32,
    ).save(cache)  # fmt: skip

    out = tmp_path / "ablations.json"
    r = _run_script(
        repo_root, "ablations.py", str(experiment_yaml),
        f"data.root={cache}", "data.num_workers=0",
        "backbone.width=32", "backbone.depth=2", "backbone.head_dim=16",
        "backbone.text_dim=16", "backbone.in_channels=4", "backbone.mlp_ratio=2.0",
        "train.warmup_steps=1", "train.batch_size=2", "--steps", "3", "--out", str(out),
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    rows = json.loads(out.read_text())
    assert {row["group"] for row in rows} == {"attn", "ffn"}
    assert any(row["variant"] == "backbone.ffn=dense" for row in rows)
