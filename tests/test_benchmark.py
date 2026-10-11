import json
import shutil
import subprocess
import sys

import pytest
import torch

from imggen.config import BackboneConfig


def tiny_backbone(**kw):
    defaults = dict(width=32, depth=1, head_dim=16, in_channels=4, text_dim=16)
    defaults.update(kw)
    return BackboneConfig(**defaults)


def test_benchmark_config_returns_expected_fields():
    from scripts.benchmark import benchmark_config

    cfg = tiny_backbone()
    result = benchmark_config(
        cfg, batch_size=2, resolution=4, device=torch.device("cpu"), warmup_steps=1, measure_steps=3
    )
    assert result["device"] == "cpu"
    assert result["latency_s_per_step"] > 0
    assert result["throughput_samples_per_s"] > 0
    assert result["peak_vram_mb"] is None  # no CUDA in this sandbox
    assert result["total_params"] == result["active_params"]  # dense FFN


def test_benchmark_config_throughput_matches_batch_over_latency():
    from scripts.benchmark import benchmark_config

    cfg = tiny_backbone()
    result = benchmark_config(
        cfg, batch_size=4, resolution=4, device=torch.device("cpu"),
        warmup_steps=1, measure_steps=3, world_size=2,
    )  # fmt: skip
    # latency_s_per_step is itself rounded for display, so reconstructing throughput from it
    # only approximately recovers the value computed internally from the unrounded latency.
    expected = 4 * 2 / result["latency_s_per_step"]
    assert result["throughput_samples_per_s"] == pytest.approx(expected, rel=1e-2)


def test_benchmark_config_moe_reports_lower_active_than_total_params():
    from scripts.benchmark import benchmark_config

    cfg = tiny_backbone(
        ffn="moe", moe={"num_experts": 4, "top_k": 2, "shared_experts": 1}, mlp_ratio=4.0
    )
    result = benchmark_config(
        cfg, batch_size=2, resolution=4, device=torch.device("cpu"), warmup_steps=1, measure_steps=2
    )
    assert result["active_params"] < result["total_params"]


def test_benchmark_config_accepts_a_prebuilt_model():
    from imggen.backbones import build_backbone
    from scripts.benchmark import benchmark_config

    cfg = tiny_backbone()
    model = build_backbone(cfg)
    before = {id(p) for p in model.parameters()}
    result = benchmark_config(
        cfg, batch_size=2, resolution=4, device=torch.device("cpu"),
        warmup_steps=1, measure_steps=2, model=model,
    )  # fmt: skip
    assert {id(p) for p in model.parameters()} == before  # same object, not rebuilt
    assert result["latency_s_per_step"] > 0


# ---------------------------------------------------------------- CLI


def _run(repo_root, *args):
    return subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "benchmark.py"), *args],
        capture_output=True, text=True,
    )  # fmt: skip


def test_cli_with_explicit_width_depth(repo_root, tmp_path):
    out = tmp_path / "bench.json"
    r = _run(
        repo_root, "--width", "32", "--depth", "1", "--head-dim", "16", "--in-channels", "4",
        "--text-dim", "16", "--resolution", "4", "--batch-size", "2", "--warmup-steps", "1",
        "--measure-steps", "2", "--out", str(out),
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    data = json.loads(out.read_text())
    assert data["device"] == "cpu" and data["peak_vram_mb"] is None


def test_cli_with_named_size(repo_root, tmp_path):
    out = tmp_path / "bench.json"
    r = _run(
        repo_root, "--size", "nano", "--head-dim", "16", "--in-channels", "4", "--text-dim", "16",
        "--resolution", "4", "--batch-size", "2", "--warmup-steps", "1", "--measure-steps", "2",
        "--out", str(out),
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    data = json.loads(out.read_text())
    assert data["config"]["width"] == 256 and data["config"]["depth"] == 8  # nano rung


def test_cli_requires_size_or_width_and_depth(repo_root):
    r = _run(repo_root, "--batch-size", "2")
    assert r.returncode != 0
    assert "--size" in r.stderr or "--width" in r.stderr


def test_cli_warns_about_cpu_numbers(repo_root, tmp_path):
    r = _run(
        repo_root, "--width", "32", "--depth", "1", "--head-dim", "16", "--in-channels", "4",
        "--text-dim", "16", "--resolution", "4", "--batch-size", "2", "--warmup-steps", "1",
        "--measure-steps", "2", "--out", str(tmp_path / "bench.json"),
    )  # fmt: skip
    assert "not a proxy for GPU behavior" in r.stderr


# ---------------------------------------------------------------- real torchrun launch


def test_benchmark_under_real_torchrun_with_fsdp_exits_cleanly(repo_root, tmp_path):
    """Regression: benchmark.py never called dist.destroy_process_group(), which hung under a
    real torchrun launch (found by actually launching it, not by inspection)."""
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not available in this environment")

    out = tmp_path / "bench.json"
    r = subprocess.run(
        [
            "torchrun", "--standalone", "--nproc-per-node=2",
            str(repo_root / "scripts" / "benchmark.py"),
            "--width", "32", "--depth", "1", "--head-dim", "16", "--in-channels", "4",
            "--text-dim", "16", "--resolution", "4", "--batch-size", "2",
            "--warmup-steps", "1", "--measure-steps", "2",
            "--parallel-strategy", "fsdp", "--out", str(out),
        ],
        capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    data = json.loads(out.read_text())
    assert data["config"]["world_size"] == 2
