"""Real multi-process tests (gloo backend, CPU, fork start method for speed) for FSDP2, tensor
parallelism, DDP, and checkpoint resharding across world sizes. See ADR 0008 for why fork (not
torch.multiprocessing.spawn's default "spawn" method) is used here: this sandbox has one CPU
core, and re-importing torch in a freshly spawned interpreter costs ~15s per process versus
~1s for a fork — the tests are equivalent either way, since nothing here touches CUDA.
"""

import itertools
import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from pydantic import ValidationError

from imggen.config import BackboneConfig, ParallelConfig, load_config
from imggen.data.shards import Manifest, ShardWriter
from imggen.layers.attention import Attention
from imggen.layers.ffn import DenseFFN
from imggen.parallel import (
    apply_tensor_parallel,
    build_device_mesh,
    is_distributed_run,
    parallelize,
)
from imggen.registry import ATTENTION
from imggen.train import run_training

_PORTS = itertools.count(29600)

TINY_OVERRIDES = [
    "backbone.width=48", "backbone.depth=2", "backbone.head_dim=16",
    "backbone.text_dim=16", "backbone.in_channels=4", "backbone.mlp_ratio=2.0",
]  # fmt: skip


def _write_tiny_cache(root, n=8, dim=16, latent_shape=(4, 4, 4), length=3, seed=0):
    writer = ShardWriter(root, shard_size=n)
    g = torch.Generator().manual_seed(seed)
    for i in range(n):
        writer.add(
            f"{i:04d}", torch.randn(*latent_shape, generator=g) * 0.5,
            torch.randn(length, dim, generator=g), torch.ones(length, dtype=torch.bool), f"c{i}",
        )  # fmt: skip
    num_shards = writer.close()
    Manifest(
        num_samples=n, num_shards=num_shards, shard_size=n, latent_channels=latent_shape[0],
        spatial_compression=8, text_dim=dim, resolution=32,
    ).save(root)  # fmt: skip


def _run_distributed(fn, world_size, args=(), port=None):
    """Forks ``world_size`` processes, each running ``fn(rank, world_size, *args)`` inside an
    initialized gloo process group; rank 0's return value comes back via a temp file (not a
    ``multiprocessing.Queue`` directly — a large payload of tensors, as the gate suite's
    gradient-parity test needs, hits a `resource_sharer`/file-descriptor edge case over a Queue;
    small dict/scalar results, as every step-8 test used, happen not to trigger it, which is why
    this surfaced only once step 9 needed to move real tensors). Raises if any process exits
    non-zero."""
    ctx = mp.get_context("fork")
    status_queue = ctx.Queue()
    port = port or next(_PORTS)
    result_path = tempfile.mktemp(suffix=".pt")

    def wrapper(rank):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        try:
            result = fn(rank, world_size, *args)
            if rank == 0:
                torch.save(result, result_path)
                status_queue.put("ok")
        except BaseException as exc:  # noqa: BLE001 - re-raise in the parent with context
            if rank == 0:
                status_queue.put(f"error: {exc!r}")
            raise
        finally:
            dist.barrier()
            dist.destroy_process_group()

    procs = [ctx.Process(target=wrapper, args=(r,)) for r in range(world_size)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    status = status_queue.get(timeout=5)
    if status != "ok" or any(p.exitcode != 0 for p in procs):
        pytest.fail(f"distributed run failed: {status}, exit codes {[p.exitcode for p in procs]}")
    try:
        return torch.load(result_path, weights_only=False)
    finally:
        Path(result_path).unlink(missing_ok=True)


# ---------------------------------------------------------------- config validation (no dist)


def test_ep_size_must_divide_num_experts():
    load_config(
        "configs/experiment/proxy_baseline.yaml",
        ["backbone.ffn=moe", "backbone.moe={num_experts: 8, top_k: 2}", "parallel.ep_size=4"],
    )
    with pytest.raises(ValidationError, match="ep_size"):
        load_config(
            "configs/experiment/proxy_baseline.yaml",
            ["backbone.ffn=moe", "backbone.moe={num_experts: 8, top_k: 2}", "parallel.ep_size=3"],
        )


def test_ep_size_ignored_for_dense_ffn():
    load_config("configs/experiment/proxy_baseline.yaml", ["parallel.ep_size=5"])  # no raise


def test_is_distributed_run_reads_world_size_env(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert is_distributed_run() is False
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert is_distributed_run() is False
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert is_distributed_run() is True


def test_init_distributed_picks_gloo_without_cuda_and_nccl_with(monkeypatch):
    from imggen.parallel import init_distributed

    calls = []
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "init_process_group", lambda backend: calls.append(backend))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    init_distributed()
    assert calls == ["gloo"]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda i: calls.append(("set_device", i)))
    monkeypatch.setenv("LOCAL_RANK", "1")
    init_distributed()
    assert calls == ["gloo", ("set_device", 1), "nccl"]


def test_init_distributed_respects_explicit_backend_and_is_a_noop_when_already_initialized(
    monkeypatch,
):
    from imggen.parallel import init_distributed

    calls = []
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "init_process_group", lambda backend: calls.append(backend))
    init_distributed(backend="gloo")
    assert calls == ["gloo"]

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    init_distributed()  # must not call init_process_group again
    assert calls == ["gloo"]


def test_parallelize_raises_when_nothing_to_do():
    cfg = ParallelConfig(strategy="none", tp_size=1)
    with pytest.raises(ValueError, match="nothing to do"):
        parallelize(nn.Linear(2, 2), cfg, device=torch.device("cpu"))


def test_parallelize_raises_for_ddp_plus_tp():
    cfg = ParallelConfig(strategy="ddp", tp_size=2)
    with pytest.raises(NotImplementedError, match="DDP"):
        parallelize(nn.Linear(2, 2), cfg, device=torch.device("cpu"))


# ---------------------------------------------------------------- multi-process: FSDP2


def _fsdp_shards_parameters(rank, world_size):
    from torch.distributed.tensor import DTensor

    from imggen.backbones import build_backbone
    from imggen.parallel import apply_fsdp

    cfg = BackboneConfig(width=48, depth=2, head_dim=16, text_dim=16, in_channels=4, mlp_ratio=2.0)
    model = build_backbone(cfg)
    global_numel = sum(p.numel() for p in model.parameters())
    mesh = build_device_mesh(tp_size=1)
    apply_fsdp(model, mesh["dp"])
    local_numel = sum(
        p.to_local().numel() if isinstance(p, DTensor) else p.numel() for p in model.parameters()
    )
    is_dtensor = all(isinstance(p, DTensor) for p in model.parameters())
    return {"global": global_numel, "local": local_numel, "all_dtensor": is_dtensor}


@pytest.mark.dist
def test_fsdp_shards_parameters_across_two_ranks():
    result = _run_distributed(_fsdp_shards_parameters, world_size=2)
    assert result["all_dtensor"]
    assert result["local"] < result["global"]  # each rank holds less than the full model


def _fsdp_training_runs(rank, world_size, cache_dir, run_dir):
    torch.manual_seed(0)
    cfg = load_config(
        "configs/experiment/proxy_baseline.yaml",
        [
            f"data.root={cache_dir}", "data.num_workers=0", *TINY_OVERRIDES,
            "train.steps=4", "train.batch_size=4", "train.lr=1e-3", "train.warmup_steps=1",
            "train.log_every=1000", "train.ckpt_every=4", "parallel.strategy=fsdp",
        ],
    )  # fmt: skip
    state = run_training(cfg, run_dir, device=torch.device("cpu"))
    return {"step": state.step, "loss": state.last_loss}


@pytest.mark.dist
def test_fsdp_training_runs_end_to_end(tmp_path):
    cache = tmp_path / "cache"
    _write_tiny_cache(cache)
    result = _run_distributed(
        _fsdp_training_runs, world_size=2, args=(str(cache), str(tmp_path / "run"))
    )
    assert result["step"] == 4
    assert torch.isfinite(torch.tensor(result["loss"]))
    ckpts = sorted((tmp_path / "run" / "checkpoints").iterdir())
    assert [p.name for p in ckpts] == ["step_00000004"]


def _fsdp_forward_backward_warnings(rank, world_size):
    import warnings

    from imggen.backbones import build_backbone
    from imggen.parallel import apply_fsdp, build_device_mesh

    torch.manual_seed(0)
    # patch_size=1 (the default — TINY_OVERRIDES above doesn't set it either) is the condition
    # that made imggen.backbones.dit.unpatchify return a genuine view before the ADR 0012 fix.
    cfg = BackboneConfig(width=32, depth=2, head_dim=16, text_dim=16, in_channels=4, mlp_ratio=2.0)
    model = build_backbone(cfg)
    apply_fsdp(model, build_device_mesh(tp_size=1)["dp"])
    x = torch.randn(2, 4, 8, 8)
    t = torch.rand(2)
    text = torch.randn(2, 5, 16)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model(x, t, text).pow(2).mean().backward()
    return [str(w.message) for w in caught]


@pytest.mark.dist
def test_fsdp_forward_backward_raises_no_view_tensor_warning():
    """Regression for ADR 0012: FSDP2 warns when a wrapped module returns a view tensor, which
    can silently drop the pre-backward hook (all-gather) and risk wrong gradients. This fired on
    a real torchrun launch before unpatchify grew a trailing .contiguous(); this test fails loud
    if that regresses, rather than relying on noticing the warning in some future log again."""
    messages = _run_distributed(_fsdp_forward_backward_warnings, world_size=2)
    assert not any("view tensor" in m for m in messages), messages


# ---------------------------------------------------------------- multi-process: tensor parallel


def _tp_matches_attention_reference(rank, world_size):
    torch.manual_seed(0)
    width, heads, hd = 32, 4, 8
    attn = Attention(width, heads, ATTENTION.build("softmax", head_dim=hd))
    x = torch.randn(2, 6, width)
    with torch.no_grad():
        ref = attn(x)
    mesh = build_device_mesh(tp_size=world_size)
    apply_tensor_parallel_single_module = attn  # parallelize just this module, not a full DiT
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )

    parallelize_module(
        apply_tensor_parallel_single_module,
        mesh["tp"],
        {"q_proj": ColwiseParallel(), "kv_proj": ColwiseParallel(), "out_proj": RowwiseParallel()},
    )
    with torch.no_grad():
        out = attn(x)
    out = out.full_tensor() if hasattr(out, "full_tensor") else out
    return (out - ref).abs().max().item()


@pytest.mark.dist
def test_tensor_parallel_attention_matches_single_process_reference():
    max_diff = _run_distributed(_tp_matches_attention_reference, world_size=2)
    assert max_diff < 1e-5


def _tp_matches_ffn_reference(rank, world_size):
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )

    torch.manual_seed(0)
    ffn = DenseFFN(32, 128)
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        ref = ffn(x)
    mesh = build_device_mesh(tp_size=world_size)
    parallelize_module(ffn, mesh["tp"], {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()})
    with torch.no_grad():
        out = ffn(x)
    out = out.full_tensor() if hasattr(out, "full_tensor") else out
    return (out - ref).abs().max().item()


@pytest.mark.dist
def test_tensor_parallel_dense_ffn_matches_single_process_reference():
    max_diff = _run_distributed(_tp_matches_ffn_reference, world_size=2)
    assert max_diff < 1e-5


def _full_dit_with_tp_matches_reference(rank, world_size):
    from imggen.backbones import build_backbone

    torch.manual_seed(0)
    cfg = BackboneConfig(width=32, depth=2, head_dim=8, text_dim=16, in_channels=4, mlp_ratio=2.0)
    model = build_backbone(cfg)
    x = torch.randn(2, 4, 8, 8)
    t = torch.rand(2)
    text = torch.randn(2, 5, 16)
    with torch.no_grad():
        ref = model(x, t, text)
    mesh = build_device_mesh(tp_size=world_size)
    apply_tensor_parallel(model, mesh["tp"])
    with torch.no_grad():
        out = model(x, t, text)
    out = out.full_tensor() if hasattr(out, "full_tensor") else out
    return (out - ref).abs().max().item()


@pytest.mark.dist
def test_full_dit_tensor_parallel_matches_single_process_reference():
    max_diff = _run_distributed(_full_dit_with_tp_matches_reference, world_size=2)
    assert max_diff < 1e-4


# ------------------------------------------------------- multi-process: reshard checkpoints


def _train_then_checkpoint(rank, world_size, cache_dir, run_dir, steps, strategy):
    torch.manual_seed(0)
    cfg = load_config(
        "configs/experiment/proxy_baseline.yaml",
        [
            f"data.root={cache_dir}", "data.num_workers=0", *TINY_OVERRIDES,
            f"train.steps={steps}", "train.batch_size=4", "train.lr=1e-3", "train.warmup_steps=1",
            "train.log_every=1000", f"train.ckpt_every={steps}", f"parallel.strategy={strategy}",
        ],
    )  # fmt: skip
    state = run_training(cfg, run_dir, device=torch.device("cpu"))
    return {"step": state.step, "loss": state.last_loss}


@pytest.mark.dist
def test_checkpoint_from_two_rank_fsdp_resumes_single_process(tmp_path):
    cache, run_dir = tmp_path / "cache", tmp_path / "run"
    _write_tiny_cache(cache)
    first = _run_distributed(
        _train_then_checkpoint, world_size=2, args=(str(cache), str(run_dir), 4, "fsdp")
    )
    assert first["step"] == 4

    torch.manual_seed(0)
    cfg2 = load_config(
        "configs/experiment/proxy_baseline.yaml",
        [
            f"data.root={cache}", "data.num_workers=0", *TINY_OVERRIDES,
            "train.steps=7", "train.batch_size=4", "train.lr=1e-3", "train.warmup_steps=1",
            "train.log_every=1000", "train.ckpt_every=7", "parallel.strategy=none",
        ],
    )  # fmt: skip
    state = run_training(cfg2, run_dir, device=torch.device("cpu"))
    assert state.step == 7  # continued from 4, not restarted
    ckpts = sorted(p.name for p in (run_dir / "checkpoints").iterdir())
    assert ckpts == ["step_00000004", "step_00000007"]


@pytest.mark.dist
def test_checkpoint_from_single_process_resumes_under_two_rank_fsdp(tmp_path):
    cache, run_dir = tmp_path / "cache", tmp_path / "run"
    _write_tiny_cache(cache)
    torch.manual_seed(0)
    cfg1 = load_config(
        "configs/experiment/proxy_baseline.yaml",
        [
            f"data.root={cache}", "data.num_workers=0", *TINY_OVERRIDES,
            "train.steps=3", "train.batch_size=4", "train.lr=1e-3", "train.warmup_steps=1",
            "train.log_every=1000", "train.ckpt_every=3", "parallel.strategy=none",
        ],
    )  # fmt: skip
    state1 = run_training(cfg1, run_dir, device=torch.device("cpu"))
    assert state1.step == 3

    second = _run_distributed(
        _train_then_checkpoint, world_size=2, args=(str(cache), str(run_dir), 6, "fsdp")
    )
    assert second["step"] == 6  # continued from 3, not restarted
    ckpts = sorted((run_dir / "checkpoints").iterdir())
    assert [p.name for p in ckpts] == ["step_00000003", "step_00000006"]


# ---------------------------------------------------------------- multi-process: DDP


def _ddp_trains_with_correct_param_names(rank, world_size, cache_dir, run_dir):
    torch.manual_seed(0)
    cfg = load_config(
        "configs/experiment/proxy_baseline.yaml",
        [
            f"data.root={cache_dir}", "data.num_workers=0", *TINY_OVERRIDES,
            "backbone.mup={enabled: true, base_width: 48}",  # exercises name-based classify()
            "train.steps=3", "train.batch_size=4", "train.lr=1e-3", "train.warmup_steps=1",
            "train.log_every=1000", "train.ckpt_every=3", "parallel.strategy=ddp",
        ],
    )  # fmt: skip
    state = run_training(cfg, run_dir, device=torch.device("cpu"))
    return {"step": state.step, "loss": state.last_loss}


@pytest.mark.dist
def test_ddp_trains_and_mup_classification_survives_the_module_prefix(tmp_path):
    cache = tmp_path / "cache"
    _write_tiny_cache(cache)
    result = _run_distributed(
        _ddp_trains_with_correct_param_names, world_size=2, args=(str(cache), str(tmp_path / "run"))
    )
    assert result["step"] == 3
    assert torch.isfinite(torch.tensor(result["loss"]))
