"""Sharded training: FSDP2 (data-parallel, parameter/gradient/optimizer sharding), tensor
parallelism via DTensor, and DDP — each behind ``ParallelConfig.strategy`` / ``tp_size``.

Nothing here changes single-process behavior: with ``strategy="none"`` and ``tp_size=1``
(the default), ``parallelize`` is never called and no process group is touched — this is the
same code path steps 1-7 already exercise and test.

Verified with real multi-process runs (gloo backend, 2 ranks, CPU — this sandbox has one GPU-less
core, so that's what's available; see ADR 0008 for exact numbers):
- FSDP2 (`fully_shard`, applied per ``DiTBlock`` then the whole model) shards parameters into
  DTensors and trains.
- Tensor parallelism (`parallelize_module` with `ColwiseParallel`/`RowwiseParallel`) reproduces
  the single-process forward pass to float32 precision, for both attention and the dense FFN.
  This required one real fix, not just configuration: see the comment in
  ``imggen.layers.attention.Attention.forward`` about the kv_proj output layout — a combined
  QKV-style projection needs its output laid out (head, k/v, dim), not (k/v, head, dim), or a
  contiguous tensor-parallel shard splits k from v instead of splitting heads.
- A checkpoint saved under FSDP with world_size=2 loads correctly into a single-process
  (world_size=1) model, and vice versa — the actual "resume on a different GPU count" promise,
  not just an assumption about DCP's design intent (see ADR 0008 for the exact test).

Not verified: real GPU collectives (NCCL), TP combined with linear attention or MoE, and anything
about *speed* — this environment cannot measure any of that.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

from imggen.backbones.dit import DiT, DiTBlock
from imggen.config import ParallelConfig

log = logging.getLogger("imggen.parallel")


def is_distributed_run() -> bool:
    """True if launched under a distributed launcher (torchrun sets WORLD_SIZE)."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def init_distributed(backend: str | None = None) -> None:
    """Join the process group described by the launcher's environment variables (RANK,
    WORLD_SIZE, MASTER_ADDR, MASTER_PORT, LOCAL_RANK — set by torchrun or an equivalent).

    ``backend`` defaults to "nccl" when CUDA is available (the correct, fast choice for GPU-GPU
    collectives) and "gloo" otherwise (CPU, or this sandbox's tests). Also calls
    ``torch.cuda.set_device(LOCAL_RANK)`` when on CUDA — without it every process on a
    multi-GPU node defaults to device 0, which is wrong (and was the actual bug here: nothing
    called this function at all before this fix, so a real torchrun launch would have crashed
    with "Default process group has not been initialized" the moment ``parallelize()`` ran).
    No-op if a process group is already initialized (e.g. a test harness set one up itself).
    """
    if dist.is_initialized():
        return
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group(backend=backend)


def build_device_mesh(tp_size: int) -> DeviceMesh:
    """A 1D ("dp",) mesh, or 2D ("dp", "tp") when ``tp_size`` > 1. FSDP shards along "dp";
    tensor parallelism uses "tp". ``world_size`` must be divisible by ``tp_size``."""
    world_size = dist.get_world_size()
    if world_size % tp_size:
        raise ValueError(f"world_size ({world_size}) must be divisible by tp_size ({tp_size})")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if tp_size == 1:
        return init_device_mesh(device_type, (world_size,), mesh_dim_names=("dp",))
    dp_size = world_size // tp_size
    return init_device_mesh(device_type, (dp_size, tp_size), mesh_dim_names=("dp", "tp"))


def _attention_tp_plan() -> dict[str, object]:
    return {
        "q_proj": ColwiseParallel(),
        "kv_proj": ColwiseParallel(),
        "out_proj": RowwiseParallel(),
    }


def apply_tensor_parallel(model: DiT, tp_mesh: DeviceMesh) -> DiT:
    """Shards every block's self- and cross-attention projections and dense FFN across
    ``tp_mesh``. ``backbone.num_heads`` must be divisible by the mesh size (checked at config
    load, ADR 0008); MoE's FFN is not yet covered (that's expert parallelism, planned but not
    implemented — see ADR 0008) so this plan skips ``ffn`` for MoE blocks. ``head_dim`` is
    untouched — TP splits *heads*, not the vector each head produces (ADR 0002/0005)."""
    from imggen.layers.ffn import DenseFFN

    for block in model.blocks:
        assert isinstance(block, DiTBlock)
        parallelize_module(block.attn, tp_mesh, _attention_tp_plan())
        parallelize_module(block.cross_attn, tp_mesh, _attention_tp_plan())
        if isinstance(block.ffn, DenseFFN):
            parallelize_module(
                block.ffn, tp_mesh, {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()}
            )
    return model


def apply_fsdp(model: DiT, dp_mesh: DeviceMesh) -> DiT:
    """FSDP2: shard each block, then the whole model (so cross-block buffers/embeddings are
    covered too). Order matters — wrapping the outer model first would leave nothing left for
    the per-block calls to shard independently."""
    from torch.distributed._composable.fsdp import fully_shard

    for block in model.blocks:
        fully_shard(block, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)
    return model


def apply_ddp(model: DiT, device: torch.device) -> nn.Module:
    device_ids = [device.index] if device.type == "cuda" else None
    return torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)


def parallelize(model: DiT, cfg: ParallelConfig, *, device: torch.device) -> nn.Module:
    """Applies ``cfg`` to ``model`` in place (TP first, then FSDP/DDP) and returns it. Only
    called when ``cfg.strategy != "none"`` or ``cfg.tp_size > 1`` — the caller
    (``imggen.train.run_training``) skips this entirely, and never touches ``torch.distributed``,
    for the single-process default.

    Validates the strategy/tp_size combination before touching ``torch.distributed`` at all, so
    a misuse (calling this with nothing to do, or an unsupported combination) is a plain
    ``ValueError`` rather than a confusing failure three collective calls deep."""
    if cfg.strategy == "none" and cfg.tp_size == 1:
        raise ValueError("parallelize() called with nothing to do (strategy='none', tp_size=1)")
    if cfg.strategy == "ddp" and cfg.tp_size > 1:
        raise NotImplementedError("DDP + tensor parallel is not supported; use strategy=fsdp")

    if cfg.tp_size > 1:
        mesh = build_device_mesh(cfg.tp_size)
        apply_tensor_parallel(model, mesh["tp"])
        dp_mesh = mesh["dp"]
    else:
        dp_mesh = build_device_mesh(cfg.tp_size)  # a plain 1D ("dp",) mesh

    if cfg.strategy == "fsdp":
        return apply_fsdp(model, dp_mesh)
    if cfg.strategy == "ddp":
        return apply_ddp(model, device)
    return model  # TP alone, no data-parallel wrapper
