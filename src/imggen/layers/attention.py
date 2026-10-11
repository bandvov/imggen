"""Attention cores and the projection wrapper.

A *core* maps per-head tensors to per-head output and is the swappable part:

    core(q, k, v, key_mask=None) -> out      # q: (B,H,N,D)  k,v: (B,H,M,D)  out: (B,H,N,D)
    key_mask: (B, M) bool, True = attend to this key (padding mask), or None

Cores are registered in ATTENTION and built as ``ATTENTION.build(name, head_dim=D)``. They hold
no projections, so every core has the same parameter count inside ``Attention``.
The optional ``scale`` (softmax logit scale, default 1/sqrt(D)) is the hook muP needs (1/D).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from imggen.registry import ATTENTION


@ATTENTION.register("softmax")
class SoftmaxCore(nn.Module):
    """Exact softmax attention via PyTorch SDPA (uses flash/efficient kernels when available)."""

    def __init__(self, head_dim: int, scale: float | None = None) -> None:
        super().__init__()
        self.scale = scale if scale is not None else head_dim**-0.5

    def forward(self, q, k, v, key_mask=None):
        mask = None if key_mask is None else key_mask[:, None, None, :]
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)


@ATTENTION.register("linear")
class LinearCore(nn.Module):
    """ReLU-kernel linear attention, O(N) in sequence length (SANA-style).

    out_i = relu(q_i) (sum_j relu(k_j) v_j^T) / (relu(q_i) . sum_j relu(k_j) + eps)
    Computed in fp32 for stability. ``scale`` is accepted for interface parity and unused.
    """

    def __init__(self, head_dim: int, scale: float | None = None, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, q, k, v, key_mask=None):
        dtype = q.dtype
        q, k, v = F.relu(q.float()), F.relu(k.float()), v.float()
        if key_mask is not None:
            m = key_mask[:, None, :, None].to(k.dtype)
            k, v = k * m, v * m
        kv = torch.einsum("bhmd,bhme->bhde", k, v)  # (B,H,D,D)
        k_sum = k.sum(dim=2)  # (B,H,D)
        num = torch.einsum("bhnd,bhde->bhne", q, kv)
        den = torch.einsum("bhnd,bhd->bhn", q, k_sum).unsqueeze(-1) + self.eps
        return (num / den).to(dtype)


class Attention(nn.Module):
    """Projections + QK RMSNorm around a swappable core. Self-attention when ``context`` is None."""

    def __init__(self, width: int, num_heads: int, core: nn.Module, qk_norm: bool = True) -> None:
        super().__init__()
        assert width % num_heads == 0
        self.num_heads, self.head_dim = num_heads, width // num_heads
        self.q_proj = nn.Linear(width, width)
        self.kv_proj = nn.Linear(width, 2 * width)
        self.out_proj = nn.Linear(width, width)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.core = core

    def forward(self, x, context=None, key_mask=None):
        ctx = x if context is None else context
        b, n, _ = x.shape
        d = self.head_dim
        q_flat = self.q_proj(x)
        kv_flat = self.kv_proj(ctx)
        # Heads, not head_dim, is what tensor parallelism shards (ADR 0008): after a
        # ColwiseParallel q_proj/kv_proj, the local projection carries only this rank's share
        # of heads, so `self.num_heads` (the *global* count) would be wrong here. Deriving the
        # local head count from the actual projected width keeps this module TP-transparent —
        # unsharded, it's exactly `self.num_heads`.
        h = q_flat.shape[-1] // d
        q = q_flat.view(b, n, h, d).transpose(1, 2)
        # kv_proj packs (head, k/v, d) per row — head outermost — NOT (k/v, head, d). A
        # contiguous slice of the output then always covers whole heads together with their
        # k and v, which is what makes ColwiseParallel's plain contiguous shard correct here;
        # (k/v, head, d) would instead hand one rank all-k and another all-v. This is just a
        # row relabelling of an otherwise arbitrary, freshly-initialized weight, so it changes
        # nothing about unsharded (single-process) behavior.
        m = kv_flat.shape[1]
        kv = kv_flat.view(b, m, h, 2, d)
        k = kv[..., 0, :].transpose(1, 2)
        v = kv[..., 1, :].transpose(1, 2)
        out = self.core(self.q_norm(q), self.k_norm(k), v, key_mask)
        return self.out_proj(out.transpose(1, 2).reshape(b, n, h * d))
