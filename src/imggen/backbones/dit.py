"""Latent DiT with shared adaLN ("adaLN-single"), cross-attention text conditioning, fixed 2D
sin-cos positions, and per-layer swappable self-attention (see BackboneConfig.attn_pattern).

forward(x, t, text, text_mask=None):
    x: (B, C, H, W) latents, t: (B,) in [0, 1], text: (B, L, text_dim), text_mask: (B, L) bool
    (True = real token; every row needs at least one True) -> (B, C, H, W)

Weight init lives in ``init_weights()``; pass a ``MuP`` (imggen.mup) to switch to muP's
role-based init, selected automatically from ``cfg.mup.enabled``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from imggen.config import BackboneConfig
from imggen.layers.attention import Attention
from imggen.layers.ffn import DenseFFN  # noqa: F401  (ensures registration)
from imggen.layers.moe import MoEFFN  # noqa: F401
from imggen.mup import MuP
from imggen.registry import ATTENTION, BACKBONE, FFN


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def sincos_2d(dim: int, gh: int, gw: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Fixed 2D sin-cos position embedding, (gh*gw, dim); works for any grid size."""
    assert dim % 4 == 0

    def one_d(d: int, pos: torch.Tensor) -> torch.Tensor:
        omega = 1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float32, device=device) / (d / 2))
        out = pos.float()[:, None] * omega[None]
        return torch.cat([out.sin(), out.cos()], dim=1)

    ys, xs = torch.meshgrid(
        torch.arange(gh, device=device), torch.arange(gw, device=device), indexing="ij"
    )
    emb = torch.cat([one_d(dim // 2, ys.reshape(-1)), one_d(dim // 2, xs.reshape(-1))], dim=1)
    return emb.to(dtype)


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """(B,C,H,W) -> (B, gh*gw, p*p*C), inverse of ``unpatchify``."""
    b, c, h, w = x.shape
    x = x.view(b, c, h // p, p, w // p, p).permute(0, 2, 4, 3, 5, 1)
    return x.reshape(b, (h // p) * (w // p), p * p * c)


def unpatchify(x: torch.Tensor, gh: int, gw: int, p: int, c: int) -> torch.Tensor:
    """``.contiguous()`` matters here, not just for memory layout: with ``patch_size=1`` (this
    project's default), the permute+reshape below is reshape-compatible without a copy, so the
    model's returned output would otherwise be a genuine *view* into ``final_proj``'s output.
    FSDP2 explicitly warns about returning a view from a wrapped module's forward — its
    pre-backward hook (which triggers the all-gather) can silently fail to fire on a later
    in-place op on that view, risking wrong gradients. Found via a real ``torchrun`` launch
    (ADR 0012); a follow-up gradient-parity test at this exact config (patch_size=1, under FSDP)
    confirms gradients are correct both before and after this fix, but the fix removes the risk
    by construction rather than relying on that empirical check continuing to hold under future
    changes to the training loop."""
    b = x.shape[0]
    x = x.view(b, gh, gw, p, p, c).permute(0, 5, 1, 3, 2, 4)
    return x.reshape(b, c, gh * p, gw * p).contiguous()


class DiTBlock(nn.Module):
    def __init__(self, cfg: BackboneConfig, layer_idx: int) -> None:
        super().__init__()
        w, heads, hd = cfg.width, cfg.num_heads, cfg.head_dim
        self.attn_kind = cfg.attn_for_layer(layer_idx)
        self.norm1 = nn.LayerNorm(w, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(w, heads, ATTENTION.build(self.attn_kind, head_dim=hd))
        self.norm_cross = nn.LayerNorm(w, eps=1e-6)
        self.cross_attn = Attention(w, heads, ATTENTION.build("softmax", head_dim=hd))
        self.norm2 = nn.LayerNorm(w, elementwise_affine=False, eps=1e-6)
        self.ffn = FFN.build(cfg.ffn, width=w, hidden=cfg.ffn_hidden, moe=cfg.moe)
        # per-block offsets to the shared modulation: shift1, scale1, gate1, shift2, scale2, gate2
        self.mod_table = nn.Parameter(torch.zeros(6, w))

    def forward(self, x, mod, text, text_mask, hw):
        s1, c1, g1, s2, c2, g2 = (self.mod_table[None] + mod).unbind(1)  # each (B, w)
        h = self.norm1(x) * (1 + c1[:, None]) + s1[:, None]
        x = x + g1[:, None] * self.attn(h)
        x = x + self.cross_attn(self.norm_cross(x), context=text, key_mask=text_mask)
        h = self.norm2(x) * (1 + c2[:, None]) + s2[:, None]
        return x + g2[:, None] * self.ffn(h, hw)


@BACKBONE.register("dit")
class DiT(nn.Module):
    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        w, p, c = cfg.width, cfg.patch_size, cfg.in_channels
        self.patch_embed = nn.Conv2d(c, w, kernel_size=p, stride=p)
        self.t_embedder = nn.Sequential(nn.Linear(cfg.time_freq_dim, w), nn.SiLU(), nn.Linear(w, w))
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(w, 6 * w))  # shared adaLN projection
        self.text_proj = nn.Sequential(
            nn.Linear(cfg.text_dim, w), nn.GELU(approximate="tanh"), nn.Linear(w, w)
        )
        self.blocks = nn.ModuleList(DiTBlock(cfg, i) for i in range(cfg.depth))
        self.final_table = nn.Parameter(torch.zeros(2, w))  # shift, scale
        self.final_norm = nn.LayerNorm(w, elementwise_affine=False, eps=1e-6)
        self.final_proj = nn.Linear(w, p * p * c)
        parametrization = MuP(cfg.mup.base_width) if cfg.mup.enabled else None
        self.init_weights(parametrization)

    def zero_init_modules(self) -> list[nn.Module]:
        """The adaLN-Zero / identity-at-init set: shared modulation, cross-attn output, head."""
        return [self.t_block[1], self.final_proj] + [b.cross_attn.out_proj for b in self.blocks]

    def init_weights(self, parametrization: MuP | None = None) -> None:
        """Xavier for linears (or muP's role-based init, when ``parametrization`` is given);
        zero-init the shared modulation, cross-attn output and head so every block is the
        identity at init (adaLN-Zero behaviour)."""
        if parametrization is not None:
            parametrization.init_(self, self.zero_init_modules())
            return  # muP's init already covers every parameter, MoE included
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for blk in self.blocks:  # FFNs with their own parameters (MoE experts, router)
            if hasattr(blk.ffn, "init_weights"):
                blk.ffn.init_weights()
        conv_w = self.patch_embed.weight
        nn.init.xavier_uniform_(conv_w.view(conv_w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)
        for m in self.zero_init_modules():
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, t, text, text_mask=None):
        cfg = self.cfg
        b, c, height, width = x.shape
        p = cfg.patch_size
        if height % p or width % p:
            raise ValueError(f"H and W ({height}x{width}) must be divisible by patch_size {p}")
        gh, gw = height // p, width // p

        h = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, w)
        h = h + sincos_2d(cfg.width, gh, gw, device=h.device, dtype=h.dtype)
        t_emb = self.t_embedder(timestep_embedding(t * 1000.0, cfg.time_freq_dim).to(h.dtype))
        mod = self.t_block(t_emb).view(b, 6, cfg.width)
        ctx = self.text_proj(text)
        for blk in self.blocks:
            h = blk(h, mod, ctx, text_mask, (gh, gw))

        shift, scale = (self.final_table[None] + t_emb[:, None]).unbind(1)
        h = self.final_norm(h) * (1 + scale[:, None]) + shift[:, None]
        return unpatchify(self.final_proj(h), gh, gw, p, c)
