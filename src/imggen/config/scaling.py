"""Size estimates and width sweeps derived from a BackboneConfig."""

from __future__ import annotations

from collections.abc import Iterable

from .schema import BackboneConfig


def estimate_params(cfg: BackboneConfig) -> dict[str, int]:
    """Exact parameter count of ``imggen.backbones.dit.DiT`` (weights, biases and norm scales).

    Kept in sync with the module by tests/test_dit.py and tests/test_moe.py, which compare this
    formula with an instantiated model. Layout (w=width, r=ffn_hidden, hd=head_dim,
    C=in_channels, p=patch_size, T=text_dim, F=time_freq_dim), per block:
      self-attention   4w^2 + 4w + 2hd   (q, kv, out projections + q/k RMSNorm scales)
      cross-attention  4w^2 + 4w + 2hd
      FFN (dense)      2wr + r + w
      FFN (moe)        router w*E + E experts of (2*w*eh + eh + w) + one shared FFN of
                       hidden shared_experts*eh (absent if shared_experts=0)
      norms            2w                (cross-attention pre-norm)
      modulation       6w                (per-block adaLN table; the 6w^2 projection is shared)
    plus shared: t_block 6w^2+6w, final table 2w, patch embed C*p^2*w + w,
    timestep MLP F*w + w^2 + 2w, text projector T*w + w^2 + 2w, head w*p^2*C + p^2*C.

    ``total`` counts every parameter; ``active`` counts what one token uses (MoE: router,
    shared FFN and top_k experts per block). They are equal for a dense FFN.
    """
    w, d, r, hd = cfg.width, cfg.depth, cfg.ffn_hidden, cfg.head_dim
    c, p2, t, f = cfg.in_channels, cfg.patch_size**2, cfg.text_dim, cfg.time_freq_dim
    if cfg.ffn == "dense":
        ffn_total = ffn_active = 2 * w * r + r + w
    elif cfg.ffn == "moe":
        m, eh = cfg.moe, cfg.expert_hidden
        expert = 2 * w * eh + eh + w
        sh = m.shared_experts * eh
        common = w * m.num_experts + (2 * w * sh + sh + w if sh else 0)
        ffn_total = common + m.num_experts * expert
        ffn_active = common + m.top_k * expert
    else:
        raise NotImplementedError(f"no parameter formula for ffn {cfg.ffn!r}")
    attn = 4 * w * w + 4 * w + 2 * hd
    out = {
        "attention": d * attn,
        "cross_attention": d * attn,
        "ffn": d * ffn_total,
        "norms": d * 2 * w,
        "modulation": d * 6 * w + (6 * w * w + 6 * w) + 2 * w,
        "embedders": (c * p2 * w + w) + (f * w + w + w * w + w) + (t * w + w + w * w + w),
        "head": w * p2 * c + p2 * c,
    }
    out["total"] = sum(out.values())
    out["active"] = out["total"] - d * (ffn_total - ffn_active)
    return out


def format_params(n: int) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.3g}{unit}"
    return str(n)


def width_sweep(base: BackboneConfig, widths: Iterable[int]) -> list[BackboneConfig]:
    """Same depth/head_dim/flavor at different widths (for muP coordinate and transfer checks)."""
    return [base.replace(width=w) for w in widths]
