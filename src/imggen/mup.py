"""Maximal-update parametrization (muP): tune hyperparameters at a small proxy width and reuse
them, unchanged, at any larger width in the size ladder.

Reference: Yang & Hu et al., "Tensor Programs V" / "Tuning Large Neural Networks via Zero-Shot
Hyperparameter Transfer" (2022), Adam column of their Table 3. Three parameter roles:

  role     dimension that scales with width      init std     Adam LR multiplier
  input    only fan_out (patch/time/text embed)   1/sqrt(fan_in)     1
  hidden   both fan_in and fan_out                1/sqrt(fan_in)     1/width_mult
  output   only fan_in (e.g. the final head)       1/fan_in          1/width_mult

``width_mult = width / base_width``. Note input and hidden share the *same* init (both are the
standard 1/fan_in-variance init) — only their LR differs. Output is the one role whose init
also changes (an extra 1/fan_in factor), which is why output layers are often zero- or
near-zero-initialized in practice. Biases and norm scales (RMSNorm/LayerNorm weight) are
``FIXED``: left at their construction default (zero for biases, ones for norm scales), LR
multiplier 1 — their fan_in doesn't scale with width, so muP has nothing to adjust.

**Attention logit scale is unaffected by this module.** The usual muP correction (1/d instead
of 1/sqrt(d)) is needed only when head_dim itself scales with width; this project fixes
head_dim (see ADR 0002) and only adds heads, so the standard 1/sqrt(head_dim) softmax scale
already satisfies muP. If a future variant scales head_dim with width, this file's
``SoftmaxCore.scale`` argument is the hook to revisit.

**One global width_mult.** Every "hidden"-scaling dimension in this architecture (ffn_hidden via
mlp_ratio, MoE expert_hidden) is a fixed proportion of ``width``, so a single model-level
width_mult is correct; a layer whose width-dependence broke that proportionality would need its
own multiplier (not needed here — see ADR 0005 for the reasoning this relies on).
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum, auto
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from imggen.backbones.dit import DiT


class ParamRole(Enum):
    INPUT = auto()
    HIDDEN = auto()
    OUTPUT = auto()
    FIXED = auto()  # biases, norm scales, additive tables — no width-dependent scaling


def _mark(roles: dict[int, ParamRole], module: nn.Module, weight_role: ParamRole) -> None:
    roles[id(module.weight)] = weight_role
    if getattr(module, "bias", None) is not None:
        roles[id(module.bias)] = ParamRole.FIXED


def classify(model: DiT) -> dict[str, ParamRole]:
    """Assign every parameter in a DiT a muP role, by its known structural position.

    Raises if any parameter is left unclassified — a new parameter kind must be given a role
    explicitly, never fall through to a default.
    """
    from imggen.layers.ffn import DenseFFN
    from imggen.layers.moe import MoEFFN

    by_id: dict[int, ParamRole] = {}
    _mark(by_id, model.patch_embed, ParamRole.INPUT)
    _mark(by_id, model.t_embedder[0], ParamRole.INPUT)
    _mark(by_id, model.t_embedder[2], ParamRole.HIDDEN)
    _mark(by_id, model.t_block[1], ParamRole.HIDDEN)
    _mark(by_id, model.text_proj[0], ParamRole.INPUT)
    _mark(by_id, model.text_proj[2], ParamRole.HIDDEN)
    _mark(by_id, model.final_proj, ParamRole.OUTPUT)
    by_id[id(model.final_table)] = ParamRole.FIXED

    for blk in model.blocks:
        for attn in (blk.attn, blk.cross_attn):
            _mark(by_id, attn.q_proj, ParamRole.HIDDEN)
            _mark(by_id, attn.kv_proj, ParamRole.HIDDEN)
            _mark(by_id, attn.out_proj, ParamRole.HIDDEN)
            for norm in (attn.q_norm, attn.k_norm):
                if hasattr(norm, "weight"):
                    by_id[id(norm.weight)] = ParamRole.FIXED
        by_id[id(blk.mod_table)] = ParamRole.FIXED
        by_id[id(blk.norm_cross.weight)] = ParamRole.FIXED
        by_id[id(blk.norm_cross.bias)] = ParamRole.FIXED

        ffn = blk.ffn
        if isinstance(ffn, DenseFFN):
            _mark(by_id, ffn.fc1, ParamRole.HIDDEN)
            _mark(by_id, ffn.fc2, ParamRole.HIDDEN)
        elif isinstance(ffn, MoEFFN):
            by_id[id(ffn.router.weight)] = ParamRole.OUTPUT  # fan_out = num_experts, fixed
            by_id[id(ffn.fc1_weight)] = ParamRole.HIDDEN
            by_id[id(ffn.fc1_bias)] = ParamRole.FIXED
            by_id[id(ffn.fc2_weight)] = ParamRole.HIDDEN
            by_id[id(ffn.fc2_bias)] = ParamRole.FIXED
            by_id[id(ffn.route_bias)] = ParamRole.FIXED
            if ffn.shared is not None:
                _mark(by_id, ffn.shared.fc1, ParamRole.HIDDEN)
                _mark(by_id, ffn.shared.fc2, ParamRole.HIDDEN)
        else:
            raise NotImplementedError(f"muP has no role rules for ffn type {type(ffn).__name__}")

    roles: dict[str, ParamRole] = {}
    missing: list[str] = []
    for name, p in model.named_parameters():
        if id(p) not in by_id:
            missing.append(name)
        else:
            roles[name] = by_id[id(p)]
    if missing:
        raise ValueError(f"muP: no role assigned for parameter(s) {missing}")
    return roles


def _fan_in(name: str, param: torch.Tensor, model: DiT) -> int:
    """Input dimension of a weight matrix. Every weight in this model follows the PyTorch
    Linear convention (..., out, in) — fan_in is the last dimension — except the patch-embed
    Conv2d, whose fan_in is in_channels * patch_size^2 and is computed from the config."""
    if name == "patch_embed.weight":
        return model.cfg.in_channels * model.cfg.patch_size**2
    return param.shape[-1]


def _init_std(role: ParamRole, fan_in: int) -> float:
    if role in (ParamRole.INPUT, ParamRole.HIDDEN):
        return fan_in**-0.5
    if role is ParamRole.OUTPUT:
        return 1.0 / fan_in
    raise ValueError(f"no init rule for role {role}")  # FIXED params never reach here


def lr_multiplier(role: ParamRole, width_mult: float) -> float:
    return 1.0 / width_mult if role in (ParamRole.HIDDEN, ParamRole.OUTPUT) else 1.0


class MuP:
    """Bound to one ``base_width``; ``width_mult(width) = width / base_width``."""

    def __init__(self, base_width: int) -> None:
        if base_width <= 0:
            raise ValueError(f"base_width must be positive, got {base_width}")
        self.base_width = base_width

    def width_mult(self, width: int) -> float:
        return width / self.base_width

    @torch.no_grad()
    def init_(self, model: DiT, zero_modules: Iterable[nn.Module]) -> None:
        """Re-initialize every weight by its muP role. ``zero_modules`` (the adaLN-Zero /
        identity-at-init set: shared modulation, cross-attention output, the head) are forced
        to zero regardless of role — that is an architectural choice independent of muP, and
        muP's LR multiplier still applies to them afterward via their classified role."""
        roles = classify(model)
        zero_ids = {id(p) for m in zero_modules for p in m.parameters()}
        width_mult = self.width_mult(model.cfg.width)
        for name, p in model.named_parameters():
            if id(p) in zero_ids or name.endswith(".bias"):
                nn.init.zeros_(p)
                continue
            role = roles[name]
            if role is ParamRole.FIXED:
                continue  # leave norm scales / additive tables at their construction default
            std = _init_std(role, _fan_in(name, p, model))
            nn.init.normal_(p, mean=0.0, std=std)
        _ = width_mult  # init variance doesn't depend on width_mult; kept for clarity/symmetry

    def param_groups(
        self, model: DiT, lr: float, weight_decay: float = 0.0
    ) -> list[dict[str, object]]:
        """Adam-ready parameter groups: each role's LR is ``lr * lr_multiplier(role)``. Weight
        decay is applied only to HIDDEN/OUTPUT weights (matrices), not to INPUT or FIXED params
        (biases, norm scales), following common decoupled-weight-decay practice."""
        roles = classify(model)
        width_mult = self.width_mult(model.cfg.width)
        buckets: dict[tuple[float, float], list[nn.Parameter]] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            role = roles[name]
            mult = lr_multiplier(role, width_mult)
            wd = weight_decay if role in (ParamRole.HIDDEN, ParamRole.OUTPUT) else 0.0
            buckets.setdefault((mult, wd), []).append(p)
        return [
            {"params": params, "lr": lr * mult, "weight_decay": wd}
            for (mult, wd), params in sorted(buckets.items())
        ]


def activation_rms_by_width(
    base_cfg, widths: Iterable[int], *, seed: int = 0, batch: int = 4, length: int = 6
) -> dict[int, float]:
    """Mean |activation| leaving the first block's self-attention, at init, for each width —
    the standard muP "coordinate check": under correct parametrization this stays roughly flat
    across widths; under standard parametrization it grows with width. One data point per width,
    a few forward passes; the full statistical version (many seeds, several layers, a fit line)
    belongs to the step-9 test gate, not this diagnostic.
    """
    from imggen.backbones.dit import DiT
    from imggen.config import width_sweep

    out: dict[int, float] = {}
    for cfg in width_sweep(base_cfg, widths):
        torch.manual_seed(seed)
        model = DiT(cfg).eval()
        captured: dict[str, torch.Tensor] = {}

        def hook(_module, _inp, output, _store=captured):
            _store["act"] = output

        handle = model.blocks[0].attn.register_forward_hook(hook)
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(batch, cfg.in_channels, 8, 8, generator=g)
        t = torch.rand(batch, generator=g)
        text = torch.randn(batch, length, cfg.text_dim, generator=g)
        model(x, t, text)
        handle.remove()
        out[cfg.width] = captured["act"].abs().mean().item()
    return out
