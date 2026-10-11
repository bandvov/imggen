"""Typed experiment config. Unknown keys are rejected so typos fail loudly."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .sizes import get_size


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(_Base):
    name: str = "debug"
    seed: int = 0
    output_dir: str = "runs"
    deterministic: bool = False


class ComponentConfig(_Base):
    """A registry entry plus constructor kwargs: ``registry.build(name, **args)``.

    New component types need no schema change.
    """

    name: str
    args: dict[str, Any] = Field(default_factory=dict)

    def build(self, registry: Any) -> Any:
        return registry.build(self.name, **self.args)


class MoEConfig(_Base):
    """Sparse mixture-of-experts FFN settings; only used when ``backbone.ffn == "moe"``."""

    num_experts: int = Field(8, gt=0)
    top_k: int = Field(2, gt=0)
    shared_experts: int = Field(1, ge=0)  # always-on experts, merged into one dense FFN
    # Hidden size of each expert. None = split the dense FFN's compute across the active experts,
    # i.e. about ffn_hidden // (shared_experts + top_k), rounded down to a multiple of 8.
    expert_hidden: int | None = Field(None, gt=0)
    balance: Literal["aux_loss", "bias", "none"] = "aux_loss"
    aux_loss_weight: float = Field(0.01, ge=0)  # balance="aux_loss"
    bias_update_rate: float = Field(1e-3, gt=0)  # balance="bias" (loss-free balancing)

    @model_validator(mode="after")
    def _check(self) -> MoEConfig:
        if self.top_k > self.num_experts:
            raise ValueError(f"top_k ({self.top_k}) cannot exceed num_experts ({self.num_experts})")
        return self

    def resolve_expert_hidden(self, ffn_hidden: int) -> int:
        if self.expert_hidden is not None:
            return self.expert_hidden
        return max(8, ffn_hidden // (self.shared_experts + self.top_k) // 8 * 8)


class MuPConfig(_Base):
    """Maximal-update parametrization. Tune hyperparameters at ``base_width`` (the proxy rung);
    a swept width then reuses the same learning rate without retuning (see mup.py)."""

    enabled: bool = False
    base_width: int = Field(384, gt=0)  # the proxy rung's width


class BackboneConfig(_Base):
    """Backbone shape. Give ``size`` (a named rung) or explicit ``width`` and ``depth``.

    Explicit width/depth win over ``size``. If the result no longer equals the named rung,
    ``size`` is cleared so the resolved config never carries a misleading label.
    """

    name: str = "dit"  # BACKBONE registry key
    size: str | None = None  # key of imggen.config.sizes.SIZES
    width: int = Field(gt=0)
    depth: int = Field(gt=0)
    head_dim: int = Field(64, gt=0)  # fixed across sizes (muP-friendly)
    mlp_ratio: float = Field(4.0, gt=0)
    patch_size: int = Field(1, gt=0)
    in_channels: int = Field(32, gt=0)  # latent channels; must match the autoencoder
    text_dim: int = Field(2048, gt=0)  # text-encoder embedding dim; must match the text encoder
    time_freq_dim: int = Field(256, gt=0)  # sinusoidal timestep feature size
    attn: str = "softmax"  # ATTENTION registry key, used for every layer by default
    attn_pattern: list[str] = Field(default_factory=list)  # repeating per-layer override
    ffn: str = "dense"  # FFN registry key
    moe: MoEConfig = Field(default_factory=MoEConfig)  # ignored unless ffn == "moe"
    mup: MuPConfig = Field(default_factory=MuPConfig)

    @model_validator(mode="before")
    @classmethod
    def _resolve_size(cls, data: Any) -> Any:
        if not isinstance(data, dict) or data.get("size") is None:
            return data
        spec = get_size(data["size"])
        out = dict(data)
        out.setdefault("width", spec.width)
        out.setdefault("depth", spec.depth)
        out["size"] = (
            spec.name if (out["width"], out["depth"]) == (spec.width, spec.depth) else None
        )
        return out

    @model_validator(mode="after")
    def _check(self) -> BackboneConfig:
        if self.width % self.head_dim:
            raise ValueError(
                f"width ({self.width}) must be divisible by head_dim ({self.head_dim})"
            )
        if self.width % 4 or self.time_freq_dim % 2:
            raise ValueError("width must be divisible by 4 and time_freq_dim by 2")
        if abs(self.width * self.mlp_ratio - self.ffn_hidden) > 1e-6:
            raise ValueError(
                f"width * mlp_ratio ({self.width * self.mlp_ratio}) must be an integer"
            )
        return self

    @property
    def num_heads(self) -> int:
        return self.width // self.head_dim

    @property
    def ffn_hidden(self) -> int:
        return round(self.width * self.mlp_ratio)

    @property
    def expert_hidden(self) -> int:
        return self.moe.resolve_expert_hidden(self.ffn_hidden)

    def attn_for_layer(self, i: int) -> str:
        if not self.attn_pattern:
            return self.attn
        return self.attn_pattern[i % len(self.attn_pattern)]

    def replace(self, **changes: Any) -> BackboneConfig:
        """Validated copy. ``replace(size="b")`` swaps the whole rung; ``replace(width=512)``
        changes only the width (and clears the size label)."""
        base = self.model_dump()
        if changes.get("size") is not None:
            base.pop("width")
            base.pop("depth")
        return type(self).model_validate({**base, **changes})


class DataConfig(_Base):
    root: str = "data/latents"
    resolution: int = Field(512, gt=0)
    num_workers: int = Field(4, ge=0)
    shard_size: int = Field(256, gt=0)  # samples per shard, used by scripts/precompute.py


class TrainConfig(_Base):
    steps: int = Field(1000, gt=0)
    batch_size: int = Field(32, gt=0)
    grad_accum: int = Field(1, gt=0)
    lr: float = Field(1e-4, gt=0)
    weight_decay: float = Field(0.0, ge=0)
    adam_beta1: float = Field(0.9, ge=0, lt=1)
    adam_beta2: float = Field(0.95, ge=0, lt=1)
    adam_eps: float = Field(1e-8, gt=0)
    grad_clip_norm: float = Field(1.0, gt=0)
    warmup_steps: int = Field(100, ge=0)
    lr_schedule: Literal["constant", "cosine", "linear"] = "cosine"
    min_lr_ratio: float = Field(0.0, ge=0, lt=1)
    precision: Literal["fp32", "bf16", "fp16"] = "bf16"
    ema_decay: float = Field(0.999, ge=0, lt=1)
    cfg_dropout: float = Field(0.0, ge=0, le=1)  # prob. of swapping a caption for the null one
    log_every: int = Field(10, gt=0)
    ckpt_every: int = Field(1000, gt=0)

    @model_validator(mode="after")
    def _check(self) -> TrainConfig:
        if self.warmup_steps >= self.steps:
            raise ValueError(f"warmup_steps ({self.warmup_steps}) must be < steps ({self.steps})")
        return self


class ParallelConfig(_Base):
    strategy: Literal["none", "ddp", "fsdp"] = "none"
    tp_size: int = Field(1, gt=0)  # tensor-parallel degree; must divide backbone.num_heads
    ep_size: int = Field(1, gt=0)  # expert-parallel degree (plan only, ADR 0008); must divide
    # backbone.moe.num_experts when backbone.ffn == "moe"


class ExperimentConfig(_Base):
    run: RunConfig = Field(default_factory=RunConfig)
    backbone: BackboneConfig
    autoencoder: ComponentConfig
    text_encoder: ComponentConfig
    objective: ComponentConfig
    data: DataConfig = Field(default_factory=DataConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)

    @model_validator(mode="after")
    def _check_tp(self) -> ExperimentConfig:
        heads, tp = self.backbone.num_heads, self.parallel.tp_size
        if heads % tp:
            raise ValueError(f"num_heads ({heads}) must be divisible by parallel.tp_size ({tp})")
        ep = self.parallel.ep_size
        if self.backbone.ffn == "moe" and self.backbone.moe.num_experts % ep:
            raise ValueError(
                f"backbone.moe.num_experts ({self.backbone.moe.num_experts}) must be "
                f"divisible by parallel.ep_size ({ep})"
            )
        return self
