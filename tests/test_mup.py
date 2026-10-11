import math

import pytest
import torch

from imggen.backbones.dit import DiT
from imggen.config import SIZES, BackboneConfig, estimate_params
from imggen.mup import MuP, ParamRole, _fan_in, activation_rms_by_width, classify, lr_multiplier


def small(**kw) -> BackboneConfig:
    base = dict(
        width=128, depth=2, text_dim=48, in_channels=4, mup={"enabled": True, "base_width": 128}
    )
    return BackboneConfig(**{**base, **kw})


def inputs(cfg, b=3, h=8, w=8, length=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(b, cfg.in_channels, h, w, generator=g),
        torch.rand(b, generator=g),
        torch.randn(b, length, cfg.text_dim, generator=g),
    )


# ---------------------------------------------------------------- classify()


def test_every_parameter_of_a_dense_model_is_classified():
    cfg = small()
    roles = classify(DiT(cfg))
    names = {n for n, _ in DiT(cfg).named_parameters()}
    assert set(roles) == names


def test_every_parameter_of_a_moe_model_is_classified():
    cfg = small(ffn="moe", moe={"num_experts": 4, "top_k": 2, "shared_experts": 1})
    model = DiT(cfg)
    roles = classify(model)
    assert set(roles) == {n for n, _ in model.named_parameters()}


def test_biases_and_norm_scales_are_fixed():
    roles = classify(DiT(small()))
    for name, role in roles.items():
        if name.endswith(".bias") or "norm" in name or name.endswith("mod_table"):
            assert role is ParamRole.FIXED, name


def test_known_layers_get_expected_roles():
    roles = classify(DiT(small()))
    expect = {
        "patch_embed.weight": ParamRole.INPUT,
        "t_embedder.0.weight": ParamRole.INPUT,
        "t_embedder.2.weight": ParamRole.HIDDEN,
        "t_block.1.weight": ParamRole.HIDDEN,
        "text_proj.0.weight": ParamRole.INPUT,
        "text_proj.2.weight": ParamRole.HIDDEN,
        "final_proj.weight": ParamRole.OUTPUT,
        "blocks.0.attn.q_proj.weight": ParamRole.HIDDEN,
        "blocks.0.attn.out_proj.weight": ParamRole.HIDDEN,
        "blocks.0.cross_attn.out_proj.weight": ParamRole.HIDDEN,
    }
    for name, role in expect.items():
        assert roles[name] is role, name


def test_moe_router_is_output_experts_are_hidden():
    cfg = small(ffn="moe", moe={"num_experts": 6, "top_k": 2, "shared_experts": 1})
    roles = classify(DiT(cfg))
    assert roles["blocks.0.ffn.router.weight"] is ParamRole.OUTPUT
    assert roles["blocks.0.ffn.fc1_weight"] is ParamRole.HIDDEN
    assert roles["blocks.0.ffn.fc2_weight"] is ParamRole.HIDDEN
    assert roles["blocks.0.ffn.shared.fc1.weight"] is ParamRole.HIDDEN


def test_classify_raises_on_unrecognised_ffn(monkeypatch):
    import torch.nn as nn

    from imggen.registry import FFN

    @FFN.register("mystery")
    class MysteryFFN(nn.Module):
        def __init__(self, width, hidden, moe=None):
            super().__init__()
            self.fc = nn.Linear(width, width)

        def forward(self, x, hw=None):
            return self.fc(x)

    try:
        with pytest.raises(NotImplementedError, match="MysteryFFN"):
            classify(DiT(small(ffn="mystery")))
    finally:
        FFN.unregister("mystery")


# ---------------------------------------------------------------- fan_in / init std


def test_fan_in_matches_shape_for_linear_layers():
    cfg = small(text_dim=48, time_freq_dim=64)
    model = DiT(cfg)
    for name, p in model.named_parameters():
        if name in ("blocks.0.attn.q_proj.weight", "blocks.0.attn.out_proj.weight"):
            assert _fan_in(name, p, model) == cfg.width
    tw = dict(model.named_parameters())["t_embedder.0.weight"]
    assert _fan_in("t_embedder.0.weight", tw, model) == cfg.time_freq_dim


def test_fan_in_of_patch_embed_uses_config_not_flattened_shape():
    cfg = small(patch_size=2, in_channels=5)
    model = DiT(cfg)
    p = model.patch_embed.weight  # shape (width, in_channels, p, p)
    assert _fan_in("patch_embed.weight", p, model) == 5 * 2 * 2


@pytest.mark.parametrize("width", [128, 512])
def test_output_layer_init_std_is_tighter_than_hidden(width):
    # final_proj is in the zero-init set (always exactly 0), so use the MoE router instead —
    # it is the other OUTPUT-role weight and is not zero-initialized.
    torch.manual_seed(0)
    cfg = BackboneConfig(
        width=width, depth=1, text_dim=32, in_channels=4, ffn="moe",
        moe={"num_experts": 8, "top_k": 2, "shared_experts": 0},
        mup={"enabled": True, "base_width": 128},
    )  # fmt: skip
    model = DiT(cfg)
    hidden_std = model.blocks[0].attn.q_proj.weight.std().item()
    output_std = model.blocks[0].ffn.router.weight.std().item()
    assert output_std < hidden_std  # 1/fan_in vs 1/sqrt(fan_in): strictly tighter for width > 1
    assert output_std == pytest.approx(1.0 / width, rel=0.25)
    assert hidden_std == pytest.approx(width**-0.5, rel=0.2)


def test_input_and_hidden_layers_share_the_same_init_formula_at_equal_fan_in():
    # t_embedder.2 (hidden, fan_in=width) vs a synthetic input layer with the same fan_in:
    # both should draw from std = fan_in^-0.5, i.e. sampled stds should be statistically equal.
    torch.manual_seed(0)
    width = 256
    cfg = BackboneConfig(
        width=width, depth=1, text_dim=width, in_channels=4,
        mup={"enabled": True, "base_width": 128},
    )  # fmt: skip
    model = DiT(cfg)
    hidden = model.t_embedder[2].weight  # HIDDEN, fan_in = width
    inp = model.text_proj[0].weight  # INPUT, fan_in = text_dim = width (matched on purpose)
    assert hidden.std().item() == pytest.approx(inp.std().item(), rel=0.15)
    assert hidden.std().item() == pytest.approx(width**-0.5, rel=0.15)


def test_biases_and_norm_scales_are_untouched_by_mup_init():
    cfg = small()
    model = DiT(cfg)
    for name, p in model.named_parameters():
        if name.endswith(".bias"):
            assert (p == 0).all(), name
    for blk in model.blocks:
        assert torch.equal(blk.attn.q_norm.weight, torch.ones_like(blk.attn.q_norm.weight))
        assert torch.equal(blk.norm_cross.weight, torch.ones_like(blk.norm_cross.weight))


def test_zero_init_set_stays_zero_under_mup():
    model = DiT(small())
    for m in model.zero_init_modules():
        assert (m.weight == 0).all() and (m.bias == 0).all()


def test_mup_output_is_exactly_zero_at_init_same_as_standard():
    cfg = small()
    assert DiT(cfg)(*inputs(cfg)).abs().max() == 0


def test_mup_model_has_same_param_count_as_standard():
    on = BackboneConfig(size="s", mup={"enabled": True})
    off = BackboneConfig(size="s", mup={"enabled": False})
    assert estimate_params(on) == estimate_params(off)


# ---------------------------------------------------------------- LR multiplier / param_groups


def test_lr_multiplier_is_one_at_base_width():
    for role in ParamRole:
        assert lr_multiplier(role, width_mult=1.0) == pytest.approx(1.0)


@pytest.mark.parametrize("width_mult", [0.5, 2.0, 4.0])
def test_lr_multiplier_scales_hidden_and_output_only(width_mult):
    assert lr_multiplier(ParamRole.INPUT, width_mult) == 1.0
    assert lr_multiplier(ParamRole.FIXED, width_mult) == 1.0
    assert lr_multiplier(ParamRole.HIDDEN, width_mult) == pytest.approx(1.0 / width_mult)
    assert lr_multiplier(ParamRole.OUTPUT, width_mult) == pytest.approx(1.0 / width_mult)


def test_width_mult_matches_ratio():
    mup = MuP(base_width=384)
    assert mup.width_mult(384) == pytest.approx(1.0)
    assert mup.width_mult(768) == pytest.approx(2.0)
    assert mup.width_mult(192) == pytest.approx(0.5)


def test_base_width_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        MuP(base_width=0)


def test_param_groups_cover_every_trainable_parameter_exactly_once():
    cfg = small()
    model = DiT(cfg)
    groups = MuP(cfg.mup.base_width).param_groups(model, lr=1e-3)
    covered = [p for g in groups for p in g["params"]]
    assert len(covered) == len(list(model.parameters()))
    assert {id(p) for p in covered} == {id(p) for p in model.parameters()}


def test_param_groups_lr_values_match_role_multipliers():
    width, base_width = 512, 128
    cfg = BackboneConfig(
        width=width, depth=1, text_dim=32, in_channels=4,
        mup={"enabled": True, "base_width": base_width},
    )  # fmt: skip
    model = DiT(cfg)
    mup = MuP(base_width)
    groups = mup.param_groups(model, lr=1e-2)
    seen_lrs = {round(g["lr"], 8) for g in groups}
    width_mult = width / base_width
    assert seen_lrs == {round(1e-2, 8), round(1e-2 / width_mult, 8)}


def test_param_groups_weight_decay_skips_biases_and_input_layers():
    cfg = small()
    model = DiT(cfg)
    groups = MuP(cfg.mup.base_width).param_groups(model, lr=1e-3, weight_decay=0.1)
    # every bias parameter must land in a zero-weight-decay group
    decayed_ids = {id(p) for g in groups if g["weight_decay"] > 0 for p in g["params"]}
    for name, p in model.named_parameters():
        if name.endswith(".bias"):
            assert id(p) not in decayed_ids, name


def test_optimizer_accepts_param_groups_and_trains_without_nan():
    cfg = small()
    model = DiT(cfg).train()
    for p in model.parameters():  # break the zero-init identity so every path gets a gradient
        if not p.abs().sum() > 0:
            with torch.no_grad():
                p.add_(torch.randn_like(p) * 0.02)
    opt = torch.optim.AdamW(MuP(cfg.mup.base_width).param_groups(model, lr=1e-3, weight_decay=0.01))
    x, t, text = inputs(cfg)
    for _ in range(3):
        opt.zero_grad()
        model(x, t, text).pow(2).mean().backward()
        opt.step()
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), name


# ---------------------------------------------------------------- integration


@pytest.mark.parametrize("name", list(SIZES))
def test_mup_enabled_matches_param_count_for_every_rung(name):
    cfg = BackboneConfig(size=name, mup={"enabled": True})
    with torch.device("meta"):
        model = DiT(cfg)
    assert sum(p.numel() for p in model.parameters()) == estimate_params(cfg)["total"]


def test_disabled_by_default_and_standard_path_unaffected():
    cfg = BackboneConfig(size="proxy")
    assert cfg.mup.enabled is False
    torch.manual_seed(0)
    a = DiT(cfg)
    torch.manual_seed(0)
    b = DiT(cfg)  # default path must remain bit-identical across construction
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert torch.equal(pa, pb), n


def test_replace_preserves_mup_settings():
    cfg = BackboneConfig(size="proxy", mup={"enabled": True, "base_width": 384})
    bigger = cfg.replace(size="b")
    assert bigger.mup.enabled is True and bigger.mup.base_width == 384


def test_softmax_attention_scale_is_unaffected_by_mup():
    """Documents the design decision in mup.py: head_dim is fixed across the ladder, so muP's
    1/d attention correction is unnecessary here; scale must stay 1/sqrt(head_dim) regardless
    of width or whether muP is enabled."""
    for width in (128, 512, 2048):
        cfg = BackboneConfig(
            width=width, depth=1, text_dim=32, in_channels=4, head_dim=64,
            mup={"enabled": True, "base_width": 128},
        )  # fmt: skip
        model = DiT(cfg)
        assert model.blocks[0].attn.core.scale == pytest.approx(64**-0.5)


# ---------------------------------------------------------------- diagnostic helper


def test_activation_rms_by_width_returns_one_finite_value_per_width():
    base = BackboneConfig(
        width=384, depth=1, text_dim=32, in_channels=4,
        mup={"enabled": True, "base_width": 384},
    )  # fmt: skip
    widths = [128, 256, 384]
    out = activation_rms_by_width(base, widths, seed=0)
    assert set(out) == set(widths)
    assert all(math.isfinite(v) and v > 0 for v in out.values())


def test_activation_rms_by_width_is_deterministic_given_a_seed():
    base = BackboneConfig(
        width=128, depth=1, text_dim=32, in_channels=4,
        mup={"enabled": True, "base_width": 128},
    )  # fmt: skip
    a = activation_rms_by_width(base, [128, 256], seed=3)
    b = activation_rms_by_width(base, [128, 256], seed=3)
    assert a == b
