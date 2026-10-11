import pytest
import torch

from imggen.backbones import build_backbone
from imggen.backbones.dit import DiT, patchify, sincos_2d, unpatchify
from imggen.config import SIZES, BackboneConfig, estimate_params
from imggen.layers.attention import LinearCore, SoftmaxCore
from imggen.registry import BACKBONE, RegistryError


def small(**kw) -> BackboneConfig:
    base = dict(width=128, depth=2, text_dim=48, in_channels=4)
    return BackboneConfig(**{**base, **kw})


def inputs(cfg, b=2, h=8, w=12, length=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, cfg.in_channels, h, w, generator=g)
    t = torch.rand(b, generator=g)
    text = torch.randn(b, length, cfg.text_dim, generator=g)
    return x, t, text


def randomize_zeros(model, seed=0):
    """Zero-init params (identity-at-init) hide bugs; give them small random values."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if not p.abs().sum() > 0:
                p.copy_(torch.randn(p.shape, generator=g) * 0.02)
    return model


def count(model):
    return sum(p.numel() for p in model.parameters())


def test_registered_and_built_from_config():
    assert BACKBONE.get("dit") is DiT
    assert isinstance(build_backbone(small()), DiT)


@pytest.mark.parametrize("p,hw", [(1, (8, 12)), (2, (8, 12)), (2, (6, 10))])
def test_output_shape_matches_input(p, hw):
    cfg = small(patch_size=p)
    x, t, text = inputs(cfg, h=hw[0], w=hw[1])
    assert DiT(cfg)(x, t, text).shape == x.shape


def test_indivisible_grid_rejected():
    cfg = small(patch_size=2)
    x, t, text = inputs(cfg, h=7, w=8)
    with pytest.raises(ValueError, match="divisible"):
        DiT(cfg)(x, t, text)


def test_output_is_exactly_zero_at_init():
    cfg = small()
    out = DiT(cfg)(*inputs(cfg))
    assert out.abs().max() == 0


@pytest.mark.parametrize("attn", ["softmax", "linear"])
def test_every_parameter_gets_gradient(attn):
    cfg = small(attn=attn, patch_size=2)
    model = randomize_zeros(DiT(cfg))
    model(*inputs(cfg)).pow(2).mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_hybrid_pattern_builds_expected_cores():
    cfg = small(depth=4, attn_pattern=["linear", "linear", "softmax"])
    model = DiT(cfg)
    kinds = [type(b.attn.core) for b in model.blocks]
    assert kinds == [LinearCore, LinearCore, SoftmaxCore, LinearCore]
    assert all(isinstance(b.cross_attn.core, SoftmaxCore) for b in model.blocks)


def test_unknown_attention_name_fails_with_registry_error():
    with pytest.raises(RegistryError, match="unknown attention"):
        DiT(small(attn="nope"))


def test_softmax_and_linear_have_identical_param_counts():
    assert count(DiT(small(attn="softmax"))) == count(DiT(small(attn="linear")))


def test_padded_text_tokens_do_not_change_output():
    cfg = small()
    model = randomize_zeros(DiT(cfg)).eval()
    x, t, text = inputs(cfg, length=6)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]] * 2).bool()
    with torch.no_grad():
        a = model(x, t, text, mask)
        text2 = text.clone()
        text2[:, 4:] = torch.randn(2, 2, cfg.text_dim) * 20
        b = model(x, t, text2, mask)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_samples_in_a_batch_are_independent():
    cfg = small()
    model = randomize_zeros(DiT(cfg)).eval()
    x, t, text = inputs(cfg)
    with torch.no_grad():
        both = model(x, t, text)
        first = model(x[:1], t[:1], text[:1])
    torch.testing.assert_close(both[:1], first, rtol=1e-4, atol=1e-5)


def test_output_depends_on_time_and_text():
    cfg = small()
    model = randomize_zeros(DiT(cfg)).eval()
    x, t, text = inputs(cfg)
    with torch.no_grad():
        base = model(x, t, text)
        assert not torch.allclose(base, model(x, (t + 0.5) % 1, text), atol=1e-4)
        assert not torch.allclose(base, model(x, t, text + 1.0), atol=1e-4)


def test_same_seed_same_weights_and_output():
    cfg = small()
    torch.manual_seed(3)
    a = DiT(cfg)
    torch.manual_seed(3)
    b = DiT(cfg)
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert torch.equal(pa, pb), n


def test_bf16_autocast_forward_is_finite():
    cfg = small(attn_pattern=["linear", "softmax"])
    model = randomize_zeros(DiT(cfg))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(*inputs(cfg))
    assert torch.isfinite(out.float()).all() and out.abs().sum() > 0


def test_patchify_unpatchify_roundtrip():
    x = torch.randn(2, 5, 8, 12)
    tokens = patchify(x, 2)
    assert tokens.shape == (2, 4 * 6, 2 * 2 * 5)
    torch.testing.assert_close(unpatchify(tokens, 4, 6, 2, 5), x)


def test_sincos_2d_layout_is_row_major_and_distinct():
    emb = sincos_2d(8, gh=3, gw=4)
    assert emb.shape == (12, 8)
    assert emb[0].tolist() == [0, 0, 1, 1, 0, 0, 1, 1]  # position (0,0): sin=0, cos=1
    assert len({tuple(r.tolist()) for r in emb}) == 12
    assert torch.equal(emb[4, 4:], emb[0, 4:]) and not torch.equal(emb[4, :4], emb[0, :4])  # y step
    assert torch.equal(emb[1, :4], emb[0, :4]) and not torch.equal(emb[1, 4:], emb[0, 4:])  # x step


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"patch_size": 2},
        {"mlp_ratio": 3.5},
        {"head_dim": 32, "text_dim": 64, "in_channels": 8},
        {"time_freq_dim": 128, "attn": "linear"},
    ],
)
def test_estimate_equals_instantiated_count(kw):
    cfg = small(**kw)
    assert count(DiT(cfg)) == estimate_params(cfg)["total"]


@pytest.mark.parametrize("name", list(SIZES))
def test_estimate_equals_count_for_every_rung(name):
    cfg = BackboneConfig(size=name)
    with torch.device("meta"):  # shapes only; no memory for the multi-billion rungs
        model = DiT(cfg)
    assert count(model) == estimate_params(cfg)["total"]
