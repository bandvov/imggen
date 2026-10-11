import pytest
import torch

from imggen.backbones import build_backbone
from imggen.config import BackboneConfig
from imggen.registry import SAMPLER
from imggen.samplers.euler import euler_sample


def small_model(randomize=False, seed=0, **kw):
    defaults = dict(width=32, depth=1, head_dim=16, text_dim=16, in_channels=4)
    defaults.update(kw)
    cfg = BackboneConfig(**defaults)
    torch.manual_seed(seed)
    model = build_backbone(cfg)
    if randomize:
        # DiT's adaLN-Zero init (ADR 0003) makes every output exactly 0 regardless of input;
        # tests that need the *conditioning* to matter (guidance, step count) need a model
        # that isn't the identity, same trick test_dit.py/test_moe.py use.
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in model.parameters():
                if not p.abs().sum() > 0:
                    p.copy_(torch.randn(p.shape, generator=g) * 0.02)
    return model, cfg


def test_registered_under_sampler():
    assert SAMPLER.get("euler") is euler_sample


def test_output_shape_matches_requested_latent_shape():
    model, cfg = small_model()
    text = torch.randn(3, 5, cfg.text_dim)
    mask = torch.ones(3, 5, dtype=torch.bool)
    out = euler_sample(model, (3, cfg.in_channels, 6, 6), text, mask, steps=4)
    assert out.shape == (3, cfg.in_channels, 6, 6)


def test_output_has_no_grad_requirement():
    model, cfg = small_model()
    text = torch.randn(2, 3, cfg.text_dim)
    out = euler_sample(model, (2, cfg.in_channels, 4, 4), text, steps=3)
    assert not out.requires_grad


def test_deterministic_given_a_generator():
    model, cfg = small_model()
    text = torch.randn(2, 3, cfg.text_dim)
    g1 = torch.Generator().manual_seed(0)
    out1 = euler_sample(model, (2, cfg.in_channels, 4, 4), text, steps=5, generator=g1)
    g2 = torch.Generator().manual_seed(0)
    out2 = euler_sample(model, (2, cfg.in_channels, 4, 4), text, steps=5, generator=g2)
    assert torch.equal(out1, out2)


def test_different_seeds_give_different_starting_noise_and_output():
    model, cfg = small_model()
    text = torch.randn(2, 3, cfg.text_dim)
    g1 = torch.Generator().manual_seed(0)
    out1 = euler_sample(model, (2, cfg.in_channels, 4, 4), text, steps=5, generator=g1)
    g2 = torch.Generator().manual_seed(1)
    out2 = euler_sample(model, (2, cfg.in_channels, 4, 4), text, steps=5, generator=g2)
    assert not torch.equal(out1, out2)


def test_more_steps_changes_the_trajectory_but_not_the_shape():
    model, cfg = small_model(randomize=True)
    text = torch.randn(2, 3, cfg.text_dim)
    out_few = euler_sample(
        model, (2, cfg.in_channels, 4, 4), text, steps=2, generator=torch.Generator().manual_seed(0)
    )
    out_many = euler_sample(
        model,
        (2, cfg.in_channels, 4, 4),
        text,
        steps=20,
        generator=torch.Generator().manual_seed(0),
    )
    assert out_few.shape == out_many.shape
    assert not torch.equal(out_few, out_many)  # different step count integrates differently


def test_return_trajectory_gives_steps_plus_one_snapshots():
    model, cfg = small_model()
    text = torch.randn(2, 3, cfg.text_dim)
    traj = euler_sample(model, (2, cfg.in_channels, 4, 4), text, steps=6, return_trajectory=True)
    assert len(traj) == 7  # initial noise + one snapshot per step
    assert all(t.shape == (2, cfg.in_channels, 4, 4) for t in traj)


def test_guidance_scale_one_matches_no_guidance_path():
    model, cfg = small_model()
    text = torch.randn(2, 3, cfg.text_dim)
    null_emb = torch.randn(1, cfg.text_dim)
    null_mask = torch.ones(1, dtype=torch.bool)
    g1 = torch.Generator().manual_seed(0)
    plain = euler_sample(model, (2, cfg.in_channels, 4, 4), text, steps=5, generator=g1)
    g2 = torch.Generator().manual_seed(0)
    guided = euler_sample(
        model, (2, cfg.in_channels, 4, 4), text, steps=5, generator=g2,
        guidance_scale=1.0, null_emb=null_emb, null_mask=null_mask,
    )  # fmt: skip
    torch.testing.assert_close(plain, guided)


def test_guidance_scale_requires_null_embedding():
    model, cfg = small_model()
    text = torch.randn(2, 3, cfg.text_dim)
    with pytest.raises(ValueError, match="null_emb"):
        euler_sample(model, (2, cfg.in_channels, 4, 4), text, steps=3, guidance_scale=2.0)


def test_guidance_scale_changes_output_when_not_one():
    model, cfg = small_model(randomize=True)
    text = torch.randn(2, 3, cfg.text_dim)
    null_emb = torch.randn(1, cfg.text_dim)
    null_mask = torch.ones(1, dtype=torch.bool)
    out_1 = euler_sample(
        model, (2, cfg.in_channels, 4, 4), text, steps=5, guidance_scale=1.0,
        null_emb=null_emb, null_mask=null_mask, generator=torch.Generator().manual_seed(0),
    )  # fmt: skip
    out_3 = euler_sample(
        model, (2, cfg.in_channels, 4, 4), text, steps=5, guidance_scale=3.0,
        null_emb=null_emb, null_mask=null_mask, generator=torch.Generator().manual_seed(0),
    )  # fmt: skip
    assert not torch.equal(out_1, out_3)


def test_works_without_a_text_mask():
    model, cfg = small_model()
    text = torch.randn(2, 3, cfg.text_dim)
    out = euler_sample(model, (2, cfg.in_channels, 4, 4), text, text_mask=None, steps=3)
    assert out.shape == (2, cfg.in_channels, 4, 4)


def test_hybrid_and_linear_attention_backbones_also_sample():
    for attn_kw in ({"attn": "linear"}, {"attn_pattern": ["linear", "softmax"]}):
        model, cfg = small_model(randomize=True, depth=2, **attn_kw)
        text = torch.randn(2, 3, cfg.text_dim)
        out = euler_sample(model, (2, cfg.in_channels, 4, 4), text, steps=3)
        assert torch.isfinite(out).all()
