import pytest
import torch

from imggen.layers.attention import Attention, LinearCore, SoftmaxCore
from imggen.registry import ATTENTION

CORES = ["softmax", "linear"]


def qkv(b=2, h=3, n=7, m=5, d=8, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(b, h, length, d, generator=g, dtype=dtype) for length in (n, m, m)]


def test_cores_registered():
    assert ATTENTION.get("softmax") is SoftmaxCore
    assert ATTENTION.get("linear") is LinearCore


def test_softmax_matches_reference_and_custom_scale():
    q, k, v = qkv(dtype=torch.float64)
    for scale in (8**-0.5, 1 / 8):  # 1/sqrt(d) default, and the 1/d muP hook
        ref = torch.softmax(q @ k.transpose(-1, -2) * scale, -1) @ v
        core = SoftmaxCore(8) if scale == 8**-0.5 else SoftmaxCore(8, scale=scale)
        torch.testing.assert_close(core(q, k, v), ref)


def test_linear_matches_quadratic_reference():
    q, k, v = qkv()
    w = torch.relu(q) @ torch.relu(k).transpose(-1, -2)
    ref = (w / (w.sum(-1, keepdim=True) + 1e-6)) @ v
    torch.testing.assert_close(LinearCore(8)(q, k, v), ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name", CORES)
def test_key_mask_ignores_masked_keys(name):
    core = ATTENTION.build(name, head_dim=8)
    q, k, v = qkv(m=6)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]]).bool()
    out = core(q, k, v, mask)
    torch.testing.assert_close(out[:1], core(q[:1], k[:1, :, :4], v[:1, :, :4]))
    torch.testing.assert_close(out[1:], core(q[1:], k[1:, :, :5], v[1:, :, :5]))
    k2, v2 = k.clone(), v.clone()
    k2[0, :, 4:], v2[0, :, 4:], k2[1, :, 5:], v2[1, :, 5:] = 1e3, -1e3, 1e3, -1e3
    torch.testing.assert_close(core(q, k2, v2, mask), out)


@pytest.mark.parametrize("name", CORES)
def test_bf16_in_bf16_out_finite(name):
    q, k, v = (t.bfloat16() for t in qkv())
    out = ATTENTION.build(name, head_dim=8)(q, k, v)
    assert out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()


def make_attn(name, width=64, heads=4):
    return Attention(width, heads, ATTENTION.build(name, head_dim=width // heads))


def test_wrapper_param_parity_across_cores():
    counts = {n: sum(p.numel() for p in make_attn(n).parameters()) for n in CORES}
    assert len(set(counts.values())) == 1


def test_wrapper_self_and_cross_shapes():
    attn = make_attn("softmax")
    x, ctx = torch.randn(2, 10, 64), torch.randn(2, 6, 64)
    assert attn(x).shape == (2, 10, 64)
    assert attn(x, context=ctx).shape == (2, 10, 64)


def test_same_weights_different_cores_give_different_outputs():
    a, b = make_attn("softmax"), make_attn("linear")
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 10, 64)
    assert not torch.allclose(a(x), b(x), atol=1e-3)


@pytest.mark.parametrize("name", CORES)
def test_padded_context_has_no_influence(name):
    attn = make_attn(name)
    x, ctx = torch.randn(2, 10, 64), torch.randn(2, 6, 64)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]] * 2).bool()
    out = attn(x, context=ctx, key_mask=mask)
    ctx2 = ctx.clone()
    ctx2[:, 4:] = torch.randn(2, 2, 64) * 50
    torch.testing.assert_close(attn(x, context=ctx2, key_mask=mask), out)


@pytest.mark.parametrize("name", CORES)
def test_all_parameters_receive_gradient(name):
    attn = make_attn(name)
    attn(torch.randn(2, 10, 64), context=torch.randn(2, 6, 64)).pow(2).sum().backward()
    for pname, p in attn.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, pname
