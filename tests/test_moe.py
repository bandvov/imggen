from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from pydantic import ValidationError

from imggen.backbones.dit import DiT
from imggen.config import SIZES, BackboneConfig, MoEConfig, estimate_params, load_config
from imggen.layers.ffn import DenseFFN
from imggen.layers.moe import (
    MoEFFN,
    collect_aux_loss,
    moe_layers,
    moe_metrics,
    step_router_bias,
)
from imggen.registry import FFN

REPO = Path(__file__).resolve().parents[1]


def make_moe(width=64, hidden=128, **kw) -> MoEFFN:
    return MoEFFN(width, hidden, MoEConfig(**kw))


def small(**kw) -> BackboneConfig:
    moe = kw.pop("moe", {"num_experts": 4, "top_k": 2, "shared_experts": 1})
    base = dict(width=128, depth=2, text_dim=48, in_channels=4, ffn="moe", moe=moe)
    return BackboneConfig(**{**base, **kw})


def inputs(cfg, b=2, h=8, w=12, length=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(b, cfg.in_channels, h, w, generator=g),
        torch.rand(b, generator=g),
        torch.randn(b, length, cfg.text_dim, generator=g),
    )


def randomize_zeros(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if not p.abs().sum() > 0:
                p.copy_(torch.randn(p.shape, generator=g) * 0.02)
    return model


def count(model):
    return sum(p.numel() for p in model.parameters())


def naive_moe(m: MoEFFN, x: torch.Tensor) -> torch.Tensor:
    """Token-by-token reference implementation of MoEFFN.forward."""
    flat = x.reshape(-1, x.shape[-1])
    out = torch.zeros_like(flat)
    for t in range(flat.shape[0]):
        probs = (flat[t] @ m.router.weight.T).softmax(-1)
        scores = probs + m.route_bias if m.moe.balance == "bias" else probs
        idx = scores.topk(m.moe.top_k).indices
        w = probs[idx] / probs[idx].sum()
        for weight, e in zip(w, idx.tolist(), strict=True):
            h = F.gelu(flat[t] @ m.fc1_weight[e].T + m.fc1_bias[e], approximate="tanh")
            out[t] += weight * (h @ m.fc2_weight[e].T + m.fc2_bias[e])
    out = out.view_as(x)
    return out if m.shared is None else out + m.shared(x)


# ---------------------------------------------------------------- parity with dense


def test_registered():
    assert FFN.get("moe") is MoEFFN


def test_single_expert_moe_equals_dense_ffn_exactly():
    torch.manual_seed(0)
    dense = DenseFFN(64, 128)
    moe = make_moe(num_experts=1, top_k=1, shared_experts=0, expert_hidden=128)
    with torch.no_grad():
        moe.fc1_weight[0].copy_(dense.fc1.weight)
        moe.fc1_bias[0].copy_(dense.fc1.bias)
        moe.fc2_weight[0].copy_(dense.fc2.weight)
        moe.fc2_bias[0].copy_(dense.fc2.bias)
    x = torch.randn(3, 20, 64)
    assert torch.equal(moe(x), dense(x))


def test_single_expert_moe_dit_equals_dense_dit():
    dense_cfg = small(ffn="dense")
    moe_cfg = small(moe={"num_experts": 1, "top_k": 1, "shared_experts": 0, "expert_hidden": 512})
    assert moe_cfg.ffn_hidden == 512
    dense = randomize_zeros(DiT(dense_cfg)).eval()
    moe = DiT(moe_cfg).eval()
    shared = {k: v for k, v in dense.state_dict().items() if ".ffn." not in k}
    moe.load_state_dict(shared, strict=False)
    with torch.no_grad():
        for bd, bm in zip(dense.blocks, moe.blocks, strict=True):
            bm.ffn.fc1_weight[0].copy_(bd.ffn.fc1.weight)
            bm.ffn.fc1_bias[0].copy_(bd.ffn.fc1.bias)
            bm.ffn.fc2_weight[0].copy_(bd.ffn.fc2.weight)
            bm.ffn.fc2_bias[0].copy_(bd.ffn.fc2.bias)
        x, t, text = inputs(dense_cfg)
        torch.testing.assert_close(moe(x, t, text), dense(x, t, text), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------- routing / dispatch


@pytest.mark.parametrize(
    "kw",
    [
        {"num_experts": 4, "top_k": 2, "shared_experts": 1},
        {"num_experts": 6, "top_k": 3, "shared_experts": 0},
        {"num_experts": 3, "top_k": 1, "shared_experts": 2},
        {"num_experts": 4, "top_k": 2, "shared_experts": 1, "balance": "bias"},
    ],
)
def test_matches_token_by_token_reference(kw):
    torch.manual_seed(1)
    m = make_moe(**kw)
    with torch.no_grad():
        m.router.weight.normal_(std=0.5)  # decisive routing
        if kw.get("balance") == "bias":
            m.route_bias.copy_(torch.tensor([0.3, -0.2, 0.1, 0.0]))
    x = torch.randn(2, 9, 64)
    torch.testing.assert_close(m(x), naive_moe(m, x), rtol=1e-4, atol=1e-5)


def test_route_picks_k_distinct_experts_with_normalised_weights():
    m = make_moe(num_experts=6, top_k=3)
    idx, weights, probs = m.route(torch.randn(50, 64))
    assert idx.shape == (50, 3) and weights.shape == (50, 3)
    assert all(len(set(row)) == 3 for row in idx.tolist())
    torch.testing.assert_close(weights.sum(-1), torch.ones(50))
    torch.testing.assert_close(probs.sum(-1), torch.ones(50))
    assert (weights >= 0).all()


def test_bias_changes_selection_but_not_combine_weights():
    torch.manual_seed(2)
    m = make_moe(num_experts=4, top_k=2, balance="bias")
    with torch.no_grad():
        m.router.weight.normal_(std=0.5)
    flat = torch.randn(40, 64)
    idx0, _, _ = m.route(flat)
    with torch.no_grad():
        m.route_bias.copy_(torch.tensor([5.0, 0.0, 0.0, -5.0]))
    idx, weights, probs = m.route(flat)
    assert not torch.equal(idx, idx0)  # selection moved
    assert (idx == 0).any(dim=-1).all() and not (idx == 3).any()
    picked = probs.gather(-1, idx)
    torch.testing.assert_close(weights, picked / picked.sum(-1, keepdim=True))  # unbiased


def test_single_token_and_hw_argument():
    m = make_moe()
    assert m(torch.randn(1, 1, 64), hw=(1, 1)).shape == (1, 1, 64)


def test_route_runs_in_fp32_under_bf16_autocast():
    m = make_moe()
    flat = torch.randn(16, 64)
    ref = m.route(flat)[2]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        idx, weights, probs = m.route(flat)
    assert probs.dtype == torch.float32 and weights.dtype == torch.float32
    torch.testing.assert_close(probs, ref)


# ---------------------------------------------------------------- balancing


def test_last_load_is_a_distribution_over_experts():
    m = make_moe(num_experts=4, top_k=2)
    m(torch.randn(2, 30, 64))
    assert m.last_load.shape == (4,)
    torch.testing.assert_close(m.last_load.sum(), torch.tensor(1.0))


def test_aux_loss_matches_formula_and_only_trains_router():
    torch.manual_seed(3)
    m = make_moe(num_experts=4, top_k=2, aux_loss_weight=0.05).train()
    with torch.no_grad():
        m.router.weight.normal_(std=0.5)
    x = torch.randn(2, 30, 64)
    m(x)
    idx, _, probs = m.route(x.reshape(-1, 64))
    f = torch.bincount(idx.reshape(-1), minlength=4).float()
    f = f / f.sum()
    expected = 0.05 * 4 * (f * probs.mean(0)).sum()
    torch.testing.assert_close(m.aux_loss, expected)
    m.aux_loss.backward()
    assert m.router.weight.grad.abs().sum() > 0
    assert m.fc1_weight.grad is None  # aux loss reaches only the router


def test_aux_loss_absent_in_eval_and_for_other_balance_modes():
    x = torch.randn(2, 10, 64)
    m = make_moe().eval()
    m(x)
    assert m.aux_loss is None
    for mode in ("bias", "none"):
        m = make_moe(balance=mode).train()
        m(x)
        assert m.aux_loss is None


def test_minimising_aux_loss_evens_out_a_skewed_router():
    torch.manual_seed(4)
    m = make_moe(num_experts=4, top_k=1, shared_experts=0, aux_loss_weight=1.0).train()
    x = torch.randn(4, 64, 64) + 0.25  # tokens must differ, or the router cannot split them
    with torch.no_grad():
        m.router.weight.normal_(std=0.02)
        m.router.weight[0] += 0.08  # expert 0 wins about 95% of tokens
    m(x)
    before = m.last_load.max().item() * 4
    opt = torch.optim.SGD([m.router.weight], lr=0.5)
    for _ in range(300):
        opt.zero_grad()
        m(x)
        m.aux_loss.backward()
        opt.step()
    m(x)
    after = m.last_load.max().item() * 4
    assert before > 3.5 and after < 1.3, (before, after)


def test_bias_balancing_evens_out_a_skewed_router():
    torch.manual_seed(5)
    m = make_moe(
        num_experts=4, top_k=1, shared_experts=0, balance="bias", bias_update_rate=0.02
    ).train()
    with torch.no_grad():
        m.route_bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))  # expert 0 wins every token
    x = torch.randn(4, 64, 64)
    m(x)
    assert m.last_load.max().item() * 4 == pytest.approx(4.0)
    m.load_accum.zero_()
    for _ in range(80):
        m(x)
        step_router_bias(m)
    m(x)
    assert m.last_load.max().item() * 4 < 1.6, m.last_load
    assert (m.load_accum > 0).all()  # accumulator only holds the last forward


def test_step_router_bias_skips_other_modes_and_resets_accumulator():
    m = make_moe(balance="aux_loss").train()
    m(torch.randn(1, 20, 64))
    before = m.route_bias.clone()
    step_router_bias(m)
    assert torch.equal(m.route_bias, before)
    b = make_moe(balance="bias").train()
    b(torch.randn(1, 20, 64))
    assert b.load_accum.sum() > 0
    step_router_bias(b)
    assert b.load_accum.sum() == 0 and b.route_bias.abs().sum() > 0


def test_state_dict_keeps_route_bias_but_not_accumulator():
    sd = make_moe(balance="bias").state_dict()
    assert "route_bias" in sd and "load_accum" not in sd


# ---------------------------------------------------------------- inside the DiT


def test_dit_with_moe_shape_zero_init_and_gradients():
    cfg = small(patch_size=2)
    model = DiT(cfg).train()
    x, t, text = inputs(cfg)
    assert model(x, t, text).abs().max() == 0  # identity at init still holds
    randomize_zeros(model)
    model(x, t, text).pow(2).mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_collect_aux_loss_and_metrics():
    cfg = small()
    model = randomize_zeros(DiT(cfg)).train()
    assert len(moe_layers(model)) == cfg.depth
    assert collect_aux_loss(model).item() == 0  # nothing recorded yet
    model(*inputs(cfg))
    aux = collect_aux_loss(model)
    parts = [m.aux_loss for _, m in moe_layers(model)]
    torch.testing.assert_close(aux, torch.stack(parts).sum())
    assert aux.requires_grad
    metrics = moe_metrics(model)
    assert "moe/max_over_mean" in metrics and "moe/blocks.0.ffn/entropy" in metrics
    assert all(v >= 0 for v in metrics.values())
    model.eval()
    model(*inputs(cfg))
    assert collect_aux_loss(model).item() == 0


def test_training_step_with_aux_loss_updates_router_and_experts():
    cfg = small()
    model = randomize_zeros(DiT(cfg)).train()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x, t, text = inputs(cfg)
    loss = model(x, t, text).pow(2).mean() + collect_aux_loss(model)
    loss.backward()
    opt.step()
    changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert any("router" in n for n in changed) and any("fc1_weight" in n for n in changed)


def test_bf16_autocast_forward_is_finite_and_hybrid_attention_composes():
    cfg = small(attn_pattern=["linear", "softmax"])
    model = randomize_zeros(DiT(cfg))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(*inputs(cfg))
    assert torch.isfinite(out.float()).all() and out.abs().sum() > 0


def test_same_seed_same_routing():
    cfg = small()
    loads = []
    for _ in range(2):
        torch.manual_seed(7)
        model = randomize_zeros(DiT(cfg)).eval()
        model(*inputs(cfg))
        loads.append(torch.stack([m.last_load for _, m in moe_layers(model)]))
    assert torch.equal(loads[0], loads[1])


# ---------------------------------------------------------------- config + estimates


def test_expert_hidden_derivation():
    proxy = BackboneConfig(size="proxy", ffn="moe")
    assert proxy.expert_hidden == 512  # 1536 // (1 shared + 2 routed)
    assert (1 + 2) * proxy.expert_hidden == proxy.ffn_hidden  # equal active FFN compute
    assert BackboneConfig(size="nano", ffn="moe").expert_hidden == 336  # 1024 // 3 -> mult. of 8
    explicit = BackboneConfig(size="proxy", ffn="moe", moe={"expert_hidden": 96})
    assert explicit.expert_hidden == 96


def test_moe_config_validation_and_roundtrip():
    with pytest.raises(ValidationError, match="top_k"):
        MoEConfig(num_experts=2, top_k=3)
    with pytest.raises(ValidationError):
        MoEConfig(balance="magic")
    cfg = BackboneConfig(size="s", ffn="moe", moe={"num_experts": 16, "top_k": 4})
    assert BackboneConfig.model_validate(cfg.model_dump()) == cfg
    assert cfg.replace(size="b").moe.num_experts == 16


def test_moe_section_is_ignored_for_dense_ffn():
    a = BackboneConfig(size="proxy", moe={"num_experts": 99, "top_k": 2})
    assert estimate_params(a) == estimate_params(BackboneConfig(size="proxy"))


@pytest.mark.parametrize(
    "moe",
    [
        {"num_experts": 4, "top_k": 2, "shared_experts": 1},
        {"num_experts": 4, "top_k": 2, "shared_experts": 0},
        {"num_experts": 8, "top_k": 1, "shared_experts": 2, "expert_hidden": 72},
        {"num_experts": 1, "top_k": 1, "shared_experts": 0},
    ],
)
def test_estimate_matches_instantiated_total_and_active(moe):
    cfg = small(moe=moe)
    model = DiT(cfg)
    est = estimate_params(cfg)
    assert count(model) == est["total"]
    experts = model.blocks[0].ffn
    per_expert = sum(
        t[0].numel()
        for t in (experts.fc1_weight, experts.fc1_bias, experts.fc2_weight, experts.fc2_bias)
    )  # one expert's slice of the stacked parameters
    unused = cfg.depth * (cfg.moe.num_experts - cfg.moe.top_k) * per_expert
    assert est["active"] == est["total"] - unused
    assert est["active"] <= est["total"]


@pytest.mark.parametrize("name", list(SIZES))
def test_estimate_equals_count_for_every_rung_with_moe(name):
    cfg = BackboneConfig(size=name, ffn="moe")
    with torch.device("meta"):
        model = DiT(cfg)
    est = estimate_params(cfg)
    assert count(model) == est["total"] and est["active"] < est["total"]


def test_moe_ladder_active_params_stay_close_to_dense():
    for name in SIZES:
        dense = estimate_params(BackboneConfig(size=name))["total"]
        active = estimate_params(BackboneConfig(size=name, ffn="moe"))["active"]
        assert 0.9 * dense < active < 1.1 * dense, name  # + router, - rounding of expert size


def test_every_shipped_experiment_config_builds_a_backbone():
    from imggen.backbones import build_backbone

    files = sorted((REPO / "configs" / "experiment").glob("*.yaml"))
    assert {f.stem for f in files} >= {"proxy_baseline", "proxy_moe"}
    for f in files:
        cfg = load_config(f).backbone
        with torch.device("meta"):
            model = build_backbone(cfg)
        assert count(model) == estimate_params(cfg)["total"], f.name
