import shutil

import pytest
import torch
import torch.nn as nn

from imggen.backbones import build_backbone
from imggen.config import BackboneConfig, ComponentConfig, load_config
from imggen.data.shards import Manifest, ShardWriter
from imggen.objectives import build_objective
from imggen.objectives.flow_matching import FlowMatchingObjective
from imggen.registry import OBJECTIVE
from imggen.train import (
    TrainState,
    autocast_context,
    build_param_groups,
    cycle,
    run_training,
    select_device,
)
from imggen.utils.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from imggen.utils.ema import EMA
from imggen.utils.schedule import build_lr_lambda

TINY_BACKBONE = dict(
    width=48, depth=1, head_dim=16, text_dim=16, in_channels=4, mlp_ratio=2.0
)  # fmt: skip


def small_backbone(**kw) -> BackboneConfig:
    return BackboneConfig(**{**TINY_BACKBONE, **kw})


def write_tiny_cache(root, n=8, dim=16, latent_shape=(4, 4, 4), length=3, seed=0):
    writer = ShardWriter(root, shard_size=n)
    g = torch.Generator().manual_seed(seed)
    for i in range(n):
        writer.add(
            f"{i:04d}",
            torch.randn(*latent_shape, generator=g) * 0.5,
            torch.randn(length, dim, generator=g),
            torch.ones(length, dtype=torch.bool),
            f"caption {i}",
        )
    num_shards = writer.close()
    Manifest(
        num_samples=n, num_shards=num_shards, shard_size=n, latent_channels=latent_shape[0],
        spatial_compression=8, text_dim=dim, resolution=32,
    ).save(root)  # fmt: skip


def tiny_experiment_config(root, tmp_path, **overrides):
    base = {
        "data.root": str(root), "data.num_workers": 0,
        "backbone.width": 48, "backbone.depth": 1, "backbone.head_dim": 16,
        "backbone.text_dim": 16, "backbone.in_channels": 4, "backbone.mlp_ratio": 2.0,
        "train.steps": 20, "train.batch_size": 4, "train.lr": 3e-3, "train.warmup_steps": 2,
        "train.log_every": 1000, "train.ckpt_every": 20, "run.output_dir": str(tmp_path / "runs"),
    }  # fmt: skip
    base.update(overrides)
    return load_config(
        "configs/experiment/proxy_baseline.yaml", [f"{k}={v}" for k, v in base.items()]
    )


# ---------------------------------------------------------------- flow matching objective


def test_registered_and_buildable():
    assert OBJECTIVE.get("flow_matching") is FlowMatchingObjective
    obj = build_objective(ComponentConfig(name="flow_matching", args={"t_dist": "uniform"}))
    assert isinstance(obj, FlowMatchingObjective)


def test_unknown_t_dist_rejected():
    with pytest.raises(ValueError, match="t_dist"):
        FlowMatchingObjective(t_dist="nope")


@pytest.mark.parametrize("t_dist", ["uniform", "logit_normal"])
def test_sample_t_is_in_open_unit_interval(t_dist):
    obj = FlowMatchingObjective(t_dist=t_dist, min_t=1e-3)
    t = obj.sample_t(200, "cpu", torch.float32, generator=torch.Generator().manual_seed(0))
    assert t.shape == (200,)
    assert (t > 0).all() and (t < 1).all()
    assert t.min() >= 1e-3 - 1e-6 and t.max() <= 1 - 1e-3 + 1e-6


def test_logit_normal_concentrates_more_than_uniform_at_the_middle():
    g = torch.Generator().manual_seed(0)
    uniform = FlowMatchingObjective(t_dist="uniform").sample_t(
        5000, "cpu", torch.float32, generator=g
    )
    logit = FlowMatchingObjective(t_dist="logit_normal").sample_t(
        5000, "cpu", torch.float32, generator=g
    )
    mid = lambda t: ((t > 0.4) & (t < 0.6)).float().mean().item()  # noqa: E731
    assert mid(logit) > mid(uniform)


def test_loss_shape_and_gradients_reach_the_model():
    cfg = small_backbone()
    model = build_backbone(cfg)
    obj = FlowMatchingObjective()
    latent = torch.randn(3, cfg.in_channels, 4, 4)
    text = torch.randn(3, 5, cfg.text_dim)
    mask = torch.ones(3, 5, dtype=torch.bool)
    loss, logs = obj.loss(model, latent, text, mask)
    assert loss.ndim == 0 and loss.item() > 0
    assert set(logs) == {"loss", "t_mean", "target_std", "pred_std"}
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_interpolation_matches_data_and_noise_at_the_endpoints():
    latent = torch.randn(2, 4, 3, 3)
    noise = torch.randn(2, 4, 3, 3)
    for t_val, expect in [(0.0, noise), (1.0, latent)]:
        t = torch.full((2,), t_val)
        t_ = t.view(2, 1, 1, 1)
        x_t = (1 - t_) * noise + t_ * latent
        torch.testing.assert_close(x_t, expect)


def test_loss_is_deterministic_given_a_generator():
    cfg = small_backbone()
    model = build_backbone(cfg)
    obj = FlowMatchingObjective()
    latent = torch.randn(2, cfg.in_channels, 4, 4)
    text = torch.randn(2, 4, cfg.text_dim)
    mask = torch.ones(2, 4, dtype=torch.bool)
    a, _ = obj.loss(model, latent, text, mask, generator=torch.Generator().manual_seed(1))
    b, _ = obj.loss(model, latent, text, mask, generator=torch.Generator().manual_seed(1))
    assert a.item() == pytest.approx(b.item())


def test_a_few_gradient_steps_reduce_loss_on_a_fixed_batch():
    torch.manual_seed(0)
    cfg = small_backbone()
    model = build_backbone(cfg)
    obj = FlowMatchingObjective()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    latent = torch.randn(4, cfg.in_channels, 4, 4) * 0.5
    text = torch.randn(4, 3, cfg.text_dim)
    mask = torch.ones(4, 3, dtype=torch.bool)
    g = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(150):
        opt.zero_grad()
        loss, _ = obj.loss(model, latent, text, mask, generator=g)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10 * 0.6


# ---------------------------------------------------------------- EMA


def test_ema_shadow_tracks_parameter_updates():
    m = nn.Linear(4, 4)
    ema = EMA(m, decay=0.9, warmup=False)
    before = ema.shadow["weight"].clone()
    with torch.no_grad():
        m.weight.add_(1.0)
    ema.update(m, step=100)  # warmup off -> decay stays exactly 0.9
    torch.testing.assert_close(ema.shadow["weight"], before * 0.9 + (before + 1.0) * 0.1)


def test_ema_warmup_uses_a_smaller_decay_early_on():
    m = nn.Linear(4, 4)
    ema = EMA(m, decay=0.999, warmup=True)
    assert ema._effective_decay(0) == pytest.approx(1 / 10)
    assert ema._effective_decay(50) < 0.999
    assert ema._effective_decay(10**7) == pytest.approx(0.999)


def test_ema_skips_non_floating_point_buffers():
    class WithCounter(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2)
            self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    m = WithCounter()
    ema = EMA(m)
    assert "count" not in ema.shadow
    assert "fc.weight" in ema.shadow


def test_average_parameters_context_manager_restores_afterward():
    m = nn.Linear(4, 4)
    ema = EMA(m, decay=0.5, warmup=False)
    with torch.no_grad():
        m.weight.add_(5.0)
    ema.update(m, step=100)
    trained = m.weight.clone()
    with ema.average_parameters(m) as swapped:
        assert torch.equal(swapped.weight, ema.shadow["weight"])
        assert not torch.equal(swapped.weight, trained)
    assert torch.equal(m.weight, trained)


def test_ema_state_dict_roundtrip():
    m = nn.Linear(4, 4)
    ema = EMA(m)
    with torch.no_grad():
        m.weight.add_(1.0)
    ema.update(m, step=1)
    restored = EMA(nn.Linear(4, 4))
    restored.load_state_dict(ema.state_dict())
    torch.testing.assert_close(restored.shadow["weight"], ema.shadow["weight"])


# ---------------------------------------------------------------- LR schedule


def test_lr_lambda_warmup_then_holds_or_decays():
    for schedule in ("constant", "cosine", "linear"):
        f = build_lr_lambda(10, 100, schedule)
        assert f(0) == pytest.approx(0.1)  # (0+1)/10
        assert f(9) == pytest.approx(1.0)
        assert f(99) >= 0.0


def test_cosine_and_linear_decay_to_min_lr_ratio_by_the_end():
    for schedule in ("cosine", "linear"):
        f = build_lr_lambda(0, 100, schedule, min_lr_ratio=0.1)
        assert f(100) == pytest.approx(0.1, abs=1e-9)  # progress clamped to 1.0 beyond total_steps
        assert f(0) == pytest.approx(1.0, abs=1e-3)


def test_constant_never_decays_after_warmup():
    f = build_lr_lambda(5, 50, "constant")
    assert f(10) == f(49) == 1.0


def test_warmup_must_be_less_than_total_steps():
    with pytest.raises(ValueError, match="warmup_steps"):
        build_lr_lambda(100, 100, "cosine")


def test_unknown_schedule_rejected():
    with pytest.raises(ValueError, match="lr_schedule"):
        build_lr_lambda(1, 10, "exponential")


# ---------------------------------------------------------------- checkpointing


def _tiny_train_objects(lr=1e-3):
    m = nn.Linear(4, 4)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, build_lr_lambda(2, 20))
    ema = EMA(m)
    return m, opt, sched, ema


def test_save_and_load_checkpoint_restores_everything(tmp_path):
    m, opt, sched, ema = _tiny_train_objects()
    for step in range(5):
        opt.zero_grad()
        m(torch.randn(2, 4)).sum().backward()
        opt.step()
        sched.step()
        ema.update(m, step)
    path = tmp_path / "checkpoints" / "step_00000005"
    save_checkpoint(path, model=m, optimizer=opt, scheduler=sched, ema=ema, step=5)

    m2, opt2, sched2, ema2 = _tiny_train_objects()
    restored_step = load_checkpoint(path, model=m2, optimizer=opt2, scheduler=sched2, ema=ema2)
    assert restored_step == 5
    torch.testing.assert_close(m.weight, m2.weight)
    torch.testing.assert_close(ema.shadow["weight"], ema2.shadow["weight"])
    assert sched.get_last_lr() == sched2.get_last_lr()
    o1, o2 = opt.state_dict()["state"], opt2.state_dict()["state"]
    for k in o1:
        torch.testing.assert_close(o1[k]["exp_avg"], o2[k]["exp_avg"])


def test_find_latest_checkpoint_picks_highest_step(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    for step in (10, 30, 20):
        (ckpt_dir / f"step_{step:08d}").mkdir(parents=True)
    assert find_latest_checkpoint(ckpt_dir).name == "step_00000030"


def test_find_latest_checkpoint_missing_or_empty(tmp_path):
    assert find_latest_checkpoint(tmp_path / "nope") is None
    empty = tmp_path / "checkpoints"
    empty.mkdir()
    (empty / "not_a_checkpoint").mkdir()
    assert find_latest_checkpoint(empty) is None


# ---------------------------------------------------------------- train.py helpers


def test_select_device_defaults_to_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device().type == "cpu"
    assert select_device("cpu").type == "cpu"


def test_build_param_groups_without_mup_splits_by_ndim():
    model = build_backbone(small_backbone())
    groups = build_param_groups(model, lr=1e-3, weight_decay=0.1, mup=None)
    assert len(groups) == 2
    decayed_ids = {id(p) for g in groups if g["weight_decay"] > 0 for p in g["params"]}
    for p in model.parameters():
        assert (id(p) in decayed_ids) == (p.ndim >= 2)


def test_build_param_groups_with_mup_delegates_to_mup_param_groups():
    from imggen.mup import MuP

    cfg = small_backbone(mup={"enabled": True, "base_width": 48})
    model = build_backbone(cfg)
    mup = MuP(cfg.mup.base_width)
    a = build_param_groups(model, lr=1e-3, weight_decay=0.0, mup=mup)
    b = mup.param_groups(model, lr=1e-3, weight_decay=0.0)
    assert [g["lr"] for g in a] == [g["lr"] for g in b]


@pytest.mark.parametrize(
    "precision,device_type,expect_dtype",
    [("fp32", "cpu", None), ("bf16", "cpu", torch.bfloat16), ("fp16", "cpu", torch.bfloat16)],
)
def test_autocast_context_precision_and_cpu_fp16_fallback(precision, device_type, expect_dtype):
    ctx = autocast_context(torch.device(device_type), precision)
    if expect_dtype is None:
        assert ctx.fast_dtype is None or not ctx._enabled
    else:
        assert ctx.fast_dtype == expect_dtype


def test_cycle_repeats_the_iterable_indefinitely():
    it = cycle([1, 2, 3])
    assert [next(it) for _ in range(7)] == [1, 2, 3, 1, 2, 3, 1]


def test_cycle_raises_instead_of_hanging_on_an_empty_loader():
    # Regression: an empty iterable used to spin forever without yielding (found via a CLI test
    # with batch_size > dataset size and drop_last=True, which produces zero batches).
    with pytest.raises(ValueError, match="no batches"):
        next(cycle([]))


def test_run_training_fails_fast_when_batch_size_exceeds_the_dataset(tmp_path):
    root = tmp_path / "cache"
    write_tiny_cache(root, n=4)
    cfg = tiny_experiment_config(root, tmp_path, **{"train.batch_size": 32, "train.steps": 3})
    with pytest.raises(ValueError, match="no batches"):
        run_training(cfg, tmp_path / "run")


# ---------------------------------------------------------------- run_training (integration)


def test_run_training_reduces_loss_on_a_tiny_fixed_cache(tmp_path):
    root = tmp_path / "cache"
    write_tiny_cache(root, n=8)
    cfg = tiny_experiment_config(root, tmp_path, **{"train.steps": 120, "train.batch_size": 4})
    torch.manual_seed(0)
    state = run_training(cfg, tmp_path / "run")
    assert isinstance(state, TrainState) and state.step == 120
    assert state.last_loss < 1.0  # started around 1.0-1.2; a healthy run drops well below it


def test_run_training_writes_checkpoint_at_ckpt_every_and_final_step(tmp_path):
    root = tmp_path / "cache"
    write_tiny_cache(root, n=8)
    cfg = tiny_experiment_config(
        root, tmp_path, **{"train.steps": 15, "train.ckpt_every": 5, "train.batch_size": 4}
    )
    run_training(cfg, tmp_path / "run")
    ckpts = sorted(p.name for p in (tmp_path / "run" / "checkpoints").iterdir())
    assert ckpts == ["step_00000005", "step_00000010", "step_00000015"]


def test_run_training_resumes_from_checkpoint_and_reaches_new_target(tmp_path):
    root = tmp_path / "cache"
    write_tiny_cache(root, n=8)
    run_dir = tmp_path / "run"
    cfg1 = tiny_experiment_config(
        root, tmp_path, **{"train.steps": 10, "train.ckpt_every": 10, "train.batch_size": 4}
    )
    torch.manual_seed(0)
    state1 = run_training(cfg1, run_dir)
    assert state1.step == 10

    cfg2 = tiny_experiment_config(
        root, tmp_path, **{"train.steps": 20, "train.ckpt_every": 20, "train.batch_size": 4}
    )
    state2 = run_training(cfg2, run_dir)
    assert state2.step == 20
    ckpts = sorted(p.name for p in (run_dir / "checkpoints").iterdir())
    assert ckpts == ["step_00000010", "step_00000020"]


def test_run_training_already_past_target_skips_without_touching_data(tmp_path):
    root = tmp_path / "cache"
    write_tiny_cache(root, n=8)
    run_dir = tmp_path / "run"
    cfg = tiny_experiment_config(
        root, tmp_path, **{"train.steps": 10, "train.ckpt_every": 10, "train.batch_size": 4}
    )
    run_training(cfg, run_dir)
    shutil.rmtree(root)  # if run_training needed the dataloader again, this would now crash
    state = run_training(cfg, run_dir)  # same target: nothing left to do
    assert state.step == 10


def test_run_training_respects_grad_accum(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    write_tiny_cache(root, n=8)
    cfg = tiny_experiment_config(
        root, tmp_path,
        **{"train.steps": 5, "train.grad_accum": 3, "train.batch_size": 2, "train.ckpt_every": 5},
    )  # fmt: skip
    calls = []
    import imggen.train as train_mod

    real_loss = train_mod.build_objective(cfg.objective).loss

    def counting_loss(model, latent, text_emb, mask, **kw):
        calls.append(1)
        return real_loss(model, latent, text_emb, mask, **kw)

    monkeypatch.setattr(
        train_mod,
        "build_objective",
        lambda _cfg: type("O", (), {"loss": staticmethod(counting_loss)})(),
    )
    run_training(cfg, tmp_path / "run")
    assert len(calls) == 5 * 3  # one micro-batch forward per grad_accum, per optimizer step


def test_run_training_with_moe_backbone_does_not_crash(tmp_path, caplog):
    root = tmp_path / "cache"
    write_tiny_cache(root, n=8)
    cfg = tiny_experiment_config(
        root, tmp_path,
        **{
            "train.steps": 6, "train.ckpt_every": 6, "train.batch_size": 4, "train.log_every": 3,
            "backbone.ffn": "moe",
            "backbone.moe": '{"num_experts": 4, "top_k": 2, "shared_experts": 1}',
        },
    )  # fmt: skip
    import logging

    with caplog.at_level(logging.INFO, logger="imggen.train"):
        state = run_training(cfg, tmp_path / "run")
    assert state.step == 6
    assert any("moe/" in r.message for r in caplog.records)


# ---------------------------------------------------------------- classifier-free guidance (CFG)


def test_run_training_requires_null_embedding_when_cfg_dropout_enabled(tmp_path):
    root = tmp_path / "cache"
    write_tiny_cache(root, n=8)  # written directly via ShardWriter: no null_embedding.pt
    cfg = tiny_experiment_config(root, tmp_path, **{"train.cfg_dropout": 0.2, "train.steps": 3})
    with pytest.raises(FileNotFoundError, match="null_embedding"):
        run_training(cfg, tmp_path / "run")


def test_run_training_with_cfg_dropout_and_a_real_null_embedding(tmp_path):
    from imggen.data.shards import save_null_embedding

    root = tmp_path / "cache"
    write_tiny_cache(root, n=8)
    save_null_embedding(root, torch.zeros(1, 16), torch.ones(1, dtype=torch.bool))
    cfg = tiny_experiment_config(
        root, tmp_path, **{"train.cfg_dropout": 0.3, "train.steps": 15, "train.batch_size": 4}
    )
    state = run_training(cfg, tmp_path / "run")
    assert state.step == 15
    assert torch.isfinite(torch.tensor(state.last_loss))


def test_run_training_cfg_dropout_zero_does_not_require_null_embedding(tmp_path):
    root = tmp_path / "cache"
    write_tiny_cache(root, n=8)  # still no null_embedding.pt
    cfg = tiny_experiment_config(root, tmp_path, **{"train.cfg_dropout": 0.0, "train.steps": 3})
    state = run_training(cfg, tmp_path / "run")  # must not raise
    assert state.step == 3
