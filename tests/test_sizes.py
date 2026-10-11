import subprocess
import sys

import pytest
from pydantic import ValidationError

from imggen.config import (
    SIZES,
    BackboneConfig,
    dump_config,
    estimate_params,
    format_params,
    load_config,
    width_sweep,
)


@pytest.mark.parametrize("name", list(SIZES))
def test_every_rung_is_valid(name):
    cfg = BackboneConfig(size=name)
    assert (cfg.width, cfg.depth) == (SIZES[name].width, SIZES[name].depth)
    assert cfg.size == name and cfg.width % cfg.head_dim == 0


def test_ladder_is_monotonic():
    cfgs = [BackboneConfig(size=n) for n in SIZES]
    params = [estimate_params(c)["total"] for c in cfgs]
    assert params == sorted(set(params))
    assert [c.width for c in cfgs] == sorted(c.width for c in cfgs)
    assert [c.depth for c in cfgs] == sorted(c.depth for c in cfgs)


def test_size_is_case_insensitive_and_unknown_lists_options():
    assert BackboneConfig(size="XL").size == "xl"
    with pytest.raises(ValidationError, match=r"unknown size.*available.*proxy"):
        BackboneConfig(size="huge")


def test_explicit_dims_override_and_clear_label():
    assert BackboneConfig(size="proxy", width=512).size is None
    assert BackboneConfig(size="proxy", depth=16).depth == 16
    same = BackboneConfig(size="proxy", width=384, depth=12)  # matches the rung -> label kept
    assert same.size == "proxy"


def test_needs_size_or_dims():
    with pytest.raises(ValidationError, match="width"):
        BackboneConfig()
    assert BackboneConfig(width=192, depth=4).size is None


def test_mlp_ratio_must_give_integer_hidden():
    with pytest.raises(ValidationError, match="integer"):
        BackboneConfig(width=384, depth=4, mlp_ratio=2.3333)


def test_dump_load_roundtrip(experiment_yaml, tmp_path):
    for overrides in ([], ["backbone.size=b"], ["backbone.width=512"]):
        cfg = load_config(experiment_yaml, overrides)
        f = tmp_path / "r.yaml"
        f.write_text(dump_config(cfg))
        assert load_config(f) == cfg


def test_cli_size_override_and_width_override(experiment_yaml):
    assert load_config(experiment_yaml).backbone.size == "proxy"
    b = load_config(experiment_yaml, ["backbone.size=b"]).backbone
    assert (b.size, b.width, b.depth) == ("b", 768, 16)
    w = load_config(experiment_yaml, ["backbone.width=512"]).backbone
    assert (w.size, w.width, w.depth) == (None, 512, 12)


def test_estimate_params_hand_computed_proxy():
    # proxy: w=384 d=12 r=1536 hd=64 C=32 p=1 T=2048 F=256, worked out by hand:
    # per block attn 4w^2+4w+2hd = 591,488; ffn 2wr+r+w = 1,181,568; norms 2w; table 6w.
    est = estimate_params(BackboneConfig(size="proxy"))
    assert est == {
        "attention": 7_097_856,
        "cross_attention": 7_097_856,
        "ffn": 14_178_816,
        "norms": 9_216,
        "modulation": 915_456,
        "embedders": 1_193_856,
        "head": 12_320,
        "total": 30_505_376,
        "active": 30_505_376,  # dense: every parameter is used per token
    }


def test_estimate_rejects_unknown_ffn():
    with pytest.raises(NotImplementedError, match="ffn 'mix'"):
        estimate_params(BackboneConfig(size="proxy", ffn="mix"))


def test_format_params():
    assert [format_params(n) for n in (950, 31_850_496, 707_788_800, 6_794_772_480)] == [
        "950", "31.9M", "708M", "6.79B",
    ]  # fmt: skip


def test_width_sweep_keeps_depth_and_flavor():
    base = BackboneConfig(size="proxy", attn="linear")
    sweep = width_sweep(base, [128, 256, 512, 1024])
    assert [c.width for c in sweep] == [128, 256, 512, 1024]
    assert {(c.depth, c.head_dim, c.attn) for c in sweep} == {(12, 64, "linear")}
    assert all(c.size is None for c in sweep)


def test_replace_size_swaps_whole_rung():
    b = BackboneConfig(size="proxy").replace(size="b")
    assert (b.size, b.width, b.depth) == ("b", 768, 16)
    assert BackboneConfig(size="b").replace(depth=8).size is None


def test_tp_size_must_divide_heads(experiment_yaml):
    load_config(experiment_yaml, ["parallel.tp_size=2"])  # proxy: 6 heads
    with pytest.raises(ValidationError, match="tp_size"):
        load_config(experiment_yaml, ["parallel.tp_size=4"])
    load_config(experiment_yaml, ["backbone.size=nano", "parallel.tp_size=4"])  # 4 heads


def test_sizes_script_prints_ladder(repo_root):
    r = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "sizes.py")], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    assert "proxy" in r.stdout and "30.5M" in r.stdout and "4xl" in r.stdout
