import pytest
import yaml
from pydantic import ValidationError

from imggen.config import (
    ConfigError,
    apply_overrides,
    config_hash,
    dump_config,
    load_config,
)


def test_loads_composed_experiment(experiment_yaml):
    cfg = load_config(experiment_yaml)
    assert cfg.run.name == "proxy_baseline"  # experiment overrides base
    assert cfg.backbone.width == 384  # from backbone/proxy.yaml
    assert cfg.autoencoder.name == "toy"  # from base.yaml
    assert cfg.backbone.num_heads == 6


def test_overrides_are_typed_and_nested(experiment_yaml):
    cfg = load_config(
        experiment_yaml,
        ["backbone.width=512", "train.lr=3e-4", "backbone.attn_pattern=[linear,linear,softmax]"],
    )
    assert cfg.backbone.width == 512
    assert cfg.train.lr == pytest.approx(3e-4)
    assert [cfg.backbone.attn_for_layer(i) for i in range(5)] == [
        "linear", "linear", "softmax", "linear", "linear",
    ]  # fmt: skip


def test_attn_for_layer_defaults_to_attn(experiment_yaml):
    cfg = load_config(experiment_yaml, ["backbone.attn=linear"])
    assert {cfg.backbone.attn_for_layer(i) for i in range(12)} == {"linear"}


def test_unknown_key_rejected(experiment_yaml):
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        load_config(experiment_yaml, ["backbone.widht=512"])


def test_width_must_divide_head_dim(experiment_yaml):
    with pytest.raises(ValidationError, match="divisible"):
        load_config(experiment_yaml, ["backbone.width=100"])


def test_bad_override_syntax_and_non_section():
    with pytest.raises(ConfigError, match="key.path=value"):
        apply_overrides({}, ["nokeyvalue"])
    with pytest.raises(ConfigError, match="not a section"):
        apply_overrides({"a": 1}, ["a.b=2"])


def test_circular_base_detected(tmp_path):
    (tmp_path / "a.yaml").write_text("base: [b.yaml]\n")
    (tmp_path / "b.yaml").write_text("base: [a.yaml]\n")
    with pytest.raises(ConfigError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_dump_roundtrip_and_hash(experiment_yaml, tmp_path):
    cfg = load_config(experiment_yaml)
    out = tmp_path / "resolved.yaml"
    out.write_text(dump_config(cfg))
    again = load_config(out)
    assert again == cfg
    assert config_hash(again) == config_hash(cfg)
    assert config_hash(load_config(experiment_yaml, ["train.lr=2e-4"])) != config_hash(cfg)


def test_component_config_builds_via_registry(experiment_yaml):
    from imggen.registry import Registry

    reg = Registry("autoencoder")
    reg.add("toy", lambda **kw: ("built", kw))
    cfg = load_config(experiment_yaml, ["autoencoder.args={scale: 32}"])
    assert cfg.autoencoder.build(reg) == ("built", {"scale": 32})


def test_base_string_form(tmp_path):
    (tmp_path / "root.yaml").write_text(yaml.safe_dump({"x": {"a": 1, "b": 2}}))
    (tmp_path / "child.yaml").write_text("base: root.yaml\nx:\n  b: 3\n")
    from imggen.config import load_raw

    assert load_raw(tmp_path / "child.yaml") == {"x": {"a": 1, "b": 3}}


# ---------------------------------------------------------------- component-swap merge (ADR 0013)


def test_switching_component_name_clears_the_old_args(tmp_path):
    import yaml

    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump({"autoencoder": {"name": "toy", "args": {"latent_channels": 32, "x": 1}}})
    )
    (tmp_path / "child.yaml").write_text(
        "base: base.yaml\nautoencoder:\n  name: sd_vae\n  args:\n    model_id: some/repo\n"
    )
    from imggen.config import load_raw

    merged = load_raw(tmp_path / "child.yaml")
    assert merged["autoencoder"] == {"name": "sd_vae", "args": {"model_id": "some/repo"}}
    assert "latent_channels" not in merged["autoencoder"]["args"]  # the bug this guards against


def test_same_component_name_still_merges_args(tmp_path):
    import yaml

    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump({"autoencoder": {"name": "toy", "args": {"latent_channels": 32, "seed": 1}}})
    )
    (tmp_path / "child.yaml").write_text(
        "base: base.yaml\nautoencoder:\n  name: toy\n  args:\n    seed: 2\n"
    )
    from imggen.config import load_raw

    merged = load_raw(tmp_path / "child.yaml")
    # seed overridden, latent_channels (not mentioned in the override) survives from base
    assert merged["autoencoder"]["args"] == {"latent_channels": 32, "seed": 2}


def test_real_components_experiment_config_has_no_leaked_toy_args():
    from imggen.config import load_config

    cfg = load_config("configs/experiment/proxy_real_components.yaml")
    assert cfg.autoencoder.name == "sd_vae"
    assert set(cfg.autoencoder.args) == {"model_id", "dtype", "sample"}
    assert cfg.text_encoder.name == "t5"
    assert set(cfg.text_encoder.args) == {"model_id", "max_length", "dtype"}
    assert cfg.backbone.in_channels == 4 and cfg.backbone.text_dim == 512


def test_non_component_dicts_are_unaffected_by_the_name_args_special_case():
    # BackboneConfig has a "name" field but no sibling "args" — must keep merging normally.
    from imggen.config import load_config

    cfg = load_config(
        "configs/experiment/proxy_baseline.yaml",
        ["backbone.moe={num_experts: 16}"],  # only one MoEConfig field touched
    )
    assert cfg.backbone.moe.num_experts == 16
    assert cfg.backbone.moe.top_k == 2  # default, survived the merge
