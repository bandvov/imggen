"""YAML loading with ``base:`` composition and dotted CLI overrides."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

from .schema import ExperimentConfig


class ConfigError(ValueError):
    pass


def _is_component_dict(d: object) -> bool:
    """True for a raw dict shaped like ``ComponentConfig`` (``{"name": ..., "args": {...}}``) —
    the pattern ``autoencoder``/``text_encoder``/``objective`` sections use. Narrow on purpose:
    ``BackboneConfig`` also has a ``name`` field but no ``args`` sibling, so it's unaffected."""
    return isinstance(d, dict) and "name" in d and "args" in d


def _deep_merge(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in other.items():
        base_val = out.get(key)
        if (
            _is_component_dict(val)
            and _is_component_dict(base_val)
            and val.get("name") != base_val.get("name")
        ):
            # Switching components (e.g. autoencoder.name: toy -> sd_vae): args are constructor
            # kwargs for a *specific* class, so merging them across a name change would leak the
            # old implementation's args into the new one's constructor call (found while writing
            # configs/components/sd_vae_t5.yaml — composing it onto base.yaml's toy/hash defaults
            # left `latent_channels`/`num_downsamples` in autoencoder.args, which
            # StableDiffusionVAE.__init__ doesn't accept, a TypeError waiting to happen the
            # moment the config was actually built, not at load time). Same name: still merge,
            # so a plain `autoencoder.args.foo=1` override keeps every other arg's default.
            out[key] = copy.deepcopy(val)
        elif isinstance(val, dict) and isinstance(base_val, dict):
            out[key] = _deep_merge(base_val, val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_raw(path: str | Path, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML file, first merging every file listed under ``base:`` (paths are
    relative to the file that names them; later entries override earlier ones)."""
    path = Path(path).resolve()
    if path in _stack:
        chain = " -> ".join(p.name for p in (*_stack, path))
        raise ConfigError(f"circular 'base' reference: {chain}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    bases = data.pop("base", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: dict[str, Any] = {}
    for b in bases:
        merged = _deep_merge(merged, load_raw(path.parent / b, (*_stack, path)))
    return _deep_merge(merged, data)


def apply_overrides(raw: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides; values are parsed as YAML (so lists/numbers work)."""
    out = copy.deepcopy(raw)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override {item!r} must look like key.path=value")
        key, _, value = item.partition("=")
        node = out
        *parents, leaf = key.strip().split(".")
        for p in parents:
            nxt = node.setdefault(p, {})
            if not isinstance(nxt, dict):
                raise ConfigError(f"override {item!r}: {p!r} is not a section")
            node = nxt
        node[leaf] = yaml.safe_load(value)
    return out


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> ExperimentConfig:
    return ExperimentConfig.model_validate(apply_overrides(load_raw(path), overrides))


def dump_config(cfg: ExperimentConfig) -> str:
    return yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)


def config_hash(cfg: ExperimentConfig, length: int = 8) -> str:
    blob = json.dumps(cfg.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:length]
