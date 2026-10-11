# 0001: Typed YAML configs and name registries

**Status:** accepted (roadmap step 1)

## Decision
- One typed config (`ExperimentConfig`, pydantic, `extra="forbid"`, frozen). YAML files compose
  through `base:` lists; CLI overrides use dotted `key.path=value`. No Hydra dependency.
- Swappable parts (attention, FFN, backbone, objective, autoencoder, text encoder, sampler,
  dataset, metric) live in per-kind registries. A component is one module with
  `@KIND.register("name")`; `discover()` imports all modules under the component subpackages.
- Components that are not core structure (autoencoder, text encoder, objective) are configured
  as `ComponentConfig(name, args)`, so adding one needs no schema change.
- Per-layer attention is `attn` plus an optional repeating `attn_pattern` list of registry keys
  (e.g. `[linear, linear, linear, softmax]`), not a hardcoded string alphabet.
- Every run writes `config.resolved.yaml`, `git.json`, `env.json` into
  `runs/<name>/<timestamp>_<confighash>/`.

## Why
Typos and stale keys fail at load time; experiments are reproducible from one directory;
new variants (attention kinds, AEs, objectives) do not touch shared code.

## Consequences / follow-ups
- Registry names in configs are only validated when built. Step 9 tests should build every
  shipped config end to end.
- `BackboneConfig` fields will grow with the size ladder (step 2) and muP (step 5).
- Adding a required top-level config section means updating `configs/base.yaml` and the tests.
