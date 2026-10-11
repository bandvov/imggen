# 0013: Real pretrained components — SD/SDXL VAE and T5

**Status:** accepted. Closes ADR 0006's long-standing caveat for the user's own machine; this
project's own sandbox still has no network access to huggingface.co, so the `from_pretrained`
path is implemented and wired up but genuinely untested here — see "Verification" below for
exactly what that does and doesn't cover.

## Decisions
- **`imggen.autoencoders.sd_vae.StableDiffusionVAE`** ("sd_vae" in `AUTOENCODER`): wraps
  `diffusers.AutoencoderKL`. `encode` uses the latent distribution's *mode* by default
  (`sample=False`) — deterministic, matching this project's established frozen-AE convention
  (ADR 0006, same choice `ToyAutoencoder` makes) — with `sample=True` available to match how SD
  itself trains (sampling from the posterior adds regularization). Applies the VAE's
  `scaling_factor` on encode, inverts it on decode, exactly like real SD/SDXL pipelines do.
  `spatial_compression` is inferred from `len(down_block_types) - 1` and overridable.
- **`imggen.text.t5_encoder.T5TextEncoder`** ("t5" in `TEXT_ENCODER`): wraps
  `transformers.T5EncoderModel` + tokenizer. `text_dim` is read from the model's own config
  (`d_model`), not hardcoded — correct for any T5 variant without extra configuration.
- **Both are optional dependencies** (`pip install -e ".[real-components]"` —
  `diffusers`, `transformers`, `sentencepiece`): most of this project, including every default
  config, uses the ADR 0006 toy/hash placeholders and never imports either.
- **Both accept injected underlying models** (`_vae` / `_model`+`_tokenizer`) specifically so
  their logic is testable without network access: a tiny, locally-constructed (no download)
  `AutoencoderKL`/`T5EncoderModel` exercises the real library code the production path also
  uses, just skipping the network call. This is how every test in `tests/test_real_components.py`
  works — not mocks of the wrapper's own logic, real diffusers/transformers objects.
- **`configs/components/sd_vae_t5.yaml`** + **`configs/experiment/proxy_real_components.yaml`**:
  a ready-to-compose overlay and a full runnable experiment config
  (`stabilityai/sdxl-vae` + `google/t5-v1_1-small`, `backbone.in_channels=4`,
  `backbone.text_dim=512`).

## A real config-loader bug, found while writing the example config
Composing `configs/components/sd_vae_t5.yaml` onto `configs/base.yaml` left
`latent_channels`/`num_downsamples` (the *toy* AE's args) sitting in the merged
`autoencoder.args`, alongside the new `model_id`/`dtype`/`sample` — confirmed directly:
`load_config(...).autoencoder.args` contained both sets, which `StableDiffusionVAE.__init__`
doesn't accept (`TypeError: unexpected keyword argument 'latent_channels'`, waiting to happen the
moment the config was actually built, not at load time). Root cause: `_deep_merge`
(`imggen/config/loader.py`) merges nested dicts unconditionally, and `args` is semantically "the
constructor kwargs for *this specific* component" — merging it across a `name` change leaks the
previous implementation's args into the new one's constructor call.

Fix, narrowly scoped: when merging two dicts that both look like a `ComponentConfig`
(`{"name": ..., "args": {...}}`) and the override's `name` differs from the base's, `args` is
*replaced* wholesale instead of merged. Deliberately narrow — `BackboneConfig` also has a `name`
field but no `args` sibling, so it's unaffected, and a same-name override (e.g. just
`autoencoder.args.foo=1`) still merges normally, preserving every other arg's default. Verified
this doesn't regress anything that relies on partial same-name merges (e.g.
`backbone.moe={num_experts: 16}` still keeps `top_k`'s default) by running the full suite before
and after: 327/327 both times, then the new tests on top.

## Verification
- `tests/test_real_components.py` (21 tests): shapes, determinism (mode vs sample), frozen/eval
  state, dtype validation, registry wiring (`build_autoencoder`/`build_text_encoder`), scaling
  factor correctness (checked against the raw `AutoencoderKL` call directly, not just shape).
- `tests/test_config.py` (5 new tests): the merge bug, reproduced and fixed; confirmed the fix
  doesn't affect same-name partial merges or non-`ComponentConfig` sections.
- Full suite: 352/352, `ruff check` clean.
- **What is not, and cannot be, tested here**: the actual `from_pretrained` download path, real
  SDXL-VAE or T5-v1.1-small weights, real image reconstruction quality, real caption
  understanding, or anything about generation quality with real components. All of that needs a
  machine with network access to huggingface.co — this project's own sandbox doesn't have one.

## How to actually use this (on a machine with network access)
```bash
pip install -e ".[real-components]"
python scripts/precompute.py configs/experiment/proxy_real_components.yaml \
    data.root=/path/to/cache --num-samples 2000
python scripts/train_toy_autoencoder.py --out ...   # NOT needed — sd_vae is already pretrained;
                                                       # this script is toy-AE-specific (ADR 0011)
python scripts/train.py configs/experiment/proxy_real_components.yaml data.root=/path/to/cache
python scripts/sample.py <run_dir>/checkpoints/step_N configs/experiment/proxy_real_components.yaml \
    data.root=/path/to/cache --caption "..." --guidance-scale 4 --out sample.pt
```
First run downloads and caches both models locally (standard `huggingface_hub` cache behavior);
subsequent runs reuse the cache. `configs/components/sd_vae_t5.yaml`'s comments cover the
`subfolder` option for repos that bundle the VAE inside a full pipeline rather than standalone.

## Not done / follow-ups
- **Nothing here has been run with real weights, anywhere, by anyone, yet.** This is infrastructure
  and tests, not a result — the gap is closed in principle, not in practice until someone runs it.
- No other VAE (e.g. DC-AE, the original plan's high-compression choice) or text encoder (Gemma,
  CLIP) is wired up — same pattern, one new registered class each, whenever needed.
- `dtype="float16"`/`"bfloat16"` are implemented but untested (no GPU here to meaningfully
  exercise mixed precision on these specific wrappers).
- The `_vae`/`_model`+`_tokenizer` injection hooks are a testing convenience, not validated as a
  good long-term API if a legitimate non-testing use for them ever shows up (e.g. sharing one
  loaded model across multiple wrapper instances).
