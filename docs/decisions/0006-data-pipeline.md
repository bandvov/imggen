# 0006: Data pipeline — cached latents and text embeddings

**Status:** accepted (roadmap step 6)

## Decisions
- **Two new component kinds, same registry pattern as everything else:** `Autoencoder`
  (`encode`/`decode`, `latent_channels`, `spatial_compression`) and `TextEncoder`
  (`encode(list[str]) -> (embeddings, mask)`, `text_dim`), built via `ComponentConfig` exactly
  like the objective/other components (ADR 0001).
- **No network access to a real pretrained AE or text encoder.** This sandbox can reach package
  registries (pypi, npm, github) but not a model hub, so step 6 ships two small, frozen,
  deterministic placeholders instead of downloading weights:
  - `ToyAutoencoder` (registered `"toy"`): a strided-conv encoder/decoder, configurable
    `latent_channels` and `num_downsamples` (so `spatial_compression = 2**num_downsamples`).
    Reconstructions are not expected to be good — it exists to give the pipeline real shapes and
    real (if crude) numbers to move.
  - `HashEmbeddingTextEncoder` (registered `"hash"`): whitespace tokens → sha256 hash → a fixed
    random embedding table → a fixed random per-token MLP. No notion of language; two unrelated
    tokens are unrelated. Deterministic given `seed` (sha256, not Python's salted `hash()`), so
    reruns of precompute reproduce the same vectors.
  - Swapping either for a real pretrained model is a new registered class under a new name,
    pointed at by `autoencoder.name` / `text_encoder.name`; nothing else changes.
- **`ToyShapesDataset`** (registered `"toy_shapes"` in `DATASET`): procedurally generated
  shapes-on-canvas images with a templated caption, standing in for "a small, clean dataset"
  since no real image dataset can be downloaded here either.
- **Shard format:** plain `tarfile`, no new dependency. Each sample is three files under one key —
  `{key}.latent.pth`, `{key}.text_emb.pth`, `{key}.json` (mask + caption) — the same
  key-with-extension convention WebDataset uses, so a move to the `webdataset` package for
  large-scale streaming later is a loader change, not a re-export of the cache.
  `ShardWriter` rolls to a new `.tar` every `data.shard_size` samples; `Manifest` (JSON) records
  counts plus `latent_channels`/`text_dim`/`spatial_compression` for validation.
- **Validate early, not deep in training.** `Manifest.check()` compares the cache's
  `latent_channels`/`text_dim` against `backbone.in_channels`/`backbone.text_dim` and raises
  `ManifestMismatch` immediately; `build_dataloader(cfg, backbone=...)` calls it at dataloader
  construction, so a stale or wrong cache fails before the first training step, not inside a
  matmul shape error three layers deep. This resolves the open item from ADR 0003.
- **`LatentShardDataset`** is a map-style `Dataset` (random access, works with a shuffling
  `DataLoader`), not an iterable WebDataset-style stream — appropriate at this scale. It builds
  its shard/key index once at construction by listing tar members, and opens each shard's tar
  handle lazily so it is safe with multi-worker `DataLoader`s (`__getstate__` drops open handles
  before pickling to a worker process).
- **`scripts/precompute.py`** encodes the toy dataset through the configured (frozen) AE and text
  encoder under `no_grad`, writes shards + manifest, and takes the same config + dotted-override
  CLI shape as `train.py`/`sizes.py`.
- `configs/base.yaml` now points `autoencoder`/`text_encoder` at `toy`/`hash` (was the ADR
  0001 placeholder names `dcae`/`gemma`, never registered); `data.resolution` dropped from 512 to
  64 as a fast default for this toy pipeline.

## Not done / follow-ups
- **No real pretrained AE or text encoder.** This is the load-bearing caveat of this step: nothing
  trained on this cache would produce meaningful images. Getting a real DC-AE / T5 / Gemma
  encoder in requires either network access to a model hub or the user supplying local weights.
- **No caption source beyond the toy templates.** A real dataset needs only to yield
  `(image, caption)` pairs, so `precompute.py`'s dataset line is the only change needed — but nothing
  here does dataset cleaning, filtering, deduplication, or synthetic recaptioning (mentioned in
  the original plan) yet.
- **No resolution bucketing / aspect-ratio handling.** Every image is encoded at one fixed
  `data.resolution`; multi-resolution or aspect-ratio-bucketed training needs more here.
- **Sharding is local-disk only**, no cloud storage backend.
- **Speed is unmeasured** — the per-expert-style Python loop in `ShardWriter`/`LatentShardDataset`
  hasn't been profiled at real dataset scale (thousands+ of shards).
