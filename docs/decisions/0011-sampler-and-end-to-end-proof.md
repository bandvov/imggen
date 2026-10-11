# 0011: Sampler, classifier-free guidance, and the first real end-to-end proof

**Status:** accepted (post-roadmap — closing the loop before adding anything new)

## Why this, now
After step 10, every result in this project was "the loss goes down" on placeholder components.
Nothing had shown the model could generate the *right* thing — a wrong conditioning path, a sign
error in the velocity target, or swapped null/conditional embeddings would all still produce a
plausible-looking loss curve. This step closes that gap: a sampler, classifier-free guidance, a
briefly-trained (not just randomly-initialized) toy autoencoder, and an actual measurement of
whether a caption produces the right image.

## Decisions
- **`imggen.samplers.euler.euler_sample`** ("euler" in `SAMPLER`): fixed-step Euler integration of
  the predicted velocity from `t=0` (noise) to `t=1`, matching the training objective's straight-
  line path (ADR 0007). A higher-order solver (Heun, DPM-Solver) would need fewer steps for the
  same accuracy but isn't built — this is deliberately the simplest correct thing.
- **Classifier-free guidance**, both halves:
  - *Training*: `imggen.guidance.apply_caption_dropout` replaces a caption's embedding with a
    saved "null" (empty-string) embedding with probability `train.cfg_dropout`. **Default is 0.0**
    — not because CFG doesn't matter, but because turning it on by default would have required
    every test cache in the suite (most built directly via `ShardWriter`, not through
    `precompute.py`) to carry a null embedding, breaking ~280 passing tests for a capability
    those tests don't exercise. Opt in per-config once a real cache has one.
  - *Sampling*: `imggen.guidance.combine_predictions` (`uncond + scale * (cond - uncond)`).
  - The null embedding is precomputed once by `precompute.py` (empty-string encode) and saved
    (`imggen.data.shards.save_null_embedding`), not recomputed by loading the text encoder again
    during training or sampling — consistent with ADR 0006's "training reads only the cache"
    design.
- **The toy autoencoder gets briefly trained** (`scripts/train_toy_autoencoder.py`,
  `train_toy_autoencoder()`). It was *always* random noise before this — ADR 0006 explicitly
  deferred this, correctly, since a real AE needs weights this sandbox can't download. This is
  still not a real AE; it's the minimum needed for "does the pipeline produce the right image" to
  be checkable at all. Reconstruction MSE: 1.41 → 0.015 over 300 steps in the calibration run.
- **`build_autoencoder` gained a generic `weights_path` hook** (`cfg.args.weights_path`, popped
  before construction, loaded via `load_state_dict` after): any registered autoencoder can use
  pretrained weights this way, not just the toy one.
- **`scripts/sample.py`**: loads a checkpoint (EMA weights by default), builds the AE and text
  encoder, samples, decodes, saves image tensors. No image-file format — this project has no
  PNG/JPEG encoder, so output is a `.pt` of `[-1, 1]`-range `(3, H, W)` tensors.
- **The actual proof** (`tests/test_e2e.py`, `@pytest.mark.slow`): train the toy AE, train a tiny
  DiT with CFG dropout on 150 toy-shapes samples (250 steps, ~35s total), sample all 15
  color×shape captions with guidance scale 4, decode, and check the decoded image's center color
  is nearest the caption's color. **Calibration result: 12/15 correct** (chance is 3/15 for 5
  colors). The one systematic miss was yellow vs. green — they share two of three RGB channels,
  a model-capacity limit of this toy setup, not a pipeline bug. Test threshold set at 8/15,
  comfortable margin below the observed 12/15.
- Manually verified beyond the automated test, at a slightly larger scale (width 192, depth 4,
  400 steps): **13/15 correct**, and `scripts/sample.py`'s CLI path independently reproduced
  "a red circle" → red center, "a blue square" → blue center.

## Verification
All new code has unit tests (guidance: 9, sampler: 12, toy-AE trainer: 4, CFG-dropout wiring: 3,
null-embedding/weights_path: 3, CLI smoke tests for `sample.py` and `train_toy_autoencoder.py`: 2)
plus the end-to-end test above. Full suite: 314/314, `ruff check` clean. The default (cfg_dropout
disabled) path was confirmed unaffected by running the full pre-existing suite before adding any
new tests.

## Not done / follow-ups
- **Still no real autoencoder or text encoder** — this is the load-bearing caveat, unchanged from
  ADR 0006. Nothing here says anything about real image quality; the proof above is specifically
  that *the pipeline's mechanics are wired correctly*, on a problem simple enough to grade by eye
  (five flat colors, three shapes).
- **Euler only.** No adaptive step size, no higher-order solver.
- **No image-file export.** Output is raw tensors; viewing a result still needs a few lines of
  Python (`tensor.permute(1,2,0)` + a plotting call) or a follow-up PNG-writing script.
- **CFG dropout's effect wasn't ablated** — the end-to-end test uses guidance_scale=4 throughout;
  whether that's a good default, or how much CFG itself (versus just more training) drives the
  12/15 result, is untested.
- **GPU and a real AE/text-encoder are still the two things most worth doing next**, per the
  recommendation already on record — this step didn't change that, it just made sure the
  scaffolding they'll plug into actually works.
