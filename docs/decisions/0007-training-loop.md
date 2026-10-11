# 0007: Single-process training loop

**Status:** accepted (roadmap step 7); the checkpoint-format decision below is superseded by ADR 0008

## Decisions
- **Objective:** `FlowMatchingObjective` (registered `"flow_matching"`), rectified-flow style:
  straight-line interpolation `x_t = (1-t) x_0 + t x_1` between noise and the (precomputed, cached)
  latent, MSE between predicted and true velocity `x_1 - x_0`. Two timestep distributions —
  `uniform` (default) and `logit_normal` (SD3-style) — which one helps is left to the step-10
  ablations, not decided here.
- **Training reads only the cache.** No autoencoder or text encoder is built or run during
  training; both were already applied once by `precompute.py` (ADR 0006). This keeps the loop
  simple and fast.
- **Checkpointing (superseded by ADR 0008 — kept here for history):** model weights went through
  `torch.distributed.checkpoint` (DCP); optimizer/scheduler/EMA/step went through plain
  `torch.save`/`torch.load`, because a freshly constructed optimizer has no per-parameter state
  until its first `step()`, and DCP's in-place load has nothing to fill for it. Step 8 found a
  simpler fix (`get_state_dict`/`set_state_dict`) that solves this for optimizer state too and
  moved it onto DCP alongside the model — see ADR 0008.
- **Resume is explicit, not automatic.** `create_run_dir` still makes a fresh timestamped
  directory every invocation (ADR 0001); `scripts/train.py --run-dir <path>` reuses an existing
  one instead and resumes its latest `checkpoints/step_*` if present. `run_training` checks
  `state.step >= cfg.train.steps` **before** building the dataloader, so re-running an
  already-finished config touches neither the AE/text-encoder-derived cache nor GPU time
  (verified: deleting the cache after a finished run doesn't break a second call with the same
  target step count).
- **EMA** (`imggen/utils/ema.py`) tracks only floating-point state (parameters and buffers);
  shadow weights swap in via a context manager (`ema.average_parameters(model)`) for eval/sampling
  and swap back out on exit. A standard warmup formula (`min(decay, (1+step)/(10+step))`) avoids
  freezing noisy early updates in at full strength.
- **LR schedule:** linear warmup then `constant` / `cosine` / `linear` decay to `min_lr_ratio`,
  expressed as a `LambdaLR` multiplier (`imggen/utils/schedule.py`). `TrainConfig` gained
  `adam_beta1/2`, `adam_eps`, `grad_clip_norm`, `lr_schedule`, `min_lr_ratio`, and now rejects
  `warmup_steps >= steps` at config-load time.
- **Parameter groups:** `build_param_groups` uses `MuP.param_groups` when `backbone.mup.enabled`
  (ADR 0005), otherwise the common split — no weight decay on any parameter with fewer than 2
  dimensions (biases, norm scales) — so muP and the default path share one code path in the
  training loop, differing only in which grouping function is called.
- **Precision:** `torch.autocast` wraps the forward+loss (not the backward pass) for `bf16`/`fp16`;
  `fp32` disables it. `fp16` on CPU has no working autocast support, so it logs a warning and
  falls back to `bf16` for that run rather than silently producing wrong numbers or crashing —
  this sandbox has no GPU, so `fp16` is untested on real hardware.
- **MoE integration:** the loss adds `collect_aux_loss(model)` every step; `step_router_bias` runs
  once per optimizer step (both from ADR 0004); `moe_metrics` is logged alongside the loss.
- **Verified the loop is actually correct**, not just that it runs: a fixed 4-8-sample batch with
  `lr=3e-3` drives the loss from ~1.0-1.2 down to the 0.03-0.6 range over 120-400 steps (both via
  `run_training` directly and via the full `precompute.py` → `scripts/train.py` CLI path). This is
  the "overfit test" the original plan put at roadmap step 9's test gate — done here too, early,
  because step 7 is exactly the code that test would be exercising, and it seemed better to know
  now than to defer discovering a broken loop until step 9 formalizes the gate.

## Not done / follow-ups
- **No sampler.** Nothing here turns a trained model back into an image — that needs an ODE
  solver (Euler at least) integrating the predicted velocity from t=0 to t=1, still unbuilt.
  Correctness so far rests on the loss going down, not on inspecting a generated sample.
- **fp16 is untested on real hardware** (no GPU in this sandbox); the CPU fallback path is tested,
  the GPU path is not.
- **No FSDP/TP** — that's step 8, and per above, optimizer-state checkpointing will need to move
  to DCP's format when it lands.
- **The "overfit test" above is a smoke check at tiny scale** (4-8 samples, sub-100M-equivalent
  proxy-ish widths), not the rigorous, gated version step 9 is meant to formalize with a fixed
  numeric threshold and CI enforcement.
