# 0003: DiT block, attention interface, conditioning

**Status:** accepted (roadmap step 3)

## Decisions
- **Swappable attention core.** A core maps per-head `(q, k, v, key_mask) -> out` and is looked up in
  the `ATTENTION` registry (`softmax` = SDPA, `linear` = ReLU-kernel, O(N)). Projections, QK
  RMSNorm and the output projection live in a shared `Attention` wrapper, so every core has the
  same parameter count and checkpoints stay comparable. Per-layer choice via
  `backbone.attn` / `backbone.attn_pattern`.
- **Text conditioning by cross-attention** (always softmax; text is short), not joint attention.
  Text is projected once to `width`, so per-block K/V projections are `width -> width`.
- **Shared adaLN ("adaLN-single").** One `SiLU -> Linear(w, 6w)` on the timestep embedding is shared
  by all blocks; each block adds a learned `(6, w)` table. This avoids a `6w^2` projection per
  block (about a third of a per-block adaLN-Zero DiT's parameters at the proxy width). Trade-off:
  less per-layer freedom; a per-block variant is an ablation candidate.
- **Identity at init.** Shared modulation, cross-attention output and the head are zero-initialised,
  so the network outputs exactly 0 at init and every block starts as the identity.
- **Positions:** fixed 2D sin-cos, computed for any grid size. Pure-linear-attention models have
  no other positional signal until a conv/RoPE variant exists.
- **`init_weights()` is separate** so muP (step 5) can replace it (see ADR 0005: it now does,
  behind `cfg.mup.enabled`). Softmax scale is exposed on the core (`scale`) for muP's `1/d` hook,
  though ADR 0005 found it unneeded here since `head_dim` is fixed across the ladder.
- **`estimate_params` is exact**, and tests compare it with the instantiated module for every rung
  (on the `meta` device, so multi-billion rungs cost no memory).
- torch is now a core dependency; the config package still imports without touching torch.

## Not done yet / follow-ups
- Mix-FFN (depthwise conv) for linear attention; RoPE. Both fit the existing `ffn(x, hw)` and
  attention interfaces.
- `in_channels` / `text_dim` must match the autoencoder and text encoder: checked when those are
  built (step 6).
- Under bf16 autocast, QK RMSNorm runs with fp32 weights on bf16 activations (PyTorch warns it
  cannot use the fused kernel). Measure before fixing (step 7/10).
- Not benchmarked: no GPU here, so speed and memory of softmax vs linear are unmeasured.
