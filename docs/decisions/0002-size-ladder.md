# 0002: Backbone size ladder

**Status:** accepted (roadmap step 2)

## Decision
- A backbone shape is `(width, depth)`; heads, FFN hidden size and parameter estimates are derived.
- `head_dim` is fixed at 64, so width grows by adding heads (muP expects this).
- Named rungs live in `config/sizes.py` (`nano`, `proxy`, `s`, `b`, `l`, `xl`, `2xl`, `3xl`, `4xl`).
  Width grows faster than depth. Any other size is just explicit `width`/`depth`; no code change.
- Config: `backbone.size=b`, or explicit `width`/`depth`. Explicit values win, and the `size`
  label is cleared when the result no longer equals the rung, so resolved configs never mislabel.
  Shipped YAML sets `size` only, so `backbone.size=b` on the CLI swaps the whole rung.
- `BackboneConfig.replace()` and `width_sweep()` make validated variants for the muP coordinate
  check (step 9) and transfer check (step 10).
- `parallel.tp_size` must divide the head count (checked at config load).
- Parameter counts come from `estimate_params`, exact for the DiT of step 3 (see ADR 0003).

## Consequences / follow-ups
- (Done in step 3) `estimate_params` is tested against the instantiated model for every rung.
- Step 4 (MoE) must extend the estimate to total vs active parameters.
- Depth transfer under muP is not assumed: step 5/10 verify it separately.
- Very large rungs are placeholders on paper. Real head counts matter for TP: `2xl` has 24 heads
  (TP up to 8), `xl` has 20 (TP up to 4). `scripts/sizes.py` prints the max TP per rung.
