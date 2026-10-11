# 0004: Sparse MoE FFN

**Status:** accepted (roadmap step 4)

## Decisions
- **Same FFN interface.** `FFN.build(name, width=, hidden=, moe=)` and `ffn(x, hw)`. `ffn="moe"`
  selects `MoEFFN`; dense stays the default. `backbone.moe` (a `MoEConfig`) is ignored for dense.
- **Routing.** Softmax over all experts, top-k, renormalise the k probabilities. With k=1 the
  weight is exactly 1, so a 1-expert MoE equals a dense FFN (tested exactly, layer and model).
  Router math runs in fp32 even under bf16 autocast.
- **Dropless dispatch.** Assignments are sorted by expert and processed per expert; no capacity
  factor, no dropped tokens. Simple and correct; the per-expert loop needs a host sync on counts.
- **Defaults:** 8 experts, top-2, one always-on shared expert. `expert_hidden` defaults to
  `ffn_hidden // (shared + top_k)` rounded down to a multiple of 8, so active FFN compute is about
  equal to the dense FFN. This makes the step-10 "dense vs MoE at equal active FLOPs" ablation
  meaningful. Explicit `expert_hidden` overrides it.
- **Stacked expert parameters** (`fc1_weight` is `(E, hidden, width)`, and so on), so FSDP and expert
  parallelism (step 8) can shard along the expert dimension.
- **Balancing, both implemented:**
  - `aux_loss` (default): Switch-style loss, stored weighted in `aux_loss`; the trainer adds
    `collect_aux_loss(model)`. It acts through token-dependent router logits, so it cannot split
    tokens the router sees as identical.
  - `bias`: loss-free selection bias, updated by `step_router_bias(model)` once per optimizer step.
    Affects selection only, never the combine weights. Needs fp32 buffers and, with several ranks,
    an all-reduce of `load_accum` (step 8).
  - `none`.
- **Observability:** `moe_metrics(model)` returns per-layer `max_over_mean`, `min_over_mean` and
  load entropy for logging.
- **Estimator** returns `total` and `active` parameters (equal for dense), exact for both FFNs.

## Follow-ups
- The trainer (step 7) must call `collect_aux_loss` / `step_router_bias` and log `moe_metrics`.
- Speed is unmeasured. The loop-based dispatch will likely lose to a fused/grouped GEMM on GPU;
  measure in step 10 before optimising. `torch.compile` compatibility is untested.
- Expert parallelism and the all-reduce for bias balancing belong to step 8.
- Whether MoE helps at this scale (vs dense at equal active parameters) is an open question for
  the step-10 ablation, not something this step shows.
