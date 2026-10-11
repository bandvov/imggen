# 0008: Sharded training — FSDP2, tensor parallelism, DDP

**Status:** accepted (roadmap step 8)

## Decisions
- **`imggen/parallel.py`**, gated by `ParallelConfig` (`strategy: none|ddp|fsdp`, `tp_size`, new
  `ep_size`). `imggen.train.run_training` only calls `parallelize()` when
  `strategy != "none" or tp_size > 1`; the default path never imports or touches
  `torch.distributed`, so every test from steps 1-7 keeps passing unmodified (verified: full
  suite, 250/250, after every change in this step).
- **FSDP2** (`torch.distributed._composable.fsdp.fully_shard`), applied per `DiTBlock` then the
  whole model, per the original plan. Composable API: it mutates the module in place rather than
  wrapping it in a container, so parameter *names* stay identical to the unsharded model — this
  matters for muP (below).
- **Tensor parallelism** via DTensor (`torch.distributed.tensor.parallel.parallelize_module`,
  `ColwiseParallel`/`RowwiseParallel`) on every block's attention projections and dense FFN.
  `backbone.num_heads` divisible by `tp_size` was already checked at config-load time (ADR 0002);
  MoE's FFN is skipped by the TP plan (see the expert-parallelism plan below).
- **A real correctness bug, found and fixed, not just configured around:** `kv_proj`'s combined
  output packed as `(k/v, head, dim)`. A contiguous `ColwiseParallel` shard of that layout hands
  one rank all of *k* and another all of *v* — not a per-head split. Fixed by repacking to
  `(head, k/v, dim)` (`Attention.forward` in `imggen/layers/attention.py`), a pure relabelling of
  an otherwise arbitrarily-initialized weight with no effect on unsharded behavior (the full
  step 1-7 suite passed before and after). `Attention.forward` also now derives its head count
  from the actual projected width rather than a fixed `self.num_heads`, making it transparent to
  TP sharding. Found by writing the parity test below, not by inspection — the first version of
  that test failed with a 0.56 max-abs-diff, not a rounding-level mismatch.
- **Checkpointing upgraded to `get_state_dict`/`set_state_dict`**
  (`torch.distributed.checkpoint.state_dict`), replacing ADR 0007's plain-`torch.save` split for
  optimizer/EMA. Verified directly: a *never-stepped* optimizer's `get_state_dict` result already
  has real per-parameter entries, which is exactly what a fresh DCP `load()` needs to fill
  in-place — solving ADR 0007's "fresh optimizer has no state" problem more simply, and this same
  mechanism is what makes checkpoints resharding-safe under FSDP. Model, optimizer, and EMA all go
  through DCP now; the LR scheduler and step counter (small, non-parameter-shaped) stay on plain
  `torch.save`.
- **The actual promise, tested both directions, not assumed:** a checkpoint saved after training
  under 2-rank FSDP was loaded and resumed correctly in a single process (world_size 2 → 1), and
  the reverse (1 → 2, under FSDP) — both continue from the correct step and keep training. This is
  `test_checkpoint_from_two_rank_fsdp_resumes_single_process` and its mirror in
  `tests/test_parallel.py`.
- **Build order matters and is now enforced by ADR, not just comments:** `parallelize()` must run
  *before* the optimizer is built. FSDP2 and TP replace parameter *objects* (with DTensor-wrapped
  ones); an optimizer built from the pre-parallelize parameters would hold stale references (this
  broke on the first integration attempt — `get_state_dict` raised `KeyError` because the
  optimizer's parameters weren't the model's current ones). DDP is the opposite case: it preserves
  parameter identity but renames everything with a `module.` prefix on its own
  `named_parameters()`. `run_training` handles this by using `model.module` (DDP's own inner
  reference) as the source for muP's name-based role classification when the strategy is DDP, and
  the parallelized `model` itself otherwise (FSDP/TP keep clean names already).
- **DDP** (`torch.nn.parallel.DistributedDataParallel`) implemented and tested — it was an unused
  enum value in `ParallelConfig.strategy` since ADR 0001; this step is what makes it work.
- **Expert parallelism: planned, not implemented**, per the original roadmap's own phrasing
  ("plan expert parallelism... because MoE plus FSDP has known wrinkles"). `parallel.ep_size` exists
  and is validated (`backbone.moe.num_experts % ep_size == 0`, mirroring the `tp_size`/`num_heads`
  check), so configs can express the intent, but no sharding of the expert dimension is wired up.
  Real EP needs all-to-all token dispatch across ranks — `MoEFFN.forward`'s current per-expert loop
  (ADR 0004) assumes every expert is local, and rewriting that dispatch is a substantial piece of
  work in its own right, not a config flag. This is the wrinkle the roadmap called out by name.

## Verification method and its limits
Every distributed claim in this ADR was checked with real multi-process runs — gloo backend, CPU,
2 ranks — not just written against the API and assumed correct (the kv_proj bug is exactly what
that assumption would have missed). This sandbox has one CPU core and no GPU, so:
- **Speed is not measured or claimed anywhere in this step.** FSDP/TP on one core are strictly
  slower than single-process; that's expected and irrelevant to the correctness this step checks.
- **NCCL, multi-GPU, and anything larger than 2 ranks are untested.**
- **fork, not `torch.multiprocessing.spawn`,** was used for the test harness (`tests/test_parallel.py`)
  to keep the suite fast (~15s → ~1s per process start on this machine) — safe here since nothing
  touches CUDA; a real multi-GPU launch normally goes through `torchrun`, not either of these.
- **TP was not tested combined with linear attention or MoE.** The TP plan explicitly skips MoE
  FFNs; linear attention's core has no learned parameters of its own to shard beyond what's
  already in the (tested) `Attention` wrapper, but the combination hasn't been run.
- **2D parallelism (TP + FSDP together) was exercised through `parallelize()`'s mesh-building code
  path but not given its own dedicated correctness test** (only TP-alone and FSDP-alone each have
  one); the two are independent, well-established compositions in `torch.distributed`, but "each
  part is correct" isn't the same claim as "the composition is correct."

## Follow-ups
- Expert parallelism's all-to-all dispatch (above).
- 2D (TP+FSDP) parity test.
- GPU/NCCL validation whenever hardware is available.
- Tune, don't just enable — per the original plan, TP/FSDP only pay off at large width; nothing
  in this step tunes sharding strategy against actual throughput, since throughput can't be
  measured here.
