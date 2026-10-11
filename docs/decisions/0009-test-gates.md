# 0009: Test suite and gates

**Status:** accepted (roadmap step 9 — "nothing moves to step 10 until these pass")

## Decisions
- **`tests/test_gates.py`**, one file, `@pytest.mark.gate` on everything (`pytest -m gate` runs
  just this). It states the roadmap's five step-9 claims as explicit, individually-named tests:
  muP coordinate check, FSDP parity, softmax-vs-linear parity, dense-vs-1-expert-MoE parity,
  overfit, determinism. Two of these (softmax-vs-linear, dense-vs-MoE) already had thorough unit
  coverage from the steps that introduced them (ADR 0003, ADR 0004) — this file restates them as
  their own gate-level assertion rather than duplicating that coverage, with a comment pointing
  back to it.
- **The muP coordinate check is now rigorous**, not the step-5 diagnostic (which that ADR itself
  flagged as informal — one hook, one seed, at init). This version trains for 10 steps (where
  step 5 found the effect actually shows up) across widths [128, 256, 512, 1024], 3 seeds each,
  and compares the coefficient of variation of a post-training activation scale across widths.
  Calibration run (recorded in the test's own comment, not re-derived at test time): muP CV =
  0.144, standard CV = 1.855 — a 13x separation, not a borderline effect. Thresholds (0.3 and 1.0)
  sit with wide margin on both sides of the measured numbers.
- **The FSDP parity test is a gradient-equivalence proof on a fixed shared batch**, not a
  comparison of two independently launched training runs. Two independent runs would conflate the
  claim with unrelated sources of variance — data shuffling order and how much incidental RNG
  each code path consumes differ between single- and multi-process launches, discovered directly:
  an early version of this test compared full training-run losses and got a 19% relative
  difference, which turned out to be a shuffling/RNG mismatch, not an FSDP bug (confirmed by then
  fixing the comparison down to a single shared, fixed batch, split exactly the way
  `DistributedSampler` splits it — see the ``ep_size``/`DistributedSampler` fix below — which
  reproduced the single-process gradient to 8.9e-8, i.e. float32 precision). Comparing gradients
  on identical, hand-fixed data isolates exactly the claim that matters: that FSDP's sharded
  backward pass reconstructs the same gradient a single process would compute on the combined
  data. Real threshold used: 1e-5.
- **A real, separate correctness gap found and fixed while building the parity test**:
  `imggen.data.loader.build_dataloader` had no rank-aware data sharding. Under FSDP with more
  than one rank, every rank was silently training on the *same* full batch rather than a
  `1/world_size` slice of it — "data parallel" in name only. Fixed with `DistributedSampler`
  (shuffle handled by the sampler, not `DataLoader`'s own `shuffle=`); verified directly that two
  ranks get disjoint index sets whose union covers the whole dataset. `batch_size` in `TrainConfig`
  is now documented as *per-rank*: the effective global batch under `strategy=fsdp`/`ddp` is
  `batch_size * world_size`, the same convention every standard PyTorch DDP/FSDP training loop
  uses.
- **A second real bug found while building this test, in the test harness itself, not the
  library**: `tests/test_parallel.py::_run_distributed` passed results back from the forked
  worker to the test process through a `multiprocessing.Queue`. That works for the small
  dict/scalar results step 8's tests used, but a large payload of real tensors (the full gradient
  dict this test needs) hits a `resource_sharer`/file-descriptor edge case and fails
  reproducibly (`FileNotFoundError` on a stale socket) — not flaky, reproduced identically on
  retry. Fixed by handing results back through a temp file (`torch.save`/`torch.load`) instead of
  the Queue, with only a small status string going through the Queue. This is a fix to shared test
  infrastructure, so every existing `test_parallel.py` test benefits too — reran all 13 after the
  change, still pass.
- **Overfit test has a fixed numeric threshold** (tail-average loss < 0.25 over the last 20 of 600
  steps on a 4-sample fixed batch), replacing ADR 0007's descriptive smoke check. Calibrated
  against 5 seeds (range 0.056-0.155 at 600 steps) before picking the threshold, so it isn't
  tuned to pass by coincidence on the one seed the test actually uses.
- **Determinism test** trains a tiny model twice from the same seed through the real
  `precompute.py`-shaped cache and `run_training` path, and checks the final step and loss match
  — the operationally-relevant level (would a real reproducibility claim about this codebase hold
  up), not just the unit-level "same seed, same initial weights" already covered elsewhere.

## Verification
Every test in this file was run and passed, individually and as the full file together, before
being called done — this matters especially for the two multi-process ones (muP check: ~63s;
FSDP parity: ~3s once the harness bug above was fixed). Full suite after this step: 269/269,
~2 minutes total, `ruff check` clean.

## Not done / follow-ups
- **The muP coordinate check tests width transfer's *symptom* (activation flatness), not the
  actual promise** (a learning rate tuned at the proxy width stays *optimal* at larger widths).
  That needs a small LR sweep at multiple widths and comparing where the optimum falls — closer to
  what the original plan called the step-10 "transfer check." Done in ADR 0010 (with caveats).
- **Depth transfer remains untested**, consistent with ADR 0005 — muP's guarantee here is about
  width only.
- **The FSDP parity test covers world_size=2 only** and gloo/CPU, not real multi-GPU/NCCL (ADR
  0008's limits still apply).
- **`DistributedSampler`'s shuffle is seeded from a fixed `epoch=0`**, never incremented via
  `set_epoch()` — correct for the parity test (which wants determinism) and for short single-pass
  runs, but a long real training run cycling through the same dataset repeatedly (via
  `imggen.train.cycle`) will reshuffle identically every pass rather than getting a fresh shuffle
  per epoch. Worth fixing before a long real run; not fixed here since it doesn't affect the gate
  claims themselves.
