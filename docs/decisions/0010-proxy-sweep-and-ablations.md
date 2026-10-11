# 0010: Proxy sweep, muP transfer check, first ablations

**Status:** accepted (roadmap step 10). Results are real measurements from this sandbox, but
small-scale and CPU-only — read "Limits" before drawing conclusions from any number below.

## Setup (what "proxy" means here)
- **Not the plan's ~50M proxy.** CPU-only, one core: the sweep/ablation model is width 192, depth 4,
  head_dim 32 — **2.75M parameters**. The transfer check goes to width 768 (~4x). Everything below
  is a scaled-down stand-in that exercises the real code paths (`imggen.sweep` reuses the training
  loop's building blocks), not a tuned result for the real proxy.
- Data: the real precomputed toy cache from step 6 (128 samples, untrained toy AE and hash text
  encoder — ADR 0006's caveat still applies: nothing here says anything about image quality).
- Raw numbers: `docs/results/step10_sweep.json`, `step10_transfer_check.json`,
  `step10_ablations.json`. Scripts: `scripts/sweep.py`, `transfer_check.py`, `ablations.py`
  (each has a CLI smoke test).

## 1. Hyperparameter sweep (width 192, 50 steps, one seed, coordinate-wise)
One knob at a time, later chunks inheriting the best value found so far (lr → 3e-3; warmup → 5;
batch size → 8). Metric: mean of the last 10 losses.

| knob | values → tail loss |
|---|---|
| lr | 3e-4 → 0.534, 1e-3 → 0.243, **3e-3 → 0.194**, 1e-2 → 0.359 |
| weight_decay (lr 3e-3) | 0 → 0.19367, 0.01 → 0.19365, 0.1 → 0.19353 |
| warmup_steps | 1 → 0.203, **5 → 0.197**, 20 → 0.210 |
| batch_size | 4 → 0.312, 8 → 0.223, 16 → 0.227 |
| t_dist | uniform → 0.223, **logit_normal → 0.147** |

Reading it honestly:
- **lr** is the only knob with a clear signal (a proper U-shape, optimum ~3e-3 on this grid).
- **weight decay does nothing measurable** at 50 steps on a 2.75M model — differences are in the
  4th decimal. That's a statement about this short/small regime, not evidence weight decay is
  useless at scale.
- **warmup**: small differences (~3-7%), one seed; treat as "5 is fine", not "5 is optimal".
- **batch size is confounded**: the cache has 128 samples and every point runs a fixed 50 steps,
  so a larger batch also sees more distinct data. The table can't separate the two effects.
- **logit_normal vs uniform** (0.147 vs 0.223) is the most interesting hint, and the least
  established: one seed, 50 steps. Needs multiple seeds before it changes any default.
- **"Init scale" was not swept as its own knob.** muP fixes init variance by formula (ADR 0005),
  so there is nothing free to sweep; the muP-on/off comparison in section 2 is the practical
  stand-in. This substitutes for the plan's item rather than fulfilling it literally.
- No default in `configs/` was changed on the strength of these numbers.

## 2. muP transfer check (the deferred promise from ADRs 0005 and 0009)
Base width 192; widths 192/384/768; LR grid {3e-4, 1e-3, 3e-3, 1e-2}; 2 seeds; fixed synthetic
batches (pure optimization dynamics). Mean tail loss:

| width | muP: best lr (loss) | standard: best lr (loss) |
|---|---|---|
| 192 (base) | 1e-3 (0.862) | 1e-3 (0.843) |
| 384 | **3e-3** (0.833) | 1e-3 (0.931) |
| 768 | **3e-3** (0.855) | 1e-3 (1.031) |

At width 768 specifically, by LR: muP 3e-4 → 1.18, 1e-3 → 1.00, 3e-3 → **0.855**, 1e-2 → 1.40;
standard 3e-4 → 1.09, 1e-3 → **1.03**, 3e-3 → 1.31, 1e-2 → **8.94** (diverging).

What the data supports:
- **Under muP the same LR (3e-3) is best at both 384 and 768**, and the loss at the optimum is about
  the same at all three widths (0.862 / 0.833 / 0.855).
- **Under standard parametrization the good range shrinks leftward with width.** At 192, lr 1e-3 and
  3e-3 are nearly tied (0.843 vs 0.849); at 384, 3e-3 is 0.14 worse than 1e-3; at 768 it is 0.27
  worse, and 1e-2 blows up (8.94 vs muP's 1.40 at the same LR). At every width above base, muP at
  its best LR beats standard at its best.
- **What it does *not* show:** the standard argmin did *not* move on this coarse (half-decade) grid —
  it stayed at 1e-3 throughout, so "the optimal LR drifts without muP" is not what was measured;
  the *upper edge* of the good range moved. And muP's own argmin moved one grid step between 192 and
  384 (1e-3 → 3e-3), but at 192 those two LRs are a tie within noise (0.862 vs 0.865), so I read it
  as a flat optimum, not drift — a finer grid would settle this.
- At the base width the LR multipliers are exactly 1, but muP and standard still differ slightly
  (0.862 vs 0.843) because muP's init (normal, 1/sqrt(fan_in)) is not Xavier uniform.

**Confounds to keep in mind:** step counts differ by width (40 at 192, 30 at 384, 15 at 768) to fit
the per-command time budget, so *within-width* muP-vs-standard comparisons are clean but
*across-width* loss comparisons within one condition are not (fewer steps at 768 flatters neither
side equally). Two seeds, 4-point grid, depth fixed at 4 (depth transfer still untested, ADR 0005).
Verdict: consistent with muP working as intended and clearly better-behaved than standard; not a
precise measurement of transfer.

## 3. First ablations (width 192, depth 4, 80 steps, lr 3e-3, batch 8, one seed)
| variant | tail loss | final loss | CPU s/step | params (total / active) |
|---|---|---|---|---|
| softmax attention (= dense FFN row) | 0.2260 | 0.2105 | 0.125 | 2.75M / 2.75M |
| linear attention | 0.2232 | 0.2093 | 0.126 | 2.75M / 2.75M |
| hybrid [linear, linear, softmax, linear] | 0.2249 | 0.2092 | 0.124 | 2.75M / 2.75M |
| MoE (8 experts, top-2, 1 shared) | 0.2174 | 0.2122 | 0.173 | 5.13M / 2.76M |

- **Attention kind: no detectable difference** (tail loss within 1.3%, one seed). Expected here —
  the latent grid is 8x8 = 64 tokens, far too short for linear attention's O(N) advantage to show
  in loss or CPU time. This is a null result at a scale that can't test the hypothesis, not evidence
  the variants are equivalent at 1024px.
- **MoE vs dense at equal active parameters** (2.76M vs 2.75M — ADR 0004's sizing rule holds):
  tail loss 4% lower for MoE, final loss 1% higher. Mixed, one seed, not significant. The cost is
  clear though: **~40% slower per step on CPU** (ADR 0004's unoptimized per-expert loop) and 1.9x the
  total parameters (memory).
- The dense and softmax rows are the same configuration (identical numbers, as expected).

## Limits (what this step did *not* deliver)
- **No latency, VRAM or throughput numbers** — the plan asked for them; there is no GPU here. CPU
  seconds/step is reported but is not a proxy for GPU behavior (different bottlenecks).
- Single seed everywhere except the transfer check's two; no confidence intervals.
- ~2.75M-parameter model, 15-80 steps per run, 128-sample toy data: these are directional signals
  about mechanics, not tuning results for the real proxy.
- Depth transfer, TP/FSDP interaction with muP, and expert parallelism remain untested.

## Incidental bug found and fixed
`imggen.train.cycle()` spun forever on an empty loader — reachable whenever `batch_size` exceeds the
dataset size with `drop_last=True` (zero batches). Found when a CLI smoke test with a 4-sample cache
and the default `batch_size=32` never returned. It now raises `ValueError("data loader produced no
batches...")`; regression tests cover `cycle([])` and `run_training` end to end. This affected
`run_training` since step 7, not just the new scripts.

## Follow-ups
- Repeat the sweep's two suggestive signals (logit_normal, warmup) with several seeds.
- Finer LR grid and equal step counts across widths for a cleaner transfer measurement.
- Measure attention/MoE speed on a GPU at realistic token counts before deciding anything.
