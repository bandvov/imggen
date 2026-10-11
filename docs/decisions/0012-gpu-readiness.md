# 0012: GPU readiness — torchrun, benchmarking, and two real bugs found by actually launching it

**Status:** accepted. This sandbox still has no GPU; everything numeric below is CPU-only and
labeled as such. The point of this step was making the untested path actually work, not measuring
GPU performance that can't be measured here.

## Why this, now
Steps 8 and 9 built and correctness-tested FSDP/TP/DDP using a hand-rolled fork-based multi-process
harness (`tests/test_parallel.py::_run_distributed`) that manually calls
`dist.init_process_group`. That harness is not how anyone would actually launch this project — real
multi-GPU training goes through `torchrun`. Nobody had tried that until this step, and it did not
work.

## Bug 1: nothing ever called `init_distributed()`
`imggen.parallel.init_distributed()` and `is_distributed_run()` existed since ADR 0008 but were
never called from `scripts/train.py` or anywhere else reachable from a real launch. Confirmed
directly: `torch.distributed.get_world_size()` with no process group raises
`ValueError: Default process group has not been initialized, please make sure to call
init_process_group.` — exactly the crash a real `torchrun --nproc-per-node=2 scripts/train.py
... parallel.strategy=fsdp` launch hit, reproduced before fixing anything.

Fix, in `scripts/train.py`:
- Call `init_distributed()` when `is_distributed_run()`, before anything touches
  `torch.distributed`.
- `init_distributed()` itself gained backend auto-selection (`nccl` when CUDA is available, `gloo`
  otherwise — it previously hardcoded `gloo`, which is correct for this sandbox's tests but wrong,
  slow collectives on real GPU hardware) and now calls `torch.cuda.set_device(LOCAL_RANK)` on CUDA,
  without which every process on a multi-GPU node would default to device 0.
- Only rank 0 picks or creates the run directory; the path is broadcast
  (`dist.broadcast_object_list`) to every other rank. Without this, every rank would
  independently call `create_run_dir()` and either land on different directories (breaking
  checkpointing, which is a collective every rank must call together) or race to create the same
  one and crash on `FileExistsError`.
- Verified against the real `torchrun` binary (available in this sandbox), not a simulation:
  `torchrun --standalone --nproc-per-node=2 scripts/train.py ... parallel.strategy=fsdp` now runs,
  trains, and checkpoints correctly. Regression test:
  `tests/test_cli.py::test_train_script_under_real_torchrun_with_fsdp`.

## Bug 2 (found while fixing bug 1): a genuine FSDP2 correctness warning
The real `torchrun` launch printed:
> FSDP2-wrapped module (FSDPDiT) returned a view tensor... An in-place op on this view will
> silently drop the pre-backward hook and skip the all-gather, which can cause backward to fail or
> produce wrong gradients.

This is not boilerplate noise — I had been filtering stderr with a pattern that happened to include
`register_pre_backward` in several step-8/9 command invocations, which hid this warning from view
without my understanding what it meant. Only seeing the real, unfiltered `torchrun` output surfaced
it.

Root cause, confirmed directly: `imggen.backbones.dit.unpatchify`'s `permute` + `reshape` is
reshape-compatible without a copy specifically when `patch_size=1` (this project's default) —
verified `out.is_contiguous()` is `False` and `out` shares storage with the input at `p=1`, `True`
and no sharing at `p=2`. Fix: `.contiguous()` on the final return.

How I know this didn't silently corrupt gradients, before or after the fix:
- `tests/test_gates.py::test_fsdp_gradient_matches_single_process_on_the_combined_batch` already
  used the default `patch_size=1` and already matched the single-process reference to 8.9e-8 —
  both before this fix (it was passing all along) and after (re-ran it, still passes). Empirically,
  autograd's own view-backward machinery handled this correctly regardless of FSDP's hook timing
  concern; nothing downstream of the model's output does an in-place op that would have exploited
  the gap FSDP was warning about.
- Fixed the root cause anyway rather than relying on that holding — a new regression test,
  `tests/test_parallel.py::test_fsdp_forward_backward_raises_no_view_tensor_warning`, asserts no
  such warning fires; confirmed it actually catches the bug by temporarily reverting the fix and
  watching the test fail, then restoring it.

## Bug 3 (found while building the benchmark script): a real hang under torchrun
`scripts/benchmark.py`, once written, hung indefinitely under a real `torchrun` launch with no
error — `timeout` was the only reason it didn't run forever. Cause: the script initialized a
process group (via `init_distributed()`) but never called `dist.destroy_process_group()` before
exiting; torchrun's elastic agent waits for clean teardown. Fixed with an explicit
`destroy_process_group()` at the end of `main()` when a process group was initialized. Verified by
timing the launch before (killed by a 30s timeout, no output after the one result line) and after
(exits in ~8s). Regression test:
`tests/test_benchmark.py::test_benchmark_under_real_torchrun_with_fsdp_exits_cleanly`.

## `scripts/benchmark.py`
Measures median per-step latency (forward+backward, `torch.cuda.synchronize()` around each
timed step so numbers reflect device completion, not kernel-launch return), throughput
(`batch_size * world_size / latency`), and peak VRAM (`torch.cuda.max_memory_allocated()`, null
on CPU — never silently presented as a GPU number). Works single-process or under `torchrun` with
`--parallel-strategy fsdp|ddp --tp-size N`. A design bug caught during testing, before it shipped:
the first version built a parallelized model only to validate the sharding plan, then discarded it
and benchmarked a fresh *unsharded* model instead — defeating the entire point of a distributed
benchmark. Fixed by threading the actual (possibly parallelized) model through to the timed
section; `tests/test_benchmark.py::test_benchmark_config_accepts_a_prebuilt_model` checks the same
object is what gets benchmarked, not a rebuilt one.

Launch recipes:
```
# Single process (this sandbox, or a single GPU)
python scripts/benchmark.py --size xl --batch-size 8 --out docs/results/bench_xl.json

# Multi-GPU, once hardware exists
torchrun --standalone --nproc-per-node=N scripts/train.py configs/experiment/proxy_baseline.yaml \
    data.root=/path/to/cache parallel.strategy=fsdp parallel.tp_size=2
torchrun --standalone --nproc-per-node=N scripts/benchmark.py --size xl \
    --parallel-strategy fsdp --tp-size 2
```

## CPU numbers gathered here (not a substitute for GPU numbers — see above)
`docs/results/step12_benchmarks.json`. Width 256, depth 6, batch 8, 256 latent tokens (16x16),
median of 15 steps after 3 warmup:

| variant | latency (s/step) | throughput (samples/s) |
|---|---|---|
| softmax attention | 0.961 | 8.32 |
| linear attention | 0.884 | 9.05 |
| dense FFN | 0.948 | 8.44 |
| MoE FFN (8 experts, top-2, equal active params) | 1.053 | 7.60 |

Linear attention is ~8% faster than softmax at 256 tokens on CPU — directionally consistent with
its O(N) complexity, but 256 tokens is still short and this is one CPU core, not evidence about
GPU behavior at the resolutions (1024+ tokens) where the difference is expected to matter. MoE is
~11% slower per step than dense at equal *active* parameters, consistent with ADR 0004's known,
unoptimized per-expert Python dispatch loop — expected to matter less on GPU with a fused/grouped
kernel, which doesn't exist here to test.

## Not done / follow-ups
- No real GPU numbers exist anywhere in this project. That remains the single biggest gap for
  actually answering the speed/resource questions the original plan asked.
- `peak_vram_mb` is implemented but entirely unverified — there's no CUDA device to confirm
  `torch.cuda.max_memory_allocated()` reports what it's supposed to in this codebase's actual
  usage pattern.
- Multi-node (as opposed to multi-process-on-one-node `--standalone`) launches are untested.
- The benchmark script measures training-step latency only, not sampling/inference latency
  (`imggen.samplers.euler_sample`, ADR 0011) — a different, also-unmeasured number.
