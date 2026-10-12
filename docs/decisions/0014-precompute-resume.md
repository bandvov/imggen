# 0014: `precompute.py --resume`

**Status:** accepted. **Written but not executed in this session** — see "Verification status"
below before trusting this ADR the way earlier ones can be trusted.

## Why
A user hit this directly: `precompute.py` has no resume support, so killing it partway (a
Windows path mistake, a network hiccup downloading real HF weights, anything) meant starting
over from sample 0. Re-running the same command *is* safe (confirmed earlier: `ShardWriter`
always begins at `shard-00000.tar` and `tarfile.open(path, "w")` truncates, so the cache ends up
correct, just with earlier progress redone) — but for a large real run with pretrained
components, redoing finished work is exactly the cost `--resume` removes.

## Design
- **`imggen.data.shards.validate_shard(path) -> int | None`**: opens a shard, groups members by
  key, requires each key have exactly the `{latent.pth, text_emb.pth, json}` triplet, and
  actually deserializes every tensor and parses every json (not just checks the tar lists
  members or the byte count looks right — a truncated tar can still list members, and a
  truncated pickle can still be the "right" size short of loading). Returns the sample count if
  everything checks out, else `None`.
- **`find_resume_point(root) -> (next_shard_idx, samples_already_done)`**: trusts every shard
  except the last (only the last can be mid-write — every earlier one was already closed before
  the writer moved to the next). Validates the last one; if it doesn't check out, discards it
  *entirely* as one file and resumes from its index, rather than trying to salvage a prefix of
  it. Worst case this redoes up to one shard's worth of samples (`shard_size`, default 256) — a
  deliberate simplicity-over-completeness tradeoff. Raises if a **non-last** shard fails
  validation — that should never happen from a normal interruption, and resuming past it would
  silently paper over real corruption.
- **`ShardWriter` gained `start_shard_idx`/`start_num_samples`** so it can continue numbering
  instead of always starting at shard 0, sample 0.
- **A progress sidecar** (`.precompute_progress.json`, `save_resume_meta`/`load_resume_meta`/
  `check_resume_meta`/`clear_resume_meta`) records `num_samples`, `dataset_seed`, `resolution`,
  `shard_size`, and the AE/text-encoder config — everything that determines *what* sample index
  N actually is, or what encoding is in the cache. `batch_size` and worker counts are
  deliberately excluded (they don't change what gets written, only how fast) so an unrelated
  tuning edit between runs doesn't block a legitimate resume. Written early (before the encode
  loop starts, every run, resumed or not — so a run that itself gets interrupted still leaves
  something to check against next time) and cleared only on full success.
- **`scripts/precompute.py --resume`**: skips entirely (no AE/text-encoder load) if a complete
  `manifest.json` already exists — resuming an already-finished cache is a no-op, not a redo.
  Otherwise checks the sidecar (raises `ResumeMismatch` with a clear message on any mismatch),
  finds the resume point, and only encodes `Subset(dataset, range(start, num_samples))` — the
  deterministic toy dataset means index N is the same sample across runs given the same seed, so
  skipping the already-done prefix and continuing with sequential keys is correct and doesn't
  need to touch finished work.
- Without `--resume` (the default): byte-identical to before this change. `--resume` is
  opt-in, not an always-on behavior change.

## Verification status — read this before trusting the rest of this ADR
**The sandbox this project has been built in was reset mid-session**, wiping the Python
environment (torch, the installed package, everything) while this feature was half-implemented.
Per the user's explicit instruction afterward ("do not install anything... just make code
changes"), **none of this has been executed** — not the unit tests, not the CLI tests, not even
a basic import. What *has* been done:
- Every changed file passes `python3 -m py_compile` (syntax-valid).
- `ast.parse` confirms every new name referenced by the tests is actually defined at module
  level in `imggen/data/shards.py`.
- Every test's expected shard-count/sample-count arithmetic was hand-traced against
  `ShardWriter`'s actual chunking logic (e.g. `_write_shards(n=7, shard_size=3)` → shards of
  sizes `[3, 3, 1]`, confirmed against the pre-existing, previously-passing
  `test_shard_writer_splits_into_expected_shard_count`) and against `configs/base.yaml`'s actual
  default values (for the mismatch-detection test's hand-written sidecar).
- All new/changed lines checked under the 100-char limit by hand (`ruff` isn't installed either).

This is a meaningfully weaker verification bar than every other ADR in this project, which was
always "run the tests, show the numbers." **The tests exist and are believed correct by careful
reading, not confirmed by execution.** Run `pytest tests/test_data.py tests/test_cli.py -k
resume` (and then the full suite) before trusting this feature, and report back anything that
fails — the most likely failure modes, if any, are an off-by-one in shard/sample counting or a
detail of `tarfile`/`torch.load` behavior that reads correctly but doesn't execute as assumed.

## Not done / follow-ups
- No resume support for anything other than `precompute.py` (training already resumes via
  checkpoints, ADR 0007/0008 — this was specifically the gap in the data-caching stage).
- `find_resume_point`'s "discard the whole last shard" is simple but wasteful for a very large
  `shard_size` on a slow (e.g. real pretrained component) precompute — a smaller `shard_size`
  bounds the worst case at the cost of more, smaller tar files.
- No progress bar / ETA — `--resume` tells you where it's continuing from
  (`log.info("resuming from sample %d/%d", ...)`) but not how long the rest will take.
