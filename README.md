# imggen

Fast, low-resource image generation. See `docs/decisions/` for why things are the way they are.

## Quickstart
```bash
pip install -e ".[dev]"
pip install -e ".[real-components]"   # optional: real SD/SDXL VAE + T5 (needs network to huggingface.co)
pytest
pytest -m gate          # just the roadmap step-9 gate (muP check, FSDP/attn/MoE parity, overfit, determinism)
python scripts/train.py configs/experiment/proxy_baseline.yaml backbone.size=b
python scripts/sizes.py [--ffn moe]      # print the size ladder (total and active params)
python scripts/mup_check.py              # coarse muP diagnostic across a width sweep
python scripts/sweep.py <config> data.root=/tmp/cache --only train.lr   # coordinate-wise hparam sweep
python scripts/transfer_check.py         # muP LR-transfer check across widths
python scripts/ablations.py <config> data.root=/tmp/cache   # attention / MoE ablations
python scripts/train_toy_autoencoder.py --out /tmp/toy_ae.pt   # train the toy AE (else: random noise)
python scripts/sample.py <run_dir>/checkpoints/step_N <config> \
    --caption "a red circle" --ae-weights /tmp/toy_ae.pt --guidance-scale 4 --out /tmp/sample.pt
python scripts/precompute.py configs/experiment/proxy_baseline.yaml \
    data.root=/tmp/cache --num-samples 512   # cache latents + text embeddings once
python scripts/train.py configs/experiment/proxy_baseline.yaml data.root=/tmp/cache
python scripts/train.py configs/experiment/proxy_baseline.yaml data.root=/tmp/cache \
    --run-dir runs/proxy_baseline/<ts>_<hash> train.steps=2000   # resume, more steps
# Sharded training (needs a distributed launcher):
torchrun --standalone --nproc-per-node=N scripts/train.py configs/experiment/proxy_baseline.yaml \
    data.root=/tmp/cache parallel.strategy=fsdp parallel.tp_size=2
python scripts/benchmark.py --size xl --batch-size 8   # latency/throughput/VRAM
# Real pretrained components instead of the toy/hash placeholders (needs network + the extra above):
python scripts/precompute.py configs/experiment/proxy_real_components.yaml data.root=/tmp/cache
python scripts/train.py configs/experiment/proxy_real_components.yaml data.root=/tmp/cache
torchrun --standalone --nproc-per-node=N scripts/benchmark.py --size xl --parallel-strategy fsdp
python scripts/train.py configs/experiment/proxy_baseline.yaml --print-config
```

## Layout
```
configs/        YAML; experiments compose base.yaml + backbone/*.yaml via `base:`
src/imggen/
  config/       typed schema, loader, size ladder (sizes.py), estimates (scaling.py)
  registry.py   per-kind registries + discover()
  layers/       attention cores (softmax, linear), ffn (dense, moe)   [registered]
  backbones/    dit (registered); build_backbone(cfg)
  mup.py        muP: parameter roles, init, LR groups (backbone.mup.enabled)
  autoencoders/ toy (untrained placeholder) + sd_vae (real, diffusers AutoencoderKL) [registered]
  text/         hash (untrained placeholder) + t5 (real, transformers T5EncoderModel) [registered]
  data/         toy_shapes dataset, tar-shard cache (shards.py), build_dataloader(cfg, per-rank under FSDP/DDP)
  objectives/   flow_matching (registered); build_objective(cfg)
  train.py      run_training(cfg, run_dir): the training loop itself
  sweep.py      run_sweep_point(): short in-memory training runs for sweeps/ablations
  samplers/     euler (registered); euler_sample() integrates noise -> latent
  (scripts/benchmark.py: latency/throughput/VRAM, single-process or under torchrun)
  guidance.py   classifier-free guidance: caption dropout (training), combine_predictions (sampling)
  utils/        ...also ema.py, schedule.py, checkpoint.py (DCP via get_state_dict/set_state_dict)
  parallel.py   FSDP2, tensor parallelism (DTensor), DDP — parallel.strategy / tp_size
  objectives/ autoencoders/ text/ samplers/ data/ eval/
  utils/        seed, rank-aware logging, run directories
scripts/        train.py (loop arrives in roadmap step 7)
tests/          pytest; markers: slow, gpu, dist
docs/decisions/ ADR-style decision log
```

## Adding a component
```python
# src/imggen/layers/softmax_attention.py
from imggen.registry import ATTENTION


@ATTENTION.register("softmax")
class SoftmaxAttention: ...
```
Then select it in YAML (`backbone.attn: softmax`). No other file changes.
