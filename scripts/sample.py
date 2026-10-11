"""Samples from a trained checkpoint: caption(s) in, decoded images out (saved as .pt tensors —
this project has no image-file encoder; values are in [-1, 1], shape (3, H, W) per sample).

Usage: python scripts/sample.py <run_dir>/checkpoints/step_00000400 \
    configs/experiment/proxy_baseline.yaml \
    --caption "a red circle" --ae-weights /tmp/toy_ae.pt --out /tmp/sample.pt
"""

from __future__ import annotations

import argparse
import logging

import torch

from imggen.autoencoders import build_autoencoder
from imggen.backbones import build_backbone
from imggen.config import ComponentConfig, load_config
from imggen.data.shards import load_null_embedding
from imggen.samplers.euler import euler_sample
from imggen.text import build_text_encoder
from imggen.utils import setup_logging
from imggen.utils.checkpoint import load_checkpoint
from imggen.utils.ema import EMA
from imggen.utils.schedule import build_lr_lambda

log = logging.getLogger("imggen.sample")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", help="a run's checkpoints/step_XXXXXXXX directory")
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--caption", action="append", required=True, help="repeatable")
    ap.add_argument("--ae-weights", default=None, help="overrides autoencoder.args.weights_path")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-ema", action="store_true", default=True)
    ap.add_argument("--out", required=True, help="path to save decoded images (torch.save)")
    args = ap.parse_args(argv)

    setup_logging()
    overrides = list(args.overrides)
    if args.ae_weights:
        overrides.append(f"autoencoder.args.weights_path={args.ae_weights}")
    cfg = load_config(args.config, overrides)

    model = build_backbone(cfg.backbone)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, build_lr_lambda(1, cfg.train.steps + 1))
    ema = EMA(model)
    step = load_checkpoint(
        args.checkpoint, model=model, optimizer=opt, scheduler=scheduler, ema=ema
    )
    log.info("loaded checkpoint at step %d", step)
    if args.use_ema:
        ema.copy_to(model)
    model.eval()

    ae_cfg = cfg.autoencoder
    if args.ae_weights and "weights_path" not in ae_cfg.args:
        ae_cfg = ComponentConfig(
            name=ae_cfg.name, args={**ae_cfg.args, "weights_path": args.ae_weights}
        )
    ae = build_autoencoder(ae_cfg).eval()
    text_encoder = build_text_encoder(cfg.text_encoder).eval()

    text_emb, text_mask = text_encoder.encode(args.caption)
    b = len(args.caption)
    kwargs = {}
    if args.guidance_scale != 1.0:
        null_emb, null_mask = load_null_embedding(cfg.data.root)
        kwargs = {"null_emb": null_emb, "null_mask": null_mask}

    g = torch.Generator().manual_seed(args.seed)
    latent_shape = (b, cfg.backbone.in_channels, 8, 8)
    latent = euler_sample(
        model, latent_shape, text_emb, text_mask, steps=args.steps,
        guidance_scale=args.guidance_scale, generator=g, **kwargs,
    )  # fmt: skip
    images = ae.decode(latent)
    torch.save({"captions": args.caption, "images": images}, args.out)
    log.info("saved %d image(s) to %s", b, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
