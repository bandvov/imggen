"""Precompute latents and text embeddings once, into the sharded cache ``build_dataloader`` reads.

Encodes the toy shapes dataset (see imggen.data.toy_dataset — a real dataset needs only to yield
(image, caption) pairs). The AE and text encoder are frozen, so this always runs under no_grad.

Usage: python scripts/precompute.py configs/experiment/proxy_baseline.yaml [key.path=value ...]
       python scripts/precompute.py configs/experiment/proxy_baseline.yaml \
           data.root=/tmp/cache train.batch_size=64  # any dotted override, incl. dataset size
       python scripts/precompute.py configs/experiment/proxy_baseline.yaml \
           data.root=/tmp/cache --num-samples 2000 --resume   # continue after an interruption
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from imggen.autoencoders import build_autoencoder
from imggen.config import dump_config, load_config
from imggen.data.shards import (
    MANIFEST_NAME,
    Manifest,
    ShardWriter,
    check_resume_meta,
    clear_resume_meta,
    find_resume_point,
    save_null_embedding,
    save_resume_meta,
)
from imggen.data.toy_dataset import ToyShapesDataset
from imggen.text import build_text_encoder
from imggen.utils import setup_logging

log = logging.getLogger("imggen.precompute")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. data.root=/tmp/cache")
    ap.add_argument("--num-samples", type=int, default=512, help="toy dataset size")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dataset-seed", type=int, default=0)
    ap.add_argument(
        "--resume", action="store_true",
        help="continue a cache an earlier, interrupted run of this command left in data.root, "
        "instead of starting over from scratch",
    )  # fmt: skip
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    setup_logging()
    log.info("config:\n%s", dump_config(cfg))

    # These must match the interrupted run's for a resume to be safe: num_samples/dataset_seed
    # fix *which* sample is at each index, and the AE/text-encoder config fixes what encoding
    # each cached latent/embedding actually is. batch_size and worker settings aren't included —
    # they don't change what gets written, only how fast.
    resume_fingerprint = dict(
        num_samples=args.num_samples, dataset_seed=args.dataset_seed,
        resolution=cfg.data.resolution, shard_size=cfg.data.shard_size,
        autoencoder={"name": cfg.autoencoder.name, "args": cfg.autoencoder.args},
        text_encoder={"name": cfg.text_encoder.name, "args": cfg.text_encoder.args},
    )  # fmt: skip

    if args.resume and Path(cfg.data.root, MANIFEST_NAME).exists():
        log.info("%s already has a complete manifest; nothing to resume", cfg.data.root)
        return 0

    start_shard_idx, start_num_samples = 0, 0
    if args.resume:
        check_resume_meta(cfg.data.root, **resume_fingerprint)
        start_shard_idx, start_num_samples = find_resume_point(cfg.data.root)
        if start_num_samples:
            log.info("resuming from sample %d/%d", start_num_samples, args.num_samples)

    save_resume_meta(cfg.data.root, **resume_fingerprint)

    ae = build_autoencoder(cfg.autoencoder).eval()
    text_encoder = build_text_encoder(cfg.text_encoder).eval()

    dataset = ToyShapesDataset(
        size=cfg.data.resolution, length=args.num_samples, seed=args.dataset_seed
    )
    remaining = Subset(dataset, range(start_num_samples, args.num_samples))
    loader = DataLoader(
        remaining, batch_size=args.batch_size, shuffle=False, num_workers=0
    )  # raw images: no benefit from workers here, and keeps the sandbox quiet

    writer = ShardWriter(
        cfg.data.root, shard_size=cfg.data.shard_size,
        start_shard_idx=start_shard_idx, start_num_samples=start_num_samples,
    )  # fmt: skip
    written = start_num_samples
    with torch.no_grad():
        for images, captions in loader:
            latents = ae.encode(images)
            text_emb, mask = text_encoder.encode(list(captions))
            for i in range(images.shape[0]):
                n = int(mask[i].sum())
                writer.add(f"{written:08d}", latents[i], text_emb[i, :n], mask[i, :n], captions[i])
                written += 1
    num_shards = writer.close()

    Manifest(
        num_samples=written,
        num_shards=num_shards,
        shard_size=cfg.data.shard_size,
        latent_channels=ae.latent_channels,
        spatial_compression=ae.spatial_compression,
        text_dim=text_encoder.text_dim,
        resolution=cfg.data.resolution,
        autoencoder={"name": cfg.autoencoder.name, "args": cfg.autoencoder.args},
        text_encoder={"name": cfg.text_encoder.name, "args": cfg.text_encoder.args},
    ).save(cfg.data.root)

    with torch.no_grad():
        null_emb, null_mask = text_encoder.encode([""])
    save_null_embedding(cfg.data.root, null_emb[0], null_mask[0])
    clear_resume_meta(cfg.data.root)

    log.info("wrote %d samples across %d shard(s) to %s", written, num_shards, cfg.data.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
