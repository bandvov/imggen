"""Briefly trains the toy autoencoder (imggen.autoencoders.toy.ToyAutoencoder) for pixel
reconstruction on the toy shapes dataset.

Every other script treats the autoencoder as frozen and pretrained (ADR 0006's placeholder
disclaimer), which is honest for a real AE but means ToyAutoencoder starts as random noise —
useless for actually looking at a decoded sample. This is still not a real AE (see ADR 0006);
it's the minimum needed for "does the pipeline produce the right image" to be checkable at all.

Usage: python scripts/train_toy_autoencoder.py --out /tmp/toy_ae.pt --steps 300
"""

from __future__ import annotations

import argparse
import logging

import torch
import torch.nn.functional as F

from imggen.autoencoders.toy import ToyAutoencoder
from imggen.data.toy_dataset import ToyShapesDataset
from imggen.utils import setup_logging

log = logging.getLogger("imggen.train_toy_autoencoder")


def train_toy_autoencoder(
    ae: ToyAutoencoder,
    *,
    steps: int = 300,
    batch_size: int = 16,
    lr: float = 3e-3,
    resolution: int = 64,
    dataset_size: int = 256,
    seed: int = 0,
) -> list[float]:
    ae.requires_grad_(
        True
    )  # ToyAutoencoder defaults to frozen; this is the one place that trains it
    dataset = ToyShapesDataset(size=resolution, length=dataset_size, seed=seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    opt = torch.optim.AdamW(ae.parameters(), lr=lr)

    def batches():
        while True:
            yield from loader

    data_iter = batches()
    losses = []
    ae.train()
    for _ in range(steps):
        images, _captions = next(data_iter)
        opt.zero_grad()
        recon = ae.decoder(ae.encoder(images))
        loss = F.mse_loss(recon, images)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    ae.eval()
    ae.requires_grad_(False)  # restore the frozen contract every other script relies on
    return losses


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="path to save the trained weights (state_dict)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--latent-channels", type=int, default=16)
    ap.add_argument("--num-downsamples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    setup_logging()
    torch.manual_seed(args.seed)
    ae = ToyAutoencoder(
        latent_channels=args.latent_channels, num_downsamples=args.num_downsamples, seed=args.seed
    )
    losses = train_toy_autoencoder(
        ae, steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        resolution=args.resolution, seed=args.seed,
    )  # fmt: skip
    log.info(
        "reconstruction MSE: first=%.4f last=%.4f (tail avg=%.4f)",
        losses[0], losses[-1], sum(losses[-10:]) / min(10, len(losses)),
    )  # fmt: skip
    torch.save(ae.state_dict(), args.out)
    log.info("saved trained toy autoencoder to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
