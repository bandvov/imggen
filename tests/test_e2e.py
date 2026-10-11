"""The actual proof this pipeline works end to end, not just that losses go down: train the toy
autoencoder to reconstruct, train a tiny DiT with classifier-free guidance on the toy shapes
dataset, sample from text, decode, and check that the decoded image's dominant color matches
what the caption asked for — something a falling training loss cannot by itself demonstrate (a
wrong conditioning path, a sign error in the velocity target, or swapped null/conditional
embeddings would all still show a plausible-looking loss curve).

Chance accuracy with 5 colors is 3/15 captions. A calibration run (recorded here, not re-derived
by the test) got 12/15 in ~35s; the threshold below (8/15) leaves comfortable margin for a
different machine or a different single seed while still being a real correctness bar, not a
rubber stamp. The one systematic miss in that run was yellow vs. green — they share two of three
RGB channels — which is a model-capacity/scale limit of this toy setup, not a pipeline bug.
"""

from __future__ import annotations

import pytest
import torch

from imggen.autoencoders.toy import ToyAutoencoder
from imggen.config import load_config
from imggen.data.shards import ShardWriter, save_null_embedding
from imggen.data.toy_dataset import ToyShapesDataset
from imggen.text.hash_embedding import HashEmbeddingTextEncoder
from imggen.train import run_training

pytestmark = pytest.mark.slow

COLORS = {
    "red": (1.0, -1.0, -1.0), "green": (-1.0, 1.0, -1.0), "blue": (-1.0, -1.0, 1.0),
    "yellow": (1.0, 1.0, -1.0), "white": (1.0, 1.0, 1.0),
}  # fmt: skip
SHAPES = ("circle", "square", "triangle")


def _build_cache(
    tmp_path, ae: ToyAutoencoder, text_encoder: HashEmbeddingTextEncoder, n: int, seed: int
):
    dataset = ToyShapesDataset(size=32, length=n, seed=seed)
    root = tmp_path / "cache"
    writer = ShardWriter(root, shard_size=n)
    with torch.no_grad():
        for i in range(n):
            img, caption = dataset[i]
            latent = ae.encode(img.unsqueeze(0))[0]
            emb, mask = text_encoder.encode([caption])
            k = int(mask[0].sum())
            writer.add(f"{i:04d}", latent, emb[0, :k], mask[0, :k], caption)
    writer.close()
    from imggen.data.shards import Manifest

    Manifest(
        num_samples=n, num_shards=1, shard_size=n, latent_channels=ae.latent_channels,
        spatial_compression=ae.spatial_compression, text_dim=text_encoder.text_dim, resolution=32,
    ).save(root)  # fmt: skip
    with torch.no_grad():
        null_emb, null_mask = text_encoder.encode([""])
    save_null_embedding(root, null_emb[0], null_mask[0])
    return root


def test_captions_produce_the_right_color_end_to_end(tmp_path):
    from scripts.train_toy_autoencoder import train_toy_autoencoder

    torch.manual_seed(0)
    ae = ToyAutoencoder(latent_channels=8, num_downsamples=2, seed=0)
    ae_losses = train_toy_autoencoder(
        ae, steps=150, batch_size=8, resolution=32, dataset_size=64, seed=0
    )
    assert sum(ae_losses[-10:]) / 10 < ae_losses[0] * 0.3  # reconstruction actually improved

    text_encoder = HashEmbeddingTextEncoder(text_dim=48, seed=0)
    cache_root = _build_cache(tmp_path, ae, text_encoder, n=150, seed=1)

    cfg = load_config(
        "configs/experiment/proxy_baseline.yaml",
        [
            f"data.root={cache_root}", "data.num_workers=0",
            "backbone.width=96", "backbone.depth=3", "backbone.head_dim=16",
            "backbone.in_channels=8", "backbone.text_dim=48", "backbone.mlp_ratio=4.0",
            "train.steps=250", "train.batch_size=16", "train.lr=3e-3", "train.warmup_steps=5",
            "train.cfg_dropout=0.15", "train.log_every=1000", "train.ckpt_every=250",
            f"run.output_dir={tmp_path / 'runs'}",
        ],
    )  # fmt: skip
    state = run_training(cfg, tmp_path / "run")
    assert state.step == 250

    from imggen.backbones import build_backbone
    from imggen.data.shards import load_null_embedding
    from imggen.samplers.euler import euler_sample
    from imggen.utils.checkpoint import load_checkpoint
    from imggen.utils.ema import EMA
    from imggen.utils.schedule import build_lr_lambda

    model = build_backbone(cfg.backbone)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, build_lr_lambda(5, 251))
    ema = EMA(model)
    load_checkpoint(
        tmp_path / "run" / "checkpoints" / "step_00000250",
        model=model, optimizer=opt, scheduler=scheduler, ema=ema,
    )  # fmt: skip
    ema.copy_to(model)
    model.eval()
    null_emb, null_mask = load_null_embedding(cache_root)

    correct = 0
    for color in COLORS:
        for shape in SHAPES:
            caption = f"a {color} {shape}"
            text_emb, text_mask = text_encoder.encode([caption])
            g = torch.Generator().manual_seed(abs(hash(caption)) % 1000)
            latent = euler_sample(
                model, (1, 8, 4, 4), text_emb, text_mask, steps=20, guidance_scale=4.0,
                null_emb=null_emb, null_mask=null_mask, generator=g,
            )  # fmt: skip
            image = ae.decode(latent)[0]
            center = image[:, 14:18, 14:18].mean(dim=(1, 2))
            nearest = min(COLORS, key=lambda c: (center - torch.tensor(COLORS[c])).norm().item())
            correct += nearest == color

    assert correct >= 8, f"only {correct}/15 captions produced the right color (chance is 3/15)"
