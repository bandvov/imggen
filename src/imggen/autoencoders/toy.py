"""A small, untrained, strided-conv autoencoder.

This is a placeholder that satisfies the ``Autoencoder`` interface end to end (real shapes, a
real if crude reconstruction) so the data pipeline, training loop and tests can run without
network access to a real pretrained checkpoint. It is not the DC-AE described in the project
plan and its reconstructions are not expected to be good — swap in a real high-compression AE
by registering it under a new name and pointing ``autoencoder.name`` at it (see ADR 0006);
nothing else in the data pipeline needs to change.

``spatial_compression = 2 ** num_downsamples``. Frozen (``requires_grad_(False)``) by default,
since this project treats latents as precomputed and cached, not trained end to end.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from imggen.autoencoders.base import Autoencoder
from imggen.registry import AUTOENCODER


def _conv_stack(channels: list[int], stride: int, transpose: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(channels) - 1):
        cin, cout = channels[i], channels[i + 1]
        last = i == len(channels) - 2
        if transpose:
            layers.append(nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=stride, padding=1))
        else:
            layers.append(nn.Conv2d(cin, cout, kernel_size=4, stride=stride, padding=1))
        if not last:
            layers.append(nn.GroupNorm(min(8, cout), cout))
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


@AUTOENCODER.register("toy")
class ToyAutoencoder(Autoencoder):
    def __init__(
        self,
        latent_channels: int = 32,
        num_downsamples: int = 3,
        hidden: int = 64,
        frozen: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.spatial_compression = 2**num_downsamples
        gen = torch.Generator().manual_seed(seed)
        with torch.random.fork_rng():
            torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen).item()))
            enc_channels = [3] + [hidden] * (num_downsamples - 1) + [latent_channels]
            dec_channels = [latent_channels] + [hidden] * (num_downsamples - 1) + [3]
            self.encoder = _conv_stack(enc_channels, stride=2, transpose=False)
            self.decoder = _conv_stack(dec_channels, stride=2, transpose=True)
        if frozen:
            self.requires_grad_(False)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
