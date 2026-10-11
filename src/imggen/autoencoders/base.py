"""Autoencoder interface. Every implementation maps pixels to a compressed latent and back.

    encode(x) -> z     x: (B, 3, H, W) in [-1, 1]   ->  z: (B, latent_channels, H/f, W/f)
    decode(z) -> x      the inverse, approximately

``f`` (``spatial_compression``) and ``latent_channels`` are read by the data pipeline (to check
they match ``backbone.in_channels``) and are not expected to change after construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Autoencoder(nn.Module, ABC):
    latent_channels: int
    spatial_compression: int

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor: ...
