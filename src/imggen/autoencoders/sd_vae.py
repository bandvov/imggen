"""Wraps ``diffusers.AutoencoderKL`` — the Stable Diffusion / SDXL VAE — to the ``Autoencoder``
interface. This is the real pretrained AE that ADR 0006 deferred: 8x spatial compression, 4
latent channels, downloaded from a Hugging Face repo. Needs ``diffusers`` installed
(``pip install diffusers``) and, for ``from_pretrained``, network access to huggingface.co —
this project's own sandbox doesn't have that (ADR 0013), so the ``from_pretrained`` path is
untested here; what's tested is everything downstream of construction, using a tiny
locally-built ``AutoencoderKL`` (no download) injected via ``_vae``.

Pixel convention matches the rest of this project: images in, images out, range [-1, 1]
(``imggen.data.toy_dataset`` already follows this).
"""

from __future__ import annotations

import torch

from imggen.autoencoders.base import Autoencoder
from imggen.registry import AUTOENCODER

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@AUTOENCODER.register("sd_vae")
class StableDiffusionVAE(Autoencoder):
    def __init__(
        self,
        model_id: str = "stabilityai/sdxl-vae",
        subfolder: str | None = None,
        scaling_factor: float | None = None,
        spatial_compression: int | None = None,
        dtype: str = "float32",
        sample: bool = False,
        frozen: bool = True,
        _vae=None,
    ) -> None:
        """``sample=False`` (default) uses the latent distribution's mode — deterministic, so
        precomputing the cache twice gives the same latents, matching this project's "frozen,
        deterministic AE" convention (ADR 0006, same choice ``ToyAutoencoder`` makes).
        ``sample=True`` matches how Stable Diffusion itself trains (sampling from the posterior
        adds regularization) at the cost of that determinism. ``_vae``: inject an
        already-constructed ``AutoencoderKL`` (e.g. for testing without network access);
        production use leaves this as ``None`` and loads ``model_id`` from the hub.
        """
        super().__init__()
        if _vae is None:
            from diffusers import AutoencoderKL

            if dtype not in _DTYPES:
                raise ValueError(f"unknown dtype {dtype!r} (expected one of {list(_DTYPES)})")
            _vae = AutoencoderKL.from_pretrained(
                model_id, subfolder=subfolder, torch_dtype=_DTYPES[dtype]
            )
        self.vae = _vae
        self.sample = sample
        self.latent_channels = self.vae.config.latent_channels
        self.spatial_compression = spatial_compression or 2 ** (
            len(self.vae.config.down_block_types) - 1
        )
        self.scaling_factor = (
            scaling_factor if scaling_factor is not None else self.vae.config.scaling_factor
        )
        self.vae.eval()
        if frozen:
            self.requires_grad_(False)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.vae.encode(x.to(self.vae.dtype)).latent_dist
        z = dist.sample() if self.sample else dist.mode()
        return (z * self.scaling_factor).float()

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = (z / self.scaling_factor).to(self.vae.dtype)
        return self.vae.decode(z).sample.float()
