"""Tests for imggen.autoencoders.sd_vae and imggen.text.t5_encoder.

Neither wrapper's from_pretrained path (the actual production path) can be tested here — this
sandbox has no network access to huggingface.co (ADR 0013). What's tested is everything
downstream of construction: a tiny, locally-built (no download) diffusers AutoencoderKL /
transformers T5EncoderModel is injected via the `_vae` / `_model`+`_tokenizer` constructor hooks,
exercising the real library code the production path also uses, just skipping the network call.
"""

import pytest
import torch

pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from diffusers import AutoencoderKL  # noqa: E402
from transformers import T5Config, T5EncoderModel  # noqa: E402

from imggen.autoencoders import build_autoencoder  # noqa: E402
from imggen.autoencoders.sd_vae import StableDiffusionVAE  # noqa: E402
from imggen.config import ComponentConfig  # noqa: E402
from imggen.registry import AUTOENCODER, TEXT_ENCODER  # noqa: E402
from imggen.text import build_text_encoder  # noqa: E402
from imggen.text.t5_encoder import T5TextEncoder  # noqa: E402


def tiny_vae(down_blocks: int = 4, latent_channels: int = 4) -> AutoencoderKL:
    return AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * down_blocks,
        up_block_types=("UpDecoderBlock2D",) * down_blocks,
        block_out_channels=(32,) * down_blocks,
        layers_per_block=1, latent_channels=latent_channels, scaling_factor=0.18215,
    )  # fmt: skip


class FakeTokenizer:
    """Matches the subset of the real tokenizer's __call__ interface T5TextEncoder.encode uses.
    A real tokenizer needs a downloaded vocab; this doesn't, so it's what the wrapper logic
    itself can be tested against without network access."""

    def __call__(self, texts, padding=True, truncation=True, max_length=64, return_tensors="pt"):
        lens = [min(len(t.split()) or 1, max_length) for t in texts]
        length = max(lens)
        ids = torch.zeros(len(texts), length, dtype=torch.long)
        mask = torch.zeros(len(texts), length, dtype=torch.long)
        for i, n in enumerate(lens):
            ids[i, :n] = torch.arange(1, n + 1)
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask}


def tiny_t5() -> T5EncoderModel:
    cfg = T5Config(vocab_size=100, d_model=32, num_layers=2, num_heads=2, d_ff=64, d_kv=16)
    return T5EncoderModel(cfg)


# ---------------------------------------------------------------- StableDiffusionVAE


def test_registered_under_autoencoder():
    assert AUTOENCODER.get("sd_vae") is StableDiffusionVAE


def test_latent_channels_and_spatial_compression_read_from_the_underlying_vae():
    ae = StableDiffusionVAE(scaling_factor=0.18215, _vae=tiny_vae(down_blocks=4))
    assert ae.latent_channels == 4
    assert ae.spatial_compression == 8  # 4 down blocks -> 2^3, matches the real SD/SDXL VAE


def test_spatial_compression_override():
    ae = StableDiffusionVAE(scaling_factor=0.18215, spatial_compression=99, _vae=tiny_vae())
    assert ae.spatial_compression == 99


def test_frozen_by_default_and_eval_mode():
    ae = StableDiffusionVAE(scaling_factor=0.18215, _vae=tiny_vae())
    assert all(not p.requires_grad for p in ae.parameters())
    assert ae.vae.training is False


def test_unfrozen_when_requested():
    ae = StableDiffusionVAE(scaling_factor=0.18215, frozen=False, _vae=tiny_vae())
    assert all(p.requires_grad for p in ae.parameters())


def test_encode_decode_shapes_and_round_trip_through_scaling_factor():
    ae = StableDiffusionVAE(scaling_factor=0.18215, _vae=tiny_vae(down_blocks=4))
    x = torch.randn(2, 3, 32, 32)
    z = ae.encode(x)
    assert z.shape == (2, 4, 4, 4)
    recon = ae.decode(z)
    assert recon.shape == x.shape


def test_scaling_factor_is_applied_and_invertible():
    ae = StableDiffusionVAE(scaling_factor=2.0, _vae=tiny_vae())
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        raw_mode = ae.vae.encode(x).latent_dist.mode()
        z = ae.encode(x)
    torch.testing.assert_close(z, raw_mode * 2.0)


def test_mode_is_deterministic_sample_is_not():
    ae_mode = StableDiffusionVAE(scaling_factor=0.18215, sample=False, _vae=tiny_vae())
    x = torch.randn(1, 3, 32, 32)
    torch.testing.assert_close(ae_mode.encode(x), ae_mode.encode(x))

    ae_sample = StableDiffusionVAE(scaling_factor=0.18215, sample=True, _vae=tiny_vae())
    torch.manual_seed(0)
    a = ae_sample.encode(x)
    torch.manual_seed(1)
    b = ae_sample.encode(x)
    assert not torch.equal(a, b)


def test_encode_runs_under_no_grad():
    ae = StableDiffusionVAE(scaling_factor=0.18215, frozen=False, _vae=tiny_vae())
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    assert not ae.encode(x).requires_grad


def test_unknown_dtype_rejected_before_any_network_call():
    with pytest.raises(ValueError, match="unknown dtype"):
        StableDiffusionVAE(dtype="not_a_dtype", model_id="irrelevant/unreachable")


def test_build_autoencoder_registry_path_uses_sd_vae():
    ae = build_autoencoder(
        ComponentConfig(name="sd_vae", args={"scaling_factor": 0.18215, "_vae": tiny_vae()})
    )
    assert isinstance(ae, StableDiffusionVAE)


# ---------------------------------------------------------------- T5TextEncoder


def test_registered_under_text_encoder():
    assert TEXT_ENCODER.get("t5") is T5TextEncoder


def test_text_dim_read_from_the_underlying_model_config():
    te = T5TextEncoder(_model=tiny_t5(), _tokenizer=FakeTokenizer())
    assert te.text_dim == 32


def test_t5_frozen_by_default_and_eval_mode():
    te = T5TextEncoder(_model=tiny_t5(), _tokenizer=FakeTokenizer())
    assert all(not p.requires_grad for p in te.parameters())
    assert te.model.training is False


def test_t5_unfrozen_when_requested():
    te = T5TextEncoder(frozen=False, _model=tiny_t5(), _tokenizer=FakeTokenizer())
    assert all(p.requires_grad for p in te.parameters())


def test_encode_shapes_and_mask():
    te = T5TextEncoder(_model=tiny_t5(), _tokenizer=FakeTokenizer())
    emb, mask = te.encode(["a red circle", "blue"])
    assert emb.shape == (2, 3, 32)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [[True, True, True], [True, False, False]]


def test_encode_is_deterministic():
    te = T5TextEncoder(_model=tiny_t5(), _tokenizer=FakeTokenizer())
    a, _ = te.encode(["same caption"])
    b, _ = te.encode(["same caption"])
    torch.testing.assert_close(a, b)


def test_t5_encode_runs_under_no_grad():
    te = T5TextEncoder(frozen=False, _model=tiny_t5(), _tokenizer=FakeTokenizer())
    emb, _ = te.encode(["text"])
    assert not emb.requires_grad


def test_max_length_is_passed_to_the_tokenizer():
    # FakeTokenizer.__call__ actually truncates to max_length (see its definition above), so
    # passing a caption longer than max_length and checking the output length is truncated
    # verifies encode() forwards self.max_length through — unlike patching an instance's
    # __call__, which Python's dunder lookup bypasses for implicit `tok(...)` invocation.
    te = T5TextEncoder(max_length=3, _model=tiny_t5(), _tokenizer=FakeTokenizer())
    emb, mask = te.encode(["one two three four five"])
    assert emb.shape[1] == 3 and mask.shape[1] == 3


def test_t5_unknown_dtype_rejected_before_any_network_call():
    with pytest.raises(ValueError, match="unknown dtype"):
        T5TextEncoder(dtype="not_a_dtype", model_id="irrelevant/unreachable")


def test_build_text_encoder_registry_path_uses_t5():
    te = build_text_encoder(
        ComponentConfig(name="t5", args={"_model": tiny_t5(), "_tokenizer": FakeTokenizer()})
    )
    assert isinstance(te, T5TextEncoder)
