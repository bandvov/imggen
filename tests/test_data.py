import json
import subprocess
import sys
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader

from imggen.autoencoders import build_autoencoder
from imggen.autoencoders.toy import ToyAutoencoder
from imggen.config import BackboneConfig, ComponentConfig, DataConfig, load_config
from imggen.data.loader import build_dataloader
from imggen.data.shards import (
    LatentShardDataset,
    Manifest,
    ManifestMismatch,
    ShardWriter,
    collate_latents,
    has_null_embedding,
    load_null_embedding,
)
from imggen.data.toy_dataset import ToyShapesDataset
from imggen.registry import AUTOENCODER, TEXT_ENCODER
from imggen.text import build_text_encoder
from imggen.text.hash_embedding import HashEmbeddingTextEncoder, _token_id

# ---------------------------------------------------------------- autoencoder


def test_toy_ae_registered_and_buildable():
    assert AUTOENCODER.get("toy") is ToyAutoencoder
    ae = build_autoencoder(ComponentConfig(name="toy", args={"latent_channels": 8}))
    assert isinstance(ae, ToyAutoencoder)


@pytest.mark.parametrize("num_downsamples,latent_channels", [(2, 4), (3, 8), (4, 16)])
def test_encode_decode_shapes(num_downsamples, latent_channels):
    ae = ToyAutoencoder(latent_channels=latent_channels, num_downsamples=num_downsamples)
    size = 2**num_downsamples * 4
    x = torch.randn(2, 3, size, size)
    z = ae.encode(x)
    assert z.shape == (
        2,
        latent_channels,
        size // ae.spatial_compression,
        size // ae.spatial_compression,
    )
    assert ae.decode(z).shape == x.shape
    assert ae.spatial_compression == 2**num_downsamples


def test_ae_is_frozen_by_default_and_deterministic_given_seed():
    ae = ToyAutoencoder(seed=7)
    assert all(not p.requires_grad for p in ae.parameters())
    a = ToyAutoencoder(seed=7)
    b = ToyAutoencoder(seed=7)
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert torch.equal(pa, pb), n
    c = ToyAutoencoder(seed=8)
    assert not torch.equal(a.encoder[0].weight, c.encoder[0].weight)


def test_ae_unfrozen_when_requested():
    ae = ToyAutoencoder(frozen=False)
    assert all(p.requires_grad for p in ae.parameters())


def test_encode_no_grad_even_if_unfrozen():
    ae = ToyAutoencoder(frozen=False)
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    z = ae.encode(x)
    assert not z.requires_grad


def test_unknown_autoencoder_name_fails_clearly():
    with pytest.raises(Exception, match="unknown autoencoder"):
        build_autoencoder(ComponentConfig(name="nope"))


# ---------------------------------------------------------------- text encoder


def test_hash_encoder_registered_and_buildable():
    assert TEXT_ENCODER.get("hash") is HashEmbeddingTextEncoder
    te = build_text_encoder(ComponentConfig(name="hash", args={"text_dim": 32}))
    assert isinstance(te, HashEmbeddingTextEncoder) and te.text_dim == 32


def test_token_id_is_stable_across_calls_and_within_vocab():
    assert _token_id("red", 1000) == _token_id("red", 1000)
    assert 0 <= _token_id("blue", 500) < 500


def test_same_caption_gives_same_embedding():
    te = HashEmbeddingTextEncoder(text_dim=16, seed=1)
    a, ma = te.encode(["a red circle"])
    b, mb = te.encode(["a red circle"])
    torch.testing.assert_close(a, b)
    assert torch.equal(ma, mb)


def test_different_captions_give_different_embeddings():
    te = HashEmbeddingTextEncoder(text_dim=16, seed=1)
    a, _ = te.encode(["a red circle"])
    b, _ = te.encode(["a blue square"])
    assert not torch.allclose(a, b)


def test_padding_mask_matches_token_counts_and_padded_slots_are_zero():
    te = HashEmbeddingTextEncoder(text_dim=8, seed=0)
    emb, mask = te.encode(["one two three", "solo"])
    assert mask.tolist() == [[True, True, True], [True, False, False]]
    assert (emb[1, 1:] == 0).all()


def test_max_length_truncates():
    te = HashEmbeddingTextEncoder(text_dim=8, max_length=3, seed=0)
    _, mask = te.encode(["a b c d e f"])
    assert mask.shape[1] == 3 and mask.all()


def test_empty_string_gets_one_token():
    te = HashEmbeddingTextEncoder(text_dim=8, seed=0)
    _, mask = te.encode([""])
    assert mask.tolist() == [[True]]


def test_text_encoder_is_frozen_and_seeded():
    a = HashEmbeddingTextEncoder(text_dim=8, seed=3)
    b = HashEmbeddingTextEncoder(text_dim=8, seed=3)
    assert all(not p.requires_grad for p in a.parameters())
    torch.testing.assert_close(a.embedding.weight, b.embedding.weight)


# ---------------------------------------------------------------- toy dataset


def test_toy_shapes_dataset_shapes_and_range():
    ds = ToyShapesDataset(size=32, length=9, seed=0)
    assert len(ds) == 9
    img, caption = ds[0]
    assert img.shape == (3, 32, 32)
    assert img.min() >= -1.0 and img.max() <= 1.0
    assert isinstance(caption, str) and len(caption.split()) == 3


def test_toy_shapes_dataset_deterministic_given_seed():
    a = ToyShapesDataset(size=16, length=5, seed=2)
    b = ToyShapesDataset(size=16, length=5, seed=2)
    for i in range(5):
        img_a, cap_a = a[i]
        img_b, cap_b = b[i]
        assert torch.equal(img_a, img_b) and cap_a == cap_b


def test_toy_shapes_dataset_has_variety():
    ds = ToyShapesDataset(size=16, length=12, seed=0)
    captions = {ds[i][1] for i in range(12)}
    assert len(captions) > 1


# ---------------------------------------------------------------- shards


def _write_shards(tmp_path, n=7, shard_size=3, dim=6, latent_shape=(4, 2, 2)):
    writer = ShardWriter(tmp_path, shard_size=shard_size)
    lengths = []
    for i in range(n):
        length = 2 + (i % 3)
        lengths.append(length)
        writer.add(
            f"{i:04d}",
            torch.full(latent_shape, float(i)),
            torch.randn(length, dim),
            torch.ones(length, dtype=torch.bool),
            f"caption {i}",
        )
    num_shards = writer.close()
    Manifest(
        num_samples=n, num_shards=num_shards, shard_size=shard_size,
        latent_channels=latent_shape[0], spatial_compression=8, text_dim=dim, resolution=64,
    ).save(tmp_path)  # fmt: skip
    return num_shards, lengths


def test_shard_writer_splits_into_expected_shard_count(tmp_path):
    num_shards, _ = _write_shards(tmp_path, n=7, shard_size=3)
    assert num_shards == 3  # ceil(7/3)
    assert sorted(p.name for p in tmp_path.glob("shard-*.tar")) == [
        "shard-00000.tar", "shard-00001.tar", "shard-00002.tar",
    ]  # fmt: skip


def test_manifest_roundtrip(tmp_path):
    m = Manifest(
        num_samples=10, num_shards=2, shard_size=5, latent_channels=32,
        spatial_compression=8, text_dim=2048, resolution=64,
        autoencoder={"name": "toy", "args": {}}, text_encoder={"name": "hash", "args": {}},
    )  # fmt: skip
    m.save(tmp_path)
    assert Manifest.load(tmp_path) == m


def test_manifest_check_raises_on_mismatch():
    m = Manifest(
        num_samples=1, num_shards=1, shard_size=1, latent_channels=32,
        spatial_compression=8, text_dim=2048, resolution=64,
    )  # fmt: skip
    m.check(in_channels=32, text_dim=2048)  # no raise
    with pytest.raises(ManifestMismatch, match="in_channels"):
        m.check(in_channels=16)
    with pytest.raises(ManifestMismatch, match="text_dim"):
        m.check(text_dim=999)


def test_dataset_reads_back_exact_values_in_written_order(tmp_path):
    n, shard_size = 7, 3
    _write_shards(tmp_path, n=n, shard_size=shard_size)
    ds = LatentShardDataset(tmp_path)
    assert len(ds) == n
    for i in range(n):
        item = ds[i]
        assert torch.equal(item["latent"], torch.full((4, 2, 2), float(i)))
        assert item["caption"] == f"caption {i}"


def test_dataset_survives_pickling_for_multiworker_use(tmp_path):
    import pickle

    _write_shards(tmp_path, n=4, shard_size=2)
    ds = LatentShardDataset(tmp_path)
    ds[0]  # opens a tar handle internally
    restored = pickle.loads(pickle.dumps(ds))
    assert restored[0]["caption"] == ds[0]["caption"]


def test_dataloader_with_multiple_workers_reads_every_sample(tmp_path):
    n = 9
    _write_shards(tmp_path, n=n, shard_size=4)
    ds = LatentShardDataset(tmp_path)
    dl = DataLoader(ds, batch_size=2, num_workers=1, collate_fn=collate_latents, shuffle=False)
    seen = sorted(cap for batch in dl for cap in batch["caption"])
    assert seen == sorted(f"caption {i}" for i in range(n))


def test_collate_pads_variable_length_text_to_batch_max():
    short = torch.tensor([True, True])
    long = torch.ones(5, dtype=torch.bool)
    batch = [
        {
            "latent": torch.zeros(4, 2, 2),
            "text_emb": torch.randn(2, 6),
            "mask": short,
            "caption": "a",
        },
        {
            "latent": torch.zeros(4, 2, 2),
            "text_emb": torch.randn(5, 6),
            "mask": long,
            "caption": "b",
        },
    ]
    out = collate_latents(batch)
    assert out["text_emb"].shape == (2, 5, 6) and out["mask"].shape == (2, 5)
    assert out["mask"][0].tolist() == [True, True, False, False, False]
    assert (out["text_emb"][0, 2:] == 0).all()


def test_build_dataloader_checks_backbone_and_yields_batches(tmp_path):
    _write_shards(tmp_path, n=6, shard_size=3, dim=2048, latent_shape=(32, 2, 2))
    cfg = DataConfig(root=str(tmp_path), num_workers=0)
    dl = build_dataloader(cfg, batch_size=3, backbone=BackboneConfig(size="proxy"), shuffle=False)
    batch = next(iter(dl))
    assert batch["latent"].shape == (3, 32, 2, 2)
    with pytest.raises(ManifestMismatch):
        build_dataloader(cfg, batch_size=3, backbone=BackboneConfig(size="proxy", text_dim=1))


def test_build_dataloader_without_backbone_skips_check(tmp_path):
    _write_shards(tmp_path, n=3, shard_size=3, dim=8, latent_shape=(4, 2, 2))
    cfg = DataConfig(root=str(tmp_path), num_workers=0)
    dl = build_dataloader(cfg, batch_size=3, shuffle=False)
    assert next(iter(dl))["latent"].shape == (3, 4, 2, 2)


# ---------------------------------------------------------------- precompute.py CLI


def test_precompute_script_writes_readable_cache(repo_root, experiment_yaml, tmp_path):
    out = tmp_path / "cache"
    r = subprocess.run(
        [
            sys.executable, str(repo_root / "scripts" / "precompute.py"), str(experiment_yaml),
            f"data.root={out}", "data.shard_size=5", "backbone.size=nano",
            "--num-samples", "13", "--batch-size", "4",
        ],
        capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["num_samples"] == 13 and manifest["num_shards"] == 3  # ceil(13/5)

    cfg = load_config(
        experiment_yaml, [f"data.root={out}", "backbone.size=nano", "data.num_workers=0"]
    )
    dl = build_dataloader(cfg.data, batch_size=4, backbone=cfg.backbone, shuffle=False)
    batch = next(iter(dl))
    assert batch["latent"].shape[1] == cfg.backbone.in_channels
    assert batch["text_emb"].shape[-1] == cfg.backbone.text_dim


def test_precompute_script_default_config_uses_registered_components(
    repo_root, experiment_yaml, tmp_path
):
    out = tmp_path / "cache"
    r = subprocess.run(
        [
            sys.executable, str(repo_root / "scripts" / "precompute.py"), str(experiment_yaml),
            f"data.root={out}", "--num-samples", "3", "--batch-size", "3",
        ],
        capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["autoencoder"]["name"] == "toy" and manifest["text_encoder"]["name"] == "hash"


def test_precompute_script_saves_a_null_embedding(repo_root, experiment_yaml, tmp_path):
    out = tmp_path / "cache"
    r = subprocess.run(
        [
            sys.executable, str(repo_root / "scripts" / "precompute.py"), str(experiment_yaml),
            f"data.root={out}", "backbone.size=nano", "--num-samples", "3", "--batch-size", "3",
        ],
        capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert has_null_embedding(out)
    emb, mask = load_null_embedding(out)
    cfg = load_config(experiment_yaml, [f"data.root={out}", "backbone.size=nano"])
    assert emb.shape[-1] == cfg.backbone.text_dim
    assert mask.dtype == torch.bool and mask.any()


def test_build_autoencoder_loads_weights_path_and_pops_it_from_args():
    ae = ToyAutoencoder(latent_channels=8, num_downsamples=2, seed=1)
    with torch.no_grad():
        ae.encoder[0].weight.add_(1.0)  # distinguish from a fresh (seed=1) construction
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/ae.pt"
        torch.save(ae.state_dict(), path)
        args = {
            "latent_channels": 8, "num_downsamples": 2, "seed": 1, "weights_path": path,
        }  # fmt: skip
        loaded = build_autoencoder(ComponentConfig(name="toy", args=args))
    torch.testing.assert_close(loaded.encoder[0].weight, ae.encoder[0].weight)
    fresh = ToyAutoencoder(latent_channels=8, num_downsamples=2, seed=1)
    assert not torch.equal(loaded.encoder[0].weight, fresh.encoder[0].weight)


def test_build_autoencoder_without_weights_path_is_unaffected():
    ae = build_autoencoder(ComponentConfig(name="toy", args={"latent_channels": 4}))
    assert isinstance(ae, ToyAutoencoder)
