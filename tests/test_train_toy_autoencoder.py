import torch

from imggen.autoencoders.toy import ToyAutoencoder
from scripts.train_toy_autoencoder import train_toy_autoencoder


def test_reconstruction_loss_decreases():
    torch.manual_seed(0)
    ae = ToyAutoencoder(latent_channels=8, num_downsamples=2, seed=0)
    losses = train_toy_autoencoder(
        ae, steps=80, batch_size=8, resolution=32, dataset_size=32, seed=0
    )
    assert len(losses) == 80
    assert sum(losses[-10:]) / 10 < losses[0] * 0.5


def test_model_is_frozen_again_after_training():
    ae = ToyAutoencoder(latent_channels=8, num_downsamples=2, seed=0)
    assert all(not p.requires_grad for p in ae.parameters())  # frozen by default (ADR 0006)
    train_toy_autoencoder(ae, steps=5, batch_size=4, resolution=32, dataset_size=8, seed=0)
    assert all(not p.requires_grad for p in ae.parameters())  # restored after training
    assert ae.training is False  # left in eval mode


def test_encode_still_runs_under_no_grad_after_training():
    ae = ToyAutoencoder(latent_channels=8, num_downsamples=2, seed=0)
    train_toy_autoencoder(ae, steps=5, batch_size=4, resolution=32, dataset_size=8, seed=0)
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    z = ae.encode(x)
    assert not z.requires_grad


def test_deterministic_given_a_seed():
    def run():
        torch.manual_seed(0)
        ae = ToyAutoencoder(latent_channels=8, num_downsamples=2, seed=0)
        return train_toy_autoencoder(
            ae, steps=10, batch_size=4, resolution=32, dataset_size=8, seed=0
        )

    assert run() == run()
