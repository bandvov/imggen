import random
import sys
import types

import pytest

from imggen.utils import derive_seed, seed_everything


def test_derive_seed_stable_and_key_sensitive():
    assert derive_seed(0, "noise", 3) == derive_seed(0, "noise", 3)
    assert derive_seed(0, "noise", 3) != derive_seed(0, "noise", 4)
    assert derive_seed(0, "noise") != derive_seed(1, "noise")
    assert 0 <= derive_seed(123, "x") < 2**32


def test_python_rng_reproducible():
    seed_everything(7)
    a = [random.random() for _ in range(3)]
    seed_everything(7)
    assert a == [random.random() for _ in range(3)]


def test_numpy_rng_reproducible():
    np = pytest.importorskip("numpy")
    seed_everything(7)
    a = np.random.rand(3)
    seed_everything(7)
    assert (a == np.random.rand(3)).all()


def test_torch_branch_with_fake_torch(monkeypatch):
    """Real torch is not installed in CI/sandbox; this only checks the calls we make."""
    calls = []
    fake = types.SimpleNamespace(
        manual_seed=lambda s: calls.append(("manual", s)),
        cuda=types.SimpleNamespace(manual_seed_all=lambda s: calls.append(("cuda", s))),
        use_deterministic_algorithms=lambda flag, warn_only=False: calls.append(("det", flag)),
        backends=types.SimpleNamespace(
            cudnn=types.SimpleNamespace(deterministic=False, benchmark=True)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    seed_everything(5, deterministic=True)
    assert calls == [("manual", 5), ("cuda", 5), ("det", True)]
    assert fake.backends.cudnn.deterministic is True and fake.backends.cudnn.benchmark is False


def test_real_torch_seeding_is_reproducible():
    torch = pytest.importorskip("torch")
    seed_everything(11)
    a = torch.randn(4)
    seed_everything(11)
    assert torch.equal(a, torch.randn(4))
