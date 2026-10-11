from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    return REPO


@pytest.fixture
def experiment_yaml(repo_root: Path) -> Path:
    return repo_root / "configs" / "experiment" / "proxy_baseline.yaml"
