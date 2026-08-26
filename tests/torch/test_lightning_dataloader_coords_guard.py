import numpy as np
import pytest

from ptycho.config.config import TrainingConfig, ModelConfig
from ptycho_torch.workflows.components import _build_lightning_dataloaders


def _make_container(has_relative: bool) -> dict:
    X = np.zeros((2, 4, 4, 1), dtype=np.float32)
    container = {
        "X": X,
        "coords_nominal": np.zeros((2, 1, 2, 1), dtype=np.float32),
    }
    if has_relative:
        container["coords_relative"] = np.zeros((2, 1, 2, 1), dtype=np.float32)
    return container


def test_coords_relative_required_when_object_big_true():
    config = TrainingConfig(model=ModelConfig(N=4, gridsize=1, object_big=True))
    container = _make_container(has_relative=False)

    with pytest.raises(ValueError, match="coords_relative is required"):
        _build_lightning_dataloaders(container, None, config, payload=None)


def test_coords_nominal_allowed_when_object_big_false():
    config = TrainingConfig(model=ModelConfig(N=4, gridsize=1, object_big=False))
    container = _make_container(has_relative=False)

    train_loader, _ = _build_lightning_dataloaders(container, None, config, payload=None)
    batch = next(iter(train_loader))
    assert batch[0]["coords_relative"] is not None
