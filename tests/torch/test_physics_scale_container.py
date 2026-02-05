import torch
from types import SimpleNamespace
from ptycho_torch.data_container_bridge import PtychoDataContainerTorch
from ptycho_torch.workflows import components


def test_container_gets_physics_scale_value():
    grouped = {
        "X_full": (torch.ones(2, 2, 2, 2)).numpy(),  # B=2, H=2, W=2, C=2
        "coords_relative": torch.zeros(2, 1, 2, 2).numpy(),
        "coords_offsets": torch.zeros(2, 1, 2, 1).numpy(),
        "nn_indices": torch.zeros(2, 2, dtype=torch.int32).numpy(),
        "Y": None,
    }
    probe = (torch.ones(2, 2, dtype=torch.complex64)).numpy()
    container = PtychoDataContainerTorch(grouped, probe)

    config = SimpleNamespace(nphotons=32.0)
    scale, source = components._attach_physics_scale(container, config, nphotons_source="config")
    assert torch.is_tensor(scale)
    assert torch.isclose(scale, torch.tensor(2.0), atol=1e-6)  # sqrt(32 / 8)
    assert source == "config"
    assert hasattr(container, "physics_scaling_constant")
