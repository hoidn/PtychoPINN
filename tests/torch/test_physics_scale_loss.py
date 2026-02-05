import torch
from ptycho_torch.model import PtychoPINN_Lightning
from ptycho_torch.config_params import ModelConfig, DataConfig, TrainingConfig, InferenceConfig


def test_poisson_loss_uses_physics_scale():
    model_cfg = ModelConfig(
        C_model=1,
        C_forward=1,
        object_big=False,
        probe_big=False,
        loss_function="Poisson",
    )
    data_cfg = DataConfig(N=64, C=1, grid_size=(1, 1))
    train_cfg = TrainingConfig(device="cpu", torch_loss_mode="poisson")
    infer_cfg = InferenceConfig()

    model = PtychoPINN_Lightning(model_cfg, data_cfg, train_cfg, infer_cfg)
    model.eval()

    x = torch.ones(1, 1, 64, 64)
    positions = torch.zeros(1, 1, 1, 2)
    probe = torch.ones(64, 64, dtype=torch.complex64)

    batch_scale_1 = (
        {
            "images": x,
            "coords_relative": positions,
            "rms_scaling_constant": torch.ones(1, 1, 1, 1),
            "physics_scaling_constant": torch.ones(1, 1, 1, 1),
            "experiment_id": torch.zeros(1, dtype=torch.long),
        },
        probe,
        torch.ones(1),
    )
    batch_scale_2 = (
        {
            "images": x,
            "coords_relative": positions,
            "rms_scaling_constant": torch.ones(1, 1, 1, 1),
            "physics_scaling_constant": torch.full((1, 1, 1, 1), 2.0),
            "experiment_id": torch.zeros(1, dtype=torch.long),
        },
        probe,
        torch.ones(1),
    )

    loss_1 = model.compute_loss(batch_scale_1)
    loss_2 = model.compute_loss(batch_scale_2)

    assert not torch.isclose(loss_1, loss_2, atol=1e-6, rtol=1e-6)
