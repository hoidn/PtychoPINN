from ptycho_torch import helper as hh
from ptycho_torch.config_params import DataConfig, ModelConfig


def test_get_padded_size_ignores_max_position_jitter():
    data_cfg = DataConfig(N=64, grid_size=(1, 1), max_neighbor_distance=0.0)
    model_cfg = ModelConfig(max_position_jitter=10)

    padded = hh.get_padded_size(data_cfg, model_cfg)

    assert padded == 64
