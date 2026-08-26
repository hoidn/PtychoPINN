import math

import pytest

from ptycho_torch import helper as hh
from ptycho_torch.config_params import DataConfig, ModelConfig


def test_get_padded_size_ignores_max_position_jitter():
    data_cfg = DataConfig(N=64, gridsize=1, group_padding_step=0.0)
    model_cfg = ModelConfig(max_position_jitter=10)

    padded = hh.get_padded_size(data_cfg, model_cfg)

    assert padded == 64


@pytest.mark.parametrize(
    ("gridsize", "group_padding_step", "expected_big_n"),
    [
        (1, 0.0, 64),
        (1, 3.0, 64),
        (1, 2.5, 64),
        (3, 0.0, 64),
        (3, 3.0, 72),   # ceil(3.0)=3, odd -> 4; 64 + 2*4
        (3, 2.5, 72),   # ceil(2.5)=3, odd -> 4; 64 + 2*4
        (3, 4.0, 72),   # ceil(4.0)=4, even; 64 + 2*4
        (3, 0.5, 68),   # ceil(0.5)=1, odd -> 2; 64 + 2*2
    ],
)
def test_get_big_n_pins_the_centered_padding_formula(
    gridsize, group_padding_step, expected_big_n
):
    data_cfg = DataConfig(N=64, gridsize=gridsize, group_padding_step=group_padding_step)
    model_cfg = ModelConfig()

    big_n = hh.get_bigN(data_cfg, model_cfg)

    offset = math.ceil(data_cfg.group_padding_step)
    if offset % 2:
        offset += 1
    assert big_n == data_cfg.N + (data_cfg.gridsize - 1) * offset
    assert big_n == expected_big_n


@pytest.mark.parametrize("group_padding_step", [-0.5, float("nan"), float("inf")])
def test_group_padding_step_rejects_non_finite_or_negative(group_padding_step):
    with pytest.raises(ValueError, match="group_padding_step must be finite and nonnegative"):
        DataConfig(N=64, gridsize=1, group_padding_step=group_padding_step)


def test_group_padding_step_defaults_to_three():
    assert DataConfig(N=64, gridsize=1).group_padding_step == 3.0
