"""portable-v3 states channel identity once, as gridsize."""
from __future__ import annotations

from dataclasses import fields


from ptycho_torch.config_params import DataConfig, ModelConfig


def test_data_config_states_channel_identity_once():
    names = {f.name for f in fields(DataConfig)}
    assert "gridsize" in names
    for retired in ("C", "grid_size"):
        assert retired not in names, f"{retired} is a duplicated channel statement"


def test_model_config_has_no_channel_twins():
    names = {f.name for f in fields(ModelConfig)}
    for retired in ("C_model", "C_forward"):
        assert retired not in names


def test_raw_selection_and_groups_per_center_are_separate():
    names = {f.name for f in fields(DataConfig)}
    assert "n_subsample" not in names, (
        "n_subsample meant raw-frame selection AND groups-per-center"
    )
