"""Deprecated flat root spellings accepted by the nested ``TrainingConfig``.

External callers (ptychodus) construct ``TrainingConfig`` with the historical
flat keyword surface. These tests pin the compatibility contract: enumerated
legacy flat keys lift into their nested owners, duplicates must agree, and
everything else still fails ``extra="forbid"``.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import pytest

from ptycho.config import (
    DataConfig,
    ModelConfig,
    SamplingConfig,
    TFLossConfig,
    TrainingConfig,
    dataclass_to_legacy_dict,
    update_legacy_dict,
)


_PTYCHODUS_MODEL = dict(
    N=64,
    gridsize=1,
    n_filters_scale=2,
    model_type='pinn',
    amp_activation='sigmoid',
    object_big=True,
    probe_big=True,
    probe_mask=False,
    pad_object=True,
    probe_scale=4.0,
    gaussian_smoothing_sigma=0.0,
)


def _ptychodus_flat_config() -> TrainingConfig:
    """Construct exactly as the external ptychodus caller does."""
    return TrainingConfig(
        model=ModelConfig(**_PTYCHODUS_MODEL),
        train_data_file=Path('train.npz'), test_data_file=None,
        batch_size=16, nepochs=2,
        mae_weight=0.0, nll_weight=1.0,
        realspace_mae_weight=0.0, realspace_weight=0.0,
        nphotons=1e6,
        positions_provided=True, probe_trainable=False,
        intensity_scale_trainable=True, output_dir=Path('out'),
    )


def _nested_config() -> TrainingConfig:
    """Construct the same configuration through the nested sections."""
    return TrainingConfig(
        model=ModelConfig(**_PTYCHODUS_MODEL),
        data=DataConfig(
            train_data_file=Path('train.npz'),
            test_data_file=None,
            nphotons=1e6,
        ),
        tf_loss=TFLossConfig(
            mae_weight=0.0,
            nll_weight=1.0,
            realspace_mae_weight=0.0,
            realspace_weight=0.0,
        ),
        batch_size=16, nepochs=2,
        positions_provided=True, probe_trainable=False,
        intensity_scale_trainable=True, output_dir=Path('out'),
    )


_ALIAS_CASES = [
    ('train_data_file', Path('train.npz'), 'data', 'train_data_file'),
    ('test_data_file', Path('test.npz'), 'data', 'test_data_file'),
    ('nphotons', 1e6, 'data', 'nphotons'),
    ('mae_weight', 0.25, 'tf_loss', 'mae_weight'),
    ('nll_weight', 0.75, 'tf_loss', 'nll_weight'),
    ('realspace_mae_weight', 0.5, 'tf_loss', 'realspace_mae_weight'),
    ('realspace_weight', 0.125, 'tf_loss', 'realspace_weight'),
    ('n_groups', 32, 'sampling', 'n_groups'),
    # n_images lifts into sampling, where SamplingConfig converts it to n_groups.
    ('n_images', 24, 'sampling', 'n_groups'),
    ('n_subsample', 8, 'sampling', 'n_subsample'),
    ('subsample_seed', 7, 'sampling', 'subsample_seed'),
    ('neighbor_count', 6, 'sampling', 'neighbor_count'),
    ('enable_oversampling', True, 'sampling', 'enable_oversampling'),
    ('neighbor_pool_size', 9, 'sampling', 'neighbor_pool_size'),
    ('sequential_sampling', True, 'sampling', 'sequential_sampling'),
]


def test_ptychodus_flat_construction_validates():
    with pytest.warns(DeprecationWarning):
        config = _ptychodus_flat_config()

    assert config.data.train_data_file == Path('train.npz')


def test_ptychodus_flat_construction_equals_nested_construction():
    with pytest.warns(DeprecationWarning):
        flat = _ptychodus_flat_config()

    assert flat.model_dump() == _nested_config().model_dump()


@pytest.mark.parametrize(
    "alias,value,section,field",
    _ALIAS_CASES,
    ids=[case[0] for case in _ALIAS_CASES],
)
def test_each_legacy_alias_lifts_into_its_nested_owner(alias, value, section, field):
    with pytest.warns(DeprecationWarning):
        config = TrainingConfig(**{alias: value})

    assert getattr(getattr(config, section), field) == value


def test_equal_flat_and_mapping_section_duplicate_is_accepted_once():
    with pytest.warns(DeprecationWarning):
        config = TrainingConfig(data={'nphotons': 1e6}, nphotons=1e6)

    assert config.data.nphotons == 1e6


def test_equal_flat_and_model_instance_section_duplicate_is_accepted_once():
    with pytest.warns(DeprecationWarning):
        config = TrainingConfig(data=DataConfig(nphotons=1e6), nphotons=1e6)

    assert config.data.nphotons == 1e6


def test_unequal_flat_and_mapping_section_duplicate_is_rejected():
    with pytest.raises(ValueError) as excinfo:
        TrainingConfig(data={'nphotons': 1e6}, nphotons=2e6)

    message = str(excinfo.value)
    assert "'nphotons'" in message and "data.nphotons" in message


def test_unequal_flat_and_model_instance_section_duplicate_is_rejected():
    with pytest.raises(ValueError) as excinfo:
        TrainingConfig(data=DataConfig(nphotons=1e6), nphotons=2e6)

    message = str(excinfo.value)
    assert "'nphotons'" in message and "data.nphotons" in message


def test_alias_use_names_its_nested_replacement_in_the_warning():
    with pytest.warns(DeprecationWarning, match=r"nphotons.*data\.nphotons"):
        TrainingConfig(nphotons=1e6)


def test_nested_construction_emits_no_deprecation_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        config = TrainingConfig(
            data=DataConfig(nphotons=1e6),
            sampling=SamplingConfig(n_groups=32),
        )

    assert config.data.nphotons == 1e6


def test_unknown_root_key_still_fails_extra_forbidden():
    with pytest.raises(ValueError) as excinfo:
        TrainingConfig(bogus_key=1)

    assert "extra_forbidden" in str(excinfo.value)


def test_unknown_root_key_alongside_alias_still_fails_extra_forbidden():
    with pytest.raises(ValueError) as excinfo:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            TrainingConfig(nphotons=1e6, bogus_key=1)

    assert "extra_forbidden" in str(excinfo.value)


def test_legacy_projection_of_flat_and_nested_configs_is_identical():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        flat = _ptychodus_flat_config()
        nested = _nested_config()

        flat_cfg: dict = {}
        nested_cfg: dict = {}
        update_legacy_dict(flat_cfg, flat)
        update_legacy_dict(nested_cfg, nested)

    assert flat_cfg == nested_cfg


def test_legacy_dict_translation_of_flat_and_nested_configs_is_identical():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        flat = _ptychodus_flat_config()
        nested = _nested_config()

        flat_projection = dataclass_to_legacy_dict(flat)
        nested_projection = dataclass_to_legacy_dict(nested)

    assert flat_projection == nested_projection
