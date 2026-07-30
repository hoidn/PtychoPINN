"""Contract checks for retired unversioned configuration entry points."""

import pytest


def test_runner_modules_do_not_expose_retired_configuration_loaders():
    import ptycho_torch.inference as inference
    import ptycho_torch.train as train

    for name in ("main", "main_lightning"):
        assert not hasattr(train, name)
    for name in ("load_all_configs", "load_and_predict", "plot_amp_and_phase"):
        assert not hasattr(inference, name)


def test_utils_do_not_rehydrate_configuration_from_unversioned_sources():
    import ptycho_torch.utils as utils

    for name in (
        "_load_single_config_from_mlflow",
        "load_all_configs_from_mlflow",
        "fix_attribute",
        "load_config_from_json",
        "validate_and_process_config",
    ):
        assert not hasattr(utils, name)


def test_lightning_runner_requires_resolved_five_record_tuple():
    from ptycho_torch import train_lightning_only

    with pytest.raises(
        TypeError,
        match="existing_config must be a five-member resolved config tuple",
    ):
        train_lightning_only.main("unused", existing_config=None)
