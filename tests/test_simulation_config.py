"""Contract tests for the canonical simulation configuration boundary."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest


def _api():
    import ptycho.config.config as config_module

    required = (
        "ProbeSimulationConfig",
        "SyntheticObjectConfig",
        "ScanSimulationConfig",
        "DetectorSimulationConfig",
        "SimulationConfig",
        "simulation_config_from_mapping",
        "simulation_config_to_dict",
        "simulation_config_sha256",
        "validate_simulation_config",
        "validate_simulation_compatibility",
    )
    missing = [name for name in required if not hasattr(config_module, name)]
    assert not missing, f"simulation config API is missing {missing}"
    return config_module


def test_config_package_exports_four_public_families_and_legacy_bridge():
    import ptycho.config as public_config

    for name in (
        "SimulationConfig",
        "ModelConfig",
        "TrainingConfig",
        "InferenceConfig",
        "PyTorchExecutionConfig",
        "update_legacy_dict",
    ):
        assert hasattr(public_config, name), name


def test_simulation_config_nested_defaults_are_independent_and_valid():
    api = _api()

    first = api.SimulationConfig()
    second = api.SimulationConfig()

    assert first.probe is not second.probe
    assert first.object is not second.object
    assert first.scan is not second.scan
    assert first.detector is not second.detector
    api.validate_simulation_config(first)


def test_simulation_config_from_mapping_converts_nested_paths_and_round_trips():
    api = _api()
    mapping = {
        "N": 128,
        "seed": 3,
        "probe": {
            "source": "custom",
            "source_path": "datasets/probe.npz",
            "transform_pipeline": "smooth:0.5|pad_extrapolate_boundary_matched:128",
            "mask_diameter": None,
        },
        "object": {
            "kind": "dead_leaves",
            "image_size": [392, 392],
            "objects_per_probe": 6,
            "diffractions_per_object": 128,
            "set_phi": True,
        },
        "scan": {
            "kind": "grid",
            "grid_size": [2, 2],
            "offset": 4,
            "outer_offset_train": 8,
            "outer_offset_test": 20,
            "train_groups": 3,
            "test_groups": 2,
            "buffer": 0,
        },
        "detector": {
            "photons_per_pattern": 1e8,
            "beamstop_diameter": 4.0,
        },
    }

    config = api.simulation_config_from_mapping(mapping)
    assert config.probe.source_path == Path("datasets/probe.npz")
    assert config.object.image_size == (392, 392)
    assert config.scan.grid_size == (2, 2)
    assert api.simulation_config_to_dict(config) == mapping
    assert api.simulation_config_from_mapping(
        api.simulation_config_to_dict(config)
    ) == config


def test_simulation_config_digest_is_stable_and_changes_with_recipe():
    api = _api()
    first = api.SimulationConfig()
    equivalent = api.simulation_config_from_mapping(
        api.simulation_config_to_dict(first)
    )
    changed = replace(first, seed=3)

    assert api.simulation_config_sha256(first) == api.simulation_config_sha256(
        equivalent
    )
    assert len(api.simulation_config_sha256(first)) == 64
    assert api.simulation_config_sha256(first) != api.simulation_config_sha256(
        changed
    )


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({"N": True}, "simulation.N"),
        ({"object": {"image_size": [392.5, 392.5]}}, "simulation.object.image_size"),
        ({"object": {"objects_per_probe": True}}, "objects_per_probe"),
        ({"detector": {"photons_per_pattern": "1e9"}}, "photons_per_pattern"),
        ({"seed": True}, "simulation.seed"),
    ],
)
def test_simulation_config_rejects_coercive_or_boolean_numeric_inputs(mapping, message):
    api = _api()
    with pytest.raises(ValueError, match=message):
        api.simulation_config_from_mapping(mapping)


def test_boundary_matched_probe_operation_must_be_terminal():
    api = _api()
    with pytest.raises(ValueError, match="must be the final operation"):
        api.simulation_config_from_mapping(
            {
                "N": 128,
                "probe": {
                    "transform_pipeline": (
                        "pad_extrapolate_boundary_matched:128|pad_preserve:128"
                    )
                },
            }
        )


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({"epochs": 10}, "simulation.epochs"),
        ({"training": {"batch_size": 4}}, "simulation.training"),
        ({"probe": {"unknown": True}}, "simulation.probe.unknown"),
        ({"object": {"optimizer": "adam"}}, "simulation.object.optimizer"),
    ],
)
def test_simulation_config_from_mapping_rejects_unknown_and_training_fields(
    mapping,
    message,
):
    api = _api()
    with pytest.raises(ValueError, match=message):
        api.simulation_config_from_mapping(mapping)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda api, cfg: replace(cfg, N=0), "simulation.N"),
        (
            lambda api, cfg: replace(
                cfg,
                object=replace(cfg.object, image_size=(392, 384)),
            ),
            "simulation.object.image_size",
        ),
        (
            lambda api, cfg: replace(
                cfg,
                scan=replace(cfg.scan, grid_size=(1, 2)),
            ),
            "simulation.scan.grid_size",
        ),
        (
            lambda api, cfg: replace(
                cfg,
                detector=replace(cfg.detector, photons_per_pattern=0),
            ),
            "simulation.detector.photons_per_pattern",
        ),
        (
            lambda api, cfg: replace(
                cfg,
                object=replace(cfg.object, objects_per_probe=0),
            ),
            "simulation.object.objects_per_probe",
        ),
        (
            lambda api, cfg: replace(
                cfg,
                object=replace(cfg.object, diffractions_per_object=0),
            ),
            "simulation.object.diffractions_per_object",
        ),
        (
            lambda api, cfg: replace(
                cfg,
                probe=api.ProbeSimulationConfig(
                    source="ideal",
                    source_path=Path("probe.npz"),
                    transform_pipeline="pad_preserve:64",
                ),
            ),
            "simulation.probe.source_path",
        ),
        (
            lambda api, cfg: replace(
                cfg,
                probe=replace(cfg.probe, transform_pipeline="pad_preserve:128"),
            ),
            "final size 128.*simulation.N 64",
        ),
    ],
)
def test_validate_simulation_config_rejects_invalid_recipes(mutator, message):
    api = _api()
    invalid = mutator(api, api.SimulationConfig())
    with pytest.raises(ValueError, match=message):
        api.validate_simulation_config(invalid)


def test_validate_simulation_compatibility_rejects_model_shape_conflicts():
    api = _api()
    simulation = api.SimulationConfig(
        N=128,
        probe=api.ProbeSimulationConfig(transform_pipeline="pad_preserve:128"),
        scan=api.ScanSimulationConfig(grid_size=(2, 2)),
    )

    with pytest.raises(ValueError, match=r"simulation.N=128.*model.N=64"):
        api.validate_simulation_compatibility(
            simulation,
            api.ModelConfig(N=64, gridsize=2),
        )
    with pytest.raises(
        ValueError,
        match=r"simulation.scan.grid_size=\(2, 2\).*model.gridsize=1",
    ):
        api.validate_simulation_compatibility(
            simulation,
            api.ModelConfig(N=128, gridsize=1),
        )


def test_simulation_config_legacy_bridge_maps_only_generation_owned_fields():
    api = _api()
    config = api.simulation_config_from_mapping(
        {
            "N": 128,
            "seed": 7,
            "probe": {
                "source": "custom",
                "source_path": "probe.npz",
                "transform_pipeline": "smooth:0.5|pad_preserve:128",
                "mask_diameter": 100.0,
            },
            "object": {
                "kind": "dead_leaves",
                "image_size": [392, 392],
                "objects_per_probe": 5,
                "diffractions_per_object": 64,
                "set_phi": True,
            },
            "scan": {
                "kind": "grid",
                "grid_size": [2, 2],
                "offset": 4,
                "outer_offset_train": 8,
                "outer_offset_test": 20,
                "train_groups": 9,
                "test_groups": 3,
                "buffer": 1,
            },
            "detector": {
                "photons_per_pattern": 1e8,
                "beamstop_diameter": 4.0,
            },
        }
    )
    legacy = {"optimizer": "leave-me-alone"}

    api.update_legacy_dict(legacy, config)

    assert legacy == {
        "optimizer": "leave-me-alone",
        "N": 128,
        "probe_source": "custom",
        "probe_npz": "probe.npz",
        "probe_transform_pipeline": "smooth:0.5|pad_preserve:128",
        "probe_mask_diameter": 100.0,
        "data_source": "dead_leaves",
        "object_class": "dead_leaves",
        "size": 392,
        "objects_per_probe": 5,
        "diff_per_object": 64,
        "set_phi": True,
        "scan_kind": "grid",
        "gridsize": 2,
        "offset": 4,
        "outer_offset_train": 8,
        "outer_offset_test": 20,
        "nimgs_train": 9,
        "nimgs_test": 3,
        "max_position_jitter": 1,
        "nphotons": 1e8,
        "beamstop_diameter": 4.0,
        "npseed": 7,
    }
