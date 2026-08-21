"""Contract tests for the canonical simulation configuration boundary."""

from __future__ import annotations

from collections import UserDict, deque
from dataclasses import FrozenInstanceError, asdict, fields, is_dataclass, replace
from decimal import Decimal
from enum import Enum, IntEnum
from fractions import Fraction
from inspect import Parameter, signature
from pathlib import Path
import re

import numpy as np
import pytest


class _ProbeSourceEnum(str, Enum):
    CUSTOM = "custom"


class _ObjectKindString(str):
    pass


class _PipelineString(str):
    pass


class _NumericEnum(IntEnum):
    FOUR = 4


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
        "dataclass_to_legacy_dict",
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


def test_probe_ideal_scale_default_and_positive_override_are_semantic():
    api = _api()
    default = api.simulation_config_from_mapping(
        {
            "probe": {
                "source": "ideal",
                "ideal_scale": 0.7,
            }
        }
    )
    overridden = api.simulation_config_from_mapping(
        {
            "probe": {
                "source": "ideal",
                "ideal_scale": 1.25,
            }
        }
    )

    assert api.simulation_config_to_dict(default)["probe"]["ideal_scale"] == 0.7
    assert api.simulation_config_to_dict(overridden)["probe"]["ideal_scale"] == 1.25
    default_payload = api.simulation_config_to_dict(default)
    overridden_payload = api.simulation_config_to_dict(overridden)
    assert {
        **overridden_payload,
        "probe": {
            **overridden_payload["probe"],
            "ideal_scale": 0.7,
        },
    } == default_payload
    assert api.simulation_config_from_mapping(
        api.simulation_config_to_dict(overridden)
    ) == overridden
    assert api.simulation_config_sha256(default) != api.simulation_config_sha256(
        overridden
    )


@pytest.mark.parametrize(
    "value",
    [0, -0.1, float("nan"), float("inf"), True, "1.0"],
    ids=("zero", "negative", "nan", "infinity", "bool", "string"),
)
def test_probe_ideal_scale_rejects_nonpositive_nonfinite_or_coercive_values(value):
    api = _api()

    with pytest.raises(
        ValueError,
        match=re.escape("simulation.probe.ideal_scale"),
    ):
        api.simulation_config_from_mapping(
            {"probe": {"source": "ideal", "ideal_scale": value}}
        )


def test_custom_probe_rejects_a_nondefault_ideal_scale():
    api = _api()

    with pytest.raises(
        ValueError,
        match=re.escape("simulation.probe.ideal_scale"),
    ):
        api.simulation_config_from_mapping(
            {
                "probe": {
                    "source": "custom",
                    "source_path": "probe.npz",
                    "ideal_scale": 1.0,
                }
            }
        )


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
            "ideal_scale": 0.7,
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
    assert type(config.probe.source) is str
    assert type(config.object.kind) is str
    assert type(config.scan.kind) is str
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
        ({"N": "64"}, "simulation.N"),
        ({"N": True}, "simulation.N"),
        ({"object": {"image_size": [392.5, 392.5]}}, "simulation.object.image_size"),
        (
            {"object": {"objects_per_probe": True}},
            "simulation.object.objects_per_probe",
        ),
        (
            {"detector": {"photons_per_pattern": "1e9"}},
            "simulation.detector.photons_per_pattern",
        ),
        ({"seed": True}, "simulation.seed"),
        ({"object": {"set_phi": 1}}, "simulation.object.set_phi"),
        ({"object": {"set_phi": "true"}}, "simulation.object.set_phi"),
        ({"object": {"kind": "LINES"}}, "simulation.object.kind"),
        ({"scan": {"offset": 1.0}}, "simulation.scan.offset"),
        (
            {"detector": {"photons_per_pattern": float("inf")}},
            "simulation.detector.photons_per_pattern",
        ),
    ],
    ids=(
        "N-string",
        "N-bool",
        "image-size-float",
        "objects-per-probe-bool",
        "photons-string",
        "seed-bool",
        "set-phi-int",
        "set-phi-string",
        "object-kind-case",
        "scan-offset-float",
        "photons-infinity",
    ),
)
def test_simulation_config_rejects_coercive_or_boolean_numeric_inputs(mapping, message):
    api = _api()
    with pytest.raises(ValueError, match=re.escape(message)):
        api.simulation_config_from_mapping(mapping)


@pytest.mark.parametrize(
    "pipeline",
    [
        b"pad_preserve:64",
        _PipelineString("pad_preserve:64"),
    ],
    ids=("bytes", "str-subclass"),
)
def test_simulation_config_from_mapping_rejects_coerced_probe_pipeline(pipeline):
    api = _api()

    with pytest.raises(
        ValueError,
        match=re.escape("simulation.probe.transform_pipeline"),
    ):
        api.simulation_config_from_mapping(
            {"probe": {"transform_pipeline": pipeline}}
        )


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
        ({"scan": {"spacing": 4}}, "simulation.scan.spacing"),
        ({"detector": {"gain": 2.0}}, "simulation.detector.gain"),
    ],
)
def test_simulation_config_from_mapping_rejects_unknown_and_training_fields(
    mapping,
    message,
):
    api = _api()
    with pytest.raises(ValueError, match=message):
        api.simulation_config_from_mapping(mapping)


def test_simulation_config_from_mapping_accepts_generic_mapping_implementations():
    api = _api()
    mapping = UserDict(
        {
            "N": 128,
            "probe": UserDict(
                {
                    "source": "custom",
                    "source_path": "probe.npz",
                    "transform_pipeline": "pad_preserve:128",
                }
            ),
        }
    )

    config = api.simulation_config_from_mapping(mapping)

    assert config.N == 128
    assert config.probe.source_path == Path("probe.npz")


@pytest.mark.parametrize(
    ("mapping", "path"),
    [
        (
            {"object": {"image_size": {392, 393}}},
            "simulation.object.image_size",
        ),
        (
            {"scan": {"grid_size": deque([1, 1])}},
            "simulation.scan.grid_size",
        ),
    ],
)
def test_simulation_config_rejects_non_sequence_dimension_pairs(mapping, path):
    api = _api()

    with pytest.raises(ValueError, match=re.escape(path)):
        api.simulation_config_from_mapping(mapping)


def test_validate_simulation_config_rejects_non_builtin_closed_domain_strings():
    api = _api()
    config = api.SimulationConfig(
        probe=api.ProbeSimulationConfig(source=_ProbeSourceEnum.CUSTOM),
        object=api.SyntheticObjectConfig(kind=_ObjectKindString("lines")),
    )

    assert config.probe.source is _ProbeSourceEnum.CUSTOM
    assert type(config.object.kind) is _ObjectKindString
    with pytest.raises(ValueError) as exc_info:
        api.validate_simulation_config(config)
    message = str(exc_info.value)
    assert "simulation.probe.source" in message
    assert "simulation.object.kind" in message


@pytest.mark.parametrize(
    ("section", "field_name", "value", "expected_sha256"),
    [
        (
            "probe",
            "mask_diameter",
            4,
            "a3a32e630aaeae47f59bc4d2c2b512ec9e9a4a07d3b1153386c4c1d71c35f3fc",
        ),
        (
            "probe",
            "mask_diameter",
            4.0,
            "0d35379b6dee35a6af4a0d35ef0fe3942053f79976b058ed0ac680407c390f45",
        ),
        (
            "detector",
            "beamstop_diameter",
            4,
            "fc05017036fd2003d46bdbb164efe08cc9ce5067980351c7932d9db5b73fbaa6",
        ),
        (
            "detector",
            "beamstop_diameter",
            4.0,
            "8c9539a5ca537ae4b21a74cb2e37229dfe4548a32cc737e4d5600bee3ec51018",
        ),
    ],
)
def test_simulation_numeric_identity_preserves_int_float_kind_and_digest(
    section,
    field_name,
    value,
    expected_sha256,
):
    api = _api()
    config = api.simulation_config_from_mapping(
        {section: {field_name: value}}
    )

    stored = api.simulation_config_to_dict(config)[section][field_name]
    assert stored == value
    assert type(stored) is type(value)
    assert api.simulation_config_sha256(config) == expected_sha256


@pytest.mark.parametrize(
    "value",
    [
        Decimal("4"),
        Fraction(4, 1),
        _NumericEnum.FOUR,
        np.int64(4),
        np.float32(4),
        np.float64(4),
    ],
    ids=(
        "decimal",
        "fraction",
        "int-enum",
        "numpy-int64",
        "numpy-float32",
        "numpy-float64",
    ),
)
def test_simulation_mapping_rejects_non_builtin_numeric_kinds(value):
    api = _api()

    with pytest.raises(
        ValueError,
        match=re.escape("simulation.probe.mask_diameter"),
    ):
        api.simulation_config_from_mapping(
            {"probe": {"mask_diameter": value}}
        )


@pytest.mark.parametrize(
    "value",
    [
        Decimal("4"),
        Fraction(4, 1),
        _NumericEnum.FOUR,
        np.int64(4),
        np.float32(4),
        np.float64(4),
    ],
    ids=(
        "decimal",
        "fraction",
        "int-enum",
        "numpy-int64",
        "numpy-float32",
        "numpy-float64",
    ),
)
def test_validate_simulation_config_rejects_non_builtin_numeric_kinds(value):
    api = _api()
    config = api.SimulationConfig(
        probe=api.ProbeSimulationConfig(mask_diameter=value)
    )

    assert config.probe.mask_diameter is value
    with pytest.raises(
        ValueError,
        match=re.escape("simulation.probe.mask_diameter"),
    ):
        api.validate_simulation_config(config)


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
                object=replace(cfg.object, set_phi=1),
            ),
            "simulation.object.set_phi",
        ),
        (
            lambda api, cfg: replace(
                cfg,
                object=replace(cfg.object, set_phi="true"),
            ),
            "simulation.object.set_phi",
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
    ids=(
        "N-zero",
        "object-nonsquare",
        "scan-nonsquare",
        "photons-zero",
        "objects-per-probe-zero",
        "diffractions-per-object-zero",
        "set-phi-int",
        "set-phi-string",
        "ideal-source-path",
        "pipeline-size-mismatch",
    ),
)
def test_validate_simulation_config_rejects_invalid_recipes(mutator, message):
    api = _api()
    invalid = mutator(api, api.SimulationConfig())
    with pytest.raises(ValueError, match=message):
        api.validate_simulation_config(invalid)


def test_direct_construction_and_replace_remain_non_validating():
    api = _api()

    directly_constructed = api.SimulationConfig(N=0)
    replaced = replace(api.SimulationConfig(), N=0)

    assert directly_constructed.N == 0
    assert replaced.N == 0
    for invalid in (directly_constructed, replaced):
        with pytest.raises(ValueError, match=re.escape("simulation.N")):
            api.validate_simulation_config(invalid)


@pytest.mark.parametrize(
    ("record_name", "field_names"),
    [
        (
            "ProbeSimulationConfig",
            (
                "source",
                "source_path",
                "transform_pipeline",
                "mask_diameter",
                "ideal_scale",
            ),
        ),
        (
            "SyntheticObjectConfig",
            (
                "kind",
                "image_size",
                "objects_per_probe",
                "diffractions_per_object",
                "set_phi",
            ),
        ),
        (
            "ScanSimulationConfig",
            (
                "kind",
                "grid_size",
                "offset",
                "outer_offset_train",
                "outer_offset_test",
                "train_groups",
                "test_groups",
                "buffer",
            ),
        ),
        (
            "DetectorSimulationConfig",
            ("photons_per_pattern", "beamstop_diameter"),
        ),
        (
            "SimulationConfig",
            ("N", "probe", "object", "scan", "detector", "seed"),
        ),
    ],
)
def test_simulation_records_retain_standard_dataclass_fields_and_signatures(
    record_name,
    field_names,
):
    api = _api()
    record_type = getattr(api, record_name)

    assert is_dataclass(record_type)
    assert is_dataclass(record_type())
    assert tuple(item.name for item in fields(record_type)) == field_names
    parameters = tuple(signature(record_type).parameters.values())
    assert tuple(item.name for item in parameters) == field_names
    assert all(item.kind is Parameter.POSITIONAL_OR_KEYWORD for item in parameters)


def test_simulation_records_retain_positional_frozen_value_semantics():
    api = _api()
    probe = api.ProbeSimulationConfig(
        "custom",
        Path("probe.npz"),
        "pad_preserve:128",
        4,
        0.9,
    )
    object_config = api.SyntheticObjectConfig(
        "dead_leaves",
        (512, 512),
        5,
        64,
        True,
    )
    scan = api.ScanSimulationConfig("grid", (2, 2), 4, 8, 20, 9, 3, 1)
    detector = api.DetectorSimulationConfig(1e8, 4.0)
    config = api.SimulationConfig(
        128,
        probe,
        object_config,
        scan,
        detector,
        7,
    )
    equivalent = api.SimulationConfig(
        128,
        probe,
        object_config,
        scan,
        detector,
        7,
    )

    assert config == equivalent
    assert hash(config) == hash(equivalent)
    assert asdict(config) == {
        "N": 128,
        "probe": {
            "source": "custom",
            "source_path": Path("probe.npz"),
            "transform_pipeline": "pad_preserve:128",
            "mask_diameter": 4,
            "ideal_scale": 0.9,
        },
        "object": {
            "kind": "dead_leaves",
            "image_size": (512, 512),
            "objects_per_probe": 5,
            "diffractions_per_object": 64,
            "set_phi": True,
        },
        "scan": {
            "kind": "grid",
            "grid_size": (2, 2),
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
        "seed": 7,
    }
    with pytest.raises(FrozenInstanceError):
        config.seed = 8
    changed = replace(config, seed=8)
    assert config.seed == 7
    assert changed.seed == 8
    assert changed != config


def test_default_simulation_canonical_dictionary_and_digest_are_exact():
    api = _api()
    config = api.SimulationConfig()

    assert api.simulation_config_to_dict(config) == {
        "N": 64,
        "seed": None,
        "probe": {
            "source": "custom",
            "source_path": None,
            "transform_pipeline": "pad_preserve:64",
            "mask_diameter": None,
            "ideal_scale": 0.7,
        },
        "object": {
            "kind": "lines",
            "image_size": [392, 392],
            "objects_per_probe": 4,
            "diffractions_per_object": 7000,
            "set_phi": False,
        },
        "scan": {
            "kind": "grid",
            "grid_size": [1, 1],
            "offset": 4,
            "outer_offset_train": 8,
            "outer_offset_test": 20,
            "train_groups": 2,
            "test_groups": 2,
            "buffer": 0,
        },
        "detector": {
            "photons_per_pattern": 1_000_000_000.0,
            "beamstop_diameter": None,
        },
    }
    assert api.simulation_config_sha256(config) == (
        "f149d2d29e2e105643f9ee44087e3e0a562b9621be24f210301194302348772d"
    )


def test_validate_simulation_compatibility_rejects_model_shape_conflicts():
    api = _api()
    simulation = api.SimulationConfig(
        N=128,
        probe=api.ProbeSimulationConfig(transform_pipeline="pad_preserve:128"),
        scan=api.ScanSimulationConfig(gridsize=2),
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
    expected_projection = {
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
    assert api.dataclass_to_legacy_dict(config) == expected_projection

    legacy = {"optimizer": "leave-me-alone"}
    api.update_legacy_dict(legacy, config)

    assert legacy == {
        "optimizer": "leave-me-alone",
        **expected_projection,
    }
