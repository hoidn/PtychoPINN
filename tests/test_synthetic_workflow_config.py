"""Contract tests for the versioned synthetic PyTorch workflow profile."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
import importlib
import json
import re

import numpy as np
import pytest


def _api():
    return importlib.import_module("ptycho.workflows.synthetic_config")


def _resolve(**kwargs):
    return _api().resolve_synthetic_workflow(
        profile="synthetic-lines",
        **kwargs,
    )


def test_profile_records_are_frozen_and_use_exactly_five_user_namespaces():
    api = _api()
    resolved = _resolve()
    namespace_records = (
        ("simulation", api.SyntheticSimulationConfig),
        ("model", api.SyntheticModelConfig),
        ("training", api.SyntheticTrainingConfig),
        ("inference", api.SyntheticInferenceConfig),
        ("workflow", api.SyntheticWorkflowConfig),
    )

    assert is_dataclass(api.ResolvedSyntheticWorkflow)
    assert is_dataclass(resolved)
    for name, record_type in namespace_records:
        record = getattr(resolved, name)
        assert type(record) is record_type
        assert is_dataclass(record_type)
        with pytest.raises(FrozenInstanceError):
            setattr(record, fields(record)[0].name, getattr(record, fields(record)[0].name))
    with pytest.raises(FrozenInstanceError):
        resolved.profile = "other"


def test_resolved_data_identity_is_nested_immutable_and_materializes_fresh():
    from ptycho_torch.config_params import DataConfig

    api = _api()
    resolved = _resolve()
    digest = api.synthetic_workflow_sha256(resolved)

    with pytest.raises(FrozenInstanceError):
        resolved.data.gridsize = 99

    assert resolved.data.gridsize == 1
    assert api.synthetic_workflow_sha256(resolved) == digest
    first = api.materialize_data_config(resolved)
    second = api.materialize_data_config(resolved)
    assert isinstance(first, DataConfig)
    assert isinstance(second, DataConfig)
    assert first is not second
    assert {item.name: getattr(first, item.name) for item in fields(first)} == {
        item.name: getattr(resolved.data, item.name)
        for item in fields(resolved.data)
    }

    first.gridsize = 99
    assert resolved.data.gridsize == 1
    assert second.gridsize == 1


def test_workflow_digest_excludes_caller_owned_output_and_probe_locations(tmp_path):
    first_probe = tmp_path / "first" / "probe.npz"
    second_probe = tmp_path / "second" / "probe.npz"
    first_probe.parent.mkdir()
    second_probe.parent.mkdir()
    probe = np.ones((64, 64), dtype=np.complex64)
    np.savez(first_probe, probeGuess=probe)
    np.savez(second_probe, probeGuess=probe)

    def resolve(probe_path, output_root):
        return _resolve(
            file_values={
                "simulation": {
                    "N": 64,
                    "probe": {
                        "source": "custom",
                        "source_path": probe_path,
                        "transform_pipeline": "smooth:1|pad_preserve:64",
                    },
                },
                "workflow": {"output_root": output_root},
            }
        )

    api = _api()
    assert api.synthetic_workflow_sha256(
        resolve(first_probe, tmp_path / "run-a")
    ) == api.synthetic_workflow_sha256(resolve(second_probe, tmp_path / "run-b"))


def test_profile_uses_full_scan_bounds_for_groups_per_center_one():
    api = _api()
    resolved = _resolve()

    assert resolved.inference.groups_per_center == 1
    assert resolved.data.x_bounds == (0.0, 1.0)
    assert resolved.data.y_bounds == (0.0, 1.0)

    materialized = api.materialize_data_config(resolved)
    assert materialized.x_bounds == (0.0, 1.0)
    assert materialized.y_bounds == (0.0, 1.0)


def test_data_snapshot_projects_every_live_default_into_identity():
    from ptycho_torch.config_params import DataConfig

    resolved = _resolve()
    live_defaults = DataConfig()
    assert {item.name for item in fields(resolved.data)} == {
        item.name for item in fields(live_defaults)
    }
    profile_owned = {
        "nphotons",
        "scale_contract_version",
        "measurement_domain",
        "N",
        "gridsize",
        "neighbor_count",
        "n_raw_frames_selected",
        "subsample_seed",
        "x_bounds",
        "y_bounds",
        "probe_scale",
        "probe_normalize",
    }
    for item in fields(live_defaults):
        if item.name not in profile_owned:
            assert getattr(resolved.data, item.name) == getattr(
                live_defaults,
                item.name,
            )


def test_profile_precedence_and_derived_data_match_the_normative_example():
    resolved = _resolve(
        file_values={"training": {"epochs": 7}},
        cli_values={
            "gridsize": 2,
            "epochs": 5,
            "training_groups": 4096,
            "validation_groups": 1024,
        },
    )

    assert resolved.training.epochs == 5
    assert resolved.simulation.train.object.diffractions_per_object == 4096
    assert resolved.simulation.test.object.diffractions_per_object == 1024
    assert resolved.data.gridsize == 2
    assert resolved.data.neighbor_count == 4
    assert resolved.data.n_raw_frames_selected == 4096
    assert resolved.training.training_groups == 4096
    assert resolved.training.validation_groups == 1024
    assert resolved.inference.groups_per_center == 1


def test_argparse_omissions_do_not_overwrite_file_or_profile_values():
    api = _api()
    cli_values = vars(
        Namespace(
            gridsize=api.UNSET,
            epochs=api.UNSET,
            batch_size=api.UNSET,
        )
    )

    resolved = _resolve(
        file_values={
            "simulation": {"gridsize": 2},
            "training": {"epochs": 7},
        },
        cli_values=cli_values,
    )

    assert resolved.data.gridsize == 2
    assert resolved.training.epochs == 7
    assert resolved.training.batch_size == 16


def test_explicit_cli_none_clears_nullable_file_values_while_unset_omits():
    api = _api()
    resolved = _resolve(
        file_values={
            "model": {"resnet_width": 32},
            "workflow": {"logger_backend": "tensorboard"},
        },
        cli_values={
            "model": {"resnet_width": None},
            "logger_backend": None,
            "epochs": api.UNSET,
        },
    )

    assert resolved.model.resnet_width is None
    assert resolved.workflow.logger_backend is None
    assert resolved.training.epochs == 50


@pytest.mark.parametrize(
    "values",
    [
        {"unknown": 1},
        {"model": {"unknown": 1}},
        {"simulation": {"probe": {"unknown": 1}}},
        {"data": {"C": 4}},
    ],
    ids=("root", "model", "nested-probe", "derived-data-namespace"),
)
def test_unknown_or_user_authored_derived_fields_fail_closed(values):
    with pytest.raises(ValueError, match="unknown|data"):
        _resolve(file_values=values)


def test_authored_historical_data_initialization_is_rejected():
    with pytest.raises(
        ValueError,
        match=r"model\.rect_s1s2_init.*(?:ones.*dose_closure|dose_closure.*ones)",
    ):
        _resolve(file_values={"model": {"rect_s1s2_init": "data"}})


def test_conflicting_duplicate_declarations_fail():
    with pytest.raises(ValueError, match=r"training\.epochs.*conflict|conflict.*training\.epochs"):
        _resolve(
            file_values={
                "epochs": 9,
                "training": {"epochs": 7},
            }
        )


def test_full_semantic_serialization_includes_unchanged_defaults_and_one_digest():
    api = _api()
    resolved = _resolve()
    payload = api.synthetic_workflow_to_dict(resolved)

    from ptycho_torch.config_params import ModelConfig

    assert set(payload) == {item.name for item in fields(resolved)}
    for name in ("simulation", "model", "training", "inference", "workflow"):
        assert set(payload[name]) == {
            item.name for item in fields(getattr(resolved, name))
        }
    assert set(payload["data"]) == {item.name for item in fields(resolved.data)}
    assert set(payload["model"]) == {
        *(item.name for item in fields(ModelConfig)),
        "amplitude_physics_gain_provenance",
    }
    assert payload["profile"] == "synthetic-lines"
    assert payload["recipe_version"] == "synthetic-lines-v1"
    assert payload["simulation"]["train"]["probe"]["ideal_scale"] == 0.7
    assert payload["data"]["K_quadrant"] == 30
    # The GS2 quality workflow reconstructs every simulated source scan.  Keep
    # those bounds explicit in the persisted DataConfig so inference does not
    # silently discard edge scans through the legacy 10--90% defaults.
    assert payload["data"]["x_bounds"] == [0.0, 1.0]
    assert payload["data"]["y_bounds"] == [0.0, 1.0]
    assert payload["model"]["amplitude_physics_gain"] is None
    assert (
        payload["model"]["amplitude_physics_gain_provenance"]
        == "pending_training_split_derivation"
    )
    assert payload["inference"]["patch_stats_limit"] is None
    assert payload["workflow"]["prefetch_factor"] is None
    digest = api.synthetic_workflow_sha256(resolved)
    assert len(digest) == 64
    assert digest == api.synthetic_workflow_sha256(_resolve())
    assert digest != api.synthetic_workflow_sha256(
        _resolve(file_values={"training": {"epochs": 7}})
    )


def test_model_snapshot_keeps_every_unchanged_live_torch_default():
    from ptycho_torch.config_params import ModelConfig

    payload = _api().synthetic_workflow_to_dict(_resolve())["model"]
    live_defaults = ModelConfig()
    profile_owned = {
        "architecture",
        "C_model",
        "C_forward",
        "object_big",
        "object_layout",
        "training_canvas",
        "training_patch_weighting",
        "probe_big",
        "amplitude_physics_gain",
        "loss_function",
    }

    for item in fields(live_defaults):
        if item.name not in profile_owned:
            assert payload[item.name] == getattr(live_defaults, item.name)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("intensity_scale", float("nan")),
        ("intensity_scale", float("inf")),
        ("amp_loss_coeff", float("-inf")),
        ("phase_loss_coeff", float("nan")),
    ],
)
def test_all_persisted_live_torch_numbers_reject_nonfinite_values(
    field_name,
    value,
):
    with pytest.raises(ValueError, match=re.escape(f"model.{field_name}")):
        _resolve(file_values={"model": {field_name: value}})


def test_optimizer_beta_zero_boundary_is_valid():
    resolved = _resolve(
        file_values={"training": {"adam_beta1": 0.0, "adam_beta2": 0}}
    )

    assert resolved.training.adam_beta1 == 0.0
    assert resolved.training.adam_beta2 == 0


@pytest.mark.parametrize("field_name", ["adam_beta1", "adam_beta2"])
@pytest.mark.parametrize("value", [-0.1, 1.0, 1.1])
def test_optimizer_betas_reject_values_outside_half_open_unit_interval(
    field_name,
    value,
):
    with pytest.raises(ValueError, match=re.escape(f"training.{field_name}")):
        _resolve(file_values={"training": {field_name: value}})


@pytest.mark.parametrize("value", [0.0, 1.0, 1.1])
def test_plateau_factor_rejects_values_outside_open_unit_interval(value):
    with pytest.raises(ValueError, match=re.escape("training.plateau_factor")):
        _resolve(file_values={"training": {"plateau_factor": value}})


def test_default_profile_materializes_the_complete_locked_recipe():
    resolved = _resolve()

    assert resolved.schema_version == "synthetic-workflow-v1"
    assert resolved.simulation.train.N == 128
    assert resolved.simulation.train.scan.kind == "nongrid"
    assert resolved.simulation.train.scan.buffer == 64
    assert resolved.simulation.train.scan.train_groups == 1
    assert resolved.simulation.test.scan.test_groups == 1
    assert resolved.simulation.train.object.image_size == (392, 392)
    assert resolved.simulation.train.object.objects_per_probe == 1
    assert resolved.simulation.train.object.set_phi is True
    assert resolved.simulation.scale_contract_version == "legacy_v1"
    assert resolved.simulation.measurement_domain == "normalized_amplitude"
    assert resolved.training.train_raw_selection == 4096
    assert resolved.training.training_groups == 1024
    assert resolved.training.validation_groups == 1024
    assert resolved.training.neighbor_count == 4
    assert resolved.training.neighbor_pool_size == 4
    assert resolved.training.enable_oversampling is False
    assert resolved.training.sequential_sampling is False
    assert resolved.training.epochs == 50
    assert resolved.training.batch_size == 16
    assert resolved.training.learning_rate == 2e-4
    assert resolved.training.scheduler == "ReduceLROnPlateau"
    assert resolved.training.torch_loss_mode == "mae"
    assert resolved.training.torch_mae_pred_l2_match_target is True
    assert resolved.model.architecture == "cnn"
    assert resolved.model.physics_forward_mode == "amplitude"
    assert resolved.data.probe_scale == 4.0
    assert resolved.data.probe_normalize is True
    assert resolved.inference.batch_size == 16
    assert resolved.inference.reconstruction_method == "barycentric"
    assert resolved.inference.patch_weighting == "probe"
    assert resolved.inference.varpro_scaling is True
    assert resolved.workflow.accelerator == "auto"
    assert resolved.workflow.devices == 1
    assert resolved.workflow.strategy == "auto"
    assert resolved.workflow.precision == "32-true"
    assert resolved.workflow.deterministic is True
    assert resolved.workflow.num_workers == 0
    assert resolved.workflow.logger_backend == "csv"
    assert resolved.workflow.checkpoint_save_top_k == 1


@pytest.mark.parametrize(
    ("gridsize", "expected"),
    [
        (
            1,
            (False, "single_patch", "independent", "central_mask", False),
        ),
        (
            2,
            (True, "grouped_patches", "relative_overlap", "probe", True),
        ),
    ],
)
def test_gridsize_derives_object_probe_and_channel_geometry(gridsize, expected):
    resolved = _resolve(cli_values={"gridsize": gridsize})

    assert (
        resolved.model.object_big,
        resolved.model.object_layout,
        resolved.model.training_canvas,
        resolved.model.training_patch_weighting,
        resolved.model.probe_big,
    ) == expected
    assert resolved.model.pad_object is True
    assert resolved.data.gridsize == gridsize


@pytest.mark.parametrize(
    ("values", "field_name"),
    [
        ({"model": {"object_big": True}}, "model.object_big"),
        (
            {
                "simulation": {"gridsize": 2},
                "model": {"object_layout": "single_patch"},
            },
            "model.object_layout",
        ),
        (
            {
                "simulation": {"gridsize": 2},
                "model": {"training_canvas": "independent"},
            },
            "model.training_canvas",
        ),
        (
            {
                "simulation": {"gridsize": 2},
                "model": {"training_patch_weighting": "central_mask"},
            },
            "model.training_patch_weighting",
        ),
        (
            {
                "simulation": {"gridsize": 2},
                "model": {"probe_big": False},
            },
            "model.probe_big",
        ),
        (
            {
                "simulation": {
                    "gridsize": 2,
                    "scan": {"grid_size": [1, 1]},
                }
            },
            "simulation.scan.grid_size",
        ),
    ],
)
def test_explicit_geometry_contradictions_name_the_offending_field(values, field_name):
    with pytest.raises(ValueError, match=re.escape(field_name)):
        _resolve(file_values=values)


@pytest.mark.parametrize(
    ("values", "field_name"),
    [
        (
            {
                "simulation": {"gridsize": 2},
                "training": {"neighbor_count": 3},
            },
            "training.neighbor_count",
        ),
        (
            {
                "training": {
                    "neighbor_count": 5,
                    "train_raw_selection": 4,
                    "training_groups": 4,
                },
            },
            "training.train_raw_selection",
        ),
        (
            {
                "simulation": {"train_patterns": 8},
                "training": {"train_raw_selection": 9},
            },
            "training.train_raw_selection",
        ),
        (
            {"simulation": {"gridsize": 2, "test_patterns": 3}},
            "simulation.test_patterns",
        ),
        (
            {
                "training": {
                    "train_raw_selection": 8,
                    "training_groups": 9,
                    "enable_oversampling": False,
                }
            },
            "training.training_groups",
        ),
        (
            {
                "simulation": {"test_patterns": 8},
                "training": {"validation_groups": 9},
            },
            "training.validation_groups",
        ),
        (
            {
                "simulation": {"gridsize": 2},
                "training": {
                    "enable_oversampling": True,
                    "neighbor_pool_size": 3,
                },
            },
            "training.neighbor_pool_size",
        ),
    ],
)
def test_sampling_inequalities_fail_closed_and_name_the_field(values, field_name):
    with pytest.raises(ValueError, match=re.escape(field_name)):
        _resolve(file_values=values)


def test_sampling_inequalities_are_inclusive_at_the_boundary():
    resolved = _resolve(
        file_values={
            "simulation": {
                "gridsize": 2,
                "train_patterns": 4,
                "test_patterns": 4,
            },
            "training": {
                "train_raw_selection": 4,
                "training_groups": 4,
                "validation_groups": 4,
                "neighbor_count": 4,
                "neighbor_pool_size": 4,
                "enable_oversampling": True,
            },
        }
    )

    assert resolved.data.gridsize == 2
    assert resolved.data.neighbor_count == 4
    assert resolved.training.training_groups == 4
    assert resolved.training.validation_groups == 4


@pytest.mark.parametrize(
    ("values", "field_name"),
    [
        ({"simulation": {"N": 96}}, "simulation.N"),
        (
            {
                "simulation": {"gridsize": 2},
                "model": {"architecture": "fno"},
            },
            "model.architecture",
        ),
        (
            {
                "simulation": {"gridsize": 2},
                "model": {"C_model": 1},
            },
            "unknown field(s) under model: C_model",
        ),
        (
            {
                "simulation": {"gridsize": 2},
                "model": {"C_forward": 1},
            },
            "unknown field(s) under model: C_forward",
        ),
    ],
)
def test_unsupported_architecture_shape_or_channel_combinations_fail(values, field_name):
    with pytest.raises(ValueError, match=re.escape(field_name)):
        _resolve(file_values=values)


def test_neuralop_uno_rejects_non_lines128_shape_before_model_construction():
    with pytest.raises(ValueError, match=re.escape("simulation.N")):
        _resolve(
            file_values={
                "simulation": {"N": 64},
                "model": {"architecture": "neuralop_uno"},
            }
        )


@pytest.mark.parametrize("source_name", ["file_values", "cli_values"])
def test_gain_provenance_is_derived_and_rejects_user_authorship(source_name):
    with pytest.raises(
        ValueError,
        match=re.escape("model.amplitude_physics_gain_provenance"),
    ):
        _resolve(
            **{
                source_name: {
                    "model": {
                        "amplitude_physics_gain_provenance": "explicit"
                    }
                }
            }
        )


def test_gain_provenance_is_unconditionally_derived_from_final_gain_contract():
    default = _resolve()
    explicit = _resolve(
        file_values={"model": {"amplitude_physics_gain": 2.0}}
    )
    cleared = _resolve(
        file_values={"model": {"amplitude_physics_gain": 2.0}},
        cli_values={"model": {"amplitude_physics_gain": None}},
    )

    assert default.model.amplitude_physics_gain is None
    assert (
        default.model.amplitude_physics_gain_provenance
        == "pending_training_split_derivation"
    )
    assert explicit.model.amplitude_physics_gain == 2.0
    assert explicit.model.amplitude_physics_gain_provenance == "explicit"
    assert cleared.model.amplitude_physics_gain is None
    assert (
        cleared.model.amplitude_physics_gain_provenance
        == "pending_training_split_derivation"
    )


def test_cnn_ci_profile_resolves_the_complete_locked_contract():
    resolved = _api().resolve_synthetic_workflow(profile="cnn-lines-ci")

    assert resolved.profile == "cnn-lines-ci"
    assert resolved.recipe_version == "cnn-lines-ci-v1"
    assert resolved.simulation.scale_contract_version == "ci_intensity_v2"
    assert resolved.simulation.measurement_domain == "count_intensity"
    assert resolved.model.architecture == "cnn"
    assert resolved.model.physics_forward_mode == "rectangular_scaled"
    assert resolved.model.cnn_output_mode == "real_imag"
    assert resolved.model.loss_function == "Poisson"
    assert resolved.model.rect_s1s2_init == "dose_closure"
    assert resolved.model.amplitude_physics_gain == 1.0
    assert (
        resolved.model.amplitude_physics_gain_provenance
        == "scale_contract_fixed"
    )
    assert resolved.training.torch_loss_mode == "poisson"
    assert resolved.training.nll is True
    assert resolved.training.gradient_clip_val == 1.0
    assert resolved.data.scale_contract_version == "ci_intensity_v2"
    assert resolved.data.measurement_domain == "count_intensity"


def test_ffno_ci_contract_uses_generator_output_mode():
    values = {
        "simulation": {
            "scale_contract_version": "ci_intensity_v2",
            "measurement_domain": "count_intensity",
        },
        "model": {
            "architecture": "ffno",
            "physics_forward_mode": "rectangular_scaled",
            "generator_output_mode": "real_imag",
            "loss_function": "Poisson",
        },
        "training": {"torch_loss_mode": "poisson", "nll": True},
    }

    resolved = _resolve(file_values=values)
    assert resolved.model.cnn_output_mode == "amp_phase"
    assert resolved.model.generator_output_mode == "real_imag"

    values["model"].update(
        cnn_output_mode="real_imag",
        generator_output_mode="amp_phase",
    )
    with pytest.raises(ValueError, match="model.generator_output_mode"):
        _resolve(file_values=values)


@pytest.mark.parametrize("source_name", ("file_values", "cli_values"))
def test_cnn_ci_profile_rejects_a_contradicting_architecture(source_name):
    with pytest.raises(ValueError, match="model.architecture.*cnn-lines-ci"):
        _api().resolve_synthetic_workflow(
            profile="cnn-lines-ci",
            **{source_name: {"model": {"architecture": "fno"}}},
        )


def test_cnn_ci_profile_rejects_a_complete_amplitude_relabel():
    with pytest.raises(ValueError, match="scale_contract_version.*cnn-lines-ci"):
        _api().resolve_synthetic_workflow(
            profile="cnn-lines-ci",
            file_values={
                "simulation": {
                    "scale_contract_version": "legacy_v1",
                    "measurement_domain": "normalized_amplitude",
                },
                "model": {
                    "physics_forward_mode": "amplitude",
                    "cnn_output_mode": "amp_phase",
                    "loss_function": "MAE",
                    "rect_s1s2_init": "ones",
                },
                "training": {"torch_loss_mode": "mae", "nll": False},
            },
        )


def test_cnn_ci_profile_accepts_matching_explicit_locks_and_ones_control():
    api = _api()
    dose_closure = api.resolve_synthetic_workflow(profile="cnn-lines-ci")
    ones = api.resolve_synthetic_workflow(
        profile="cnn-lines-ci",
        cli_values={
            "simulation": {
                "scale_contract_version": "ci_intensity_v2",
                "measurement_domain": "count_intensity",
            },
            "model": {
                "architecture": "cnn",
                "physics_forward_mode": "rectangular_scaled",
                "cnn_output_mode": "real_imag",
                "loss_function": "Poisson",
                "rect_s1s2_init": "ones",
            },
            "training": {"torch_loss_mode": "poisson", "nll": True},
        },
    )

    assert ones.model.rect_s1s2_init == "ones"
    assert api.synthetic_workflow_sha256(ones) != (
        api.synthetic_workflow_sha256(dose_closure)
    )


def test_cnn_ci_profile_rejects_supervised_mode_with_ones_initialization():
    with pytest.raises(ValueError, match="model.mode.*Unsupervised"):
        _api().resolve_synthetic_workflow(
            profile="cnn-lines-ci",
            file_values={
                "model": {
                    "mode": "Supervised",
                    "rect_s1s2_init": "ones",
                }
            },
        )


@pytest.mark.parametrize(
    "boundary_name",
    ["synthetic_workflow_to_dict", "materialize_data_config"],
)
def test_public_boundaries_revalidate_rect_s1s2_initialization_contract(
    boundary_name,
):
    api = _api()
    resolved = _resolve()
    invalid = replace(
        resolved,
        model=replace(resolved.model, rect_s1s2_init="dose_closure"),
    )

    with pytest.raises(ValueError, match="rect_s1s2_init.*requires.*coherent"):
        getattr(api, boundary_name)(invalid)


@pytest.mark.parametrize(
    "boundary_name",
    ["synthetic_workflow_to_dict", "materialize_data_config"],
)
def test_public_boundaries_revalidate_replaced_data_snapshots(boundary_name):
    api = _api()
    resolved = _resolve()
    invalid = replace(resolved, data=replace(resolved.data, neighbor_count=0))

    with pytest.raises(ValueError, match=re.escape("data.neighbor_count")):
        getattr(api, boundary_name)(invalid)


@pytest.mark.parametrize(
    ("case", "field_name"),
    [
        ("probe", "simulation.test.probe.ideal_scale"),
        ("detector", "simulation.test.detector.photons_per_pattern"),
        ("seed", "simulation.test.seed"),
        ("object", "simulation.test.object.objects_per_probe"),
        ("scan", "simulation.test.scan.buffer"),
        ("train-groups", "simulation.train.scan.train_groups"),
        ("test-groups", "simulation.train.scan.test_groups"),
    ],
)
@pytest.mark.parametrize(
    "boundary_name",
    ["synthetic_workflow_to_dict", "materialize_data_config"],
)
def test_public_boundaries_revalidate_all_split_derivations(
    case,
    field_name,
    boundary_name,
):
    api = _api()
    resolved = _resolve()
    simulation = resolved.simulation
    train = simulation.train
    test = simulation.test

    if case == "probe":
        test = replace(test, probe=replace(test.probe, ideal_scale=1.25))
    elif case == "detector":
        test = replace(
            test,
            detector=replace(
                test.detector,
                photons_per_pattern=5e8,
            ),
        )
    elif case == "seed":
        test = replace(test, seed=4)
    elif case == "object":
        test = replace(
            test,
            object=replace(test.object, objects_per_probe=2),
        )
    elif case == "scan":
        test = replace(test, scan=replace(test.scan, buffer=63))
    elif case == "train-groups":
        train = replace(
            train,
            scan=replace(train.scan, train_groups=2),
        )
    else:
        train = replace(
            train,
            scan=replace(train.scan, test_groups=2),
        )
    invalid = replace(
        resolved,
        simulation=replace(simulation, train=train, test=test),
    )

    with pytest.raises(ValueError, match=re.escape(field_name)):
        getattr(api, boundary_name)(invalid)


@pytest.mark.parametrize(
    ("values", "field_name"),
    [
        ({"model": {"loss_function": "Poisson"}}, "model.loss_function"),
        ({"training": {"nll": True}}, "training.nll"),
        (
            {
                "training": {"torch_loss_mode": "poisson", "nll": True},
                "model": {"loss_function": "MAE"},
            },
            "model.loss_function",
        ),
        (
            {
                "training": {"torch_loss_mode": "poisson", "nll": False},
                "model": {"loss_function": "Poisson"},
            },
            "training.nll",
        ),
    ],
    ids=(
        "mae-mode-poisson-model",
        "mae-mode-nll",
        "poisson-mode-mae-model",
        "poisson-mode-without-nll",
    ),
)
def test_loss_identity_fields_cannot_contradict_one_another(values, field_name):
    with pytest.raises(ValueError, match=re.escape(field_name)):
        _resolve(file_values=values)


@pytest.mark.parametrize(
    ("values", "field_name"),
    [
        (
            {"simulation": {"measurement_domain": "count_intensity"}},
            "simulation.measurement_domain",
        ),
        (
            {"simulation": {"scale_contract_version": "ci_intensity_v2"}},
            "simulation.scale_contract_version",
        ),
        (
            {"model": {"physics_forward_mode": "rectangular_scaled"}},
            "model.physics_forward_mode",
        ),
    ],
)
def test_incoherent_scaling_and_forward_pairs_fail(values, field_name):
    with pytest.raises(ValueError, match=re.escape(field_name)):
        _resolve(file_values=values)


def test_custom_probe_requires_a_source_path():
    with pytest.raises(
        ValueError,
        match=re.escape("simulation.probe.source_path"),
    ):
        _resolve(
            file_values={
                "simulation": {
                    "probe": {"source": "custom", "source_path": None}
                }
            }
        )


@pytest.mark.parametrize(
    ("source_size", "expected_pipeline"),
    [
        (64, "smooth:0.5|pad_preserve:64"),
        (32, "pad_extrapolate:64|smooth:0.5"),
    ],
)
def test_custom_probe_omitted_transform_is_source_size_aware(
    tmp_path, source_size, expected_pipeline
):
    source = tmp_path / f"probe-{source_size}.npz"
    np.savez(
        source,
        probeGuess=np.ones((1, source_size, source_size, 1), dtype=np.complex64),
    )

    resolved = _resolve(
        file_values={
            "simulation": {
                "N": 64,
                "probe": {
                    "source": "custom",
                    "source_path": source,
                    "transform_pipeline": None,
                },
            }
        }
    )

    assert resolved.simulation.train.probe.transform_pipeline == expected_pipeline
    assert resolved.simulation.test.probe.transform_pipeline == expected_pipeline


def test_larger_custom_probe_requires_explicit_downsampling_transform(tmp_path):
    source = tmp_path / "probe-128.npz"
    np.savez(source, probeGuess=np.ones((128, 128), dtype=np.complex64))

    with pytest.raises(ValueError, match="larger.*explicit.*downsampling"):
        _resolve(
            file_values={
                "simulation": {
                    "N": 64,
                    "probe": {
                        "source": "custom",
                        "source_path": source,
                        "transform_pipeline": None,
                    },
                }
            }
        )

    with pytest.raises(ValueError, match="pad_complex.*current probe size"):
        _resolve(
            file_values={
                "simulation": {
                    "N": 64,
                    "probe": {
                        "source": "custom",
                        "source_path": source,
                        "transform_pipeline": "smooth:0.5|pad_preserve:64",
                    },
                }
            }
        )


@pytest.mark.parametrize(
    ("source_size", "pipeline"),
    [
        (32, "smooth:1|pad_preserve:64"),
        (128, "smooth:0.25|interp:64"),
    ],
)
def test_explicit_custom_probe_transform_is_validated_and_preserved(
    tmp_path, source_size, pipeline
):
    source = tmp_path / f"explicit-probe-{source_size}.npz"
    np.savez(source, probeGuess=np.ones((source_size, source_size), dtype=np.complex64))

    resolved = _resolve(
        file_values={
            "simulation": {
                "N": 64,
                "probe": {
                    "source": "custom",
                    "source_path": source,
                    "transform_pipeline": pipeline,
                },
            }
        }
    )

    assert resolved.simulation.train.probe.transform_pipeline == pipeline


def test_dose_closure_rejects_an_incomplete_ci_contract():
    with pytest.raises(ValueError, match="rect_s1s2_init.*requires.*coherent"):
        _resolve(
            cli_values={"model": {"rect_s1s2_init": "dose_closure"}},
        )


def test_unknown_profile_fails_with_the_profile_name():
    with pytest.raises(ValueError, match="missing-profile"):
        _api().resolve_synthetic_workflow(profile="missing-profile")


SEALED_SYNTHETIC_LINES_IDENTITY = {
    50: {
        "digest": "67d08a43390564516da61c4da077ca4c0cbebb42b993ed017c5550cfabddb36a",
        "payload_bytes": 5023,
    },
    5: {
        "digest": "44c89ee6a6d78b3e2b839dc5b2b45f162075198eb6ce73f98f36ebed95423f8b",
        "payload_bytes": 5022,
    },
}

@pytest.mark.parametrize("epochs", (50, 5))
def test_sealed_synthetic_lines_v1_identity_is_unchanged(epochs):
    """Adding another profile must not move the established public recipe."""

    api = _api()
    cli_values = {} if epochs == 50 else {"training": {"epochs": epochs}}
    resolved = _resolve(cli_values=cli_values)
    payload = api.synthetic_workflow_to_dict(resolved)
    payload["workflow"].pop("output_root")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    expected = SEALED_SYNTHETIC_LINES_IDENTITY[epochs]
    assert resolved.profile == "synthetic-lines"
    assert resolved.recipe_version == "synthetic-lines-v1"
    assert api.synthetic_workflow_sha256(resolved) == expected["digest"]
    assert len(encoded) == expected["payload_bytes"]


def test_cnn_lines_ci_v1_identity_is_distinct_and_round_trips():
    api = _api()
    resolved = api.resolve_synthetic_workflow(profile="cnn-lines-ci")
    payload = api.synthetic_workflow_to_dict(resolved)

    assert api.synthetic_workflow_sha256(resolved) not in {
        item["digest"] for item in SEALED_SYNTHETIC_LINES_IDENTITY.values()
    }
    assert api.synthetic_workflow_sha256(payload) == (
        api.synthetic_workflow_sha256(resolved)
    )
