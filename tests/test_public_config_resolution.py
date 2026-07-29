from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import warnings

import pytest

import ptycho.config as public_config
import ptycho.params as params
from ptycho.config.config import InferenceConfig, ModelConfig, TrainingConfig
from ptycho.config.resolution import (
    resolve_inference_config,
    resolve_training_config,
    validate_inference_config_structure,
    validate_inference_resources,
    validate_model_config_structure,
    validate_runnable_training_config,
    validate_training_config_structure,
)


_ARCHITECTURES = (
    "cnn",
    "ffno",
    "fno",
    "hybrid",
    "stable_hybrid",
    "fno_vanilla",
    "neuralop_uno",
    "hybrid_resnet",
    "hybrid_resnet_ffno_ptychoblock_encoder",
    "hybrid_resnet_ptychoblock_ffno_encoder",
    "spectral_resnet_bottleneck_net",
    "spectral_resnet_bottleneck_linear_decoder",
    "hybrid_resnet_ffno_bottleneck",
    "hybrid_resnet_convnext_bottleneck",
)


def _inference_source(**updates):
    values = {
        "model_path": "models/example",
        "test_data_file": "data/test.npz",
    }
    values.update(updates)
    return values


_RESOLVER_CASES = (
    pytest.param(resolve_training_config, lambda **updates: updates, id="training"),
    pytest.param(resolve_inference_config, _inference_source, id="inference"),
)


def test_training_file_value_survives_omitted_cli_value():
    config = resolve_training_config(
        {"nepochs": 9, "model": {"N": 128}},
        {},
    )

    assert config.nepochs == 9
    assert config.model.N == 128


def test_explicit_cli_value_overrides_file_value():
    config = resolve_training_config({"nepochs": 9}, {"nepochs": 3})

    assert config.nepochs == 3


@pytest.mark.parametrize("resolver,source_factory", _RESOLVER_CASES)
def test_flat_and_nested_model_values_follow_conflict_rule(
    resolver,
    source_factory,
):
    accepted = resolver(
        source_factory(N=128, model={"N": 128}),
        {},
    )
    assert accepted.model.N == 128

    with pytest.raises(ValueError, match="N.*flat.*model"):
        resolver(source_factory(N=64, model={"N": 128}), {})


@pytest.mark.parametrize("flat,nested", [(True, 1), (1, True)])
def test_flat_and_nested_equality_is_type_exact(flat, nested):
    with pytest.raises(ValueError, match="probe_big.*flat.*model"):
        resolve_training_config(
            {"probe_big": flat, "model": {"probe_big": nested}},
            {},
        )


def test_cli_flat_model_value_overrides_file_nested_value():
    config = resolve_training_config(
        {"model": {"N": 64}},
        {"N": 128},
    )

    assert config.model.N == 128


def test_cli_nested_model_value_overrides_file_flat_value():
    config = resolve_training_config(
        {"N": 64},
        {"model": {"N": 128}},
    )

    assert config.model.N == 128


def test_inference_root_values_are_resolved():
    config = resolve_inference_config(
        _inference_source(
            n_groups=9,
            neighbor_count=7,
            subsample_seed=123,
            debug=True,
            output_dir="yaml-output",
            backend="pytorch",
        ),
        {"n_groups": 3, "debug": False},
    )

    assert config.n_groups == 3
    assert config.neighbor_count == 7
    assert config.subsample_seed == 123
    assert config.debug is False
    assert config.output_dir == Path("yaml-output")
    assert config.backend == "pytorch"


@pytest.mark.parametrize("resolver,source_factory", _RESOLVER_CASES)
def test_n_images_is_canonicalized_at_the_boundary(resolver, source_factory):
    with pytest.warns(DeprecationWarning) as caught:
        config = resolver(source_factory(n_images=7), {})

    assert len(caught) == 1
    assert config.n_groups == 7
    assert config.n_images is None


@pytest.mark.parametrize("resolver,source_factory", _RESOLVER_CASES)
def test_equal_group_aliases_are_accepted_once(resolver, source_factory):
    with pytest.warns(DeprecationWarning) as caught:
        config = resolver(
            source_factory(n_images=7, n_groups=7),
            {},
        )

    assert len(caught) == 1
    assert config.n_groups == 7
    assert config.n_images is None


@pytest.mark.parametrize("resolver,source_factory", _RESOLVER_CASES)
def test_conflicting_group_aliases_fail_without_warning(
    resolver,
    source_factory,
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="n_images.*n_groups"):
            resolver(source_factory(n_images=7, n_groups=9), {})

    assert caught == []


@pytest.mark.parametrize("resolver,source_factory", _RESOLVER_CASES)
def test_explicit_cli_group_field_wins_across_aliases(
    resolver,
    source_factory,
):
    with pytest.warns(DeprecationWarning):
        canonical = resolver(
            source_factory(n_images=7),
            {"n_groups": 11},
        )
    with pytest.warns(DeprecationWarning):
        alias = resolver(
            source_factory(n_groups=7),
            {"n_images": 11},
        )

    assert canonical.n_groups == 11
    assert alias.n_groups == 11
    assert canonical.n_images is alias.n_images is None


@pytest.mark.parametrize("resolver,source_factory", _RESOLVER_CASES)
def test_failed_resolution_does_not_emit_alias_warning(
    resolver,
    source_factory,
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="gridsize"):
            resolver(source_factory(n_images=7, gridsize=0), {})

    assert caught == []


@pytest.mark.parametrize(
    "resolver,source",
    [
        (resolve_training_config, {"zeta_unknown": 1, "alpha_unknown": 2}),
        (
            resolve_inference_config,
            _inference_source(zeta_unknown=1, alpha_unknown=2),
        ),
    ],
)
def test_unknown_root_names_fail_deterministically(resolver, source):
    with pytest.raises(ValueError) as error:
        resolver(source, {})

    message = str(error.value)
    assert message.index("alpha_unknown") < message.index("zeta_unknown")


@pytest.mark.parametrize(
    "resolver,source",
    [
        (
            resolve_training_config,
            {"model": {"zeta_unknown": 1, "alpha_unknown": 2}},
        ),
        (
            resolve_inference_config,
            _inference_source(model={"zeta_unknown": 1, "alpha_unknown": 2}),
        ),
    ],
)
def test_unknown_nested_model_names_fail_deterministically(resolver, source):
    with pytest.raises(ValueError) as error:
        resolver(source, {})

    message = str(error.value)
    assert message.index("alpha_unknown") < message.index("zeta_unknown")


@pytest.mark.parametrize("resolver,source_factory", _RESOLVER_CASES)
def test_nested_model_must_be_a_mapping(resolver, source_factory):
    with pytest.raises(ValueError, match="model.*mapping"):
        resolver(source_factory(model=[]), {})


@pytest.mark.parametrize("resolver,source_factory", _RESOLVER_CASES)
def test_only_none_means_an_absent_mapping_source(resolver, source_factory):
    resolver(source_factory(), None)

    with pytest.raises(ValueError, match="mapping"):
        resolver([], None)
    with pytest.raises(ValueError, match="mapping"):
        resolver(source_factory(), [])


@pytest.mark.parametrize(
    "resolver,file_values,cli_values",
    [
        (
            resolve_training_config,
            {"model": {"N": 64}, "nepochs": 9},
            {"N": 128, "output_dir": "cli-output"},
        ),
        (
            resolve_inference_config,
            _inference_source(model={"N": 64}, n_groups=9),
            {"N": 128, "output_dir": "cli-output"},
        ),
    ],
)
def test_source_mappings_are_unchanged(resolver, file_values, cli_values):
    original_file = deepcopy(file_values)
    original_cli = deepcopy(cli_values)

    resolver(file_values, cli_values)

    assert file_values == original_file
    assert cli_values == original_cli


def test_training_path_fields_are_materialized_as_paths():
    config = resolve_training_config(
        {
            "train_data_file": "data/train.npz",
            "test_data_file": "data/test.npz",
            "output_dir": "outputs/train",
        },
        {},
    )

    assert config.train_data_file == Path("data/train.npz")
    assert config.test_data_file == Path("data/test.npz")
    assert config.output_dir == Path("outputs/train")
    assert all(
        isinstance(value, Path)
        for value in (
            config.train_data_file,
            config.test_data_file,
            config.output_dir,
        )
    )


def test_inference_path_fields_are_materialized_as_paths():
    config = resolve_inference_config(
        _inference_source(output_dir="outputs/inference"),
        {},
    )

    assert config.model_path == Path("models/example")
    assert config.test_data_file == Path("data/test.npz")
    assert config.output_dir == Path("outputs/inference")


@pytest.mark.parametrize("resolver,source_factory", _RESOLVER_CASES)
def test_invalid_path_source_fails(resolver, source_factory):
    field = "output_dir"
    with pytest.raises(ValueError, match=field):
        resolver(source_factory(**{field: 3}), {})


@pytest.mark.parametrize("resolver,source_factory", _RESOLVER_CASES)
def test_resolvers_return_fresh_dataclasses(resolver, source_factory):
    first = resolver(source_factory(model={"N": 128}), {})
    second = resolver(source_factory(model={"N": 128}), {})

    assert first == second
    assert first is not second
    assert first.model is not second.model


def test_training_dataclass_source_is_copied_and_can_be_overridden():
    source = TrainingConfig(
        model=ModelConfig(N=64),
        train_data_file=Path("data/train.npz"),
        nepochs=9,
    )

    resolved = resolve_training_config(source, {"N": 128, "nepochs": 3})

    assert resolved is not source
    assert resolved.model is not source.model
    assert resolved.model.N == 128
    assert resolved.nepochs == 3
    assert source.model.N == 64
    assert source.nepochs == 9


def test_inference_dataclass_source_is_copied_and_can_be_overridden():
    source = InferenceConfig(
        model=ModelConfig(N=64),
        model_path=Path("models/example"),
        test_data_file=Path("data/test.npz"),
        n_groups=9,
    )

    resolved = resolve_inference_config(source, {"N": 128, "n_groups": 3})

    assert resolved is not source
    assert resolved.model is not source.model
    assert resolved.model.N == 128
    assert resolved.n_groups == 3
    assert source.model.N == 64
    assert source.n_groups == 9


def test_wrong_dataclass_family_is_rejected():
    training = TrainingConfig(model=ModelConfig())

    with pytest.raises(ValueError, match="InferenceConfig|mapping"):
        resolve_inference_config(training, {})


@pytest.mark.parametrize(
    "source,resolver",
    [
        (TrainingConfig(model=ModelConfig()), resolve_training_config),
        (
            InferenceConfig(
                model=ModelConfig(),
                model_path=Path("models/example"),
                test_data_file=Path("data/test.npz"),
            ),
            resolve_inference_config,
        ),
    ],
)
def test_mutated_dataclass_source_requires_a_model_config(source, resolver):
    source.model = {"N": 128}

    with pytest.raises(ValueError, match="model.*ModelConfig"):
        resolver(source, {})


def test_explicit_cli_patch_must_remain_a_mapping():
    source = TrainingConfig(model=ModelConfig())

    with pytest.raises(ValueError, match="explicit CLI.*mapping"):
        resolve_training_config({}, source)


def test_resolver_has_no_params_cfg_side_effect():
    before = deepcopy(params.cfg)

    resolve_training_config(
        {"model": {"N": 128}, "n_groups": 7, "backend": "pytorch"},
        {},
    )

    assert params.cfg == before


@pytest.mark.parametrize("architecture", _ARCHITECTURES)
def test_all_internal_architectures_are_structurally_valid(architecture):
    validate_model_config_structure(ModelConfig(architecture=architecture))


@pytest.mark.parametrize(
    "architecture",
    [
        "hybrid_resnet",
        "hybrid_resnet_ffno_ptychoblock_encoder",
        "hybrid_resnet_ptychoblock_ffno_encoder",
        "spectral_resnet_bottleneck_net",
        "spectral_resnet_bottleneck_linear_decoder",
        "hybrid_resnet_ffno_bottleneck",
        "hybrid_resnet_convnext_bottleneck",
    ],
)
def test_resnet_family_retains_downsample_and_width_constraints(architecture):
    with pytest.raises(ValueError, match="fno_blocks"):
        validate_model_config_structure(
            ModelConfig(architecture=architecture, fno_blocks=2)
        )
    with pytest.raises(ValueError, match="resnet_width.*divisible"):
        validate_model_config_structure(
            ModelConfig(architecture=architecture, resnet_width=30)
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("N", 32),
        ("N", True),
        ("architecture", "not-an-architecture"),
        ("architecture", b"cnn"),
        ("amp_activation", "silu"),
        ("gridsize", 0),
        ("gridsize", True),
        ("probe_big", 1),
        ("probe_scale", float("inf")),
        ("probe_mask_diameter", 0),
    ],
)
def test_model_structure_rejects_invalid_exact_types_and_domains(field, value):
    with pytest.raises(ValueError, match=field):
        validate_model_config_structure(replace(ModelConfig(), **{field: value}))


@pytest.mark.parametrize("N", [64, 128, 256])
def test_model_structure_accepts_declared_authoring_sizes(N):
    validate_model_config_structure(ModelConfig(N=N))


@pytest.mark.parametrize(
    "backend,weighting,expected_big",
    [
        ("pytorch", "probe", True),
        ("pytorch", "uniform", True),
        ("tensorflow", "central_mask", False),
    ],
)
def test_object_policy_resolution_is_backend_aware(
    backend,
    weighting,
    expected_big,
):
    config = resolve_training_config(
        {
            "backend": backend,
            "object_layout": ("grouped_patches" if expected_big else "single_patch"),
            "training_canvas": ("relative_overlap" if expected_big else "independent"),
            "training_patch_weighting": weighting,
        },
        {},
    )

    assert config.model.object_big is expected_big
    assert config.model.training_patch_weighting == weighting


def test_tensorflow_rejects_torch_weighting_policy():
    with pytest.raises(ValueError, match="TensorFlow.*central_mask"):
        resolve_training_config(
            {"backend": "tensorflow", "training_patch_weighting": "probe"},
            {},
        )


@pytest.mark.parametrize("resolver,source_factory", _RESOLVER_CASES)
@pytest.mark.parametrize(
    "field,value",
    [
        ("object_big", 1),
        ("probe_big", 1),
        ("pad_object", 1),
    ],
)
def test_resolver_object_policy_types_use_stable_value_errors(
    resolver,
    source_factory,
    field,
    value,
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match=field):
            resolver(source_factory(**{field: value}), {})

    assert caught == []


@pytest.mark.parametrize(
    "resolver,source",
    [
        (
            resolve_training_config,
            {"object_big": True, "batch_size": 0},
        ),
        (
            resolve_inference_config,
            _inference_source(object_big=True, debug=1),
        ),
    ],
)
def test_later_structural_failure_does_not_emit_object_big_warning(
    resolver,
    source,
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            resolver(source, {})

    assert caught == []


@pytest.mark.parametrize("resolver,source_factory", _RESOLVER_CASES)
def test_successful_explicit_object_big_warns_exactly_once(
    resolver,
    source_factory,
):
    with pytest.warns(DeprecationWarning, match="object_big") as caught:
        config = resolver(source_factory(object_big=True), {})

    assert len(caught) == 1
    assert config.model.object_big is True


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"batch_size": True}, "batch_size"),
        ({"nepochs": -1}, "nepochs"),
        ({"mae_weight": "0.5"}, "mae_weight"),
        ({"nll_weight": 2}, "nll_weight"),
        ({"positions_provided": 1}, "positions_provided"),
        ({"backend": "torch"}, "backend"),
        ({"optimizer": "rmsprop"}, "optimizer"),
        ({"scheduler": "Cosine"}, "scheduler"),
        ({"subsample_seed": -1}, "subsample_seed"),
    ],
)
def test_training_structure_rejects_invalid_exact_types_and_domains(
    updates,
    message,
):
    config = TrainingConfig(model=ModelConfig(), **updates)

    with pytest.raises(ValueError, match=message):
        validate_training_config_structure(config)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 16])
def test_training_structure_accepts_positive_integer_batches(batch_size):
    validate_training_config_structure(
        TrainingConfig(model=ModelConfig(), batch_size=batch_size)
    )


def test_training_structure_rejects_zero_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        validate_training_config_structure(
            TrainingConfig(model=ModelConfig(), batch_size=0)
        )


def test_structural_training_record_can_be_non_running(tmp_path):
    train_path = tmp_path / "train.npz"
    train_path.touch()
    inspectable = TrainingConfig(
        model=ModelConfig(),
        train_data_file=train_path,
        nepochs=0,
        n_groups=0,
        nphotons=0,
    )

    validate_training_config_structure(inspectable)

    with pytest.raises(ValueError, match="nepochs"):
        validate_runnable_training_config(inspectable)


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"batch_size": 0}, "batch_size"),
        ({"nphotons": 0}, "nphotons"),
        ({"n_groups": 0}, "n_groups"),
        ({"n_subsample": 0}, "n_subsample"),
    ],
)
def test_runnable_training_requires_positive_execution_values(
    tmp_path,
    updates,
    message,
):
    train_path = tmp_path / "train.npz"
    train_path.touch()
    config = TrainingConfig(
        model=ModelConfig(),
        train_data_file=train_path,
        **updates,
    )

    with pytest.raises(ValueError, match=message):
        validate_runnable_training_config(config)


def test_runnable_training_requires_existing_readable_data(
    tmp_path,
    monkeypatch,
):
    missing = TrainingConfig(
        model=ModelConfig(),
        train_data_file=tmp_path / "missing.npz",
    )
    with pytest.raises(ValueError, match="train_data_file.*exist"):
        validate_runnable_training_config(missing)

    train_path = tmp_path / "train.npz"
    train_path.touch()
    unreadable = replace(missing, train_data_file=train_path)
    from ptycho.config import resolution

    monkeypatch.setattr(resolution.os, "access", lambda path, mode: False)
    with pytest.raises(ValueError, match="train_data_file.*readable"):
        validate_runnable_training_config(unreadable)


def test_runnable_training_rejects_a_training_data_directory(tmp_path):
    train_directory = tmp_path / "train.npz"
    train_directory.mkdir()
    config = TrainingConfig(
        model=ModelConfig(),
        train_data_file=train_directory,
    )

    with pytest.raises(ValueError, match="train_data_file.*regular file"):
        validate_runnable_training_config(config)


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"n_groups": True}, "n_groups"),
        ({"neighbor_count": 0}, "neighbor_count"),
        ({"debug": 1}, "debug"),
        ({"backend": "torch"}, "backend"),
    ],
)
def test_inference_structure_rejects_invalid_exact_types_and_domains(
    updates,
    message,
):
    config = InferenceConfig(
        model=ModelConfig(),
        model_path=Path("models/example"),
        test_data_file=Path("data/test.npz"),
        **updates,
    )

    with pytest.raises(ValueError, match=message):
        validate_inference_config_structure(config)


def test_inference_structural_validation_does_not_access_resources():
    class ExplodingPath(type(Path())):
        def exists(self):
            raise AssertionError("structural validation accessed filesystem")

    config = InferenceConfig(
        model=ModelConfig(),
        model_path=ExplodingPath("models/example"),
        test_data_file=ExplodingPath("data/test.npz"),
    )

    validate_inference_config_structure(config)


def test_inference_resource_validation_checks_archive_and_data(tmp_path):
    model_path = tmp_path / "model"
    test_path = tmp_path / "test.npz"
    config = InferenceConfig(
        model=ModelConfig(),
        model_path=model_path,
        test_data_file=test_path,
    )
    test_path.touch()

    with pytest.raises(ValueError, match="model_path.*exist"):
        validate_inference_resources(config)

    model_path.mkdir()
    with pytest.raises(ValueError, match=r"wts\.h5\.zip"):
        validate_inference_resources(config)

    archive = model_path / "wts.h5.zip"
    archive.touch()
    validate_inference_resources(config)

    test_path.unlink()
    with pytest.raises(ValueError, match="test_data_file.*exist"):
        validate_inference_resources(config)

    test_path.mkdir()
    with pytest.raises(ValueError, match="test_data_file.*regular file"):
        validate_inference_resources(config)


def test_public_module_exports_the_resolution_surface():
    assert public_config.resolve_training_config is resolve_training_config
    assert public_config.resolve_inference_config is resolve_inference_config
    assert (
        public_config.validate_model_config_structure is validate_model_config_structure
    )
    assert (
        public_config.validate_training_config_structure
        is validate_training_config_structure
    )
    assert (
        public_config.validate_inference_config_structure
        is validate_inference_config_structure
    )
    assert (
        public_config.validate_runnable_training_config
        is validate_runnable_training_config
    )
    assert public_config.validate_inference_resources is validate_inference_resources


def _parse_public_training_args(monkeypatch, *argv: str):
    from ptycho.workflows.components import parse_arguments

    monkeypatch.setattr(sys, "argv", ["ptycho-train", *argv])
    return parse_arguments()


def test_training_yaml_precedence_survives_omitted_cli_value(
    monkeypatch,
    tmp_path,
):
    from ptycho.workflows.components import setup_configuration

    config_path = tmp_path / "training.yaml"
    config_path.write_text("nepochs: 9\n", encoding="utf-8")
    args = _parse_public_training_args(
        monkeypatch,
        "--config",
        str(config_path),
    )

    config = setup_configuration(args, args.config)

    assert config.nepochs == 9
    assert not hasattr(args, "nepochs")


@pytest.mark.parametrize(
    ("cli_value", "expected"),
    [("3", 3), ("50", 50)],
)
def test_training_explicit_cli_precedence_including_declared_default(
    monkeypatch,
    tmp_path,
    cli_value,
    expected,
):
    from ptycho.workflows.components import setup_configuration

    config_path = tmp_path / "training.yaml"
    config_path.write_text("nepochs: 9\n", encoding="utf-8")
    args = _parse_public_training_args(
        monkeypatch,
        "--config",
        str(config_path),
        "--nepochs",
        cli_value,
    )

    config = setup_configuration(args, args.config)

    assert config.nepochs == expected


@pytest.mark.parametrize("initially_sealed", [False, True])
def test_training_setup_preserves_legacy_mapping_and_sealed_state(
    monkeypatch,
    initially_sealed,
):
    from ptycho import params
    from ptycho.workflows.components import setup_configuration

    legacy_values = {"sentinel": "unchanged"}
    monkeypatch.setattr(params, "cfg", legacy_values.copy())
    monkeypatch.setattr(params, "_sealed", initially_sealed)

    config = setup_configuration(argparse.Namespace(), None)

    assert config.nepochs == 50
    assert params.cfg == legacy_values
    assert params._sealed is initially_sealed


def test_public_training_parser_unwraps_direct_and_optional_literal_choices(
    monkeypatch,
):
    args = _parse_public_training_args(
        monkeypatch,
        "--N",
        "128",
        "--backend",
        "pytorch",
        "--object_layout",
        "single_patch",
        "--architecture",
        "hybrid_resnet",
    )

    assert args.N == 128
    assert type(args.N) is int
    assert args.backend == "pytorch"
    assert type(args.backend) is str
    assert args.object_layout == "single_patch"
    assert type(args.object_layout) is str
    assert args.architecture == "hybrid_resnet"


def test_public_training_parser_boolean_actions_preserve_explicitness(
    monkeypatch,
):
    omitted = _parse_public_training_args(monkeypatch)

    assert not hasattr(omitted, "probe_big")
    assert not hasattr(omitted, "probe_mask")

    explicit = _parse_public_training_args(
        monkeypatch,
        "--probe_mask",
        "--no-probe_big",
    )

    assert explicit.probe_mask is True
    assert explicit.probe_big is False


def test_public_training_parser_keeps_numeric_overrides_primitive(monkeypatch):
    args = _parse_public_training_args(
        monkeypatch,
        "--mae_weight",
        "0.25",
        "--nphotons",
        "1000",
        "--gradient_clip_val",
        "2.5",
    )

    assert args.mae_weight == 0.25
    assert type(args.mae_weight) is float
    assert args.nphotons == 1000.0
    assert type(args.nphotons) is float
    assert args.gradient_clip_val == 2.5
    assert type(args.gradient_clip_val) is float


def test_public_training_parser_preserves_required_and_optional_path_values(
    monkeypatch,
    tmp_path,
):
    output_dir = tmp_path / "output"
    train_data_file = tmp_path / "train.npz"

    args = _parse_public_training_args(
        monkeypatch,
        "--output_dir",
        str(output_dir),
        "--train_data_file",
        str(train_data_file),
    )

    assert args.output_dir == output_dir
    assert isinstance(args.output_dir, Path)
    assert args.train_data_file == train_data_file
    assert isinstance(args.train_data_file, Path)


def test_public_training_parser_literal_help_keeps_existing_cli_spellings(
    monkeypatch,
    capsys,
):
    with pytest.raises(SystemExit) as error:
        _parse_public_training_args(monkeypatch, "--help")

    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--n_groups" in help_text
    assert "--n_images" in help_text
    assert "--neighbor_count" in help_text
    assert "--backend {tensorflow,pytorch}" in help_text
    assert "hybrid_resnet" in help_text
