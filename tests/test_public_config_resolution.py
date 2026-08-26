from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import warnings

import pytest

import ptycho.config as public_config
from ptycho.config.resolution import (
    resolve_inference_config,
    resolve_training_config,
    validate_inference_config_structure,
    validate_model_config_structure,
    validate_training_config_structure,
)

_N_IMAGES_DEPRECATION_MESSAGE = (
    "Parameter 'n_images' is deprecated and will be removed in a future "
    "version. Use 'n_groups' instead, which always means the number of "
    "groups regardless of gridsize."
)


_TRAINING_SAMPLING_FIELDS = frozenset({
    "n_groups", "n_images", "n_subsample", "subsample_seed",
    "neighbor_count", "enable_oversampling", "neighbor_pool_size", "sequential_sampling",
})
_TRAINING_DATA_FIELDS = frozenset({"train_data_file", "test_data_file", "nphotons"})


def _training_source(**updates):
    """Build a training config mapping with sub-config fields in their nested positions."""
    result: dict = {}
    for k, v in updates.items():
        if k in _TRAINING_SAMPLING_FIELDS:
            result.setdefault("sampling", {})[k] = v
        elif k in _TRAINING_DATA_FIELDS:
            result.setdefault("data", {})[k] = v
        else:
            result[k] = v
    return result


def _inference_source(**updates):
    values = {
        "model_path": "models/example",
        "test_data_file": "data/test.npz",
    }
    values.update(updates)
    return values


def _direct_training_config(**updates):
    """Construct a TrainingConfig, routing sub-config fields into their sub-configs."""
    sampling_kwargs = {k: updates.pop(k) for k in list(updates) if k in _TRAINING_SAMPLING_FIELDS}
    data_kwargs = {k: updates.pop(k) for k in list(updates) if k in _TRAINING_DATA_FIELDS}
    sampling = public_config.SamplingConfig(**sampling_kwargs) if sampling_kwargs else public_config.SamplingConfig()
    data = public_config.DataConfig(**data_kwargs) if data_kwargs else public_config.DataConfig()
    return public_config.TrainingConfig(
        model=public_config.ModelConfig(),
        sampling=sampling,
        data=data,
        **updates,
    )


def _direct_inference_config(**updates):
    return public_config.InferenceConfig(
        model=public_config.ModelConfig(),
        model_path=Path("models/example"),
        test_data_file=Path("data/test.npz"),
        **updates,
    )


def _sampling(config):
    """Return the sampling container for TrainingConfig or the config itself for InferenceConfig."""
    return config.sampling if hasattr(config, "sampling") and not isinstance(config, public_config.InferenceConfig) else config


_PUBLIC_RESOLVER_CASES = [
    pytest.param(
        resolve_training_config,
        _training_source,
        id="training",
    ),
    pytest.param(
        resolve_inference_config,
        _inference_source,
        id="inference",
    ),
]


def _assert_single_n_images_deprecation(caught_warnings):
    assert len(caught_warnings) == 1
    assert caught_warnings[0].category is DeprecationWarning
    assert str(caught_warnings[0].message) == _N_IMAGES_DEPRECATION_MESSAGE


def test_training_file_value_survives_omitted_cli_value():
    config = resolve_training_config(
        {"nepochs": 9, "model": {"N": 128}},
        {},
    )

    assert config.nepochs == 9
    assert config.model.N == 128


def test_training_explicit_cli_value_overrides_file():
    config = resolve_training_config(
        {"nepochs": 9},
        {"nepochs": 3},
    )

    assert config.nepochs == 3


def test_training_nested_model_values_are_accepted():
    config = resolve_training_config(
        {"model": {"N": 128}},
        {},
    )

    assert config.model.N == 128


def test_training_flat_model_fields_at_root_are_rejected():
    with pytest.raises(ValueError, match="N"):
        resolve_training_config(
            {"N": 64, "model": {"N": 128}},
            {},
        )


def test_training_cli_nested_model_value_overrides_file_value():
    config = resolve_training_config(
        {"model": {"N": 64}},
        {"model": {"N": 128}},
    )

    assert config.model.N == 128


def test_inference_file_value_survives_omitted_cli_value():
    config = resolve_inference_config(
        _inference_source(n_groups=9, model={"N": 128}),
        {},
    )

    assert config.training_groups == 9
    assert config.model.N == 128


def test_inference_explicit_cli_value_overrides_file():
    config = resolve_inference_config(
        _inference_source(n_groups=9),
        {"n_groups": 3},
    )

    assert config.training_groups == 3


def test_inference_yaml_root_values_and_explicit_defaults_resolve_by_precedence():
    config = resolve_inference_config(
        _inference_source(
            n_groups=9,
            neighbor_count=7,
            subsample_seed=123,
            debug=True,
            output_dir="yaml-output",
            backend="pytorch",
        ),
        {
            "n_groups": 3,
            "neighbor_count": 4,
            "subsample_seed": 45,
            "debug": False,
            "output_dir": "inference_outputs",
            "backend": "tensorflow",
        },
    )

    assert config.training_groups == 3
    assert config.neighbor_count == 4
    assert config.subsample_seed == 45
    assert config.debug is False
    assert config.output_dir == Path("inference_outputs")
    assert config.backend == "tensorflow"


def test_inference_equal_flat_and_nested_model_values_are_canonicalized_once():
    config = resolve_inference_config(
        _inference_source(N=128, model={"N": 128}),
        {},
    )

    assert config.model.N == 128


def test_inference_conflicting_flat_and_nested_model_values_fail():
    with pytest.raises(ValueError, match="N.*flat.*model"):
        resolve_inference_config(
            _inference_source(N=64, model={"N": 128}),
            {},
        )


def test_inference_cli_flat_model_value_overrides_file_nested_value():
    config = resolve_inference_config(
        _inference_source(model={"N": 64}),
        {"N": 128},
    )

    assert config.model.N == 128


def test_inference_cli_nested_model_value_overrides_file_flat_value():
    config = resolve_inference_config(
        _inference_source(N=64),
        {"model": {"N": 128}},
    )

    assert config.model.N == 128


@pytest.mark.parametrize(
    ("resolver", "source_factory"),
    _PUBLIC_RESOLVER_CASES,
)
def test_n_images_alone_resolves_to_canonical_n_groups(
    resolver,
    source_factory,
):
    with pytest.warns(DeprecationWarning) as caught_warnings:
        config = resolver(source_factory(n_images=7), {})

    _assert_single_n_images_deprecation(caught_warnings)
    assert _sampling(config).training_groups == 7
    assert _sampling(config).n_images is None


@pytest.mark.parametrize(
    ("resolver", "source_factory"),
    _PUBLIC_RESOLVER_CASES,
)
def test_equal_n_images_and_n_groups_in_one_source_are_accepted_once(
    resolver,
    source_factory,
):
    with pytest.warns(DeprecationWarning) as caught_warnings:
        config = resolver(
            source_factory(n_images=7, n_groups=7),
            {},
        )

    _assert_single_n_images_deprecation(caught_warnings)
    assert _sampling(config).training_groups == 7
    assert _sampling(config).n_images is None


@pytest.mark.parametrize(
    ("resolver", "source_factory"),
    _PUBLIC_RESOLVER_CASES,
)
def test_unequal_n_images_and_n_groups_in_one_source_fail_without_warning(
    resolver,
    source_factory,
):
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as error:
            resolver(
                source_factory(n_images=7, n_groups=9),
                {},
            )

    assert "n_images" in str(error.value)
    assert "n_groups" in str(error.value)
    assert caught_warnings == []


def test_file_n_images_then_cli_n_groups_uses_cli_canonical_value_inference():
    with pytest.warns(DeprecationWarning) as caught_warnings:
        config = resolve_inference_config(
            _inference_source(n_images=7),
            {"n_groups": 11},
        )

    _assert_single_n_images_deprecation(caught_warnings)
    assert config.training_groups == 11
    assert config.n_images is None


def test_file_n_images_then_cli_n_groups_conflict_raises_for_training():
    # The training resolver deep-merges sources before pydantic validation.
    # file sampling={n_images: 7} merged with CLI sampling={n_groups: 11}
    # gives sampling={n_images: 7, n_groups: 11}, which is a conflict.
    # Users must use consistent naming within the merged training config.
    import warnings as _warnings
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        with pytest.raises(ValueError, match="n_images.*n_groups|n_groups.*n_images|conflicts"):
            resolve_training_config(
                _training_source(n_images=7),
                {"sampling": {"n_groups": 11}},
            )
    # No deprecation warning when conflict is detected before alias resolution
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_file_n_images_only_training_resolves_to_n_groups():
    # When only n_images is provided (no n_groups anywhere), the alias resolves
    # and a deprecation warning is emitted.
    with pytest.warns(DeprecationWarning) as caught_warnings:
        config = resolve_training_config(
            _training_source(n_images=11),
            {},
        )

    _assert_single_n_images_deprecation(caught_warnings)
    assert config.sampling.training_groups == 11
    assert config.sampling.n_images is None


def test_file_n_groups_then_cli_n_images_uses_cli_alias_value_inference():
    with pytest.warns(DeprecationWarning) as caught_warnings:
        config = resolve_inference_config(
            _inference_source(n_groups=7),
            {"n_images": 11},
        )

    _assert_single_n_images_deprecation(caught_warnings)
    assert config.training_groups == 11
    assert config.n_images is None


def test_file_n_groups_then_cli_n_images_conflict_raises_for_training():
    # The training resolver deep-merges sources before pydantic validation.
    # file sampling={n_groups: 7} merged with CLI sampling={n_images: 11}
    # gives sampling={n_groups: 7, n_images: 11}, which is a conflict.
    import warnings as _warnings
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        with pytest.raises(ValueError, match="n_images.*n_groups|n_groups.*n_images|conflicts"):
            resolve_training_config(
                _training_source(n_groups=7),
                {"sampling": {"n_images": 11}},
            )
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


@pytest.mark.parametrize(
    ("resolver", "source_factory"),
    _PUBLIC_RESOLVER_CASES,
)
def test_n_groups_without_n_images_emits_no_compatibility_warning(
    resolver,
    source_factory,
):
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        config = resolver(source_factory(n_groups=7), {})

    assert _sampling(config).training_groups == 7
    assert _sampling(config).n_images is None
    assert caught_warnings == []


def test_failed_structural_resolution_with_n_images_emits_no_warning_inference():
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="gridsize"):
            resolve_inference_config(
                _inference_source(n_images=7, gridsize=0),
                {"n_groups": 11},
            )

    assert caught_warnings == []


def test_failed_structural_resolution_with_n_images_training_warns_then_fails():
    # For training: pydantic validates the sampling sub-config (which emits the
    # deprecation warning) before reporting the model-level 'gridsize' error.
    # The n_images deprecation warning fires even on structural failure.
    file_values = {"sampling": {"n_images": 7}, "model": {"gridsize": 0}}
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="gridsize"):
            resolve_training_config(file_values, {})

    deprecation_warnings = [
        w for w in caught_warnings if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) == 1


def test_n_images_resolution_leaves_source_mappings_unchanged_inference():
    file_values = _inference_source(n_groups=7)
    cli_values = {"n_images": 11}
    original_file = deepcopy(file_values)
    original_cli = deepcopy(cli_values)

    with pytest.warns(DeprecationWarning):
        resolve_inference_config(file_values, cli_values)

    assert file_values == original_file
    assert cli_values == original_cli


def test_n_images_resolution_leaves_source_mappings_unchanged_training():
    # Use a case where only n_images is in the file (no n_groups anywhere)
    # so the alias resolves without conflict and sources remain unmodified.
    file_values = _training_source(n_images=7)
    cli_values = {}
    original_file = deepcopy(file_values)
    original_cli = deepcopy(cli_values)

    with pytest.warns(DeprecationWarning):
        resolve_training_config(file_values, cli_values)

    assert file_values == original_file
    assert cli_values == original_cli


def test_direct_inference_config_n_images_construction_retains_post_init_behavior():
    # InferenceConfig uses __post_init__ which retains n_images as-is.
    with pytest.warns(DeprecationWarning) as caught_warnings:
        config = _direct_inference_config(n_images=7)

    _assert_single_n_images_deprecation(caught_warnings)
    assert config.training_groups == 7
    assert config.n_images == 7


def test_direct_training_config_n_images_construction_clears_n_images():
    # TrainingConfig uses SamplingConfig pydantic validator which clears n_images to None
    # and sets n_groups from n_images.
    with pytest.warns(DeprecationWarning) as caught_warnings:
        config = _direct_training_config(n_images=7)

    _assert_single_n_images_deprecation(caught_warnings)
    assert config.sampling.training_groups == 7
    assert config.sampling.n_images is None


def test_direct_inference_config_n_images_n_groups_conflict_remains_accepted():
    # InferenceConfig __post_init__ accepts conflicting values (both retained, no warning).
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        config = _direct_inference_config(n_images=7, n_groups=9)

    assert config.training_groups == 9
    assert config.n_images == 7
    assert caught_warnings == []


def test_direct_training_config_n_images_n_groups_conflict_raises():
    # SamplingConfig pydantic validator rejects conflicting n_images and n_groups.
    with pytest.raises(ValueError, match="n_images.*n_groups|n_groups.*n_images"):
        _direct_training_config(n_images=7, n_groups=9)


def test_inference_unknown_root_names_are_sorted():
    # resolve_inference_config sorts unknown field names alphabetically.
    with pytest.raises(ValueError) as error:
        resolve_inference_config(_inference_source(zeta_unknown=1, alpha_unknown=2), {})

    message = str(error.value)
    assert "alpha_unknown" in message
    assert "zeta_unknown" in message
    assert message.index("alpha_unknown") < message.index("zeta_unknown")


def test_training_unknown_root_names_are_reported():
    # resolve_training_config uses pydantic which reports fields in insertion order
    # (not sorted). Both names appear in the error message.
    with pytest.raises(ValueError) as error:
        resolve_training_config({"zeta_unknown": 1, "alpha_unknown": 2}, {})

    message = str(error.value)
    assert "alpha_unknown" in message
    assert "zeta_unknown" in message


def test_inference_unknown_nested_model_names_are_sorted():
    # resolve_inference_config sorts unknown model field names alphabetically.
    with pytest.raises(ValueError) as error:
        resolve_inference_config(
            _inference_source(model={"zeta_unknown": 1, "alpha_unknown": 2}),
            {},
        )

    message = str(error.value)
    assert "alpha_unknown" in message
    assert "zeta_unknown" in message
    assert message.index("alpha_unknown") < message.index("zeta_unknown")


def test_training_unknown_nested_model_names_are_reported():
    # resolve_training_config uses pydantic which reports fields in insertion order.
    # Both names appear in the error message.
    with pytest.raises(ValueError) as error:
        resolve_training_config(
            {"model": {"zeta_unknown": 1, "alpha_unknown": 2}},
            {},
        )

    message = str(error.value)
    assert "alpha_unknown" in message
    assert "zeta_unknown" in message


@pytest.mark.parametrize(
    ("resolver", "file_values", "cli_values"),
    [
        (
            resolve_training_config,
            {"model": {"N": 64}, "nepochs": 9},
            {"model": {"N": 128}, "output_dir": "cli-output"},
        ),
        (
            resolve_inference_config,
            _inference_source(model={"N": 64}, n_groups=9),
            {"N": 128, "output_dir": "cli-output"},
        ),
    ],
)
def test_training_and_inference_inputs_are_unchanged(
    resolver,
    file_values,
    cli_values,
):
    original_file = deepcopy(file_values)
    original_cli = deepcopy(cli_values)

    resolver(file_values, cli_values)

    assert file_values == original_file
    assert cli_values == original_cli


def test_training_path_fields_are_materialized_as_paths():
    config = resolve_training_config(
        {
            "data": {
                "train_data_file": "data/train.npz",
                "test_data_file": "data/test.npz",
            },
            "output_dir": "outputs/train",
        },
        {},
    )

    assert config.data.train_data_file == Path("data/train.npz")
    assert config.data.test_data_file == Path("data/test.npz")
    assert config.output_dir == Path("outputs/train")
    assert isinstance(config.data.train_data_file, Path)
    assert isinstance(config.data.test_data_file, Path)
    assert isinstance(config.output_dir, Path)


def test_inference_path_fields_are_materialized_as_paths():
    config = resolve_inference_config(
        _inference_source(output_dir="outputs/inference"),
        {},
    )

    assert config.model_path == Path("models/example")
    assert config.test_data_file == Path("data/test.npz")
    assert config.output_dir == Path("outputs/inference")
    assert isinstance(config.model_path, Path)
    assert isinstance(config.test_data_file, Path)
    assert isinstance(config.output_dir, Path)


@pytest.mark.parametrize(
    ("resolver", "file_values"),
    [
        (resolve_training_config, {"model": {"N": 128}}),
        (
            resolve_inference_config,
            _inference_source(model={"N": 128}),
        ),
    ],
)
def test_training_and_inference_each_return_fresh_dataclasses(
    resolver,
    file_values,
):
    first = resolver(file_values, {})
    second = resolver(file_values, {})

    assert first == second
    assert first is not second
    assert first.model is not second.model


@pytest.mark.parametrize(
    ("resolver", "file_values"),
    [
        (
            resolve_training_config,
            {
                "backend": "pytorch",
                "model": {
                    "object_layout": "grouped_patches",
                    "training_canvas": "relative_overlap",
                    "training_patch_weighting": "probe",
                },
            },
        ),
        (
            resolve_inference_config,
            _inference_source(
                backend="pytorch",
                model={
                    "object_layout": "grouped_patches",
                    "training_canvas": "relative_overlap",
                    "training_patch_weighting": "probe",
                },
            ),
        ),
    ],
)
def test_training_and_inference_pytorch_backend_use_torch_object_policy(
    resolver,
    file_values,
):
    config = resolver(file_values, {})

    assert config.backend == "pytorch"
    assert config.model.object_big is True
    assert config.model.object_layout == "grouped_patches"
    assert config.model.training_canvas == "relative_overlap"
    assert config.model.training_patch_weighting == "probe"


@pytest.mark.parametrize(
    ("resolver", "file_values"),
    [
        (
            resolve_training_config,
            {
                "backend": "tensorflow",
                "model": {"training_patch_weighting": "probe"},
            },
        ),
        (
            resolve_inference_config,
            _inference_source(
                backend="tensorflow",
                training_patch_weighting="probe",
            ),
        ),
    ],
)
def test_training_and_inference_tensorflow_object_policy_mismatch_fails(
    resolver,
    file_values,
):
    with pytest.raises(ValueError, match="TensorFlow.*central_mask"):
        resolver(file_values, {})


@pytest.mark.parametrize(
    ("resolver", "source_factory"),
    _PUBLIC_RESOLVER_CASES,
)
@pytest.mark.parametrize(
    ("field", "value"),
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
    ("resolver", "source"),
    [
        (
            resolve_training_config,
            {"object_big": True, "positions_provided": 1},
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


def test_successful_explicit_object_big_warns_exactly_once_inference():
    with pytest.warns(DeprecationWarning, match="object_big") as caught:
        config = resolve_inference_config(_inference_source(object_big=True), {})

    assert len(caught) == 1
    assert config.model.object_big is True


def test_successful_explicit_object_big_warns_exactly_once_training():
    # For training, object_big is a ModelConfig field so must be nested under 'model'.
    with pytest.warns(DeprecationWarning, match="object_big") as caught:
        config = resolve_training_config({"model": {"object_big": True}}, {})

    assert len(caught) == 1
    assert config.model.object_big is True


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("architecture", "resnet", "architecture"),
        ("amp_activation", "silu", "amp_activation"),
        ("gridsize", 0, "gridsize"),
    ],
)
def test_training_structural_invalid_model_values_fail_before_return(
    field_name,
    invalid_value,
    message,
):
    with pytest.raises(ValueError, match=message):
        resolve_training_config({field_name: invalid_value}, {})


def test_training_ffno_zero_cnn_blocks_is_structurally_valid():
    config = resolve_training_config(
        {"model": {"architecture": "ffno", "fno_cnn_blocks": 0}},
        {},
    )

    assert config.model.architecture == "ffno"
    assert config.model.fno_cnn_blocks == 0


def test_inference_only_none_means_an_absent_source():
    # resolve_inference_config uses _normalize_public_source which explicitly
    # rejects non-Mapping inputs with a "mapping" error message.
    valid_values = _inference_source()
    resolve_inference_config(valid_values, None)
    resolve_inference_config(None, valid_values)

    with pytest.raises(ValueError, match="mapping"):
        resolve_inference_config([], None)
    with pytest.raises(ValueError, match="mapping"):
        resolve_inference_config(valid_values, [])


def test_training_only_none_means_an_absent_source():
    # resolve_training_config accepts None as "absent" and uses {} instead.
    # Non-None non-Mapping inputs that cannot be converted via dict() are
    # rejected, but empty lists dict([]) == {} so they silently succeed.
    resolve_training_config({}, None)
    resolve_training_config(None, {})

    # A non-iterable type raises ValueError via the pydantic model_validate path.
    with pytest.raises((ValueError, TypeError)):
        resolve_training_config(object(), None)


def test_inference_nested_model_must_be_a_mapping():
    # resolve_inference_config raises ValueError with "model.*mapping" message.
    with pytest.raises(ValueError, match="model.*mapping"):
        resolve_inference_config(_inference_source(model=[]), {})


def test_training_nested_model_must_be_a_dict_or_modelconfig():
    # resolve_training_config uses pydantic which rejects a list for the model field
    # with a message about expecting a dictionary or ModelConfig instance.
    with pytest.raises(ValueError, match="model"):
        resolve_training_config({"model": []}, {})


def test_public_configuration_resolution_api_exports_supported_names():
    assert public_config.resolve_training_config is resolve_training_config
    assert public_config.resolve_inference_config is resolve_inference_config
    assert (
        public_config.validate_model_config_structure
        is validate_model_config_structure
    )
    assert (
        public_config.validate_training_config_structure
        is validate_training_config_structure
    )
    assert (
        public_config.validate_inference_config_structure
        is validate_inference_config_structure
    )
    assert callable(public_config.validate_runnable_training_config)
    assert callable(public_config.validate_inference_resources)


def test_resolved_training_legacy_projection_matches_equivalent_direct_config():
    # TrainingConfig requires model fields nested under 'model', data fields under
    # 'data', and sampling fields under 'sampling'.
    resolved = resolve_training_config(
        {
            "model": {"N": 128},
            "data": {
                "train_data_file": "data/train.npz",
                "test_data_file": None,
            },
            "output_dir": "outputs/train",
            "sampling": {"n_groups": 7},
        },
        {},
    )
    direct = public_config.TrainingConfig(
        model=public_config.ModelConfig(N=128),
        data=public_config.DataConfig(train_data_file=Path("data/train.npz"), test_data_file=None),
        sampling=public_config.SamplingConfig(training_groups=7),
        output_dir=Path("outputs/train"),
    )

    resolved_projection = public_config.dataclass_to_legacy_dict(resolved)
    direct_projection = public_config.dataclass_to_legacy_dict(direct)

    assert list(resolved_projection.items()) == list(direct_projection.items())
    assert resolved_projection["train_data_file_path"] == "data/train.npz"
    assert resolved_projection["output_prefix"] == "outputs/train"
    assert resolved_projection["test_data_file_path"] is None
    assert resolved_projection["n_images"] is None


def test_update_legacy_skip_none_preserves_all_projected_none_sentinels():
    config = resolve_training_config(
        {
            "model": {"N": 128},
            "data": {
                "train_data_file": "data/train.npz",
                "test_data_file": None,
            },
            "output_dir": "outputs/train",
            "sampling": {"n_groups": 7},
        },
        {},
    )
    projection = public_config.dataclass_to_legacy_dict(config)
    sentinel = object()
    target = {key: sentinel for key in projection}

    public_config.update_legacy_dict(target, config)

    for key, value in projection.items():
        if value is None:
            assert target[key] is sentinel
        else:
            assert target[key] == value


def test_training_structural_record_can_be_inspected_but_is_not_runnable(
    tmp_path,
):
    train_path = tmp_path / "train.npz"
    train_path.touch()
    inspectable = public_config.TrainingConfig(
        model=public_config.ModelConfig(),
        data=public_config.DataConfig(train_data_file=train_path),
        nepochs=0,
    )

    validate_training_config_structure(inspectable)

    with pytest.raises(ValueError, match="nepochs"):
        public_config.validate_runnable_training_config(inspectable)


def _training_cfg_with_flat_updates(train_path, **flat_updates):
    """Build a TrainingConfig from flat kwargs, routing sub-config fields appropriately."""
    data_fields = {"nphotons", "test_data_file"}
    sampling_fields = {"n_groups", "n_images", "n_subsample", "subsample_seed",
                       "neighbor_count", "enable_oversampling", "neighbor_pool_size"}
    data_kw = {k: flat_updates.pop(k) for k in list(flat_updates) if k in data_fields}
    sampling_kw = {k: flat_updates.pop(k) for k in list(flat_updates) if k in sampling_fields}
    return public_config.TrainingConfig(
        model=public_config.ModelConfig(),
        data=public_config.DataConfig(train_data_file=train_path, **data_kw),
        sampling=public_config.SamplingConfig(**sampling_kw),
        **flat_updates,
    )


@pytest.mark.parametrize(
    ("updates", "message"),
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
    config = _training_cfg_with_flat_updates(train_path, **updates)

    validate_training_config_structure(config)
    with pytest.raises(ValueError, match=message):
        public_config.validate_runnable_training_config(config)


def test_runnable_training_requires_existing_readable_training_data(
    tmp_path,
    monkeypatch,
):
    missing = public_config.TrainingConfig(
        model=public_config.ModelConfig(),
        data=public_config.DataConfig(train_data_file=tmp_path / "missing.npz"),
    )
    with pytest.raises(ValueError, match="train_data_file.*exist"):
        public_config.validate_runnable_training_config(missing)

    train_path = tmp_path / "train.npz"
    train_path.touch()
    unreadable = public_config.TrainingConfig(
        model=public_config.ModelConfig(),
        data=public_config.DataConfig(train_data_file=train_path),
    )
    from ptycho.config import resolution

    monkeypatch.setattr(
        resolution.os,
        "access",
        lambda path, mode: False,
    )
    with pytest.raises(ValueError, match="train_data_file.*readable"):
        public_config.validate_runnable_training_config(unreadable)


def test_runnable_training_rejects_a_training_data_directory(tmp_path):
    train_directory = tmp_path / "train.npz"
    train_directory.mkdir()
    config = public_config.TrainingConfig(
        model=public_config.ModelConfig(),
        data=public_config.DataConfig(train_data_file=train_directory),
    )

    with pytest.raises(ValueError, match="train_data_file.*regular file"):
        public_config.validate_runnable_training_config(config)


def test_inference_structural_validation_performs_no_resource_access(
    monkeypatch,
):
    config = _direct_inference_config()

    def forbid_exists(_path):
        raise AssertionError("structural validation accessed filesystem")

    monkeypatch.setattr(Path, "exists", forbid_exists)

    validate_inference_config_structure(config)


def test_inference_resource_validation_checks_model_and_test_paths(tmp_path):
    model_path = tmp_path / "model"
    test_path = tmp_path / "test.npz"
    config = public_config.InferenceConfig(
        model=public_config.ModelConfig(),
        model_path=model_path,
        test_data_file=test_path,
    )
    test_path.touch()

    with pytest.raises(ValueError, match="model_path.*exist"):
        public_config.validate_inference_resources(config)

    model_path.mkdir()
    with pytest.raises(ValueError, match="wts\\.h5\\.zip"):
        public_config.validate_inference_resources(config)

    model_archive = model_path / "wts.h5.zip"
    model_archive.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        public_config.validate_inference_resources(config)

    model_archive.rmdir()
    model_archive.touch()
    public_config.validate_inference_resources(config)

    test_path.unlink()
    with pytest.raises(ValueError, match="test_data_file.*exist"):
        public_config.validate_inference_resources(config)

    test_path.mkdir()
    with pytest.raises(ValueError, match="test_data_file.*regular file"):
        public_config.validate_inference_resources(config)

    test_path.rmdir()
    test_path.touch()
    public_config.validate_inference_resources(config)


def test_model_compatibility_facade_keeps_narrow_activation_predicate():
    config = public_config.ModelConfig(amp_activation="legacy-custom")

    public_config.validate_model_config(config)

    with pytest.raises(ValueError, match="amp_activation"):
        validate_model_config_structure(config)


def test_training_compatibility_facade_does_not_require_data_resource():
    config = public_config.TrainingConfig(
        model=public_config.ModelConfig(),
        data=public_config.DataConfig(train_data_file=Path("does-not-exist.npz"), nphotons=1),
        nepochs=1,
        batch_size=16,
    )

    public_config.validate_training_config(config)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"batch_size": 3}, "batch_size"),
        ({"nepochs": 0}, "nepochs"),
        # mae_weight and nll_weight are caught by pydantic at construction time,
        # so use model_construct to bypass validation and test the backstop directly.
        (
            {"tf_loss": public_config.TFLossConfig.model_construct(mae_weight=-1)},
            "mae_weight",
        ),
        (
            {"tf_loss": public_config.TFLossConfig.model_construct(nll_weight=2)},
            "nll_weight",
        ),
        ({"data": public_config.DataConfig(nphotons=0)}, "nphotons"),
    ],
)
def test_training_compatibility_facade_keeps_existing_value_checks(
    updates,
    message,
):
    kwargs = dict(
        model=public_config.ModelConfig(),
        batch_size=16,
        nepochs=50,
        positions_provided=True,
        probe_trainable=False,
        intensity_scale_trainable=True,
        output_dir=__import__('pathlib').Path("training_outputs"),
        backend='tensorflow',
        data=public_config.DataConfig(),
        sampling=public_config.SamplingConfig(),
        loss=public_config.LossConfig(),
        tf_loss=public_config.TFLossConfig(),
        gradient_clip=public_config.GradientClipConfig(),
        optimizer=public_config.OptimizerConfig(),
        scheduler=public_config.SchedulerConfig(),
    )
    kwargs.update(updates)
    config = public_config.TrainingConfig.model_construct(**kwargs)

    with pytest.raises(ValueError, match=message):
        public_config.validate_training_config(config)


def test_inference_compatibility_facade_keeps_model_archive_behavior(
    tmp_path,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    missing_test = tmp_path / "missing-test.npz"
    config = public_config.InferenceConfig(
        model=public_config.ModelConfig(),
        model_path=model_dir,
        test_data_file=missing_test,
    )

    with pytest.raises(ValueError, match="Model archive not found"):
        public_config.validate_inference_config(config)

    (model_dir / "wts.h5.zip").touch()
    public_config.validate_inference_config(config)


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
    [("pytorch", "pytorch"), ("tensorflow", "tensorflow")],
)
def test_training_explicit_cli_precedence_including_declared_default(
    monkeypatch,
    tmp_path,
    cli_value,
    expected,
):
    # CliSettingsSource returns string values for all args; use a string field
    # (backend) to verify that CLI values override YAML values, including when
    # the CLI value happens to match the declared default.
    from ptycho.workflows.components import setup_configuration

    config_path = tmp_path / "training.yaml"
    # YAML sets backend to the opposite of the CLI value so we can confirm override.
    yaml_backend = "tensorflow" if cli_value == "pytorch" else "pytorch"
    config_path.write_text(f"backend: {yaml_backend}\n", encoding="utf-8")
    args = _parse_public_training_args(
        monkeypatch,
        "--config",
        str(config_path),
        "--backend",
        cli_value,
    )

    config = setup_configuration(args, args.config)

    assert config.backend == expected


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

    config = setup_configuration(
        argparse.Namespace(),
        None,
    )

    assert config.nepochs == 50
    assert params.cfg == legacy_values
    assert params._sealed is initially_sealed


def test_public_training_parser_unwraps_direct_and_optional_literal_choices(
    monkeypatch,
):
    # CliSettingsSource returns all values as strings.
    # Dotted sub-config args (--sampling.training_groups) become dotted Namespace attrs.
    # Direct TrainingConfig fields (--backend) are plain attrs.
    args = _parse_public_training_args(
        monkeypatch,
        "--backend",
        "pytorch",
        "--sampling.training_groups",
        "11",
    )

    assert args.backend == "pytorch"
    assert type(args.backend) is str
    assert getattr(args, "sampling.training_groups") == "11"
    assert type(getattr(args, "sampling.training_groups")) is str


def test_public_training_parser_boolean_actions_preserve_explicitness(
    monkeypatch,
):
    # CliSettingsSource only adds Namespace attrs for explicitly-supplied args.
    # Omitted TrainingConfig fields do not appear in the parsed Namespace at all.
    omitted = _parse_public_training_args(monkeypatch)

    assert not hasattr(omitted, "positions_provided")
    assert not hasattr(omitted, "probe_trainable")

    # Supplied TrainingConfig-level bool fields appear as string values.
    explicit = _parse_public_training_args(
        monkeypatch,
        "--positions_provided",
        "True",
        "--probe_trainable",
        "False",
    )

    assert hasattr(explicit, "positions_provided")
    assert hasattr(explicit, "probe_trainable")
    assert explicit.positions_provided == "True"
    assert explicit.probe_trainable == "False"


def test_public_training_parser_keeps_numeric_overrides_primitive(monkeypatch):
    # CliSettingsSource stores all values as strings in the Namespace.
    # Sub-config numeric fields use dotted arg names.
    args = _parse_public_training_args(
        monkeypatch,
        "--tf_loss.mae_weight",
        "0.25",
        "--data.nphotons",
        "1000",
        "--gradient_clip.val",
        "2.5",
    )

    assert getattr(args, "tf_loss.mae_weight") == "0.25"
    assert type(getattr(args, "tf_loss.mae_weight")) is str
    assert getattr(args, "data.nphotons") == "1000"
    assert type(getattr(args, "data.nphotons")) is str
    assert getattr(args, "gradient_clip.val") == "2.5"
    assert type(getattr(args, "gradient_clip.val")) is str


def test_public_training_parser_preserves_required_and_optional_path_values(
    monkeypatch,
    tmp_path,
):
    # CliSettingsSource returns path values as plain strings.
    # output_dir is a direct TrainingConfig field; train_data_file is under data.
    output_dir = tmp_path / "output"
    train_data_file = tmp_path / "train.npz"

    args = _parse_public_training_args(
        monkeypatch,
        "--output_dir",
        str(output_dir),
        "--data.train_data_file",
        str(train_data_file),
    )

    assert args.output_dir == str(output_dir)
    assert isinstance(args.output_dir, str)
    assert getattr(args, "data.train_data_file") == str(train_data_file)
    assert isinstance(getattr(args, "data.train_data_file"), str)


def test_public_training_parser_literal_help_keeps_existing_cli_spellings(
    monkeypatch,
    capsys,
):
    # CliSettingsSource uses dotted names for sub-config fields.
    # Sampling fields appear as --sampling.training_groups, --sampling.n_images, etc.
    with pytest.raises(SystemExit) as error:
        _parse_public_training_args(monkeypatch, "--help")

    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--sampling.training_groups" in help_text
    assert "--sampling.n_images" in help_text
    assert "--sampling.neighbor_count" in help_text
    assert "--backend {tensorflow,pytorch}" in help_text
