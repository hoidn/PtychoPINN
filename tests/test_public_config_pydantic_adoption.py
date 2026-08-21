"""Installed-version feasibility and compatibility pins for public adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from enum import StrEnum
import inspect
import os
import pickle
from pathlib import Path, PurePosixPath
from typing import Annotated
import warnings

import pytest
from pydantic import (
    BeforeValidator,
    ConfigDict,
    StrictInt,
    TypeAdapter,
    ValidationError,
    with_config,
)

from ptycho.config import (
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
    dataclass_to_legacy_dict,
    resolve_inference_config,
    resolve_training_config,
    update_legacy_dict,
    validate_inference_config_structure,
    validate_model_config_structure,
    validate_training_config_structure,
)
from ptycho.config.config import (
    DataConfig,
    GradientClipConfig,
    LossConfig,
    SamplingConfig,
    TFLossConfig,
)
from ptycho.config import resolution


_EXPECTED_ADAPTER_POLICY = {
    "extra": "forbid",
    "revalidate_instances": "always",
    "validate_default": True,
}


class _StringSubclass(str):
    pass


class _GenericPathLike(os.PathLike):
    def __init__(self, value: str):
        self.value = value

    def __fspath__(self):
        return self.value


_CLOSED_STRING_CASES = [
    ("model", "model_type", "pinn"),
    ("model", "architecture", "cnn"),
    ("model", "fno_input_transform", "none"),
    ("model", "generator_output_mode", "real_imag"),
    ("model", "amp_activation", "sigmoid"),
    ("model", "object_layout", "single_patch"),
    ("model", "training_canvas", "independent"),
    ("model", "training_patch_weighting", "central_mask"),
    ("training", "backend", "tensorflow"),
    ("training_loss", "torch_loss_mode", "poisson"),
    ("training_gradient_clip", "algorithm", "norm"),
    ("training_optimizer", "algorithm", "adam"),
    ("training_scheduler", "kind", "Default"),
    ("inference", "backend", "tensorflow"),
]


def _closed_string_variants(value: str):
    enum_type = StrEnum("PublicConfigString", {"VALUE": value})
    return (_StringSubclass(value), enum_type.VALUE)


def test_strict_scalar_primitives_are_owned_by_the_bounded_leaf():
    config_dir = Path(resolution.__file__).parent
    strict_types_path = config_dir / "strict_types.py"

    assert strict_types_path.is_file()
    source = strict_types_path.read_text(encoding="utf-8")
    for name in (
        "_require_exact_int",
        "_require_exact_optional_int",
        "_require_exact_bool",
        "_require_exact_finite_number",
        "_require_exact_str",
        "_StrictPositiveInt",
        "_StrictNonNegativeInt",
        "_StrictOptionalInt",
        "_StrictBool",
        "_StrictFinitePositiveNumber",
    ):
        assert f"def {name}" in source or f"{name} =" in source


def test_public_adapter_owner_modules_obey_the_static_architecture():
    config_dir = Path(resolution.__file__).parent
    owners = {"config.py", "resolution.py", "strict_types.py"}
    forbidden = (
        "BaseModel",
        "pydantic.dataclasses",
        "validate_assignment",
        "model_dump",
        "dump_python",
        "dump_json",
    )

    for path in config_dir.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "pydantic" in source or "TypeAdapter" in source:
            assert path.name in owners
        if path.name in owners:
            assert all(token not in source for token in forbidden)
            assert "TypeAdapter(DatagenConfig" not in source


def _exact_int(value):
    if type(value) is not int:
        raise ValueError("must be an exact built-in integer")
    return value


def test_f1_installed_pydantic_revalidates_mutable_stdlib_dataclasses():
    @with_config(ConfigDict(**_EXPECTED_ADAPTER_POLICY))
    @dataclass
    class MutableRecord:
        value: Annotated[StrictInt, BeforeValidator(_exact_int)]

    adapter = TypeAdapter(MutableRecord)
    assert adapter.validate_python({"value": 2}) == MutableRecord(2)

    with pytest.raises(ValidationError):
        adapter.validate_python({"value": "2"})
    with pytest.raises(ValidationError):
        adapter.validate_python({"value": 2, "unknown": 1})

    mutated = MutableRecord(2)
    mutated.value = "2"
    with pytest.raises(ValidationError):
        adapter.validate_python(mutated, strict=True)


def test_f3_public_records_remain_reflection_neutral_stdlib_dataclasses():
    model_signature = inspect.signature(ModelConfig)
    training_signature = inspect.signature(TrainingConfig)
    inference_signature = inspect.signature(InferenceConfig)

    assert dict(ModelConfig.__pydantic_config__) == _EXPECTED_ADAPTER_POLICY
    assert dict(TrainingConfig.__pydantic_config__) == _EXPECTED_ADAPTER_POLICY
    assert dict(InferenceConfig.__pydantic_config__) == _EXPECTED_ADAPTER_POLICY
    assert "N" in {item.name for item in fields(ModelConfig)}
    assert tuple(training_signature.parameters)[:3] == (
        "model",
        "batch_size",
        "nepochs",
    )
    assert tuple(inference_signature.parameters)[:3] == (
        "model",
        "model_path",
        "test_data_file",
    )
    assert "N" in model_signature.parameters

    model = ModelConfig()
    training = TrainingConfig(model, sampling=SamplingConfig(training_groups=7))
    assert ModelConfig() == model
    assert hash(ModelConfig()) == hash(model)
    assert replace(training, batch_size=8).batch_size == 8
    assert asdict(training)["model"]["N"] == 64
    assert pickle.loads(pickle.dumps(training)) == training


def test_f4_cached_complete_adapters_report_nested_dotted_paths():
    assert isinstance(resolution._MODEL_CONFIG_ADAPTER, TypeAdapter)
    assert isinstance(resolution._TRAINING_CONFIG_ADAPTER, TypeAdapter)
    assert isinstance(resolution._INFERENCE_CONFIG_ADAPTER, TypeAdapter)

    with pytest.raises(ValueError, match=r"training\.model\.gridsize"):
        resolve_training_config({"model": {"gridsize": "2"}}, {})

    mutated = TrainingConfig(ModelConfig(), sampling=SamplingConfig(training_groups=7))
    mutated.model = "not-a-model"
    with pytest.raises(ValueError, match=r"training\.model"):
        validate_training_config_structure(mutated)


@pytest.mark.parametrize("owner,field,value", _CLOSED_STRING_CASES)
def test_mapping_boundaries_reject_closed_string_subclasses(owner, field, value):
    for invalid in _closed_string_variants(value):
        if owner == "model":
            source = {field: invalid}
            resolver = resolve_training_config
        elif owner == "training":
            source = {field: invalid}
            resolver = resolve_training_config
        elif owner == "training_loss":
            source = {"loss": {field: invalid}}
            resolver = resolve_training_config
        elif owner == "training_gradient_clip":
            source = {"gradient_clip": {field: invalid}}
            resolver = resolve_training_config
        elif owner == "training_optimizer":
            source = {"optimizer": {field: invalid}}
            resolver = resolve_training_config
        elif owner == "training_scheduler":
            source = {"scheduler": {field: invalid}}
            resolver = resolve_training_config
        else:
            source = {
                "model": {},
                "model_path": "model",
                "test_data_file": "test.npz",
                field: invalid,
            }
            resolver = resolve_inference_config

        with pytest.raises(ValueError, match=field):
            resolver(source, {})


@pytest.mark.parametrize("owner,field,value", _CLOSED_STRING_CASES)
def test_strict_instance_boundaries_reject_closed_string_subclasses(
    owner,
    field,
    value,
):
    for invalid in _closed_string_variants(value):
        if owner == "model":
            record = replace(ModelConfig(), **{field: invalid})
            validator = validate_model_config_structure
        elif owner == "training":
            record = TrainingConfig(ModelConfig(), **{field: invalid})
            validator = validate_training_config_structure
        elif owner == "training_loss":
            record = TrainingConfig(
                ModelConfig(), loss=LossConfig(**{field: invalid})
            )
            validator = validate_training_config_structure
        elif owner == "training_gradient_clip":
            record = TrainingConfig(
                ModelConfig(), gradient_clip=GradientClipConfig(**{field: invalid})
            )
            validator = validate_training_config_structure
        elif owner == "training_optimizer":
            from ptycho.config.config import OptimizerConfig
            record = TrainingConfig(
                ModelConfig(), optimizer=OptimizerConfig(**{field: invalid})
            )
            validator = validate_training_config_structure
        elif owner == "training_scheduler":
            from ptycho.config.config import SchedulerConfig
            record = TrainingConfig(
                ModelConfig(), scheduler=SchedulerConfig(**{field: invalid})
            )
            validator = validate_training_config_structure
        else:
            record = InferenceConfig(
                ModelConfig(),
                Path("model"),
                Path("test.npz"),
                **{field: invalid},
            )
            validator = validate_inference_config_structure

        with pytest.raises(ValueError, match=field):
            validator(record)


@pytest.mark.parametrize(
    "validator,record",
    [
        (
            validate_training_config_structure,
            TrainingConfig(
                ModelConfig(
                    object_layout="single_patch",
                    training_canvas="relative_overlap",
                ),
                sampling=SamplingConfig(training_groups=7),
            ),
        ),
        (
            validate_inference_config_structure,
            InferenceConfig(
                ModelConfig(
                    object_layout="single_patch",
                    training_canvas="relative_overlap",
                ),
                Path("model"),
                Path("test.npz"),
            ),
        ),
    ],
)
def test_root_validators_retain_nested_model_semantics(validator, record):
    with pytest.raises(ValueError, match="object_layout|training_canvas"):
        validator(record)


@pytest.mark.parametrize(
    "resolver,source",
    [
        (
            resolve_training_config,
            {
                "model": {
                    "object_layout": "single_patch",
                    "training_canvas": "relative_overlap",
                },
            },
        ),
        (
            resolve_inference_config,
            {
                "model": {
                    "object_layout": "single_patch",
                    "training_canvas": "relative_overlap",
                },
                "model_path": "model",
                "test_data_file": "test.npz",
            },
        ),
    ],
)
def test_root_resolvers_retain_nested_model_semantics(resolver, source):
    with pytest.raises(ValueError, match="object_layout|training_canvas"):
        resolver(source, {})


def test_f5_path_conversion_matches_the_manual_public_boundary():
    training = resolve_training_config(
        {
            "model": {},
            "data": {
                "train_data_file": "train.npz",
                "test_data_file": Path("test.npz"),
            },
            "output_dir": "outputs",
        },
        {},
    )
    inference = resolve_inference_config(
        {
            "model": {},
            "model_path": "model",
            "test_data_file": Path("test.npz"),
            "output_dir": "inference",
        },
        {},
    )

    assert training.data.train_data_file == Path("train.npz")
    assert type(training.data.train_data_file) is type(Path())
    assert training.data.test_data_file == Path("test.npz")
    assert inference.model_path == Path("model")
    assert type(inference.output_dir) is type(Path())


@pytest.mark.parametrize(
    "owner,field",
    [
        ("training_data", "train_data_file"),
        ("training_data", "test_data_file"),
        ("training", "output_dir"),
        ("inference", "model_path"),
        ("inference", "test_data_file"),
        ("inference", "output_dir"),
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        PurePosixPath("pure-path.npz"),
        _GenericPathLike("generic-pathlike.npz"),
    ],
)
def test_mapping_path_conversion_matches_path_constructor(owner, field, value):
    if owner == "training_data":
        resolved = resolve_training_config({"data": {field: value}}, {})
        assert getattr(resolved.data, field) == Path(value)
        assert type(getattr(resolved.data, field)) is type(Path())
    elif owner == "training":
        resolved = resolve_training_config({field: value}, {})
        assert getattr(resolved, field) == Path(value)
        assert type(getattr(resolved, field)) is type(Path())
    else:
        source = {
            "model": {},
            "model_path": "model",
            "test_data_file": "test.npz",
            field: value,
        }
        resolved = resolve_inference_config(source, {})
        assert getattr(resolved, field) == Path(value)
        assert type(getattr(resolved, field)) is type(Path())


@pytest.mark.parametrize(
    "owner,field,dotted_path",
    [
        ("training_data", "train_data_file", r"training\.data\.train_data_file"),
        ("training_data", "test_data_file", r"training\.data\.test_data_file"),
        ("training", "output_dir", r"training\.output_dir"),
        ("inference", "model_path", r"inference\.model_path"),
        ("inference", "test_data_file", r"inference\.test_data_file"),
        ("inference", "output_dir", r"inference\.output_dir"),
    ],
)
def test_mapping_path_conversion_rejects_bytes_with_dotted_errors(owner, field, dotted_path):
    if owner == "training_data":
        source = {"data": {field: b"bytes-path"}}
        resolver = resolve_training_config
    elif owner == "training":
        source = {field: b"bytes-path"}
        resolver = resolve_training_config
    else:
        source = {
            "model": {},
            "model_path": "model",
            "test_data_file": "test.npz",
            field: b"bytes-path",
        }
        resolver = resolve_inference_config

    with pytest.raises(ValueError, match=dotted_path):
        resolver(source, {})


@pytest.mark.parametrize(
    "owner,field,dotted_path",
    [
        ("training_data", "train_data_file", r"training\.data\.train_data_file"),
        ("training_data", "test_data_file", r"training\.data\.test_data_file"),
        ("training", "output_dir", r"training\.output_dir"),
        ("inference", "model_path", r"inference\.model_path"),
        ("inference", "test_data_file", r"inference\.test_data_file"),
        ("inference", "output_dir", r"inference\.output_dir"),
    ],
)
@pytest.mark.parametrize(
    "value",
    ["string-path", PurePosixPath("pure-path"), _GenericPathLike("pathlike")],
)
def test_strict_instance_path_validation_rejects_non_path(owner, field, dotted_path, value):
    if owner == "training_data":
        record = TrainingConfig(ModelConfig(), data=DataConfig(**{field: value}))
        validator = validate_training_config_structure
    elif owner == "training":
        record = TrainingConfig(ModelConfig(), **{field: value})
        validator = validate_training_config_structure
    else:
        values = {
            "model": ModelConfig(),
            "model_path": Path("model"),
            "test_data_file": Path("test.npz"),
            field: value,
        }
        record = InferenceConfig(**values)
        validator = validate_inference_config_structure

    with pytest.raises(ValueError, match=dotted_path):
        validator(record)


def test_f7_post_init_alias_and_default_behavior_is_stable():
    defaulted = resolve_training_config({"model": {}}, {})
    assert defaulted.sampling.training_groups == 512

    with pytest.warns(DeprecationWarning, match="n_images") as caught:
        directly_adapted = resolution._TRAINING_CONFIG_ADAPTER.validate_python(
            {"model": {}, "sampling": {"n_images": 8}}
        )
    assert directly_adapted.sampling.training_groups == 8
    assert len(caught) == 1

    with pytest.warns(DeprecationWarning, match="n_images") as caught:
        aliased = resolve_training_config({"model": {}, "sampling": {"n_images": 9}}, {})
    assert aliased.sampling.training_groups == 9
    assert len(caught) == 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        canonical = resolve_training_config(
            {"model": {}, "sampling": {"training_groups": 9, "n_images": None}},
            {},
        )
        validate_training_config_structure(canonical)
    assert canonical.sampling.training_groups == 9
    assert caught == []


@pytest.mark.parametrize(
    "record,adapter",
    [
        (
            TrainingConfig(
                ModelConfig(),
                data=DataConfig(train_data_file=Path("train.npz")),
                sampling=SamplingConfig(training_groups=7),
            ),
            lambda: resolution._TRAINING_CONFIG_ADAPTER,
        ),
        (
            InferenceConfig(
                ModelConfig(),
                Path("model"),
                Path("test.npz"),
                inference_groups=7,
            ),
            lambda: resolution._INFERENCE_CONFIG_ADAPTER,
        ),
    ],
)
def test_f8_legacy_projection_is_byte_identical(record, adapter):
    baseline_projection = pickle.dumps(dataclass_to_legacy_dict(record))
    baseline_update = {}
    update_legacy_dict(baseline_update, record)

    adapted = adapter().validate_python(asdict(record))
    adapted_update = {}
    update_legacy_dict(adapted_update, adapted)

    assert pickle.dumps(dataclass_to_legacy_dict(adapted)) == baseline_projection
    assert pickle.dumps(adapted_update) == pickle.dumps(baseline_update)
