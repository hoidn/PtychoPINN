"""Internal-safe contracts for transactional Torch configuration resolution."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import warnings

import numpy as np
import pytest

from ptycho import params
from ptycho_torch.execution_request import (
    ExecutionCapabilities,
    ExecutionRequest,
    TOPOLOGY_EXECUTION_COMPAT_FIELDS,
    normalize_execution_input,
)


INTERNAL_MODEL_FIELDS = frozenset(
    {
        "hybrid_skip_connections",
        "hybrid_downsample_steps",
        "hybrid_downsample_op",
        "hybrid_encoder_conv_hidden_scale",
        "hybrid_encoder_spectral_hidden_scale",
        "hybrid_encoder_conv_hidden_channels",
        "hybrid_encoder_spectral_hidden_channels",
        "hybrid_resnet_blocks",
        "hybrid_skip_style",
        "hybrid_resnet_bottleneck_layerscale_mode",
        "hybrid_resnet_bottleneck_layerscale_value",
        "hybrid_encoder_fusion_mode",
        "hybrid_encoder_layerscale_init",
        "hybrid_encoder_branch_gate_init",
        "hybrid_encoder_branch_select",
        "ffno_encoder_blocks",
        "ffno_encoder_modes",
        "ffno_encoder_share_weights",
        "ffno_encoder_gate_init",
        "ffno_encoder_norm",
        "ffno_encoder_mlp_ratio",
        "convnext_bottleneck_layerscale_init",
        "convnext_bottleneck_mlp_ratio",
        "convnext_bottleneck_kernel_size",
    }
)

EXPECTED_TORCH_ARCHITECTURES = frozenset(
    {
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
    }
)


def _resolve_training_patch(
    tmp_path: Path,
    updates: dict[str, object],
):
    resolution = importlib.import_module("ptycho_torch.config_resolution")
    return resolution.resolve_training_bundle(
        baseline=resolution.training_factory_baseline(),
        normalized=resolution.normalize_training_patch(
            {"n_groups": 4, **updates}
        ),
        observations=resolution.TrainingObservations(
            train_data_file=tmp_path / "train.npz",
            output_dir=tmp_path / "out",
            inferred_probe_size=64,
        ),
    )


def _resolve_inference_patch(
    tmp_path: Path,
    updates: dict[str, object],
):
    resolution = importlib.import_module("ptycho_torch.config_resolution")
    return resolution.resolve_inference_bundle(
        baseline=resolution.inference_factory_baseline(),
        normalized=resolution.normalize_inference_patch(
            {"n_groups": 4, **updates}
        ),
        observations=resolution.InferenceObservations(
            model_path=tmp_path / "model",
            test_data_file=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            inferred_probe_size=64,
        ),
    )


def test_training_registry_preserves_internal_model_surface() -> None:
    resolution = importlib.import_module("ptycho_torch.config_resolution")

    model_fields = dict(resolution._TRAINING_INPUTS_BY_OWNER)["model"]

    assert len(resolution.TRAINING_INPUT_RULES) == 148
    assert INTERNAL_MODEL_FIELDS <= frozenset(model_fields)


def test_torch_architecture_domain_is_the_exact_declared_fourteen() -> None:
    resolution = importlib.import_module("ptycho_torch.config_resolution")

    assert resolution.SUPPORTED_TORCH_ARCHITECTURES == (
        EXPECTED_TORCH_ARCHITECTURES
    )


def test_training_resolver_rejects_unsupported_architecture(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="architecture"):
        _resolve_training_patch(
            tmp_path,
            {"architecture": "not-an-architecture"},
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("batch_size", 0),
        ("batch_size", True),
        ("epochs", 0),
        ("epochs", True),
    ],
)
def test_training_resolver_rejects_governed_nonpositive_counts(
    tmp_path: Path,
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _resolve_training_patch(tmp_path, {field_name: invalid_value})


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("batch_size", 0),
        ("batch_size", True),
    ],
)
def test_inference_resolver_rejects_nonpositive_batch_size(
    tmp_path: Path,
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _resolve_inference_patch(tmp_path, {field_name: invalid_value})


@pytest.mark.parametrize(
    "updates",
    [
        {"K": 0},
        {"probe_scale": 0.0},
        {"probe_normalize": 1},
        {"momentum": 1.5},
        {"optimizer": "rmsprop"},
    ],
)
def test_training_resolver_does_not_invent_unsettled_field_policy(
    tmp_path: Path,
    updates: dict[str, object],
) -> None:
    _resolve_training_patch(tmp_path, updates)


def test_inference_resolver_does_not_invent_patch_weighting_policy(
    tmp_path: Path,
) -> None:
    _resolve_inference_patch(tmp_path, {"patch_weighting": "central_mask"})


def test_all_execution_topology_compatibility_inputs_identity_map_to_model() -> None:
    resolution = importlib.import_module("ptycho_torch.config_resolution")

    assert resolution.DEPRECATED_EXECUTION_MODEL_ALIASES == {
        name: name for name in TOPOLOGY_EXECUTION_COMPAT_FIELDS
    }

    request = ExecutionRequest(
        values={
            "hybrid_skip_style": "concat",
            "ffno_encoder_norm": "layer",
            "spectral_bottleneck_gate_mode": "per_block",
        },
        explicit_fields=frozenset(
            {
                "hybrid_skip_style",
                "ffno_encoder_norm",
                "spectral_bottleneck_gate_mode",
            }
        ),
    )
    normalized_execution = normalize_execution_input(
        request,
        mode="training",
    )
    assert normalized_execution is not None

    normalized = resolution.normalize_training_patch(
        {},
        normalized_execution=normalized_execution,
    )

    assert normalized.values["hybrid_skip_style"] == "concat"
    assert normalized.values["ffno_encoder_norm"] == "layer"
    assert normalized.values["spectral_bottleneck_gate_mode"] == "per_block"
    assert normalized.aliases == {
        "ffno_encoder_norm": ("ffno_encoder_norm",),
        "hybrid_skip_style": ("hybrid_skip_style",),
        "spectral_bottleneck_gate_mode": (
            "spectral_bottleneck_gate_mode",
        ),
    }


def test_training_and_inference_patches_reject_sorted_unknown_names() -> None:
    resolution = importlib.import_module("ptycho_torch.config_resolution")

    with pytest.raises(
        ValueError,
        match=r"unknown training input field\(s\): aaa, zzz",
    ):
        resolution.normalize_training_patch({"zzz": 1, "aaa": 2})
    with pytest.raises(
        ValueError,
        match=r"unknown inference input field\(s\): aaa, zzz",
    ):
        resolution.normalize_inference_patch({"zzz": 1, "aaa": 2})


def test_alias_conflicts_are_phase_consistent_and_input_is_not_mutated() -> None:
    resolution = importlib.import_module("ptycho_torch.config_resolution")
    training_patch = {"grid_size": (2, 2), "gridsize": 1}
    inference_patch = dict(training_patch)

    with pytest.raises(ValueError, match="gridsize.*grid_size.*conflicts"):
        resolution.normalize_training_patch(training_patch)
    with pytest.raises(ValueError, match="gridsize.*grid_size.*conflicts"):
        resolution.normalize_inference_patch(inference_patch)

    assert training_patch == {"grid_size": (2, 2), "gridsize": 1}
    assert inference_patch == training_patch


def test_training_resolution_is_return_new_and_derives_bundle_joins(
    tmp_path: Path,
) -> None:
    resolution = importlib.import_module("ptycho_torch.config_resolution")
    baseline = resolution.training_factory_baseline()
    assert baseline.training is not None
    baseline.training.training_directories.append("caller-owned")
    patch = {
        "n_groups": 4,
        "grid_size": (2, 2),
        "torch_loss_mode": "mae",
        "architecture": "hybrid_resnet_convnext_bottleneck",
        "convnext_bottleneck_layerscale_init": 0.25,
        "convnext_bottleneck_mlp_ratio": 3.0,
        "convnext_bottleneck_kernel_size": 5,
    }
    normalized = resolution.normalize_training_patch(patch)

    resolved = resolution.resolve_training_bundle(
        baseline=baseline,
        normalized=normalized,
        observations=resolution.TrainingObservations(
            train_data_file=tmp_path / "train.npz",
            output_dir=tmp_path / "out",
            inferred_probe_size=64,
        ),
    )

    assert resolved.data is not baseline.data
    assert resolved.model is not baseline.model
    assert resolved.training is not baseline.training
    assert resolved.inference is not baseline.inference
    assert resolved.training.training_directories is not (
        baseline.training.training_directories
    )
    assert baseline.training.training_directories == ["caller-owned"]
    assert resolved.data.C == 4
    assert resolved.model.C_model == 4
    assert resolved.model.C_forward == 4
    assert resolved.training.nll is False
    assert resolved.model.loss_function == "MAE"
    assert resolved.model.architecture == "hybrid_resnet_convnext_bottleneck"
    assert resolved.model.convnext_bottleneck_kernel_size == 5
    assert patch == {
        "n_groups": 4,
        "grid_size": (2, 2),
        "torch_loss_mode": "mae",
        "architecture": "hybrid_resnet_convnext_bottleneck",
        "convnext_bottleneck_layerscale_init": 0.25,
        "convnext_bottleneck_mlp_ratio": 3.0,
        "convnext_bottleneck_kernel_size": 5,
    }


def test_failed_training_resolution_exposes_no_partial_candidate(
    tmp_path: Path,
) -> None:
    resolution = importlib.import_module("ptycho_torch.config_resolution")
    baseline = resolution.training_factory_baseline()
    baseline_snapshot = (
        baseline.data.grid_size,
        baseline.data.C,
        baseline.model.C_model,
        baseline.model.C_forward,
    )
    normalized = resolution.normalize_training_patch(
        {
            "n_groups": 4,
            "grid_size": (2, 2),
            "C_model": 1,
        }
    )

    with pytest.raises(ValueError, match="C_model=.*derived from grid_size"):
        resolution.resolve_training_bundle(
            baseline=baseline,
            normalized=normalized,
            observations=resolution.TrainingObservations(
                train_data_file=tmp_path / "train.npz",
                output_dir=tmp_path / "out",
                inferred_probe_size=64,
            ),
        )

    assert (
        baseline.data.grid_size,
        baseline.data.C,
        baseline.model.C_model,
        baseline.model.C_forward,
    ) == baseline_snapshot


def test_inference_resolution_is_return_new_and_phase_aware(
    tmp_path: Path,
) -> None:
    resolution = importlib.import_module("ptycho_torch.config_resolution")
    inference_baseline = resolution.inference_factory_baseline()
    normalized = resolution.normalize_inference_patch(
        {
            "n_groups": 3,
            "gridsize": 2,
            "batch_size": 8,
        }
    )

    resolved = resolution.resolve_inference_bundle(
        baseline=inference_baseline,
        normalized=normalized,
        observations=resolution.InferenceObservations(
            model_path=tmp_path / "model",
            test_data_file=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            inferred_probe_size=128,
        ),
    )

    assert resolved.data is not inference_baseline.data
    assert resolved.model is not inference_baseline.model
    assert resolved.inference is not inference_baseline.inference
    assert resolved.data.N == 128
    assert resolved.data.C == 4
    assert resolved.model.C_model == 4
    assert resolved.model.C_forward == 4
    assert resolved.inference.batch_size == 8
    assert inference_baseline.data.N == 64
    assert inference_baseline.data.C == 1

    training_baseline = resolution.training_factory_baseline()
    with pytest.raises(ValueError, match="cannot be used for training"):
        resolution.resolve_training_bundle(
            baseline=training_baseline,
            normalized=normalized,
            observations=resolution.TrainingObservations(
                train_data_file=tmp_path / "train.npz",
                output_dir=tmp_path / "out",
                inferred_probe_size=64,
            ),
        )


@pytest.mark.parametrize(
    ("field_name", "compatibility_value", "canonical_value"),
    [
        ("learning_rate", 2e-3, 3e-3),
        ("scheduler", "Exponential", "WarmupCosine"),
        ("gradient_clip_val", 2.0, 3.0),
        ("gradient_clip_algorithm", "value", "agc"),
        ("accum_steps", 2, 3),
    ],
)
def test_canonical_training_owner_wins_over_all_optimizer_compatibility_inputs(
    field_name: str,
    compatibility_value: object,
    canonical_value: object,
) -> None:
    resolution = importlib.import_module("ptycho_torch.config_resolution")
    normalized_execution = normalize_execution_input(
        ExecutionRequest(
            values={field_name: compatibility_value},
            explicit_fields=frozenset({field_name}),
        ),
        mode="training",
    )
    assert normalized_execution is not None

    resolved, provenance = resolution.resolve_optimizer_ownership(
        training_baseline=None,
        normalized_execution=normalized_execution,
        canonical_training_patch={field_name: canonical_value},
    )

    assert getattr(resolved, field_name) == canonical_value
    assert provenance[field_name] == "canonical_override"


@pytest.fixture
def training_npz(tmp_path: Path) -> Path:
    path = tmp_path / "train.npz"
    np.savez(path, probeGuess=np.ones((64, 64), dtype=np.complex64))
    return path


def test_training_factory_resolves_execution_request_and_canonical_precedence(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    from ptycho_torch.config_factory import create_training_payload

    execution = ExecutionRequest(
        values={
            "accelerator": "cpu",
            "learning_rate": 4e-3,
            "hybrid_skip_style": "concat",
            "ffno_encoder_modes": 10,
        },
        explicit_fields=frozenset(
            {
                "accelerator",
                "learning_rate",
                "hybrid_skip_style",
                "ffno_encoder_modes",
            }
        ),
    )

    with pytest.warns(DeprecationWarning, match="topology fields"):
        payload = create_training_payload(
            train_data_file=training_npz,
            output_dir=tmp_path / "out",
            overrides={
                "n_groups": 4,
                "gridsize": 1,
                "learning_rate": 2e-3,
                "architecture": "hybrid_resnet",
            },
            execution_config=execution,
            execution_capabilities=ExecutionCapabilities(
                cuda_available=False,
                cuda_device_count=0,
            ),
        )

    assert payload.pt_model_config.hybrid_skip_style == "concat"
    assert payload.pt_model_config.ffno_encoder_modes == 10
    assert payload.pt_training_config.learning_rate == 2e-3
    assert payload.execution_config.accelerator == "cpu"
    assert payload.overrides_applied["training_config_provenance"][
        "learning_rate"
    ] == "canonical_override"
    assert payload.overrides_applied["topology_compatibility"] == {
        "ffno_encoder_modes": "ffno_encoder_modes",
        "hybrid_skip_style": "hybrid_skip_style",
    }


def test_factory_mirrors_effective_optimizer_owner_into_runtime_compatibility(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    from ptycho_torch.config_factory import create_training_payload

    compatibility_values = {
        "learning_rate": 2e-3,
        "scheduler": "Exponential",
        "gradient_clip_val": 2.0,
        "gradient_clip_algorithm": "value",
        "accum_steps": 2,
    }
    canonical_values = {
        "learning_rate": 3e-3,
        "scheduler": "WarmupCosine",
        "gradient_clip_val": 3.0,
        "gradient_clip_algorithm": "agc",
        "accum_steps": 3,
    }
    request = ExecutionRequest(
        values={"accelerator": "cpu", **compatibility_values},
        explicit_fields=frozenset(
            {"accelerator", *compatibility_values}
        ),
    )

    payload = create_training_payload(
        train_data_file=training_npz,
        output_dir=tmp_path / "out",
        overrides={"n_groups": 4, **canonical_values},
        execution_config=request,
        execution_capabilities=ExecutionCapabilities(
            cuda_available=False,
            cuda_device_count=0,
        ),
    )

    for field_name, expected in canonical_values.items():
        assert getattr(payload.pt_training_config, field_name) == expected
        assert getattr(payload.execution_config, field_name) == expected
        assert payload.overrides_applied["training_config_provenance"][
            field_name
        ] == "canonical_override"
    assert payload.overrides_applied["execution_runtime"][
        "explicit_fields"
    ] == sorted(request.explicit_fields)


def test_factory_classifies_unknown_input_before_resource_observation(
    tmp_path: Path,
) -> None:
    from ptycho_torch.config_factory import (
        create_inference_payload,
        create_training_payload,
    )

    with pytest.raises(
        ValueError,
        match=r"unknown training input field\(s\): typo",
    ):
        create_training_payload(
            train_data_file=tmp_path / "missing-train.npz",
            output_dir=tmp_path / "out",
            overrides={"n_groups": 4, "typo": True},
        )
    with pytest.raises(
        ValueError,
        match=r"unknown inference input field\(s\): typo",
    ):
        create_inference_payload(
            model_path=tmp_path / "missing-model",
            test_data_file=tmp_path / "missing-test.npz",
            output_dir=tmp_path / "out",
            overrides={"n_groups": 4, "typo": True},
        )


def test_supported_factories_do_not_call_tolerant_legacy_updater() -> None:
    import ptycho_torch.config_factory as factory

    assert "update_existing_config" not in inspect.getsource(
        factory.create_training_payload
    )
    assert "update_existing_config" not in inspect.getsource(
        factory.create_inference_payload
    )


def test_resolver_integration_preserves_internal_identity_versions(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    from ptycho_torch.artifact_schema import (
        ARTIFACT_SCHEMA_V1_VERSION,
        CURRENT_ARTIFACT_SCHEMA_VERSION,
    )
    from ptycho_torch.config_factory import create_training_payload
    from ptycho_torch.model_spec import (
        CURRENT_MODEL_SPEC_VERSION,
        MODEL_SPEC_V1_VERSION,
    )

    payload = create_training_payload(
        train_data_file=training_npz,
        output_dir=tmp_path / "out",
        overrides={
            "n_groups": 4,
            "gridsize": 1,
            "architecture": "hybrid_resnet_convnext_bottleneck",
            "convnext_bottleneck_layerscale_init": 0.25,
            "convnext_bottleneck_mlp_ratio": 3.0,
            "convnext_bottleneck_kernel_size": 5,
        },
        execution_config=ExecutionRequest(
            values={"accelerator": "cpu"},
            explicit_fields=frozenset({"accelerator"}),
        ),
    )

    assert payload.model_spec.schema_version == "torch-model-spec-v2"
    assert payload.model_spec.to_payload()["model_config"][
        "convnext_bottleneck_kernel_size"
    ] == 5
    assert MODEL_SPEC_V1_VERSION == "torch-model-spec-v1"
    assert CURRENT_MODEL_SPEC_VERSION == "torch-model-spec-v2"
    assert ARTIFACT_SCHEMA_V1_VERSION == "torch-artifact-v1"
    assert CURRENT_ARTIFACT_SCHEMA_VERSION == "torch-artifact-v2"


def test_factory_payload_construction_precedes_legacy_commit(
    training_npz: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ptycho_torch.config_factory as factory

    commits: list[object] = []

    monkeypatch.setattr(
        factory,
        "populate_legacy_params",
        lambda config: commits.append(config),
    )
    monkeypatch.setattr(
        factory,
        "derive_model_spec",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("model spec construction failed")
        ),
    )

    with pytest.raises(RuntimeError, match="model spec construction failed"):
        factory.create_training_payload(
            train_data_file=training_npz,
            output_dir=tmp_path / "out",
            overrides={"n_groups": 4},
            execution_config=ExecutionRequest(
                values={"accelerator": "cpu"},
                explicit_fields=frozenset({"accelerator"}),
            ),
        )

    assert commits == []


def test_failed_resolution_defers_warnings_and_filesystem_effects(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    from ptycho_torch.config_factory import create_training_payload

    output_dir = tmp_path / "must-not-exist"
    execution = ExecutionRequest(
        values={
            "accelerator": "cpu",
            "hybrid_skip_style": "concat",
        },
        explicit_fields=frozenset(
            {"accelerator", "hybrid_skip_style"}
        ),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="n_groups"):
            create_training_payload(
                train_data_file=training_npz,
                output_dir=output_dir,
                overrides={"n_groups": 0},
                execution_config=execution,
            )

    assert caught == []
    assert not output_dir.exists()


def test_partial_legacy_commit_failure_restores_mapping_and_seal_state(
    training_npz: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ptycho_torch.config_factory as factory

    original_mapping = params.cfg
    original_contents = dict(params.cfg)
    original_sealed = params._sealed

    def fail_after_partial_commit(_config: object) -> None:
        params.cfg["partial-transaction"] = "must roll back"
        raise RuntimeError("bridge failed")

    monkeypatch.setattr(
        factory,
        "populate_legacy_params",
        fail_after_partial_commit,
    )

    with pytest.raises(RuntimeError, match="bridge failed"):
        factory.create_training_payload(
            train_data_file=training_npz,
            output_dir=tmp_path / "out",
            overrides={"n_groups": 4},
            execution_config=ExecutionRequest(
                values={"accelerator": "cpu"},
                explicit_fields=frozenset({"accelerator"}),
            ),
        )

    assert params.cfg is original_mapping
    assert params.cfg == original_contents
    assert params._sealed is original_sealed


def test_invalid_architecture_fails_before_capabilities_warnings_or_global_commit(
    training_npz: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ptycho_torch.config_factory as factory
    import ptycho_torch.execution_request as execution_request

    original_contents = dict(params.cfg)
    original_sealed = params._sealed
    capability_calls: list[object] = []

    def reject_capability_observation() -> object:
        capability_calls.append(object())
        raise AssertionError("capabilities observed before domain validation")

    monkeypatch.setattr(
        execution_request,
        "observe_execution_capabilities",
        reject_capability_observation,
    )
    request = ExecutionRequest(
        values={"accelerator": "auto"},
        explicit_fields=frozenset({"accelerator"}),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="architecture"):
            factory.create_training_payload(
                train_data_file=training_npz,
                output_dir=tmp_path / "out",
                overrides={
                    "n_groups": 4,
                    "architecture": "bogus",
                },
                execution_config=request,
            )

    assert capability_calls == []
    assert caught == []
    assert params.cfg == original_contents
    assert params._sealed is original_sealed
