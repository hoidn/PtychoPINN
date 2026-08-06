"""Transactional normalization contracts for Torch phase configuration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import inspect
import math
from pathlib import Path
from types import MappingProxyType
import warnings

import pytest
import torch

from ptycho.config.config import (
    ModelConfig as PublicModelConfig,
    SchedulerConfig as PublicSchedulerConfig,
    TrainingConfig as PublicTrainingConfig,
)
from ptycho_torch.config_resolution import (
    INFERENCE_INPUT_RULES,
    TRAINING_INPUT_RULES,
    InferenceObservations,
    NormalizedPatch,
    TorchConfigBaseline,
    TrainingObservations,
    inference_factory_baseline,
    normalize_inference_patch,
    normalize_training_patch,
    observe_probe_size,
    resolve_inference_bundle,
    resolve_training_bundle,
    training_factory_baseline,
)
from ptycho_torch.config_params import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)
from ptycho_torch.execution_request import (
    EnvironmentResolution,
    ExecutionRequest,
    ResolutionNotice,
    normalize_execution_input,
    resolve_runtime_execution_request,
)


TRAINING_INPUTS_BY_OWNER = {
    "data": {
        "nphotons",
        "scale_contract_version",
        "measurement_domain",
        "N",
        "K",
        "K_quadrant",
        "n_subsample",
        "subsample_seed",
        "grid_size",
        "neighbor_function",
        "min_neighbor_distance",
        "max_neighbor_distance",
        "scan_pattern",
        "normalize",
        "probe_scale",
        "probe_normalize",
        "data_scaling",
        "phase_subtraction",
        "x_bounds",
        "y_bounds",
    },
    "model": {
        "mode",
        "architecture",
        "fno_modes",
        "fno_width",
        "fno_blocks",
        "fno_cnn_blocks",
        "learned_input_channels",
        "fno_input_transform",
        "max_hidden_channels",
        "resnet_width",
        "spectral_bottleneck_blocks",
        "spectral_bottleneck_modes",
        "spectral_bottleneck_share_weights",
        "spectral_bottleneck_gate_init",
        "spectral_bottleneck_gate_mode",
        "generator_output_mode",
        "cnn_output_mode",
        "use_shared_decoder",
        "intensity_scale_trainable",
        "intensity_scale",
        "max_position_jitter",
        "num_datasets",
        "n_filters_scale",
        "amp_activation",
        "batch_norm",
        "probe_mask",
        "probe_mask_tensor",
        "probe_mask_sigma",
        "probe_mask_diameter",
        "edge_pad",
        "decoder_last_c_outer_fraction",
        "decoder_last_amp_channels",
        "use_legacy_decoder_channel_override",
        "eca_encoder",
        "cbam_encoder",
        "cbam_bottleneck",
        "cbam_decoder",
        "eca_decoder",
        "spatial_decoder",
        "decoder_spatial_kernel",
        "object_big",
        "object_layout",
        "training_canvas",
        "probe_big",
        "offset",
        "training_patch_weighting",
        "physics_forward_mode",
        "rect_s1s2_trainable",
        "rect_s1s2_init",
        "amplitude_physics_gain",
        "pad_object",
        "gaussian_smoothing_sigma",
        "amp_loss",
        "phase_loss",
        "amp_loss_coeff",
        "phase_loss_coeff",
    },
    "training": {
        "training_directories",
        "framework",
        "orchestrator",
        "learning_rate",
        "epochs",
        "batch_size",
        "epochs_fine_tune",
        "fine_tune_gamma",
        "scheduler",
        "lr_warmup_epochs",
        "lr_min_ratio",
        "plateau_factor",
        "plateau_patience",
        "plateau_min_lr",
        "plateau_threshold",
        "accum_steps",
        "gradient_clip_val",
        "gradient_clip_algorithm",
        "optimizer",
        "momentum",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "log_grad_norm",
        "grad_norm_log_freq",
        "stage_1_epochs",
        "stage_2_epochs",
        "stage_3_epochs",
        "physics_weight_schedule",
        "stage_3_lr_factor",
        "torch_loss_mode",
        "torch_mae_pred_l2_match_target",
        "experiment_name",
        "notes",
        "model_name",
        "test_data_file",
        "n_groups",
    },
    "inference": {
        "middle_trim",
        "inference_batch_size",
        "experiment_number",
        "pad_eval",
        "window",
        "patch_weighting",
        "varpro_scaling",
        "log_patch_stats",
        "patch_stats_limit",
    },
    "bridge": {
        "enable_oversampling",
        "neighbor_pool_size",
        "sequential_sampling",
    },
    "derived_constraint": {
        "C",
        "C_model",
        "C_forward",
        "loss_function",
        "nll",
        "train_data_file",
        "output_dir",
    },
}

INFERENCE_INPUTS_BY_OWNER = {
    "data": {
        "N",
        "K",
        "grid_size",
        "probe_scale",
        "subsample_seed",
        "scale_contract_version",
        "measurement_domain",
    },
    "model": {
        "mode",
        "amp_activation",
        "n_filters_scale",
        "object_big",
        "object_layout",
        "training_canvas",
        "training_patch_weighting",
        "probe_big",
        "probe_mask",
        "probe_mask_tensor",
        "probe_mask_sigma",
        "probe_mask_diameter",
        "pad_object",
        "gaussian_smoothing_sigma",
    },
    "inference": {
        "batch_size",
        "patch_weighting",
        "varpro_scaling",
        "log_patch_stats",
        "patch_stats_limit",
    },
    "bridge": {
        "n_groups",
        "n_subsample",
    },
    "derived_constraint": {
        "C",
        "C_model",
        "C_forward",
        "model_path",
        "test_data_file",
        "output_dir",
    },
}

TRAINING_ALIASES = {
    "gridsize": "grid_size",
    "neighbor_count": "K",
    "model_type": "mode",
    "max_epochs": "epochs",
}
INFERENCE_ALIASES = {
    "gridsize": "grid_size",
    "neighbor_count": "K",
    "model_type": "mode",
}

def _rules_by_owner(rules):
    by_owner: dict[str, set[str]] = {}
    for rule in rules:
        by_owner.setdefault(rule.owner, set()).add(rule.canonical)
    return by_owner


def _aliases_from_rules(rules):
    return {
        alias: rule.canonical
        for rule in rules
        for alias in rule.aliases
    }


def test_training_registry_matches_literal_phase_inventory() -> None:
    assert len(TRAINING_INPUT_RULES) == 132
    assert _rules_by_owner(TRAINING_INPUT_RULES) == TRAINING_INPUTS_BY_OWNER
    assert _aliases_from_rules(TRAINING_INPUT_RULES) == TRAINING_ALIASES


def test_inference_registry_matches_literal_phase_inventory() -> None:
    assert len(INFERENCE_INPUT_RULES) == 34
    assert _rules_by_owner(INFERENCE_INPUT_RULES) == INFERENCE_INPUTS_BY_OWNER
    assert _aliases_from_rules(INFERENCE_INPUT_RULES) == INFERENCE_ALIASES


def test_training_patch_rejects_sorted_unknown_names() -> None:
    patch = {"zeta_typo": 1, "alpha_typo": 2}

    with pytest.raises(
        ValueError,
        match=r"unknown training input.*alpha_typo.*zeta_typo",
    ):
        normalize_training_patch(patch)

    assert patch == {"zeta_typo": 1, "alpha_typo": 2}


def test_inference_patch_rejects_sorted_unknown_names() -> None:
    patch = {"zeta_typo": 1, "batch_szie": 4}

    with pytest.raises(
        ValueError,
        match=r"unknown inference input.*batch_szie.*zeta_typo",
    ):
        normalize_inference_patch(patch)

    assert patch == {"zeta_typo": 1, "batch_szie": 4}


@pytest.mark.parametrize(
    ("normalizer", "patch", "canonical", "expected"),
    [
        (
            normalize_training_patch,
            {"max_epochs": 7, "epochs": 7},
            "epochs",
            7,
        ),
        (
            normalize_training_patch,
            {"neighbor_count": 4, "K": 4},
            "K",
            4,
        ),
        (
            normalize_training_patch,
            {"model_type": "Supervised", "mode": "Supervised"},
            "mode",
            "Supervised",
        ),
        (
            normalize_training_patch,
            {"gridsize": 2, "grid_size": (2, 2)},
            "grid_size",
            (2, 2),
        ),
        (
            normalize_inference_patch,
            {"neighbor_count": 4, "K": 4},
            "K",
            4,
        ),
        (
            normalize_inference_patch,
            {"model_type": "Supervised", "mode": "Supervised"},
            "mode",
            "Supervised",
        ),
        (
            normalize_inference_patch,
            {"gridsize": 2, "grid_size": (2, 2)},
            "grid_size",
            (2, 2),
        ),
    ],
)
def test_equal_alias_and_canonical_are_consumed_once(
    normalizer,
    patch,
    canonical,
    expected,
) -> None:
    before = dict(patch)

    normalized = normalizer(patch)

    assert normalized.values[canonical] == expected
    assert normalized.audit[canonical] == expected
    assert set(normalized.audit) == {canonical}
    assert normalized.aliases[canonical]
    assert patch == before


@pytest.mark.parametrize(
    ("normalizer", "patch", "alias", "canonical"),
    [
        (
            normalize_training_patch,
            {"max_epochs": 6, "epochs": 7},
            "max_epochs",
            "epochs",
        ),
        (
            normalize_training_patch,
            {"neighbor_count": 5, "K": 4},
            "neighbor_count",
            "K",
        ),
        (
            normalize_training_patch,
            {"model_type": "Supervised", "mode": "Unsupervised"},
            "model_type",
            "mode",
        ),
        (
            normalize_training_patch,
            {"gridsize": 2, "grid_size": (3, 3)},
            "gridsize",
            "grid_size",
        ),
        (
            normalize_inference_patch,
            {"neighbor_count": 5, "K": 4},
            "neighbor_count",
            "K",
        ),
        (
            normalize_inference_patch,
            {"model_type": "Supervised", "mode": "Unsupervised"},
            "model_type",
            "mode",
        ),
        (
            normalize_inference_patch,
            {"gridsize": 2, "grid_size": (3, 3)},
            "gridsize",
            "grid_size",
        ),
    ],
)
def test_unequal_alias_and_canonical_fail_without_mutating_input(
    normalizer,
    patch,
    alias,
    canonical,
) -> None:
    before = dict(patch)

    with pytest.raises(
        ValueError,
        match=rf"alias.*{alias}.*{canonical}.*conflict",
    ):
        normalizer(patch)

    assert patch == before


def test_normalized_patch_is_an_immutable_snapshot() -> None:
    patch = {"epochs": 7, "max_epochs": 7}

    normalized = normalize_training_patch(patch)
    patch["epochs"] = 99

    mapping_proxy_type = type(MappingProxyType({}))
    assert isinstance(normalized.values, mapping_proxy_type)
    assert isinstance(normalized.audit, mapping_proxy_type)
    assert isinstance(normalized.aliases, mapping_proxy_type)
    assert normalized.values["epochs"] == 7
    assert normalized.aliases["epochs"] == ("max_epochs",)
    with pytest.raises(TypeError):
        normalized.values["epochs"] = 8
    with pytest.raises(TypeError):
        normalized.audit["epochs"] = 8
    with pytest.raises(TypeError):
        normalized.aliases["epochs"] = ()
    with pytest.raises(FrozenInstanceError):
        normalized.notices = ()


def test_execution_compatibility_resolution_surface_is_absent() -> None:
    import ptycho_torch.config_resolution as resolution

    assert not hasattr(resolution, "DEPRECATED_EXECUTION_MODEL_ALIASES")
    assert not hasattr(resolution, "resolve_optimizer_ownership")
    assert "normalized_execution" not in inspect.signature(
        resolution.normalize_training_patch
    ).parameters
    assert "normalized_execution" not in inspect.signature(
        resolution.resolve_training_bundle
    ).parameters


def _training_observations(
    *,
    inferred_probe_size: int = 128,
    photon_metadata: float | None = 2.5e8,
    notices: tuple[ResolutionNotice, ...] = (),
) -> TrainingObservations:
    return TrainingObservations(
        train_data_file=Path("train.npz"),
        output_dir=Path("out"),
        inferred_probe_size=inferred_probe_size,
        photon_metadata=photon_metadata,
        notices=notices,
    )


def _inference_observations(
    *,
    inferred_probe_size: int = 128,
    notices: tuple[ResolutionNotice, ...] = (),
) -> InferenceObservations:
    return InferenceObservations(
        model_path=Path("model"),
        test_data_file=Path("test.npz"),
        output_dir=Path("recon"),
        inferred_probe_size=inferred_probe_size,
        notices=notices,
    )


def test_factory_specific_baselines_lock_all_phase_divergences() -> None:
    training = training_factory_baseline()
    inference = inference_factory_baseline()

    assert training.data.grid_size == (1, 1)
    assert training.data.C == 1
    assert training.data.K == 6
    assert training.data.nphotons == 1e9
    assert training.model.C_model == 1
    assert training.model.C_forward == 1
    assert training.model.loss_function == "Poisson"
    assert training.model.object_layout == "grouped_patches"
    assert training.model.training_canvas == "relative_overlap"
    assert training.model.training_patch_weighting == "central_mask"
    assert training.training == TrainingConfig()
    assert training.inference.batch_size == 1000
    assert training.inference.patch_weighting == "probe"
    assert training.inference.varpro_scaling is True
    assert training.inference.log_patch_stats is False
    assert training.inference.patch_stats_limit is None

    assert inference.data.grid_size == (1, 1)
    assert inference.data.C == 1
    assert inference.data.K == 4
    assert inference.data.nphotons == DataConfig().nphotons
    assert inference.data.scale_contract_version == "ci_intensity_v2"
    assert inference.data.measurement_domain == "count_intensity"
    assert inference.model.C_model == 1
    assert inference.model.C_forward == 1
    assert inference.model.object_layout == "grouped_patches"
    assert inference.model.training_canvas == "relative_overlap"
    assert inference.model.training_patch_weighting == "central_mask"
    assert inference.training is None
    assert inference.inference.batch_size == 16
    assert inference.inference.patch_weighting == "probe"
    assert inference.inference.varpro_scaling is True
    assert inference.inference.log_patch_stats is False
    assert inference.inference.patch_stats_limit is None


def test_training_factory_baseline_tracks_default_and_public_sources() -> None:
    defaults = training_factory_baseline()
    public = training_factory_baseline(
        training_baseline=PublicTrainingConfig(
            model=PublicModelConfig(),
            scheduler=PublicSchedulerConfig(kind="WarmupCosine"),
        )
    )

    assert defaults.training.scheduler == "Default"
    assert defaults.training_provenance["scheduler"] == "torch_default"
    assert public.training.scheduler == "WarmupCosine"
    assert public.training_provenance["scheduler"] == "training_baseline"


def test_training_resolution_returns_fresh_records_without_mutating_inputs() -> None:
    probe_mask = torch.ones(4, 4)
    directories = ["existing"]
    baseline = TorchConfigBaseline(
        data=DataConfig(grid_size=(1, 1), C=1, N=64, nphotons=1e9),
        model=ModelConfig(
            C_model=1,
            C_forward=1,
            probe_mask_tensor=probe_mask,
        ),
        training=TrainingConfig(
            epochs=3,
            training_directories=directories,
        ),
        inference=InferenceConfig(batch_size=8),
    )
    patch = {"epochs": 9, "n_groups": 16}
    normalized = normalize_training_patch(patch)

    resolved = resolve_training_bundle(
        baseline=baseline,
        normalized=normalized,
        observations=_training_observations(),
    )

    assert resolved.data.N == 128
    assert resolved.data.nphotons == 2.5e8
    assert resolved.training.epochs == 9
    assert resolved.training.n_groups == 16
    assert resolved.training.train_data_file == "train.npz"
    assert resolved.training.output_dir == "out"
    assert resolved.data is not baseline.data
    assert resolved.model is not baseline.model
    assert resolved.training is not baseline.training
    assert resolved.inference is not baseline.inference
    assert resolved.model.probe_mask_tensor is probe_mask
    assert resolved.training.training_directories == ["existing"]
    assert (
        resolved.training.training_directories
        is not baseline.training.training_directories
    )
    assert baseline.data.N == 64
    assert baseline.data.nphotons == 1e9
    assert baseline.training.epochs == 3
    assert baseline.training.n_groups is None
    assert baseline.training.train_data_file is None
    assert baseline.training.output_dir == "training_outputs"
    assert directories == ["existing"]
    assert patch == {"epochs": 9, "n_groups": 16}
    assert resolved.audit["amplitude_physics_gain"] == 1.0
    assert resolved.audit["learning_rate"] == baseline.training.learning_rate


def test_mutable_patch_values_are_independent_snapshots() -> None:
    x_bounds = [0.1, 0.9]
    probe_mask = torch.ones(4, 4)
    patch = {
        "x_bounds": x_bounds,
        "probe_mask_tensor": probe_mask,
        "n_groups": 1,
    }

    normalized = normalize_training_patch(patch)
    x_bounds.append(1.0)
    resolved = resolve_training_bundle(
        baseline=training_factory_baseline(),
        normalized=normalized,
        observations=_training_observations(),
    )

    normalized.values["x_bounds"].append(2.0)
    normalized.audit["x_bounds"].append(3.0)
    resolved.data.x_bounds.append(4.0)
    resolved.audit["x_bounds"].append(5.0)

    assert patch["x_bounds"] == [0.1, 0.9, 1.0]
    assert normalized.values["x_bounds"] == [0.1, 0.9, 2.0]
    assert normalized.audit["x_bounds"] == [0.1, 0.9, 3.0]
    assert resolved.data.x_bounds == [0.1, 0.9, 4.0]
    assert resolved.audit["x_bounds"] == [0.1, 0.9, 5.0]
    assert normalized.values["probe_mask_tensor"] is probe_mask
    assert normalized.audit["probe_mask_tensor"] is probe_mask
    assert resolved.model.probe_mask_tensor is probe_mask
    assert resolved.audit["probe_mask_tensor"] is probe_mask

    bridge_value = ["source"]
    bridge_bundle = replace(
        resolved,
        bridge={"mutable": bridge_value},
    )
    bridge_value.append("caller")
    bridge_bundle.bridge["mutable"].append("bundle")

    assert bridge_value == ["source", "caller"]
    assert bridge_bundle.bridge["mutable"] == ["source", "bundle"]


def test_execution_request_freezes_reconstruction_indices_through_resolution() -> None:
    source_indices = [1, 3]
    request = ExecutionRequest(
        values={
            "accelerator": "cpu",
            "recon_log_fixed_indices": source_indices,
        },
        explicit_fields=frozenset(
            {"accelerator", "recon_log_fixed_indices"}
        ),
    )

    source_indices.append(5)
    returned = request.as_dict()
    returned["recon_log_fixed_indices"].append(7)
    normalized = normalize_execution_input(request, mode="training")

    assert request.values["recon_log_fixed_indices"] == (1, 3)
    assert returned["recon_log_fixed_indices"] == [1, 3, 7]
    assert normalized is not None
    assert normalized.values["recon_log_fixed_indices"] == (1, 3)

    resolved = resolve_runtime_execution_request(
        normalized,
        mode="training",
    )

    assert resolved.config.recon_log_fixed_indices == [1, 3]
    assert isinstance(resolved.config.recon_log_fixed_indices, list)
    resolved.config.recon_log_fixed_indices.append(9)
    assert normalized.values["recon_log_fixed_indices"] == (1, 3)


def test_bare_resolved_execution_config_is_not_an_unresolved_input() -> None:
    from ptycho.config.config import PyTorchExecutionConfig

    config = PyTorchExecutionConfig(
        accelerator="cpu",
        recon_log_fixed_indices=[2, 4],
    )

    with pytest.raises(TypeError, match="resolved output carrier"):
        normalize_execution_input(config, mode="training")


def test_environment_resolution_snapshots_mutable_execution_values() -> None:
    requested_indices = [1, 2]
    resolved_indices = [3, 4]
    resolution = EnvironmentResolution(
        requested={"recon_log_fixed_indices": requested_indices},
        resolved={"recon_log_fixed_indices": resolved_indices},
        capabilities=None,
    )

    requested_indices.append(5)
    resolved_indices.append(6)

    assert resolution.requested["recon_log_fixed_indices"] == (1, 2)
    assert resolution.resolved["recon_log_fixed_indices"] == (3, 4)


def test_training_resolver_rejects_inference_normalized_patch() -> None:
    normalized = normalize_inference_patch({"n_groups": 1})

    with pytest.raises(ValueError, match=r"inference.*training"):
        resolve_training_bundle(
            baseline=training_factory_baseline(),
            normalized=normalized,
            observations=_training_observations(),
        )


def test_inference_resolver_rejects_training_normalized_patch() -> None:
    normalized = normalize_training_patch({"n_groups": 1})

    with pytest.raises(ValueError, match=r"training.*inference"):
        resolve_inference_bundle(
            baseline=inference_factory_baseline(),
            normalized=normalized,
            observations=_inference_observations(),
        )


@pytest.mark.parametrize(
    ("patch", "metadata", "expected_n", "expected_nphotons", "n_source", "p_source"),
    [
        (
            {"N": 256, "nphotons": 7.0, "n_groups": 1},
            9.0,
            256,
            7.0,
            "explicit",
            "explicit",
        ),
        (
            {"n_groups": 1},
            9.0,
            128,
            9.0,
            "observation",
            "metadata",
        ),
        (
            {"n_groups": 1},
            None,
            128,
            1e9,
            "observation",
            "declared_default",
        ),
    ],
)
def test_training_N_and_nphotons_have_declared_precedence(
    patch,
    metadata,
    expected_n,
    expected_nphotons,
    n_source,
    p_source,
) -> None:
    resolved = resolve_training_bundle(
        baseline=training_factory_baseline(),
        normalized=normalize_training_patch(patch),
        observations=_training_observations(photon_metadata=metadata),
    )

    assert resolved.data.N == expected_n
    assert resolved.data.nphotons == expected_nphotons
    assert resolved.audit["N_source"] == n_source
    assert resolved.audit["nphotons_source"] == p_source


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [
        ("scheduler", "CosineAnnealing"),
        ("gradient_clip_algorithm", "bogus"),
        ("learning_rate", 0.0),
        ("learning_rate", math.inf),
        ("accum_steps", 0),
        ("gradient_clip_val", -1.0),
        ("gradient_clip_val", math.nan),
    ],
)
def test_training_resolution_rejects_invalid_effective_owner_domain(
    field_name: str,
    invalid: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        resolve_training_bundle(
            baseline=training_factory_baseline(),
            normalized=normalize_training_patch(
                {"n_groups": 1, field_name: invalid}
            ),
            observations=_training_observations(),
        )


@pytest.mark.parametrize("field_name", ["C", "C_model", "C_forward"])
def test_training_rejects_conflicting_explicit_derived_channel(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"{field_name}.*grid_size.*4",
    ):
        resolve_training_bundle(
            baseline=training_factory_baseline(),
            normalized=normalize_training_patch(
                {
                    "grid_size": (2, 2),
                    field_name: 3,
                    "n_groups": 1,
                }
            ),
            observations=_training_observations(),
        )


def test_equal_explicit_derived_channels_are_accepted_once() -> None:
    resolved = resolve_training_bundle(
        baseline=training_factory_baseline(),
        normalized=normalize_training_patch(
            {
                "grid_size": (2, 2),
                "C": 4,
                "C_model": 4,
                "C_forward": 4,
                "n_groups": 1,
            }
        ),
        observations=_training_observations(),
    )

    assert resolved.data.C == 4
    assert resolved.model.C_model == 4
    assert resolved.model.C_forward == 4
    assert resolved.audit["C_source"] == "derived:grid_size"
    assert resolved.audit["C_model_source"] == "derived:grid_size"
    assert resolved.audit["C_forward_source"] == "derived:grid_size"


@pytest.mark.parametrize(
    ("torch_loss_mode", "loss_function", "nll"),
    [
        ("poisson", "Poisson", True),
        ("mae", "MAE", False),
    ],
)
def test_loss_identity_resolves_coherently(
    torch_loss_mode: str,
    loss_function: str,
    nll: bool,
) -> None:
    resolved = resolve_training_bundle(
        baseline=training_factory_baseline(),
        normalized=normalize_training_patch(
            {"torch_loss_mode": torch_loss_mode, "n_groups": 1}
        ),
        observations=_training_observations(),
    )

    assert resolved.training.torch_loss_mode == torch_loss_mode
    assert resolved.model.loss_function == loss_function
    assert resolved.training.nll is nll


@pytest.mark.parametrize(
    "patch",
    [
        {"torch_loss_mode": "mae", "loss_function": "Poisson"},
        {"torch_loss_mode": "mae", "nll": True},
        {"torch_loss_mode": "poisson", "loss_function": "MAE"},
        {"torch_loss_mode": "poisson", "nll": False},
    ],
)
def test_conflicting_explicit_loss_constraints_fail(patch) -> None:
    with pytest.raises(ValueError, match="loss|nll"):
        resolve_training_bundle(
            baseline=training_factory_baseline(),
            normalized=normalize_training_patch({**patch, "n_groups": 1}),
            observations=_training_observations(),
        )


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        (
            {
                "scale_contract_version": "legacy_v1",
                "measurement_domain": "count_intensity",
            },
            "Unsupported scale contract",
        ),
        ({"amplitude_physics_gain": 0.0}, "amplitude_physics_gain"),
        (
            {"rect_s1s2_init": "data"},
            "rect_s1s2_init must be 'ones' or 'dose_closure'",
        ),
        ({"rect_s1s2_init": "dose_closure"}, "Half-configured CI"),
        (
            {
                "physics_forward_mode": "rectangular_scaled",
                "torch_loss_mode": "mae",
            },
            "torch_loss_mode",
        ),
    ],
)
def test_complete_training_bundle_is_validated_before_publication(
    patch,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_training_bundle(
            baseline=training_factory_baseline(),
            normalized=normalize_training_patch({**patch, "n_groups": 1}),
            observations=_training_observations(),
        )


def test_object_policy_is_materialized_once_in_resolved_bundle() -> None:
    resolved = resolve_training_bundle(
        baseline=training_factory_baseline(),
        normalized=normalize_training_patch(
            {"object_big": False, "n_groups": 1}
        ),
        observations=_training_observations(),
    )

    assert resolved.model.object_big is False
    assert resolved.model.object_layout == "single_patch"
    assert resolved.model.training_canvas == "independent"
    assert resolved.model.training_patch_weighting == "central_mask"


def test_probe_fallback_notice_is_deferred_through_later_failure(
    tmp_path: Path,
) -> None:
    probe_observation = observe_probe_size(tmp_path / "missing.npz")
    observations = _training_observations(
        inferred_probe_size=probe_observation.value,
        notices=probe_observation.notices,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="C_model.*grid_size"):
            resolve_training_bundle(
                baseline=training_factory_baseline(),
                normalized=normalize_training_patch(
                    {"C_model": 2, "n_groups": 1}
                ),
                observations=observations,
            )

    assert caught == []


def test_successful_resolution_retains_one_deferred_probe_notice(
    tmp_path: Path,
) -> None:
    probe_observation = observe_probe_size(tmp_path / "missing.npz")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = resolve_training_bundle(
            baseline=training_factory_baseline(),
            normalized=normalize_training_patch({"n_groups": 1}),
            observations=_training_observations(
                inferred_probe_size=probe_observation.value,
                notices=probe_observation.notices,
            ),
        )

    assert resolved.data.N == 64
    assert len(resolved.notices) == 1
    assert resolved.notices[0].category is UserWarning
    assert "fallback N=64" in resolved.notices[0].message
    assert caught == []


def test_inference_resolution_is_runtime_only_and_bridge_n_subsample_stays_bridge_only() -> None:
    baseline = inference_factory_baseline()

    resolved = resolve_inference_bundle(
        baseline=baseline,
        normalized=normalize_inference_patch(
            {"n_groups": 8, "n_subsample": 19}
        ),
        observations=_inference_observations(),
    )

    assert resolved.data.N == 128
    assert resolved.data.n_subsample == baseline.data.n_subsample
    assert resolved.bridge["n_groups"] == 8
    assert resolved.bridge["n_subsample"] == 19
    assert resolved.bridge["model_path"] == Path("model")
    assert resolved.bridge["test_data_file"] == Path("test.npz")
    assert resolved.bridge["output_dir"] == Path("recon")
    assert not hasattr(resolved, "model_spec")
    assert resolved.data is not baseline.data
    assert resolved.model is not baseline.model
    assert resolved.inference is not baseline.inference


def test_explicit_inference_N_wins_over_observed_probe_size() -> None:
    resolved = resolve_inference_bundle(
        baseline=inference_factory_baseline(),
        normalized=normalize_inference_patch({"N": 256, "n_groups": 1}),
        observations=_inference_observations(inferred_probe_size=128),
    )

    assert resolved.data.N == 256
    assert resolved.audit["N_source"] == "explicit"


def test_training_factory_delegates_patch_and_bundle_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    import ptycho.params
    import ptycho_torch.config_factory as factory
    import ptycho_torch.config_bridge as bridge

    train_path = tmp_path / "train.npz"
    np.savez(
        train_path,
        probeGuess=np.ones((64, 64), dtype=np.complex64),
    )
    monkeypatch.setattr(ptycho.params, "cfg", {})
    monkeypatch.setattr(ptycho.params, "_sealed", False)

    events: list[str] = []
    captured = {}
    real_profile_gate = factory.resolve_profile_overrides
    real_exists = Path.exists
    real_observe = factory.observe_probe_size
    real_bridge_model = bridge.to_model_config
    real_bridge_training = bridge.to_training_config
    real_unseal = ptycho.params.unseal
    real_commit = factory.populate_legacy_params

    def profile_gate_spy(*args, **kwargs):
        events.append("profile_pair")
        return real_profile_gate(*args, **kwargs)

    def normalize_spy(*args, **kwargs):
        events.append("normalize")
        return normalize_training_patch(*args, **kwargs)

    def exists_spy(path):
        if path == train_path:
            events.append("file_check")
        return real_exists(path)

    def observe_spy(*args, **kwargs):
        events.append("observe")
        return real_observe(*args, **kwargs)

    def resolve_spy(*args, **kwargs):
        events.append("resolve")
        resolved = resolve_training_bundle(*args, **kwargs)
        captured["bundle"] = resolved
        return resolved

    def bridge_model_spy(*args, **kwargs):
        events.append("bridge_model")
        captured["bridge_model_inputs"] = args[:2]
        return real_bridge_model(*args, **kwargs)

    def bridge_training_spy(*args, **kwargs):
        events.append("bridge_training")
        captured["bridge_training_inputs"] = args[1:4]
        return real_bridge_training(*args, **kwargs)

    def unseal_spy():
        events.append("unseal")
        return real_unseal()

    def commit_spy(config):
        events.append("commit")
        return real_commit(config)

    monkeypatch.setattr(factory, "resolve_profile_overrides", profile_gate_spy)
    monkeypatch.setattr(
        factory,
        "normalize_training_patch",
        normalize_spy,
        raising=False,
    )
    monkeypatch.setattr(
        factory,
        "resolve_training_bundle",
        resolve_spy,
        raising=False,
    )
    monkeypatch.setattr(Path, "exists", exists_spy)
    monkeypatch.setattr(factory, "observe_probe_size", observe_spy)
    monkeypatch.setattr(bridge, "to_model_config", bridge_model_spy)
    monkeypatch.setattr(bridge, "to_training_config", bridge_training_spy)
    monkeypatch.setattr(ptycho.params, "unseal", unseal_spy)
    monkeypatch.setattr(factory, "populate_legacy_params", commit_spy)

    payload = factory.create_training_payload(
        train_data_file=train_path,
        output_dir=tmp_path / "out",
        overrides={"n_groups": 1},
    )

    assert events == [
        "profile_pair",
        "normalize",
        "file_check",
        "observe",
        "resolve",
        "bridge_model",
        "bridge_training",
        "unseal",
        "commit",
    ]
    resolved = captured["bundle"]
    assert payload.pt_data_config is resolved.data
    assert payload.pt_model_config is resolved.model
    assert payload.pt_training_config is resolved.training
    assert payload.pt_inference_config is resolved.inference
    assert captured["bridge_model_inputs"] == (
        resolved.data,
        resolved.model,
    )
    assert captured["bridge_training_inputs"] == (
        resolved.data,
        resolved.model,
        resolved.training,
    )


def test_inference_factory_delegates_patch_and_bundle_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    import ptycho.params
    import ptycho_torch.config_factory as factory
    import ptycho_torch.config_bridge as bridge

    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "wts.h5.zip").touch()
    test_path = tmp_path / "test.npz"
    np.savez(
        test_path,
        probeGuess=np.ones((12, 12), dtype=np.complex64),
    )
    monkeypatch.setattr(ptycho.params, "cfg", {})
    monkeypatch.setattr(ptycho.params, "_sealed", False)

    events: list[str] = []
    captured = {}
    real_profile_gate = factory.resolve_profile_overrides
    real_exists = Path.exists
    real_observe = factory.observe_probe_size
    real_bridge_model = bridge.to_model_config
    real_bridge_inference = bridge.to_inference_config
    real_unseal = ptycho.params.unseal
    real_commit = factory.populate_legacy_params

    def profile_gate_spy(*args, **kwargs):
        events.append("profile_pair")
        return real_profile_gate(*args, **kwargs)

    def normalize_spy(*args, **kwargs):
        events.append("normalize")
        return normalize_inference_patch(*args, **kwargs)

    tracked_paths = {
        model_path: "model_check",
        model_path / "wts.h5.zip": "checkpoint_check",
        test_path: "data_check",
    }

    def exists_spy(path):
        if path in tracked_paths:
            events.append(tracked_paths[path])
        return real_exists(path)

    def observe_spy(*args, **kwargs):
        events.append("observe")
        return real_observe(*args, **kwargs)

    def resolve_spy(*args, **kwargs):
        events.append("resolve")
        resolved = resolve_inference_bundle(*args, **kwargs)
        captured["bundle"] = resolved
        return resolved

    def bridge_model_spy(*args, **kwargs):
        events.append("bridge_model")
        captured["bridge_model_inputs"] = args[:2]
        return real_bridge_model(*args, **kwargs)

    def bridge_inference_spy(*args, **kwargs):
        events.append("bridge_inference")
        captured["bridge_inference_inputs"] = args[1:3]
        return real_bridge_inference(*args, **kwargs)

    def unseal_spy():
        events.append("unseal")
        return real_unseal()

    def commit_spy(config):
        events.append("commit")
        return real_commit(config)

    monkeypatch.setattr(factory, "resolve_profile_overrides", profile_gate_spy)
    monkeypatch.setattr(factory, "normalize_inference_patch", normalize_spy)
    monkeypatch.setattr(factory, "resolve_inference_bundle", resolve_spy)
    monkeypatch.setattr(Path, "exists", exists_spy)
    monkeypatch.setattr(factory, "observe_probe_size", observe_spy)
    monkeypatch.setattr(bridge, "to_model_config", bridge_model_spy)
    monkeypatch.setattr(bridge, "to_inference_config", bridge_inference_spy)
    monkeypatch.setattr(ptycho.params, "unseal", unseal_spy)
    monkeypatch.setattr(factory, "populate_legacy_params", commit_spy)

    payload = factory.create_inference_payload(
        model_path=model_path,
        test_data_file=test_path,
        output_dir=tmp_path / "out",
        overrides={"n_groups": 1},
    )

    assert events == [
        "profile_pair",
        "normalize",
        "model_check",
        "checkpoint_check",
        "data_check",
        "observe",
        "resolve",
        "bridge_model",
        "bridge_inference",
        "unseal",
        "commit",
    ]
    resolved = captured["bundle"]
    assert payload.pt_data_config is resolved.data
    assert payload.pt_inference_config is resolved.inference
    assert captured["bridge_model_inputs"] == (
        resolved.data,
        resolved.model,
    )
    assert captured["bridge_inference_inputs"] == (
        resolved.data,
        resolved.inference,
    )


@pytest.mark.parametrize(
    ("phase", "patch"),
    [
        ("training", {"n_groups": 0}),
        ("inference", {"n_groups": 0}),
    ],
)
def test_failed_factory_resolution_preserves_legacy_state_without_warnings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    patch: dict[str, object],
) -> None:
    import numpy as np
    import ptycho.params
    import ptycho_torch.config_factory as factory

    data_path = tmp_path / "data.npz"
    np.savez(data_path, probeGuess=np.ones((2, 3), dtype=np.complex64))
    before = {"sentinel": ["unchanged"]}
    monkeypatch.setattr(ptycho.params, "cfg", before.copy())
    monkeypatch.setattr(ptycho.params, "_sealed", True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="n_groups"):
            if phase == "training":
                factory.create_training_payload(
                    train_data_file=data_path,
                    output_dir=tmp_path / "out",
                    overrides=patch,
                )
            else:
                model_path = tmp_path / "model"
                model_path.mkdir()
                (model_path / "wts.h5.zip").touch()
                factory.create_inference_payload(
                    model_path=model_path,
                    test_data_file=data_path,
                    output_dir=tmp_path / "out",
                    overrides=patch,
                )

    assert ptycho.params.cfg == before
    assert ptycho.params._sealed is True
    assert caught == []


@pytest.mark.parametrize("initially_sealed", [False, True])
def test_partial_legacy_commit_failure_restores_mapping_object_and_seal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initially_sealed: bool,
) -> None:
    import numpy as np
    import ptycho.params
    import ptycho_torch.config_factory as factory

    data_path = tmp_path / "train.npz"
    np.savez(
        data_path,
        probeGuess=np.ones((64, 64), dtype=np.complex64),
    )
    legacy_cfg = {
        "sentinel": "unchanged",
        "prior_value": 17,
    }
    expected = dict(legacy_cfg)
    monkeypatch.setattr(ptycho.params, "cfg", legacy_cfg)
    monkeypatch.setattr(ptycho.params, "_sealed", initially_sealed)

    def partially_mutate_then_fail(_config) -> None:
        ptycho.params.cfg["sentinel"] = "mutated"
        ptycho.params.cfg["partial_only"] = True
        if initially_sealed:
            ptycho.params.unseal()
        else:
            ptycho.params.seal()
        raise RuntimeError("legacy commit failed")

    monkeypatch.setattr(
        factory,
        "populate_legacy_params",
        partially_mutate_then_fail,
    )

    with pytest.raises(RuntimeError, match="legacy commit failed"):
        factory.create_training_payload(
            train_data_file=data_path,
            output_dir=tmp_path / "out",
            overrides={"n_groups": 1},
        )

    assert ptycho.params.cfg is legacy_cfg
    assert ptycho.params.cfg == expected
    assert ptycho.params._sealed is initially_sealed


@pytest.mark.parametrize(
    (
        "field_name",
        "baseline_value",
        "canonical_value",
    ),
    [
        ("learning_rate", 0.01, 0.03),
        ("scheduler", "Default", "WarmupCosine"),
        ("gradient_clip_val", 1.0, 3.0),
        ("gradient_clip_algorithm", "norm", "agc"),
        ("accum_steps", 1, 3),
    ],
)
def test_training_owner_precedence_matrix_is_resolved_once(
    field_name: str,
    baseline_value: object,
    canonical_value: object,
) -> None:
    baseline = training_factory_baseline(
        training_baseline=TrainingConfig(**{field_name: baseline_value})
    )

    canonical = resolve_training_bundle(
        baseline=baseline,
        normalized=normalize_training_patch(
            {"n_groups": 1, field_name: canonical_value}
        ),
        observations=_training_observations(),
    )
    baseline_only = resolve_training_bundle(
        baseline=baseline,
        normalized=normalize_training_patch({"n_groups": 1}),
        observations=_training_observations(),
    )
    declared_default = resolve_training_bundle(
        baseline=training_factory_baseline(),
        normalized=normalize_training_patch({"n_groups": 1}),
        observations=_training_observations(),
    )

    assert getattr(canonical.training, field_name) == canonical_value
    assert getattr(baseline_only.training, field_name) == baseline_value
    assert getattr(declared_default.training, field_name) == getattr(
        TrainingConfig(), field_name
    )
    assert canonical.audit["training_config_provenance"][field_name] == (
        "canonical_override"
    )
    assert baseline_only.audit["training_config_provenance"][field_name] == (
        "training_baseline"
    )
    assert declared_default.audit["training_config_provenance"][field_name] == (
        "torch_default"
    )


def test_factory_audit_records_canonical_effective_values_and_alias_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    import ptycho.params
    import ptycho_torch.config_factory as factory

    data_path = tmp_path / "train.npz"
    np.savez(
        data_path,
        probeGuess=np.ones((64, 64), dtype=np.complex64),
    )
    monkeypatch.setattr(ptycho.params, "cfg", {})
    monkeypatch.setattr(ptycho.params, "_sealed", False)
    payload = factory.create_training_payload(
        train_data_file=data_path,
        output_dir=tmp_path / "out",
        overrides={
            "n_groups": 1,
            "max_epochs": 7,
            "learning_rate": 0.03,
        },
        execution_config=ExecutionRequest(
            values={"accelerator": "cpu"},
            explicit_fields=frozenset({"accelerator"}),
        ),
    )

    assert payload.overrides_applied["epochs"] == 7
    assert "max_epochs" not in payload.overrides_applied
    assert payload.overrides_applied["input_aliases"] == {
        "epochs": ("max_epochs",)
    }
    assert "topology_compatibility" not in payload.overrides_applied
    assert payload.overrides_applied["learning_rate"] == 0.03


def test_inference_factory_audit_keeps_aliases_provenance_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    import ptycho.params
    import ptycho_torch.config_factory as factory

    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "wts.h5.zip").touch()
    data_path = tmp_path / "test.npz"
    np.savez(
        data_path,
        probeGuess=np.ones((64, 64), dtype=np.complex64),
    )
    monkeypatch.setattr(ptycho.params, "cfg", {})
    monkeypatch.setattr(ptycho.params, "_sealed", False)

    payload = factory.create_inference_payload(
        model_path=model_path,
        test_data_file=data_path,
        output_dir=tmp_path / "out",
        overrides={
            "n_groups": 1,
            "gridsize": 2,
            "neighbor_count": 5,
            "model_type": "Unsupervised",
        },
        execution_config=ExecutionRequest(
            values={"accelerator": "cpu"},
            explicit_fields=frozenset({"accelerator"}),
        ),
    )

    assert payload.overrides_applied["grid_size"] == (2, 2)
    assert payload.overrides_applied["K"] == 5
    assert payload.overrides_applied["mode"] == "Unsupervised"
    assert {
        "gridsize",
        "neighbor_count",
        "model_type",
    }.isdisjoint(payload.overrides_applied)
    assert payload.overrides_applied["input_aliases"] == {
        "K": ("neighbor_count",),
        "grid_size": ("gridsize",),
        "mode": ("model_type",),
    }
    assert "topology_compatibility" not in payload.overrides_applied


@pytest.mark.parametrize("phase", ["training", "inference"])
def test_successful_factory_commits_complete_projection_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    import numpy as np
    import ptycho.params
    import ptycho_torch.config_factory as factory

    data_path = tmp_path / "data.npz"
    np.savez(
        data_path,
        probeGuess=np.ones((64, 64), dtype=np.complex64),
    )
    monkeypatch.setattr(ptycho.params, "cfg", {})
    monkeypatch.setattr(ptycho.params, "_sealed", False)
    committed = []
    real_commit = factory.populate_legacy_params

    def commit_spy(config):
        committed.append(config)
        return real_commit(config)

    monkeypatch.setattr(factory, "populate_legacy_params", commit_spy)
    if phase == "training":
        payload = factory.create_training_payload(
            train_data_file=data_path,
            output_dir=tmp_path / "out",
            overrides={"n_groups": 1},
        )
        assert committed == [payload.tf_training_config]
    else:
        model_path = tmp_path / "model"
        model_path.mkdir()
        (model_path / "wts.h5.zip").touch()
        payload = factory.create_inference_payload(
            model_path=model_path,
            test_data_file=data_path,
            output_dir=tmp_path / "out",
            overrides={"n_groups": 1},
        )
        assert committed == [payload.tf_inference_config]

    assert ptycho.params.cfg["N"] == 64
    assert ptycho.params.cfg["n_groups"] == 1
    assert ptycho.params._sealed is True
