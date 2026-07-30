#!/usr/bin/env python
"""Reproduce the frozen pre-Pydantic Torch configuration wire fixtures."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from ptycho.config.config import ModelConfig as CanonicalModelConfig
from ptycho_torch.artifact_schema import (
    ARTIFACT_SCHEMA_V1_VERSION,
    PORTABLE_V1_DATA_FIELDS,
    PORTABLE_V1_INFERENCE_FIELDS,
    PORTABLE_V1_TRAINING_FIELDS,
    encode_artifact_identity,
    to_json_payload,
)
from ptycho_torch.config_params import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)
from ptycho_torch.model_spec import (
    MODEL_SPEC_V1_MODEL_FIELDS,
    MODEL_SPEC_V1_VERSION,
    MODEL_SPEC_V2_MODEL_FIELDS,
    derive_model_spec,
)


PINNED_REVISION = "99efda11155119161d371d5d0e5ec7c33a720594"
PORTABLE_V1_FIXTURE = "pydantic_pre_migration_portable_v1.json"
PORTABLE_V2_FIXTURE = "pydantic_pre_migration_portable_v2.json"
TENSOR_MASK_FIXTURE = "pydantic_pre_migration_tensor_mask.json"
FIXTURE_ROOT = Path(__file__).resolve().parent


def _configuration_identity():
    data = DataConfig(
        nphotons=246810.5,
        scale_contract_version="legacy_v1",
        measurement_domain="normalized_amplitude",
        N=128,
        C=4,
        K=9,
        K_quadrant=11,
        n_subsample=3,
        subsample_seed=17,
        grid_size=(2, 2),
        neighbor_function="4_quadrant",
        min_neighbor_distance=0.25,
        max_neighbor_distance=2.5,
        scan_pattern="Rectangular",
        normalize="Group",
        probe_scale=3.25,
        probe_normalize=False,
        data_scaling="Max",
        phase_subtraction=False,
        x_bounds=(0.2, 0.8),
        y_bounds=(0.3, 0.7),
    )
    model = ModelConfig(
        mode="Supervised",
        architecture="ffno",
        fno_modes=7,
        fno_width=48,
        fno_blocks=3,
        fno_cnn_blocks=1,
        learned_input_channels=2,
        fno_input_transform="log1p",
        max_hidden_channels=96,
        resnet_width=24,
        spectral_bottleneck_blocks=5,
        spectral_bottleneck_modes=9,
        spectral_bottleneck_share_weights=False,
        spectral_bottleneck_gate_init=0.25,
        spectral_bottleneck_gate_mode="per_block",
        generator_output_mode="amp_phase",
        cnn_output_mode="real_imag",
        use_shared_decoder=True,
        intensity_scale_trainable=True,
        intensity_scale=4321.5,
        max_position_jitter=2,
        num_datasets=3,
        C_model=4,
        n_filters_scale=3,
        amp_activation="silu",
        batch_norm=True,
        probe_mask=False,
        probe_mask_tensor=None,
        probe_mask_sigma=1.75,
        probe_mask_diameter=23.5,
        edge_pad=5,
        decoder_last_c_outer_fraction=0.25,
        decoder_last_amp_channels=2,
        use_legacy_decoder_channel_override=True,
        eca_encoder=True,
        cbam_encoder=False,
        cbam_bottleneck=True,
        cbam_decoder=True,
        eca_decoder=True,
        spatial_decoder=True,
        decoder_spatial_kernel=5,
        object_big=None,
        object_layout="grouped_patches",
        training_canvas="relative_overlap",
        probe_big=False,
        offset=4,
        C_forward=4,
        training_patch_weighting="probe",
        physics_forward_mode="amplitude",
        rect_s1s2_trainable=False,
        rect_s1s2_init="data",
        amplitude_physics_gain=1.5,
        pad_object=False,
        gaussian_smoothing_sigma=0.75,
        loss_function="MAE",
        amp_loss="Total_Variation",
        phase_loss="Mean_Deviation",
        amp_loss_coeff=0.4,
        phase_loss_coeff=0.6,
    )
    canonical = CanonicalModelConfig(
        N=128,
        gridsize=2,
        n_filters_scale=3,
        model_type="supervised",
        architecture="ffno",
        fno_modes=7,
        fno_width=48,
        fno_blocks=3,
        fno_cnn_blocks=1,
        learned_input_channels=2,
        max_hidden_channels=96,
        resnet_width=24,
        fno_input_transform="log1p",
        generator_output_mode="amp_phase",
        amp_activation="swish",
        object_big=None,
        object_layout="grouped_patches",
        training_canvas="relative_overlap",
        training_patch_weighting="probe",
        probe_big=False,
        probe_mask=False,
        probe_mask_sigma=1.75,
        probe_mask_diameter=23.5,
        pad_object=False,
        probe_scale=3.25,
        gaussian_smoothing_sigma=0.75,
    )
    training = TrainingConfig(
        training_directories=["datasets/train-a", "datasets/train-b"],
        nll=False,
        device="cpu",
        strategy=None,
        n_devices="auto",
        framework="Default",
        orchestrator="Lightning",
        learning_rate=0.00025,
        epochs=7,
        batch_size=3,
        epochs_fine_tune=2,
        fine_tune_gamma=0.2,
        scheduler="WarmupCosine",
        lr_warmup_epochs=2,
        lr_min_ratio=0.05,
        plateau_factor=0.25,
        plateau_patience=4,
        plateau_min_lr=0.00001,
        plateau_threshold=0.001,
        num_workers=0,
        accum_steps=2,
        gradient_clip_val=0.75,
        gradient_clip_algorithm="value",
        optimizer="adamw",
        momentum=0.8,
        weight_decay=0.01,
        adam_beta1=0.85,
        adam_beta2=0.98,
        log_grad_norm=True,
        grad_norm_log_freq=2,
        stage_1_epochs=1,
        stage_2_epochs=2,
        stage_3_epochs=4,
        physics_weight_schedule="exponential",
        stage_3_lr_factor=0.25,
        torch_loss_mode="mae",
        torch_mae_pred_l2_match_target=True,
        experiment_name="pydantic-wire-fixture",
        notes="pre-migration",
        model_name="FixtureNet",
        output_dir="outputs/pydantic-wire",
        train_data_file="datasets/train.npz",
        test_data_file="datasets/test.npz",
        n_groups=13,
    )
    inference = InferenceConfig(
        middle_trim=24,
        batch_size=17,
        experiment_number=8,
        pad_eval=False,
        window=9,
        patch_weighting="uniform",
        varpro_scaling=False,
        log_patch_stats=True,
        patch_stats_limit=5,
    )
    spec = derive_model_spec(
        canonical,
        model,
        data,
        parity_scale_mode="fixed",
        parity_fixed_delta=1.25,
        parity_init_scheme="tf_glorot",
    )
    return canonical, data, model, training, inference, spec


def _project_portable_v1(
    portable_v2: Mapping[str, Any],
    *,
    data: DataConfig,
    training: TrainingConfig,
    inference: InferenceConfig,
    spec,
) -> dict[str, Any]:
    resolved_model = spec.to_model_config()
    return {
        "backend": portable_v2["backend"],
        "schema_version": ARTIFACT_SCHEMA_V1_VERSION,
        "model_spec": {
            "schema_version": MODEL_SPEC_V1_VERSION,
            "model_config": {
                name: getattr(resolved_model, name)
                for name in MODEL_SPEC_V1_MODEL_FIELDS
            },
            "parity_scale_mode": spec.parity_scale_mode,
            "parity_fixed_delta": spec.parity_fixed_delta,
            "parity_init_scheme": spec.parity_init_scheme,
        },
        "data_config": {
            name: getattr(data, name) for name in PORTABLE_V1_DATA_FIELDS
        },
        "training_config": {
            name: getattr(training, name)
            for name in PORTABLE_V1_TRAINING_FIELDS
        },
        "inference_config": {
            name: getattr(inference, name)
            for name in PORTABLE_V1_INFERENCE_FIELDS
        },
        "ci_statistics": portable_v2["ci_statistics"],
    }


def _tensor_mask_payload(
    *,
    canonical: CanonicalModelConfig,
    data: DataConfig,
    model: ModelConfig,
) -> dict[str, Any]:
    mask = torch.tensor(
        [
            [1.0 + 2.0j, -3.0 + 0.5j, complex(0.0, -4.0)],
            [5.25 - 6.5j, -7.75 + 8.125j, 9.0 + 0.0j],
        ],
        dtype=torch.complex64,
    )
    tensor_model = replace(
        model,
        probe_mask=True,
        probe_mask_tensor=mask,
    )
    tensor_canonical = replace(canonical, probe_mask=True)
    tensor_spec = derive_model_spec(
        tensor_canonical,
        tensor_model,
        data,
        parity_scale_mode="fixed",
        parity_fixed_delta=1.25,
        parity_init_scheme="tf_glorot",
    )
    return {
        "field_name": "probe_mask_tensor",
        "model_spec_field_tuple": list(MODEL_SPEC_V2_MODEL_FIELDS),
        "model_spec": tensor_spec.to_payload(),
    }


def build_fixture_payloads() -> dict[str, dict[str, Any]]:
    canonical, data, model, training, inference, spec = (
        _configuration_identity()
    )
    portable_v2 = encode_artifact_identity(
        spec,
        data,
        training,
        inference,
        ci_statistics={
            "rms_input_scale": torch.tensor([0.125, 0.25]),
            "rms_probe_scale": torch.tensor([2.0]),
        },
    )
    portable_v1 = _project_portable_v1(
        portable_v2,
        data=data,
        training=training,
        inference=inference,
        spec=spec,
    )
    return {
        PORTABLE_V1_FIXTURE: portable_v1,
        PORTABLE_V2_FIXTURE: portable_v2,
        TENSOR_MASK_FIXTURE: _tensor_mask_payload(
            canonical=canonical,
            data=data,
            model=model,
        ),
    }


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        to_json_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def build_fixture_bytes() -> dict[str, bytes]:
    return {
        name: (canonical_json(payload) + "\n").encode("utf-8")
        for name, payload in build_fixture_payloads().items()
    }


def render_stdout() -> str:
    tagged_payloads = {
        name: to_json_payload(payload)
        for name, payload in build_fixture_payloads().items()
    }
    return (
        json.dumps(
            tagged_payloads,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="print one canonical JSON mapping of fixture names to payloads",
    )
    args = parser.parse_args(argv)

    if args.stdout:
        sys.stdout.write(render_stdout())
        return 0

    for name, payload in build_fixture_bytes().items():
        (FIXTURE_ROOT / name).write_bytes(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
