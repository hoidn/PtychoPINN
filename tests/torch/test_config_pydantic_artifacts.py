"""Frozen pre-Pydantic Torch configuration wire artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch


FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "config"
PINNED_REVISION = "99efda11155119161d371d5d0e5ec7c33a720594"
FIXTURE_NAMES = (
    "pydantic_pre_migration_portable_v1.json",
    "pydantic_pre_migration_portable_v2.json",
    "pydantic_pre_migration_tensor_mask.json",
)
PRE_MIGRATION_MODEL_SPEC_V1_MODEL_FIELDS = (
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
    "C_model",
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
    "probe_big",
    "offset",
    "C_forward",
    "training_patch_weighting",
    "physics_forward_mode",
    "rect_s1s2_trainable",
    "rect_s1s2_init",
    "amplitude_physics_gain",
    "pad_object",
    "gaussian_smoothing_sigma",
    "loss_function",
    "amp_loss",
    "phase_loss",
    "amp_loss_coeff",
    "phase_loss_coeff",
)
PRE_MIGRATION_PORTABLE_V1_DATA_FIELDS = (
    "nphotons",
    "scale_contract_version",
    "measurement_domain",
    "N",
    "C",
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
)
PRE_MIGRATION_PORTABLE_V1_TRAINING_FIELDS = (
    "training_directories",
    "nll",
    "device",
    "strategy",
    "n_devices",
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
    "num_workers",
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
    "output_dir",
    "train_data_file",
    "test_data_file",
    "n_groups",
)
PRE_MIGRATION_PORTABLE_V1_INFERENCE_FIELDS = (
    "middle_trim",
    "batch_size",
    "experiment_number",
    "pad_eval",
    "window",
    "patch_weighting",
    "varpro_scaling",
    "log_patch_stats",
    "patch_stats_limit",
)


def test_pre_migration_fixture_inventory_is_complete():
    expected = {
        *FIXTURE_NAMES,
        "README.md",
        "generate_pre_migration_fixtures.py",
    }

    assert FIXTURE_ROOT.is_dir()
    assert expected <= {
        path.name for path in FIXTURE_ROOT.iterdir() if path.is_file()
    }


def _read_payload(name: str):
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _canonical_bytes(payload) -> bytes:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (serialized + "\n").encode("utf-8")


def test_generator_reproduces_each_canonical_fixture_byte_stream():
    generator = FIXTURE_ROOT / "generate_pre_migration_fixtures.py"

    result = subprocess.run(
        [sys.executable, str(generator), "--stdout"],
        cwd=Path(__file__).parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    generated = json.loads(result.stdout)

    assert tuple(sorted(generated)) == tuple(sorted(FIXTURE_NAMES))
    for name in FIXTURE_NAMES:
        expected = _canonical_bytes(generated[name])
        assert (FIXTURE_ROOT / name).read_bytes() == expected


def test_portable_v1_is_an_exact_explicit_projection_and_historical_data_is_rejected():
    from ptycho_torch.artifact_schema import (
        ARTIFACT_SCHEMA_V1_VERSION,
        PORTABLE_V1_DATA_FIELDS,
        PORTABLE_V1_INFERENCE_FIELDS,
        PORTABLE_V1_TRAINING_FIELDS,
        decode_artifact_identity,
        from_json_payload,
    )
    from ptycho_torch.model_spec import (
        MODEL_SPEC_V1_MODEL_FIELDS,
        MODEL_SPEC_V1_VERSION,
    )

    raw = _read_payload("pydantic_pre_migration_portable_v1.json")
    model_spec = raw["model_spec"]

    assert raw["backend"] == "pytorch"
    assert raw["schema_version"] == ARTIFACT_SCHEMA_V1_VERSION
    assert model_spec["schema_version"] == MODEL_SPEC_V1_VERSION
    assert MODEL_SPEC_V1_MODEL_FIELDS == (
        PRE_MIGRATION_MODEL_SPEC_V1_MODEL_FIELDS
    )
    assert PORTABLE_V1_DATA_FIELDS == PRE_MIGRATION_PORTABLE_V1_DATA_FIELDS
    assert PORTABLE_V1_TRAINING_FIELDS == (
        PRE_MIGRATION_PORTABLE_V1_TRAINING_FIELDS
    )
    assert PORTABLE_V1_INFERENCE_FIELDS == (
        PRE_MIGRATION_PORTABLE_V1_INFERENCE_FIELDS
    )
    assert tuple(model_spec["model_config"]) == tuple(
        sorted(PRE_MIGRATION_MODEL_SPEC_V1_MODEL_FIELDS)
    )
    assert tuple(raw["data_config"]) == tuple(
        sorted(PRE_MIGRATION_PORTABLE_V1_DATA_FIELDS)
    )
    assert tuple(raw["training_config"]) == tuple(
        sorted(PRE_MIGRATION_PORTABLE_V1_TRAINING_FIELDS)
    )
    assert tuple(raw["inference_config"]) == tuple(
        sorted(PRE_MIGRATION_PORTABLE_V1_INFERENCE_FIELDS)
    )

    assert model_spec["model_config"]["mode"] == "Supervised"
    assert model_spec["model_config"]["architecture"] == "ffno"
    assert model_spec["model_config"]["object_big"] is True
    assert model_spec["model_config"]["training_patch_weighting"] == "probe"
    assert model_spec["model_config"]["rect_s1s2_init"] == "data"
    assert raw["data_config"]["neighbor_function"] == "4_quadrant"
    assert raw["data_config"]["scan_pattern"] == "Rectangular"
    assert raw["training_config"]["scheduler"] == "WarmupCosine"
    assert raw["training_config"]["optimizer"] == "adamw"
    assert raw["inference_config"]["patch_weighting"] == "uniform"
    assert raw["ci_statistics"] == {
        "rms_input_scale": [0.125, 0.25],
        "rms_probe_scale": [2.0],
    }

    with pytest.raises(
        ValueError,
        match=r"data.*unsupported.*ones.*dose_closure.*historical code or retraining",
    ):
        decode_artifact_identity(from_json_payload(raw))


def test_portable_v3_is_a_frozen_historical_era_and_historical_data_is_rejected():
    from ptycho_torch.artifact_schema import (
        ARTIFACT_SCHEMA_V3_VERSION,
        decode_artifact_identity,
        from_json_payload,
    )
    from ptycho_torch.model_spec import (
        CURRENT_MODEL_SPEC_VERSION,
        MODEL_SPEC_V3_MODEL_FIELDS,
    )

    raw_v1 = _read_payload("pydantic_pre_migration_portable_v1.json")
    raw_v3 = _read_payload("pydantic_pre_migration_portable_v2.json")

    assert raw_v3["schema_version"] == ARTIFACT_SCHEMA_V3_VERSION
    assert raw_v3["model_spec"]["schema_version"] == CURRENT_MODEL_SPEC_VERSION
    assert set(raw_v3["model_spec"]["model_config"]) == set(
        MODEL_SPEC_V3_MODEL_FIELDS
    )
    assert "object_big" not in raw_v3["model_spec"]["model_config"]
    assert raw_v3["model_spec"]["model_config"]["object_layout"] == (
        "grouped_patches"
    )

    assert raw_v1["model_spec"]["model_config"]["rect_s1s2_init"] == "data"
    assert raw_v3["model_spec"]["model_config"]["rect_s1s2_init"] == "data"
    # v1 states channel identity three ways (C/grid_size/n_subsample); v3 once.
    assert set(raw_v1["data_config"]) >= {"C", "grid_size", "n_subsample"}
    assert set(raw_v3["data_config"]) >= {"gridsize", "n_raw_frames_selected"}
    assert raw_v3["training_config"] == raw_v1["training_config"]
    assert raw_v3["inference_config"] == raw_v1["inference_config"]
    assert raw_v3["ci_statistics"] == raw_v1["ci_statistics"]

    for raw in (raw_v1, raw_v3):
        with pytest.raises(
            ValueError,
            match=r"data.*unsupported.*ones.*dose_closure.*historical code or retraining",
        ):
            decode_artifact_identity(from_json_payload(raw))


def test_complex_tensor_mask_tag_decodes_exactly_and_is_defensively_copied():
    from ptycho_torch.artifact_schema import from_json_payload
    from ptycho_torch.model_spec import MODEL_SPEC_V2_MODEL_FIELDS, ModelSpec

    raw = _read_payload("pydantic_pre_migration_tensor_mask.json")
    tagged = raw["model_spec"]["model_config"]["probe_mask_tensor"]

    assert raw["field_name"] == "probe_mask_tensor"
    assert tuple(raw["model_spec_field_tuple"]) == MODEL_SPEC_V2_MODEL_FIELDS
    assert tagged == {
        "__ptychopinn_torch_tensor__": True,
        "data": {
            "imag": [2.0, 0.5, -4.0, -6.5, 8.125, 0.0],
            "real": [1.0, -3.0, 0.0, 5.25, -7.75, 9.0],
        },
        "dtype": "complex64",
        "shape": [2, 3],
    }

    compatible_raw = copy.deepcopy(raw)
    compatible_raw["model_spec"]["model_config"]["rect_s1s2_init"] = "ones"
    decoded_payload = from_json_payload(compatible_raw)
    decoded_tensor = decoded_payload["model_spec"]["model_config"][
        "probe_mask_tensor"
    ]
    expected = torch.tensor(
        [
            [1.0 + 2.0j, -3.0 + 0.5j, -4.0j],
            [5.25 - 6.5j, -7.75 + 8.125j, 9.0 + 0.0j],
        ],
        dtype=torch.complex64,
    )
    assert decoded_tensor.dtype == torch.complex64
    assert tuple(decoded_tensor.shape) == (2, 3)
    torch.testing.assert_close(decoded_tensor, expected, rtol=0.0, atol=0.0)

    spec = ModelSpec.from_payload(decoded_payload["model_spec"])
    decoded_tensor.zero_()
    first = spec.to_model_config().probe_mask_tensor
    second = spec.to_model_config().probe_mask_tensor

    torch.testing.assert_close(first, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(second, expected, rtol=0.0, atol=0.0)
    assert first.data_ptr() != second.data_ptr()
    first.add_(100.0 + 100.0j)
    torch.testing.assert_close(
        spec.to_model_config().probe_mask_tensor,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_readme_pins_revision_reproduction_command_and_fixture_hashes():
    readme = (FIXTURE_ROOT / "README.md").read_text(encoding="utf-8")

    assert PINNED_REVISION in readme
    assert (
        "python tests/fixtures/config/generate_pre_migration_fixtures.py "
        "--stdout"
    ) in readme
    for name in FIXTURE_NAMES:
        digest = hashlib.sha256((FIXTURE_ROOT / name).read_bytes()).hexdigest()
        assert f"| `{name}` | `{digest}` |" in readme
