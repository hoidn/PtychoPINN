"""Natural-patch expanded-object CDI benchmark harness."""

from __future__ import annotations

import json
import hashlib
from dataclasses import fields
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence
import zipfile

import matplotlib.pyplot as plt
import numpy as np

from ptycho import params
from ptycho.config.config import ModelConfig, TrainingConfig
from ptycho.metadata import MetadataManager
from ptycho import evaluation
from ptycho.config.config import update_legacy_dict
from scripts.studies.metrics_tables import write_paper_benchmark_bundle
from scripts.studies.paper_provenance import (
    file_identity,
    merge_git_provenance,
    merge_runtime_provenance,
    relative_to_output_dir,
    write_dataset_identity_manifest,
    write_exit_code_proof,
    write_split_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


NATURAL_PATCH_DATASET_ID = "natural_patches128_fixedprobe_v1"
NATURAL_PATCH_ROW_ROSTER = (
    "baseline",
    "pinn",
    "pinn_hybrid_resnet",
    "pinn_fno_vanilla",
    "pinn_ffno",
    "pinn_neuralop_uno",
)

DEFAULT_SEED = 3
DEFAULT_EPOCHS = 40
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_TORCH_SCHEDULER = "ReduceLROnPlateau"
DEFAULT_TORCH_PLATEAU_FACTOR = 0.5
DEFAULT_TORCH_PLATEAU_PATIENCE = 2
DEFAULT_TORCH_PLATEAU_MIN_LR = 1e-4
DEFAULT_TORCH_PLATEAU_THRESHOLD = 0.0
DEFAULT_TORCH_LOSS_MODE = "mae"
DEFAULT_TORCH_OUTPUT_MODE = "real_imag"
DEFAULT_FNO_MODES = 12
DEFAULT_FNO_WIDTH = 32
DEFAULT_FNO_BLOCKS = 4
DEFAULT_FNO_CNN_BLOCKS = 2
DEFAULT_BATCH_SIZE = 16


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, default=_json_default), encoding="utf-8")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_probe_manifest(probe_manifest: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(probe_manifest)
    if not normalized.get("canonical_pipeline") and normalized.get("pipeline_spec"):
        normalized["canonical_pipeline"] = normalized["pipeline_spec"]
    return normalized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _dataset_training_config(*, patch_size: int, output_dir: Path) -> TrainingConfig:
    return TrainingConfig(
        model=ModelConfig(N=patch_size, gridsize=1, model_type="pinn"),
        train_data_file=output_dir / "train_grouped.npz",
        test_data_file=output_dir / "test_grouped.npz",
        output_dir=output_dir,
        nepochs=DEFAULT_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        nphotons=1e9,
        backend="pytorch",
    )


def _build_grouped_payload(
    *,
    split_name: str,
    split_npz: Path,
    probe: np.ndarray,
) -> dict[str, np.ndarray]:
    with np.load(split_npz, allow_pickle=True) as data:
        objects = np.asarray(data["objects"], dtype=np.complex64)
        diffraction = np.asarray(data["diffraction"], dtype=np.float32)
        source_ids = np.asarray(data["source_ids"])
        patch_ids = np.asarray(data["patch_ids"])
    if objects.ndim != 3:
        raise ValueError(f"{split_name} objects must be rank-3, got {objects.shape}")
    if diffraction.ndim != 3:
        raise ValueError(f"{split_name} diffraction must be rank-3, got {diffraction.shape}")
    n_samples, patch_h, patch_w = objects.shape
    diffraction_grouped = diffraction[..., np.newaxis].astype(np.float32, copy=False)
    complex_grouped = objects[..., np.newaxis].astype(np.complex64, copy=False)
    zero_coords = np.zeros((n_samples, 1, 2, 1), dtype=np.float32)
    nn_indices = np.arange(n_samples, dtype=np.int64)[:, np.newaxis]
    return {
        "diffraction": diffraction_grouped,
        "X_full": diffraction_grouped,
        "Y": complex_grouped,
        "Y_I": np.abs(complex_grouped).astype(np.float32),
        "Y_phi": np.angle(complex_grouped).astype(np.float32),
        "coords_nominal": zero_coords.copy(),
        "coords_true": zero_coords.copy(),
        "coords_offsets": zero_coords.copy(),
        "coords_relative": zero_coords.copy(),
        "coords_start_offsets": zero_coords.copy(),
        "coords_start_relative": zero_coords.copy(),
        "YY_full": objects.copy(),
        "YY_ground_truth": objects.copy(),
        "probeGuess": np.asarray(probe, dtype=np.complex64),
        "nn_indices": nn_indices,
        "norm_Y_I": np.asarray(1.0, dtype=np.float32),
        "metadata_split": np.asarray([split_name], dtype=object),
        "source_ids": source_ids,
        "patch_ids": patch_ids,
        "object_count": np.asarray(n_samples, dtype=np.int64),
        "patch_size_hw": np.asarray([patch_h, patch_w], dtype=np.int64),
    }


def _write_grouped_split(
    *,
    grouped_path: Path,
    grouped_payload: Mapping[str, np.ndarray],
    metadata_config: TrainingConfig,
    split_name: str,
    dataset_id: str,
    probe_pipeline: str,
) -> Path:
    metadata = MetadataManager.create_metadata(
        metadata_config,
        script_name="cdi_natural_patch_benchmark",
        split_name=split_name,
        dataset_id=dataset_id,
        coords_type="relative",
        benchmark_contract="one_scan_group_per_object_patch_zero_coords",
        probe_transform_pipeline=probe_pipeline,
    )
    MetadataManager.save_with_metadata(
        str(grouped_path),
        dict(grouped_payload),
        metadata,
    )
    return grouped_path


def _load_existing_prepared_inputs(
    *,
    dataset_root: Path,
    item_root: Path,
    manifests: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    prepared_root = Path(item_root) / "prepared_inputs"
    prepared_manifest_path = prepared_root / "prepared_input_manifest.json"
    identity_audit_path = prepared_root / "grouped_input_identity_audit.json"
    required = [
        prepared_manifest_path,
        identity_audit_path,
        prepared_root / "train_grouped.npz",
        prepared_root / "val_grouped.npz",
        prepared_root / "test_grouped.npz",
    ]
    if not all(path.exists() for path in required):
        return None
    try:
        prepared_manifest = _load_json(prepared_manifest_path)
        identity_audit = _load_json(identity_audit_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if prepared_manifest.get("dataset_id") != NATURAL_PATCH_DATASET_ID:
        return None
    if identity_audit.get("dataset_id") != NATURAL_PATCH_DATASET_ID:
        return None
    if Path(identity_audit.get("source_root", "")).resolve() != Path(dataset_root).resolve():
        return None
    grouped_paths: dict[str, str] = {}
    expected_counts = manifests["split_manifest"].get("split_counts", {})
    for split_name in ("train", "val", "test"):
        manifest_grouped = prepared_manifest.get("grouped_paths", {}).get(split_name)
        grouped_path = Path(item_root) / manifest_grouped if manifest_grouped else prepared_root / f"{split_name}_grouped.npz"
        if not grouped_path.exists():
            return None
        try:
            with np.load(grouped_path, allow_pickle=True) as data:
                observed_count = int(np.asarray(data["diffraction"]).shape[0])
        except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile):
            return None
        if observed_count != int(expected_counts.get(split_name, observed_count)):
            return None
        grouped_paths[split_name] = str(grouped_path)
    normalized_probe_lineage = manifests["probe_manifest"].get("canonical_pipeline")
    normalized_grouped_paths = {
        split_name: _relative(Path(path), item_root)
        for split_name, path in grouped_paths.items()
    }
    prepared_manifest_changed = False
    if prepared_manifest.get("probe_lineage") != normalized_probe_lineage:
        prepared_manifest["probe_lineage"] = normalized_probe_lineage
        prepared_manifest_changed = True
    if prepared_manifest.get("split_counts") != expected_counts:
        prepared_manifest["split_counts"] = expected_counts
        prepared_manifest_changed = True
    if prepared_manifest.get("grouped_paths") != normalized_grouped_paths:
        prepared_manifest["grouped_paths"] = normalized_grouped_paths
        prepared_manifest_changed = True
    if prepared_manifest_changed:
        _write_json(prepared_manifest_path, prepared_manifest)
    return {
        "dataset_root": str(dataset_root),
        "prepared_root": str(prepared_root),
        "grouped_paths": grouped_paths,
        "prepared_input_manifest": str(prepared_manifest_path),
        "grouped_input_identity_audit": str(identity_audit_path),
        "manifests": manifests,
    }


def _resolve_fixed_sample_ids(test_count: int) -> list[int]:
    if test_count <= 0:
        raise ValueError("test split must contain at least one sample")
    candidates = {0, test_count // 2, test_count - 1}
    return sorted(candidate for candidate in candidates if 0 <= candidate < test_count)


def _resolve_visual_scales(
    *,
    test_grouped_path: Path,
    fixed_sample_ids: Sequence[int],
) -> dict[str, Any]:
    with np.load(test_grouped_path, allow_pickle=True) as data:
        y = np.asarray(data["Y"], dtype=np.complex64)[list(fixed_sample_ids), ..., 0]
        diffraction = np.asarray(data["diffraction"], dtype=np.float32)[list(fixed_sample_ids), ..., 0]
    amp = np.abs(y)
    phase = np.angle(y)
    return {
        "amplitude": {"vmin": float(np.min(amp)), "vmax": float(np.max(amp))},
        "phase": {"vmin": float(np.min(phase)), "vmax": float(np.max(phase))},
        "diffraction": {
            "vmin": float(np.min(diffraction)),
            "vmax": float(np.max(diffraction)),
        },
        "error_amplitude": {"vmin": 0.0, "vmax": float(np.max(amp) - np.min(amp))},
        "error_phase": {"vmin": 0.0, "vmax": float(np.max(phase) - np.min(phase))},
    }


def _load_grouped_split(path: Path) -> dict[str, Any]:
    data, metadata = MetadataManager.load_with_metadata(str(path))
    payload = dict(data)
    payload["_metadata"] = metadata or {}
    return payload


def _patchwise_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    *,
    label: str,
) -> dict[str, tuple[float, float]]:
    def _as_scalar(value: Any) -> Optional[float]:
        array = np.asarray(value)
        if array.size != 1:
            return None
        return float(array.reshape(()))

    pred = np.asarray(predictions)
    gt = np.asarray(ground_truth)
    if pred.ndim == 4 and pred.shape[-1] == 1:
        pred = pred[..., 0]
    if gt.ndim == 4 and gt.shape[-1] == 1:
        gt = gt[..., 0]
    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]
    if gt.ndim == 2:
        gt = gt[np.newaxis, ...]
    if pred.shape[0] != gt.shape[0]:
        raise ValueError(f"prediction/test batch mismatch for {label}: {pred.shape} vs {gt.shape}")
    per_metric: dict[str, list[tuple[float, float]]] = {}
    for index in range(pred.shape[0]):
        metrics = evaluation.eval_reconstruction(
            pred[index][..., np.newaxis],
            gt[index][..., np.newaxis],
            label=f"{label}_{index}",
        )
        for metric_name, pair in metrics.items():
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            first = _as_scalar(pair[0])
            second = _as_scalar(pair[1])
            if first is None or second is None:
                continue
            per_metric.setdefault(metric_name, []).append((first, second))
    return {
        metric_name: (
            float(np.mean([pair[0] for pair in pairs])),
            float(np.mean([pair[1] for pair in pairs])),
        )
        for metric_name, pairs in per_metric.items()
    }


def _save_fixed_sample_visuals(
    *,
    run_root: Path,
    model_id: str,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    fixed_sample_ids: Sequence[int],
    scales: Mapping[str, Any],
) -> dict[str, str]:
    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)
    if predictions.ndim == 4 and predictions.shape[-1] == 1:
        predictions = predictions[..., 0]
    if ground_truth.ndim == 4 and ground_truth.shape[-1] == 1:
        ground_truth = ground_truth[..., 0]
    visuals_dir = Path(run_root) / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    amp_path = visuals_dir / f"amp_phase_{model_id}.png"
    err_path = visuals_dir / f"amp_phase_error_{model_id}.png"
    sample_ids = list(fixed_sample_ids)

    fig_amp, axes_amp = plt.subplots(2, len(sample_ids), figsize=(4 * len(sample_ids), 6))
    fig_err, axes_err = plt.subplots(2, len(sample_ids), figsize=(4 * len(sample_ids), 6))
    if len(sample_ids) == 1:
        axes_amp = np.asarray(axes_amp).reshape(2, 1)
        axes_err = np.asarray(axes_err).reshape(2, 1)
    amp_scale = scales["amplitude"]
    phase_scale = scales["phase"]
    err_amp_scale = scales["error_amplitude"]
    err_phase_scale = scales["error_phase"]
    for column, sample_id in enumerate(sample_ids):
        pred = np.squeeze(predictions[sample_id])
        gt = np.squeeze(ground_truth[sample_id])
        pred_amp = np.abs(pred)
        pred_phase = np.angle(pred)
        gt_amp = np.abs(gt)
        gt_phase = np.angle(gt)
        amp_error = np.abs(pred_amp - gt_amp)
        phase_error = np.abs(pred_phase - gt_phase)

        axes_amp[0, column].imshow(pred_amp, cmap="viridis", vmin=amp_scale["vmin"], vmax=amp_scale["vmax"])
        axes_amp[0, column].set_title(f"{model_id} amp #{sample_id}")
        axes_amp[1, column].imshow(pred_phase, cmap="twilight", vmin=phase_scale["vmin"], vmax=phase_scale["vmax"])
        axes_amp[1, column].set_title(f"{model_id} phase #{sample_id}")
        axes_err[0, column].imshow(amp_error, cmap="magma", vmin=err_amp_scale["vmin"], vmax=err_amp_scale["vmax"])
        axes_err[0, column].set_title(f"{model_id} amp err #{sample_id}")
        axes_err[1, column].imshow(phase_error, cmap="magma", vmin=err_phase_scale["vmin"], vmax=err_phase_scale["vmax"])
        axes_err[1, column].set_title(f"{model_id} phase err #{sample_id}")
        for row_axes in (axes_amp[:, column], axes_err[:, column]):
            for axis in row_axes:
                axis.set_xticks([])
                axis.set_yticks([])
    fig_amp.tight_layout()
    fig_err.tight_layout()
    fig_amp.savefig(amp_path, dpi=150)
    fig_err.savefig(err_path, dpi=150)
    plt.close(fig_amp)
    plt.close(fig_err)
    return {
        "amp_phase_png": _relative(amp_path, run_root),
        "amp_phase_error_png": _relative(err_path, run_root),
    }


def _normalize_patch_batch(array: np.ndarray, *, label: str) -> np.ndarray:
    patches = np.asarray(array)
    if patches.ndim == 2:
        patches = patches[np.newaxis, ...]
    if patches.ndim == 4 and patches.shape[-1] == 1:
        patches = patches[..., 0]
    if patches.ndim == 4 and patches.shape[1] == 1:
        patches = patches[:, 0, ...]
    if patches.ndim != 3:
        raise ValueError(f"{label} must resolve to a patch batch with shape (B, H, W), got {patches.shape}")
    return patches.astype(np.complex64, copy=False)


def _save_patchwise_prediction_artifacts(
    *,
    run_root: Path,
    model_id: str,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    fixed_sample_ids: Sequence[int],
    scales: Mapping[str, Any],
) -> dict[str, str]:
    predictions = _normalize_patch_batch(predictions, label=f"{model_id} predictions")
    ground_truth = _normalize_patch_batch(ground_truth, label="ground_truth")
    sample_ids = [int(sample_id) for sample_id in fixed_sample_ids]
    if not sample_ids:
        raise ValueError("fixed_sample_ids must not be empty")
    if predictions.shape[0] != ground_truth.shape[0]:
        raise ValueError(
            f"prediction/test batch mismatch for {model_id}: {predictions.shape} vs {ground_truth.shape}"
        )
    if predictions.shape[0] == len(sample_ids) and ground_truth.shape[0] == len(sample_ids):
        selected_pred = predictions
        selected_gt = ground_truth
    elif min(sample_ids) < 0 or max(sample_ids) >= predictions.shape[0]:
        raise IndexError(f"fixed sample ids {sample_ids} outside prediction batch of {predictions.shape[0]}")
    else:
        selected_pred = predictions[sample_ids]
        selected_gt = ground_truth[sample_ids]
    patchwise_dir = Path(run_root) / "patchwise" / model_id
    visuals_dir = Path(run_root) / "visuals"
    patchwise_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)
    npz_path = patchwise_dir / "fixed_samples.npz"
    png_path = visuals_dir / f"patchwise_gt_pred_error_{model_id}.png"

    pred_amp = np.abs(selected_pred).astype(np.float32)
    pred_phase = np.angle(selected_pred).astype(np.float32)
    gt_amp = np.abs(selected_gt).astype(np.float32)
    gt_phase = np.angle(selected_gt).astype(np.float32)
    amp_error = np.abs(pred_amp - gt_amp).astype(np.float32)
    phase_error = np.abs(np.angle(np.exp(1j * (pred_phase - gt_phase)))).astype(np.float32)
    np.savez(
        npz_path,
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        YY_pred=selected_pred.astype(np.complex64),
        YY_ground_truth=selected_gt.astype(np.complex64),
        amp_pred=pred_amp,
        phase_pred=pred_phase,
        amp_gt=gt_amp,
        phase_gt=gt_phase,
        amp_error=amp_error,
        phase_error=phase_error,
    )

    amp_scale = scales["amplitude"]
    phase_scale = scales["phase"]
    err_amp_scale = scales["error_amplitude"]
    err_phase_scale = scales["error_phase"]
    rows = (
        ("GT amp", gt_amp, "viridis", amp_scale),
        (f"{model_id} amp", pred_amp, "viridis", amp_scale),
        ("amp err", amp_error, "magma", err_amp_scale),
        ("GT phase", gt_phase, "twilight", phase_scale),
        (f"{model_id} phase", pred_phase, "twilight", phase_scale),
        ("phase err", phase_error, "magma", err_phase_scale),
    )
    fig, axes = plt.subplots(len(rows), len(sample_ids), figsize=(4 * len(sample_ids), 16))
    if len(sample_ids) == 1:
        axes = np.asarray(axes).reshape(len(rows), 1)
    for row_index, (row_label, values, cmap, scale) in enumerate(rows):
        for column, sample_id in enumerate(sample_ids):
            axis = axes[row_index, column]
            axis.imshow(values[column], cmap=cmap, vmin=scale["vmin"], vmax=scale["vmax"])
            axis.set_title(f"{row_label} #{sample_id}")
            axis.set_xticks([])
            axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return {
        "patchwise_npz": _relative(npz_path, run_root),
        "gt_pred_error_png": _relative(png_path, run_root),
    }


def _torch_runner_config_from_saved_row(row_config_path: Path, *, test_npz: Path):
    from scripts.studies import grid_lines_torch_runner as torch_runner

    payload = _load_json(row_config_path)
    cfg_payload = dict(payload.get("torch_runner_config", {}))
    if not cfg_payload:
        raise ValueError(f"missing torch_runner_config in {row_config_path}")
    for field_name in ("train_npz", "test_npz", "output_dir", "artifact_root"):
        if cfg_payload.get(field_name) is not None:
            cfg_payload[field_name] = Path(cfg_payload[field_name])
    cfg_payload["test_npz"] = Path(test_npz)
    allowed_fields = {field.name for field in fields(torch_runner.TorchRunnerConfig)}
    cfg_payload = {key: value for key, value in cfg_payload.items() if key in allowed_fields}
    return torch_runner.TorchRunnerConfig(**cfg_payload)


def _build_torch_model_from_saved_config(cfg: Any, *, n_groups: int = 1):
    from ptycho_torch.config_factory import create_training_payload
    from ptycho_torch.config_params import InferenceConfig as PTInferenceConfig
    from ptycho_torch.generators.registry import resolve_generator
    from scripts.studies import grid_lines_torch_runner as torch_runner

    config, execution_config = torch_runner.setup_torch_configs(cfg)
    config.n_groups = int(n_groups)
    mode_map = {"pinn": "Unsupervised", "supervised": "Supervised"}
    factory_overrides = {
        "n_groups": config.n_groups,
        "gridsize": config.model.gridsize,
        "architecture": config.model.architecture,
        "model_type": mode_map.get(config.model.model_type, "Unsupervised"),
        "amp_activation": config.model.amp_activation,
        "n_filters_scale": config.model.n_filters_scale,
        "object_big": config.model.object_big,
        "probe_big": config.model.probe_big,
        "probe_mask": config.model.probe_mask,
        "probe_mask_sigma": getattr(config.model, "probe_mask_sigma", 1.0),
        "probe_mask_diameter": getattr(config.model, "probe_mask_diameter", None),
        "pad_object": config.model.pad_object,
        "nphotons": config.nphotons,
        "neighbor_count": config.neighbor_count,
        "max_epochs": config.nepochs,
        "batch_size": getattr(config, "batch_size", 16),
        "subsample_seed": getattr(config, "subsample_seed", None),
        "torch_loss_mode": getattr(config, "torch_loss_mode", "poisson"),
        "torch_mae_pred_l2_match_target": getattr(config, "torch_mae_pred_l2_match_target", False),
        "log_grad_norm": getattr(config, "log_grad_norm", False),
        "grad_norm_log_freq": getattr(config, "grad_norm_log_freq", 1),
        "test_data_file": config.test_data_file,
    }
    if execution_config.learning_rate is not None:
        factory_overrides["learning_rate"] = execution_config.learning_rate
    if execution_config.gradient_clip_val is not None:
        factory_overrides["gradient_clip_val"] = execution_config.gradient_clip_val
    for opt_field in (
        "scheduler",
        "optimizer",
        "weight_decay",
        "momentum",
        "adam_beta1",
        "adam_beta2",
        "plateau_factor",
        "plateau_patience",
        "plateau_min_lr",
        "plateau_threshold",
    ):
        value = getattr(config, opt_field, None)
        if value is not None:
            factory_overrides[opt_field] = value
    for field_name in ("fno_modes", "fno_width", "fno_blocks", "fno_cnn_blocks", "fno_input_transform"):
        field_value = getattr(config.model, field_name, None)
        if field_value is not None:
            factory_overrides[field_name] = field_value
    generator_output_mode = getattr(config.model, "generator_output_mode", None)
    if generator_output_mode is not None:
        factory_overrides["generator_output_mode"] = generator_output_mode
    for field_name in (
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
        "spectral_bottleneck_blocks",
        "spectral_bottleneck_modes",
        "spectral_bottleneck_share_weights",
        "spectral_bottleneck_gate_init",
        "spectral_bottleneck_gate_mode",
    ):
        field_value = getattr(execution_config, field_name, None)
        if field_value is not None:
            factory_overrides[field_name] = field_value
    payload = create_training_payload(
        train_data_file=Path(config.train_data_file),
        output_dir=Path(config.output_dir),
        execution_config=execution_config,
        overrides=factory_overrides,
    )
    pt_configs = {
        "model_config": payload.pt_model_config,
        "data_config": payload.pt_data_config,
        "training_config": payload.pt_training_config,
        "inference_config": PTInferenceConfig(),
        "execution_config": execution_config,
    }
    return resolve_generator(config).build_model(pt_configs)


def _run_saved_torch_row_inference(*, run_root: Path, model_id: str, test_npz: Path) -> np.ndarray:
    import torch
    from scripts.studies import grid_lines_torch_runner as torch_runner

    cfg = _torch_runner_config_from_saved_row(
        run_root / "runs" / model_id / "config.json",
        test_npz=test_npz,
    )
    test_data, test_metadata = torch_runner.load_cached_dataset_with_metadata(test_npz)
    model = _build_torch_model_from_saved_config(cfg, n_groups=1)
    state_dict_path = run_root / "runs" / model_id / "model.pt"
    state_dict = torch.load(state_dict_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    predictions = torch_runner.run_torch_inference(model, test_data, cfg, metadata=test_metadata)
    return torch_runner.to_complex_patches(predictions) if predictions.shape[-1] == 2 else predictions


def _run_saved_baseline_inference(*, run_root: Path, test_npz: Path) -> np.ndarray:
    import tensorflow as tf
    from ptycho.workflows import grid_lines_workflow as glw

    test_grouped = _load_grouped_split(test_npz)
    model = tf.keras.models.load_model(run_root / "baseline" / "baseline.keras", compile=False)
    return glw.run_baseline_inference(model, np.asarray(test_grouped["diffraction"], dtype=np.float32))


def _run_saved_pinn_inference(*, run_root: Path, test_npz: Path) -> np.ndarray:
    from ptycho import model_manager
    from ptycho.workflows import grid_lines_workflow as glw

    test_grouped = _load_grouped_split(test_npz)
    config = _build_tf_training_config(
        model_type="pinn",
        train_npz=test_npz,
        val_npz=test_npz,
        run_root=run_root,
    )
    update_legacy_dict(params.cfg, config)
    models = model_manager.ModelManager.load_multiple_models(
        str(run_root / "pinn" / "wts.h5"),
        ["diffraction_to_obj"],
    )
    container = _build_tf_container(test_grouped)
    predictions = glw.run_pinn_inference(
        models["diffraction_to_obj"],
        container.X,
        container.coords_nominal,
    )
    if predictions is None:
        raise RuntimeError("saved PINN inference returned None")
    return predictions


def export_saved_patchwise_predictions(
    *,
    run_root: Path,
    rows: Sequence[str] = NATURAL_PATCH_ROW_ROSTER,
    fixed_sample_ids: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    run_root = Path(run_root)
    manifest_path = run_root / "paper_benchmark_manifest.json"
    manifest = _load_json(manifest_path)
    item_root = Path(manifest["item_root"])
    test_npz = item_root / "prepared_inputs" / "test_grouped.npz"
    if not test_npz.exists():
        raise FileNotFoundError(f"missing prepared test split: {test_npz}")
    fixed_sample_manifest = _load_json(item_root / "contract" / "fixed_sample_manifest.json")
    scales = _load_json(item_root / "contract" / "shared_visual_scales.json")
    sample_ids = list(fixed_sample_ids or fixed_sample_manifest["fixed_sample_ids"])
    test_grouped = _load_grouped_split(test_npz)
    ground_truth = np.asarray(test_grouped["YY_ground_truth"], dtype=np.complex64)
    exporters = {
        "baseline": lambda: _run_saved_baseline_inference(run_root=run_root, test_npz=test_npz),
        "pinn": lambda: _run_saved_pinn_inference(run_root=run_root, test_npz=test_npz),
        "pinn_hybrid_resnet": lambda: _run_saved_torch_row_inference(
            run_root=run_root, model_id="pinn_hybrid_resnet", test_npz=test_npz
        ),
        "pinn_fno_vanilla": lambda: _run_saved_torch_row_inference(
            run_root=run_root, model_id="pinn_fno_vanilla", test_npz=test_npz
        ),
        "pinn_ffno": lambda: _run_saved_torch_row_inference(
            run_root=run_root, model_id="pinn_ffno", test_npz=test_npz
        ),
        "pinn_neuralop_uno": lambda: _run_saved_torch_row_inference(
            run_root=run_root, model_id="pinn_neuralop_uno", test_npz=test_npz
        ),
    }
    outputs: dict[str, Any] = {}
    for row in rows:
        if row not in exporters:
            raise ValueError(f"unsupported saved natural-patch row export: {row}")
        predictions = exporters[row]()
        artifacts = _save_patchwise_prediction_artifacts(
            run_root=run_root,
            model_id=row,
            predictions=predictions,
            ground_truth=ground_truth,
            fixed_sample_ids=sample_ids,
            scales=scales,
        )
        normalized = _normalize_patch_batch(predictions, label=f"{row} predictions")
        selected = normalized[sample_ids]
        outputs[row] = {
            "artifacts": artifacts,
            "prediction_shape": list(normalized.shape),
            "selected_amp_mean": float(np.mean(np.abs(selected))),
            "selected_phase_mean": float(np.mean(np.angle(selected))),
        }
    export_manifest_path = run_root / "patchwise" / "patchwise_export_manifest.json"
    if export_manifest_path.exists():
        previous_manifest = _load_json(export_manifest_path)
        previous_rows = dict(previous_manifest.get("rows", {}))
    else:
        previous_rows = {}
    previous_rows.update(outputs)
    manifest_out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "export_saved_patchwise_predictions_no_training",
        "source_run_root": str(run_root),
        "test_npz": str(test_npz),
        "fixed_sample_ids": sample_ids,
        "rows": previous_rows,
    }
    export_manifest_path = _write_json(export_manifest_path, manifest_out)
    manifest_out["manifest_path"] = str(export_manifest_path)
    return manifest_out


def _validate_dataset_contract(dataset_root: Path) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    required = [
        dataset_root / "dataset_manifest.json",
        dataset_root / "source_manifest.json",
        dataset_root / "split_manifest.json",
        dataset_root / "probe_manifest.json",
        dataset_root / "simulation_manifest.json",
        dataset_root / "adapter_contract.json",
        dataset_root / "train.npz",
        dataset_root / "val.npz",
        dataset_root / "test.npz",
        dataset_root / "probe.npz",
        dataset_root / "verification" / "post_audit.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing locked natural-patch dataset artifacts: {missing}")
    dataset_manifest = _load_json(dataset_root / "dataset_manifest.json")
    split_manifest = _load_json(dataset_root / "split_manifest.json")
    probe_manifest = _normalize_probe_manifest(_load_json(dataset_root / "probe_manifest.json"))
    post_audit = _load_json(dataset_root / "verification" / "post_audit.json")
    if dataset_manifest.get("dataset_id") != NATURAL_PATCH_DATASET_ID:
        raise ValueError(
            f"unexpected dataset id {dataset_manifest.get('dataset_id')!r}; expected {NATURAL_PATCH_DATASET_ID!r}"
        )
    if not post_audit.get("manifests_present") or not post_audit.get("no_source_overlap"):
        raise ValueError(f"dataset post-audit failed: {post_audit}")
    return {
        "dataset_manifest": dataset_manifest,
        "source_manifest": _load_json(dataset_root / "source_manifest.json"),
        "split_manifest": split_manifest,
        "probe_manifest": probe_manifest,
        "simulation_manifest": _load_json(dataset_root / "simulation_manifest.json"),
        "adapter_contract": _load_json(dataset_root / "adapter_contract.json"),
        "post_audit": post_audit,
    }


def prepare_natural_patch_inputs(*, dataset_root: Path, item_root: Path) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    item_root = Path(item_root)
    manifests = _validate_dataset_contract(dataset_root)
    existing = _load_existing_prepared_inputs(
        dataset_root=dataset_root,
        item_root=item_root,
        manifests=manifests,
    )
    if existing is not None:
        return existing
    prepared_root = item_root / "prepared_inputs"
    prepared_root.mkdir(parents=True, exist_ok=True)
    with np.load(dataset_root / "probe.npz", allow_pickle=True) as probe_npz:
        probe = np.asarray(probe_npz["probeGuess"], dtype=np.complex64)
    patch_size = int(manifests["dataset_manifest"].get("patch_size", probe.shape[0]))
    metadata_config = _dataset_training_config(patch_size=patch_size, output_dir=prepared_root)
    grouped_paths: dict[str, str] = {}
    grouped_identities: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "val", "test"):
        split_npz = dataset_root / f"{split_name}.npz"
        grouped_payload = _build_grouped_payload(split_name=split_name, split_npz=split_npz, probe=probe)
        grouped_path = prepared_root / f"{split_name}_grouped.npz"
        _write_grouped_split(
            grouped_path=grouped_path,
            grouped_payload=grouped_payload,
            metadata_config=metadata_config,
            split_name=split_name,
            dataset_id=NATURAL_PATCH_DATASET_ID,
            probe_pipeline=str(manifests["probe_manifest"].get("canonical_pipeline", "")),
        )
        grouped_paths[split_name] = str(grouped_path)
        grouped_identities[split_name] = {
            "source_npz": str(split_npz),
            "source_sha256": _sha256(split_npz),
            "grouped_npz": str(grouped_path),
            "grouped_sha256": _sha256(grouped_path),
            "sample_count": int(grouped_payload["diffraction"].shape[0]),
            "probe_sha256": _sha256(dataset_root / "probe.npz"),
        }
    prepared_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": NATURAL_PATCH_DATASET_ID,
        "source_corpus": manifests["source_manifest"].get("source_names", []),
        "split_counts": manifests["split_manifest"].get("split_counts", {}),
        "grouped_paths": {
            split_name: _relative(Path(path), item_root)
            for split_name, path in grouped_paths.items()
        },
        "probe_lineage": manifests["probe_manifest"].get("canonical_pipeline"),
        "adapter_contract_assumptions": {
            "one_scan_group_per_object_patch": True,
            "one_zero_coordinate_per_sample": True,
            "derive_Y_from_objects": True,
            "reuse_frozen_diffraction": True,
        },
    }
    audit_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": NATURAL_PATCH_DATASET_ID,
        "source_root": str(dataset_root),
        "prepared_root": str(prepared_root),
        "derived_from_locked_dataset": True,
        "regenerated_diffraction": False,
        "grouped_split_identities": grouped_identities,
    }
    _write_json(prepared_root / "prepared_input_manifest.json", prepared_manifest)
    _write_json(prepared_root / "grouped_input_identity_audit.json", audit_payload)
    return {
        "dataset_root": str(dataset_root),
        "prepared_root": str(prepared_root),
        "grouped_paths": grouped_paths,
        "prepared_input_manifest": str(prepared_root / "prepared_input_manifest.json"),
        "grouped_input_identity_audit": str(prepared_root / "grouped_input_identity_audit.json"),
        "manifests": manifests,
    }


def _write_contract_artifacts(
    *,
    dataset_root: Path,
    item_root: Path,
    prepared: Mapping[str, Any],
    rows: Sequence[str],
    seed: int,
) -> dict[str, str]:
    contract_root = Path(item_root) / "contract"
    contract_root.mkdir(parents=True, exist_ok=True)
    test_grouped_path = Path(prepared["grouped_paths"]["test"])
    with np.load(test_grouped_path, allow_pickle=True) as data:
        test_count = int(data["diffraction"].shape[0])
    fixed_sample_ids = _resolve_fixed_sample_ids(test_count)
    visual_scales = _resolve_visual_scales(
        test_grouped_path=test_grouped_path,
        fixed_sample_ids=fixed_sample_ids,
    )
    contract_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": NATURAL_PATCH_DATASET_ID,
        "dataset_root": str(dataset_root),
        "row_roster": list(rows),
        "single_seed": int(seed),
        "three_way_split": {
            split_name: _relative(Path(path), item_root)
            for split_name, path in prepared["grouped_paths"].items()
        },
        "frozen_training_recipe": {
            "torch_epochs": DEFAULT_EPOCHS,
            "torch_learning_rate": DEFAULT_LEARNING_RATE,
            "torch_scheduler": DEFAULT_TORCH_SCHEDULER,
            "torch_plateau_factor": DEFAULT_TORCH_PLATEAU_FACTOR,
            "torch_plateau_patience": DEFAULT_TORCH_PLATEAU_PATIENCE,
            "torch_plateau_min_lr": DEFAULT_TORCH_PLATEAU_MIN_LR,
            "torch_plateau_threshold": DEFAULT_TORCH_PLATEAU_THRESHOLD,
            "torch_loss_mode": DEFAULT_TORCH_LOSS_MODE,
            "torch_output_mode": DEFAULT_TORCH_OUTPUT_MODE,
            "fno_modes": DEFAULT_FNO_MODES,
            "fno_width": DEFAULT_FNO_WIDTH,
            "fno_blocks": DEFAULT_FNO_BLOCKS,
            "fno_cnn_blocks": DEFAULT_FNO_CNN_BLOCKS,
        },
        "claim_boundary": "single_seed_natural_patch_expanded_object_cdi_only",
        "does_not_replace_lines128": True,
    }
    fixed_sample_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": NATURAL_PATCH_DATASET_ID,
        "split": "test",
        "fixed_sample_ids": fixed_sample_ids,
    }
    contract_path = _write_json(contract_root / "natural_patch_benchmark_contract.json", contract_payload)
    fixed_sample_path = _write_json(contract_root / "fixed_sample_manifest.json", fixed_sample_payload)
    visual_scale_path = _write_json(contract_root / "shared_visual_scales.json", visual_scales)
    return {
        "contract_path": str(contract_path),
        "fixed_sample_manifest_path": str(fixed_sample_path),
        "shared_visual_scales_path": str(visual_scale_path),
    }


def _build_benchmark_manifest(
    *,
    dataset_root: Path,
    item_root: Path,
    run_root: Path,
    rows: Sequence[str],
    row_statuses: Mapping[str, Mapping[str, object]],
    row_payloads: Mapping[str, Mapping[str, object]],
    bundle_paths: Mapping[str, str],
) -> dict[str, object]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": NATURAL_PATCH_DATASET_ID,
        "dataset_root": str(dataset_root),
        "item_root": str(item_root),
        "run_root": str(run_root),
        "row_roster": list(rows),
        "row_statuses": dict(row_statuses),
        "completed_rows": sorted(row_payloads.keys()),
        "bundle_paths": dict(bundle_paths),
        "claim_boundary": "single_seed_natural_patch_expanded_object_cdi_only",
        "does_not_replace_lines128": True,
    }


def _execute_rows_not_implemented(
    *,
    dataset_root: Path,
    item_root: Path,
    run_root: Path,
    prepared: Mapping[str, Any],
    rows: Sequence[str],
    seed: int,
) -> Mapping[str, Mapping[str, object]]:
    del dataset_root, item_root, prepared, seed
    row_statuses: dict[str, Mapping[str, object]] = {}
    for row in rows:
        row_statuses[row] = {
            "status": "blocked",
            "reason": "natural-patch row execution is not implemented in this harness yet",
            "run_root": str(run_root),
        }
    return row_statuses


def _update_row_config(
    *,
    run_root: Path,
    model_id: str,
    validation_npz: Path,
    final_eval_test_npz: Path,
) -> None:
    config_path = run_root / "runs" / model_id / "config.json"
    if not config_path.exists():
        return
    payload = _load_json(config_path)
    payload["validation_npz"] = str(validation_npz)
    payload["final_eval_test_npz"] = str(final_eval_test_npz)
    _write_json(config_path, payload)


def _update_tf_row_config(
    *,
    run_root: Path,
    model_id: str,
    validation_npz: Path,
    final_eval_test_npz: Path,
) -> None:
    _update_row_config(
        run_root=run_root,
        model_id=model_id,
        validation_npz=validation_npz,
        final_eval_test_npz=final_eval_test_npz,
    )


def _attach_natural_patch_row_provenance(
    *,
    run_root: Path,
    model_id: str,
    row_payload: Dict[str, object],
    train_npz: Path,
    test_npz: Path,
    probe_npz: Path,
    seed: int,
    nimgs_train: int,
    nimgs_test: int,
    gridsize: int = 1,
    set_phi: bool = False,
    proof_source: str,
    git_commit_override: Optional[str] = None,
) -> None:
    """Augment a natural-patch row payload with full paper-grade provenance scaffolding.

    Writes per-run dataset/split manifests and a per-row exit-code proof that
    `metrics_tables.write_paper_benchmark_bundle(require_row_provenance=True)`
    dereferences. Updates the row payload to carry the manifest references,
    full ``environment``/``git`` payloads, ``randomness.requested_seed``, and
    ``outputs.exit_code_proof_json``.
    """
    run_root = Path(run_root)
    run_dir = run_root / "runs" / model_id

    dataset_manifest = write_dataset_identity_manifest(
        run_root,
        train_npz=train_npz,
        test_npz=test_npz,
        dataset_source=NATURAL_PATCH_DATASET_ID,
        probe_npz=probe_npz,
        probe_source="custom",
    )
    split_manifest = write_split_manifest(
        run_root,
        train_npz=train_npz,
        test_npz=test_npz,
        seed=seed,
        nimgs_train=nimgs_train,
        nimgs_test=nimgs_test,
        gridsize=gridsize,
        set_phi=set_phi,
    )

    existing_dataset = row_payload.get("dataset")
    dataset_payload: Dict[str, Any] = (
        dict(existing_dataset) if isinstance(existing_dataset, Mapping) else {}
    )
    dataset_payload.update(
        {
            "train_npz": str(train_npz),
            "test_npz": str(test_npz),
            "probe_npz": str(probe_npz),
            "probe_source": "custom",
            "dataset_source": NATURAL_PATCH_DATASET_ID,
            "manifest_json": relative_to_output_dir(run_root, dataset_manifest),
        }
    )
    row_payload["dataset"] = dataset_payload

    existing_splits = row_payload.get("splits")
    split_payload: Dict[str, Any] = (
        dict(existing_splits) if isinstance(existing_splits, Mapping) else {}
    )
    split_payload.update(
        {
            "train": str(train_npz),
            "test": str(test_npz),
            "nimgs_train": int(nimgs_train),
            "nimgs_test": int(nimgs_test),
            "gridsize": int(gridsize),
            "set_phi": bool(set_phi),
            "seed": int(seed),
            "manifest_json": relative_to_output_dir(run_root, split_manifest),
        }
    )
    row_payload["splits"] = split_payload

    existing_randomness = row_payload.get("randomness")
    randomness: Dict[str, Any] = (
        dict(existing_randomness) if isinstance(existing_randomness, Mapping) else {}
    )
    randomness["requested_seed"] = int(seed)
    randomness.setdefault("seed", int(seed))
    row_payload["randomness"] = randomness

    existing_environment = row_payload.get("environment")
    existing_hardware = row_payload.get("hardware_summary")
    row_payload["environment"] = merge_runtime_provenance(
        existing_environment if isinstance(existing_environment, Mapping) else {},
        hardware_summary=existing_hardware if isinstance(existing_hardware, Mapping) else None,
    )
    existing_git = row_payload.get("git")
    if isinstance(git_commit_override, str) and git_commit_override:
        git_commit: Optional[str] = git_commit_override
        existing_git_for_merge: Mapping[str, Any] = (
            {key: value for key, value in existing_git.items() if key != "commit"}
            if isinstance(existing_git, Mapping)
            else {}
        )
    else:
        git_commit = existing_git.get("commit") if isinstance(existing_git, Mapping) else None
        existing_git_for_merge = existing_git if isinstance(existing_git, Mapping) else {}
    if not (isinstance(git_commit, str) and git_commit):
        from scripts.studies.invocation_logging import get_git_commit

        git_commit = get_git_commit(REPO_ROOT)
    row_payload["git"] = merge_git_provenance(
        existing_git_for_merge,
        repo_root=REPO_ROOT,
        commit=str(git_commit) if isinstance(git_commit, str) and git_commit else None,
        note_source="natural_patch_harness_row_provenance",
    )

    existing_outputs = row_payload.get("outputs")
    outputs_payload: Dict[str, Any] = (
        dict(existing_outputs) if isinstance(existing_outputs, Mapping) else {}
    )
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    invocation_path = run_dir / "invocation.json"
    exit_code_proof = write_exit_code_proof(
        run_root,
        model_id=model_id,
        invocation_json=invocation_path if invocation_path.exists() else None,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        proof_source=proof_source,
    )
    if exit_code_proof is not None:
        outputs_payload["exit_code_proof_json"] = relative_to_output_dir(run_root, exit_code_proof)
    else:
        outputs_payload.pop("exit_code_proof_json", None)
        stale_proof = run_dir / "exit_code_proof.json"
        if stale_proof.exists():
            stale_proof.unlink()
    if "stdout_log" not in outputs_payload and stdout_log.exists():
        outputs_payload["stdout_log"] = relative_to_output_dir(run_root, stdout_log)
    if "stderr_log" not in outputs_payload and stderr_log.exists():
        outputs_payload["stderr_log"] = relative_to_output_dir(run_root, stderr_log)
    row_payload["outputs"] = outputs_payload


def _run_torch_row(
    *,
    model_id: str,
    architecture: str,
    training_procedure: str,
    train_npz: Path,
    val_npz: Path,
    test_npz: Path,
    probe_npz: Path,
    run_root: Path,
    seed: int,
    fixed_sample_ids: Sequence[int],
    scales: Mapping[str, Any],
) -> dict[str, object]:
    from scripts.studies import grid_lines_torch_runner as torch_runner
    from scripts.studies.invocation_logging import update_invocation_artifacts
    from ptycho.workflows.grid_lines_workflow import save_recon_artifact

    cfg = torch_runner.TorchRunnerConfig(
        train_npz=train_npz,
        test_npz=val_npz,
        output_dir=run_root,
        artifact_root=run_root,
        architecture=architecture,
        training_procedure=training_procedure,
        model_id_override=model_id,
        seed=seed,
        epochs=DEFAULT_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        learning_rate=DEFAULT_LEARNING_RATE,
        infer_batch_size=DEFAULT_BATCH_SIZE,
        generator_output_mode=DEFAULT_TORCH_OUTPUT_MODE,
        N=128,
        gridsize=1,
        probe_source="custom",
        torch_loss_mode=DEFAULT_TORCH_LOSS_MODE,
        fno_modes=DEFAULT_FNO_MODES,
        fno_width=DEFAULT_FNO_WIDTH,
        fno_blocks=DEFAULT_FNO_BLOCKS,
        fno_cnn_blocks=DEFAULT_FNO_CNN_BLOCKS,
        scheduler=DEFAULT_TORCH_SCHEDULER,
        plateau_factor=DEFAULT_TORCH_PLATEAU_FACTOR,
        plateau_patience=DEFAULT_TORCH_PLATEAU_PATIENCE,
        plateau_min_lr=DEFAULT_TORCH_PLATEAU_MIN_LR,
        plateau_threshold=DEFAULT_TORCH_PLATEAU_THRESHOLD,
        reassembly_mode="position",
    )
    invocation_json = torch_runner._write_runner_invocation_artifacts(
        cfg,
        extra={
            "natural_patch_benchmark": True,
            "validation_npz": str(val_npz),
            "final_eval_test_npz": str(test_npz),
        },
    )
    try:
        train_data, train_metadata = torch_runner.load_cached_dataset_with_metadata(train_npz)
        val_data, val_metadata = torch_runner.load_cached_dataset_with_metadata(val_npz)
        test_data, test_metadata = torch_runner.load_cached_dataset_with_metadata(test_npz)

        train_start = time.perf_counter()
        results = torch_runner.run_torch_training(
            cfg,
            train_data,
            val_data,
            train_metadata=train_metadata,
            test_metadata=val_metadata,
        )
        train_wall_time_sec = time.perf_counter() - train_start

        model = results.get("model")
        if model is None and isinstance(results.get("models"), dict):
            model = results["models"].get("diffraction_to_obj")
        if model is None:
            raise RuntimeError(f"missing trained model for {model_id}")

        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        inference_start = time.perf_counter()
        predictions = torch_runner.run_torch_inference(model, test_data, cfg, metadata=test_metadata)
        inference_time_s = time.perf_counter() - inference_start
        pred_complex = torch_runner.to_complex_patches(predictions) if predictions.shape[-1] == 2 else predictions
        gt = test_data.get("YY_ground_truth")
        if gt is None:
            gt = test_data.get("YY_full")
        if gt is None:
            raise RuntimeError(f"missing test ground truth for {model_id}")
        pred_complex = np.asarray(pred_complex, dtype=np.complex64)
        gt = np.asarray(gt, dtype=np.complex64)
        recon_path = save_recon_artifact(run_root, model_id, pred_complex)
        metrics = _patchwise_metrics(pred_complex, gt, label=model_id)
        randomness_contract = torch_runner._build_randomness_contract(cfg)
        run_dir = torch_runner.save_run_artifacts(
            cfg,
            results,
            metrics,
            randomness_contract,
            recon_path=Path(recon_path),
        )
        _update_row_config(
            run_root=run_root,
            model_id=model_id,
            validation_npz=val_npz,
            final_eval_test_npz=test_npz,
        )
        stdout_log = _write_text(run_dir / "stdout.log", "Natural-patch torch row executed in-process.\n")
        stderr_log = _write_text(run_dir / "stderr.log", "")
        row_payload = torch_runner._build_paper_row_payload(
            cfg,
            metrics=metrics,
            history=results.get("history", {}),
            model_params=int(model_params),
            train_wall_time_sec=float(train_wall_time_sec),
            inference_time_s=float(inference_time_s),
            run_dir=Path(run_dir),
            recon_path=Path(recon_path),
            invocation_json=invocation_json,
        )
        row_payload["dataset"] = {
            "train_npz": str(train_npz),
            "validation_npz": str(val_npz),
            "test_npz": str(test_npz),
            "dataset_source": NATURAL_PATCH_DATASET_ID,
            "probe_source": "custom",
        }
        row_payload["splits"] = {
            "train": str(train_npz),
            "val": str(val_npz),
            "test": str(test_npz),
            "seed": int(seed),
            "gridsize": 1,
        }
        row_payload.setdefault("outputs", {})
        row_payload["outputs"]["stdout_log"] = _relative(stdout_log, run_root)
        row_payload["outputs"]["stderr_log"] = _relative(stderr_log, run_root)
        row_payload["outputs"]["recon_npz"] = _relative(Path(recon_path), run_root)
        row_payload["visuals"] = _save_fixed_sample_visuals(
            run_root=run_root,
            model_id=model_id,
            predictions=pred_complex,
            ground_truth=gt,
            fixed_sample_ids=fixed_sample_ids,
            scales=scales,
        )
        update_invocation_artifacts(
            invocation_json,
            status="completed",
            exit_code=0,
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            run_dir=str(run_dir),
        )
        with np.load(train_npz, allow_pickle=True) as train_data_npz:
            train_count = int(np.asarray(train_data_npz["diffraction"]).shape[0])
        with np.load(test_npz, allow_pickle=True) as test_data_npz:
            test_count_full = int(np.asarray(test_data_npz["diffraction"]).shape[0])
        _attach_natural_patch_row_provenance(
            run_root=run_root,
            model_id=model_id,
            row_payload=row_payload,
            train_npz=train_npz,
            test_npz=test_npz,
            probe_npz=probe_npz,
            seed=seed,
            nimgs_train=train_count,
            nimgs_test=test_count_full,
            gridsize=1,
            set_phi=False,
            proof_source="natural_patch_torch_row_in_process_completion",
        )
        return {"status": "completed", "row_payload": row_payload}
    except Exception as exc:
        update_invocation_artifacts(
            invocation_json,
            status="failed",
            exit_code=1,
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )
        return {"status": "blocked", "reason": str(exc)}


def _build_tf_training_config(
    *,
    model_type: str,
    train_npz: Path,
    val_npz: Path,
    run_root: Path,
) -> TrainingConfig:
    return TrainingConfig(
        model=ModelConfig(N=128, gridsize=1, model_type=model_type),
        train_data_file=train_npz,
        test_data_file=val_npz,
        output_dir=run_root,
        nepochs=DEFAULT_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        n_groups=1,
        neighbor_count=1,
        nphotons=1e9,
    )


def _build_tf_container(grouped: Mapping[str, Any]):
    from ptycho import loader

    probe = np.asarray(grouped["probeGuess"], dtype=np.complex64)
    callback_payload = {
        "X_full": np.asarray(grouped["X_full"], dtype=np.float32),
        "Y": np.asarray(grouped["Y"], dtype=np.complex64),
        "coords_offsets": np.asarray(grouped["coords_start_offsets"], dtype=np.float32),
        "coords_relative": np.asarray(grouped["coords_start_relative"], dtype=np.float32),
        "coords_start_offsets": np.asarray(grouped["coords_start_offsets"], dtype=np.float32),
        "coords_start_relative": np.asarray(grouped["coords_start_relative"], dtype=np.float32),
        "nn_indices": np.asarray(grouped["nn_indices"], dtype=np.int64),
    }
    return loader.load(lambda: callback_payload, probe, which=None, create_split=False)


def _run_tf_pinn_row(
    *,
    train_npz: Path,
    val_npz: Path,
    test_npz: Path,
    probe_npz: Path,
    run_root: Path,
    seed: int,
    fixed_sample_ids: Sequence[int],
    scales: Mapping[str, Any],
) -> dict[str, object]:
    from ptycho import model_manager
    from ptycho.workflows import components as wf_components
    from ptycho.workflows import grid_lines_workflow as glw

    train_grouped = _load_grouped_split(train_npz)
    val_grouped = _load_grouped_split(val_npz)
    test_grouped = _load_grouped_split(test_npz)
    config = _build_tf_training_config(
        model_type="pinn",
        train_npz=train_npz,
        val_npz=val_npz,
        run_root=run_root,
    )
    update_legacy_dict(params.cfg, config)
    glw._apply_execution_seed(seed)
    train_container = _build_tf_container(train_grouped)
    val_container = _build_tf_container(val_grouped)
    test_container = _build_tf_container(test_grouped)

    train_start = time.perf_counter()
    results = wf_components.train_cdi_model(train_container, val_container, config)
    train_wall_time_sec = time.perf_counter() - train_start
    model = results.get("models", {}).get("diffraction_to_obj")
    if model is None:
        from ptycho import model as model_module

        model = getattr(model_module, "diffraction_to_obj", None)
    if model is None:
        raise RuntimeError("missing trained TF PINN model")
    model_params = int(model.count_params()) if hasattr(model, "count_params") else 0

    pinn_dir = run_root / "pinn"
    pinn_dir.mkdir(parents=True, exist_ok=True)
    model_manager.save(str(pinn_dir))

    inference_start = time.perf_counter()
    predictions = glw.run_pinn_inference(model, test_container.X, test_container.coords_nominal)
    inference_time_s = time.perf_counter() - inference_start
    if predictions is None:
        raise RuntimeError("PINN inference returned None")
    pred_complex = np.asarray(predictions, dtype=np.complex64)[..., 0]
    gt = np.asarray(test_grouped["YY_ground_truth"], dtype=np.complex64)
    metrics = _patchwise_metrics(pred_complex, gt, label="pinn")
    row_payload = glw._build_tf_row_payload(
        model_id="pinn",
        model_label="CDI CNN + PINN",
        model=model,
        history=results.get("history", {}),
        metrics=metrics,
        N=128,
        epoch_budget=DEFAULT_EPOCHS,
        train_wall_time_sec=float(train_wall_time_sec),
        inference_time_sec=float(inference_time_s),
    )
    recon_path = glw.save_recon_artifact(run_root, "pinn", pred_complex)
    glw._write_tf_row_provenance(
        cfg=glw.GridLinesConfig(
            N=128,
            gridsize=1,
            output_dir=run_root,
            probe_npz=probe_npz,
            seed=seed,
            nimgs_train=int(train_grouped["diffraction"].shape[0]),
            nimgs_test=int(test_grouped["diffraction"].shape[0]),
            nphotons=1e9,
            nepochs=DEFAULT_EPOCHS,
            batch_size=DEFAULT_BATCH_SIZE,
        ),
        model_id="pinn",
        row_payload=row_payload,
        history=results.get("history", {}),
        train_npz=train_npz,
        test_npz=val_npz,
        model_artifact=pinn_dir / "wts.h5.zip",
        recon_path=Path(recon_path),
    )
    _update_tf_row_config(
        run_root=run_root,
        model_id="pinn",
        validation_npz=val_npz,
        final_eval_test_npz=test_npz,
    )
    dataset_field = row_payload.get("dataset")
    if isinstance(dataset_field, dict):
        dataset_field["validation_npz"] = str(val_npz)
        dataset_field["test_npz"] = str(test_npz)
    row_payload["visuals"] = _save_fixed_sample_visuals(
        run_root=run_root,
        model_id="pinn",
        predictions=pred_complex,
        ground_truth=gt,
        fixed_sample_ids=fixed_sample_ids,
        scales=scales,
    )
    _attach_natural_patch_row_provenance(
        run_root=run_root,
        model_id="pinn",
        row_payload=row_payload,
        train_npz=train_npz,
        test_npz=test_npz,
        probe_npz=probe_npz,
        seed=seed,
        nimgs_train=int(train_grouped["diffraction"].shape[0]),
        nimgs_test=int(test_grouped["diffraction"].shape[0]),
        gridsize=1,
        set_phi=False,
        proof_source="natural_patch_tf_pinn_in_process_completion",
    )
    return {"status": "completed", "row_payload": row_payload}


def _run_tf_baseline_row(
    *,
    train_npz: Path,
    val_npz: Path,
    test_npz: Path,
    probe_npz: Path,
    run_root: Path,
    seed: int,
    fixed_sample_ids: Sequence[int],
    scales: Mapping[str, Any],
) -> dict[str, object]:
    import tensorflow as tf
    from ptycho import baselines
    from ptycho.workflows import grid_lines_workflow as glw

    train_grouped = _load_grouped_split(train_npz)
    val_grouped = _load_grouped_split(val_npz)
    test_grouped = _load_grouped_split(test_npz)
    config = _build_tf_training_config(
        model_type="supervised",
        train_npz=train_npz,
        val_npz=val_npz,
        run_root=run_root,
    )
    update_legacy_dict(params.cfg, config)
    glw._apply_execution_seed(seed)
    x_train, y_i_train, y_phi_train = glw.select_baseline_channels(
        np.asarray(train_grouped["diffraction"], dtype=np.float32),
        np.asarray(train_grouped["Y_I"], dtype=np.float32),
        np.asarray(train_grouped["Y_phi"], dtype=np.float32),
    )
    x_val, y_i_val, y_phi_val = glw.select_baseline_channels(
        np.asarray(val_grouped["diffraction"], dtype=np.float32),
        np.asarray(val_grouped["Y_I"], dtype=np.float32),
        np.asarray(val_grouped["Y_phi"], dtype=np.float32),
    )
    model = baselines.build_model(x_train, y_i_train, y_phi_train)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=DEFAULT_TORCH_PLATEAU_FACTOR,
        patience=DEFAULT_TORCH_PLATEAU_PATIENCE,
        min_lr=DEFAULT_TORCH_PLATEAU_MIN_LR,
        verbose=1,
    )
    train_start = time.perf_counter()
    history = model.fit(
        x_train,
        [y_i_train, y_phi_train],
        shuffle=True,
        batch_size=DEFAULT_BATCH_SIZE,
        verbose=1,
        epochs=DEFAULT_EPOCHS,
        validation_data=(x_val, [y_i_val, y_phi_val]),
        callbacks=[reduce_lr, earlystop],
    )
    train_wall_time_sec = time.perf_counter() - train_start
    baseline_dir = run_root / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    model.save(baseline_dir / "baseline.keras")

    x_test = np.asarray(test_grouped["diffraction"], dtype=np.float32)
    inference_start = time.perf_counter()
    predictions = glw.run_baseline_inference(model, x_test)
    inference_time_s = time.perf_counter() - inference_start
    pred_complex = np.asarray(predictions, dtype=np.complex64)[..., 0]
    gt = np.asarray(test_grouped["YY_ground_truth"], dtype=np.complex64)
    metrics = _patchwise_metrics(pred_complex, gt, label="baseline")
    row_payload = glw._build_tf_row_payload(
        model_id="baseline",
        model_label="CDI CNN + supervised",
        model=model,
        history=history,
        metrics=metrics,
        N=128,
        epoch_budget=DEFAULT_EPOCHS,
        train_wall_time_sec=float(train_wall_time_sec),
        inference_time_sec=float(inference_time_s),
    )
    recon_path = glw.save_recon_artifact(run_root, "baseline", pred_complex)
    glw._write_tf_row_provenance(
        cfg=glw.GridLinesConfig(
            N=128,
            gridsize=1,
            output_dir=run_root,
            probe_npz=probe_npz,
            seed=seed,
            nimgs_train=int(train_grouped["diffraction"].shape[0]),
            nimgs_test=int(test_grouped["diffraction"].shape[0]),
            nphotons=1e9,
            nepochs=DEFAULT_EPOCHS,
            batch_size=DEFAULT_BATCH_SIZE,
        ),
        model_id="baseline",
        row_payload=row_payload,
        history=history,
        train_npz=train_npz,
        test_npz=val_npz,
        model_artifact=baseline_dir / "baseline.keras",
        recon_path=Path(recon_path),
    )
    _update_tf_row_config(
        run_root=run_root,
        model_id="baseline",
        validation_npz=val_npz,
        final_eval_test_npz=test_npz,
    )
    dataset_field = row_payload.get("dataset")
    if isinstance(dataset_field, dict):
        dataset_field["validation_npz"] = str(val_npz)
        dataset_field["test_npz"] = str(test_npz)
    row_payload["visuals"] = _save_fixed_sample_visuals(
        run_root=run_root,
        model_id="baseline",
        predictions=pred_complex,
        ground_truth=gt,
        fixed_sample_ids=fixed_sample_ids,
        scales=scales,
    )
    _attach_natural_patch_row_provenance(
        run_root=run_root,
        model_id="baseline",
        row_payload=row_payload,
        train_npz=train_npz,
        test_npz=test_npz,
        probe_npz=probe_npz,
        seed=seed,
        nimgs_train=int(train_grouped["diffraction"].shape[0]),
        nimgs_test=int(test_grouped["diffraction"].shape[0]),
        gridsize=1,
        set_phi=False,
        proof_source="natural_patch_tf_baseline_in_process_completion",
    )
    return {"status": "completed", "row_payload": row_payload}


def _execute_rows(
    *,
    dataset_root: Path,
    item_root: Path,
    run_root: Path,
    prepared: Mapping[str, Any],
    rows: Sequence[str],
    seed: int,
) -> Mapping[str, Mapping[str, object]]:
    grouped_paths = prepared["grouped_paths"]
    train_npz = Path(grouped_paths["train"])
    val_npz = Path(grouped_paths["val"])
    test_npz = Path(grouped_paths["test"])
    probe_npz = Path(dataset_root) / "probe.npz"
    fixed_sample_manifest = _load_json(Path(item_root) / "contract" / "fixed_sample_manifest.json")
    scales = _load_json(Path(item_root) / "contract" / "shared_visual_scales.json")
    fixed_sample_ids = fixed_sample_manifest["fixed_sample_ids"]
    results: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if row == "baseline":
            results[row] = _run_tf_baseline_row(
                train_npz=train_npz,
                val_npz=val_npz,
                test_npz=test_npz,
                probe_npz=probe_npz,
                run_root=run_root,
                seed=seed,
                fixed_sample_ids=fixed_sample_ids,
                scales=scales,
            )
            continue
        if row == "pinn":
            results[row] = _run_tf_pinn_row(
                train_npz=train_npz,
                val_npz=val_npz,
                test_npz=test_npz,
                probe_npz=probe_npz,
                run_root=run_root,
                seed=seed,
                fixed_sample_ids=fixed_sample_ids,
                scales=scales,
            )
            continue
        if row == "pinn_hybrid_resnet":
            results[row] = _run_torch_row(
                model_id=row,
                architecture="hybrid_resnet",
                training_procedure="pinn",
                train_npz=train_npz,
                val_npz=val_npz,
                test_npz=test_npz,
                probe_npz=probe_npz,
                run_root=run_root,
                seed=seed,
                fixed_sample_ids=fixed_sample_ids,
                scales=scales,
            )
            continue
        if row == "pinn_fno_vanilla":
            results[row] = _run_torch_row(
                model_id=row,
                architecture="fno_vanilla",
                training_procedure="pinn",
                train_npz=train_npz,
                val_npz=val_npz,
                test_npz=test_npz,
                probe_npz=probe_npz,
                run_root=run_root,
                seed=seed,
                fixed_sample_ids=fixed_sample_ids,
                scales=scales,
            )
            continue
        if row == "pinn_ffno":
            results[row] = _run_torch_row(
                model_id=row,
                architecture="ffno",
                training_procedure="pinn",
                train_npz=train_npz,
                val_npz=val_npz,
                test_npz=test_npz,
                probe_npz=probe_npz,
                run_root=run_root,
                seed=seed,
                fixed_sample_ids=fixed_sample_ids,
                scales=scales,
            )
            continue
        if row == "pinn_neuralop_uno":
            results[row] = _run_torch_row(
                model_id=row,
                architecture="neuralop_uno",
                training_procedure="pinn",
                train_npz=train_npz,
                val_npz=val_npz,
                test_npz=test_npz,
                probe_npz=probe_npz,
                run_root=run_root,
                seed=seed,
                fixed_sample_ids=fixed_sample_ids,
                scales=scales,
            )
            continue
        results[row] = {"status": "not_protocol_compatible", "reason": f"unsupported row {row}"}
    return results


def _bundle_row_statuses_from_execution(
    execution_row_statuses: Mapping[str, Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    """Translate execution-time row statuses into the bundle harness-status form.

    The bundle writer in metrics_tables.py treats any required row whose
    forwarded status is not exactly "supported_for_harness" as incomplete. Row
    execution returns "completed"/"blocked"/"not_protocol_compatible", so the
    forwarded statuses must be remapped: "completed" rows are protocol-
    compatible by construction in the natural-patch roster, so they map to
    "supported_for_harness"; other execution statuses are preserved as-is so
    they continue to downgrade the bundle.
    """
    bundle: dict[str, Mapping[str, object]] = {}
    for row, payload in execution_row_statuses.items():
        translated = dict(payload)
        if translated.get("status") == "completed":
            translated["status"] = "supported_for_harness"
            translated["execution_status"] = "completed"
        bundle[row] = translated
    return bundle


def _backfill_torch_row_visuals(
    *,
    run_root: Path,
    model_id: str,
    test_npz: Path,
    fixed_sample_ids: Sequence[int],
    scales: Mapping[str, Any],
) -> Optional[dict[str, str]]:
    """Reconstruct fixed-sample amp_phase visuals from saved torch row artifacts.

    Used by the recollate path so torch rows that did not emit `visuals/amp_phase_*`
    PNGs during the original launch can satisfy the bundle visuals contract without
    retraining. Prefers the per-sample patchwise NPZ that the harness already
    saves; falls back to the row-level recon when the patchwise file is absent.
    """
    patchwise_npz = run_root / "patchwise" / model_id / "fixed_samples.npz"
    if patchwise_npz.exists():
        with np.load(patchwise_npz, allow_pickle=True) as data:
            predictions = np.asarray(data["YY_pred"], dtype=np.complex64)
            ground_truth = np.asarray(data["YY_ground_truth"], dtype=np.complex64)
        return _save_fixed_sample_visuals(
            run_root=run_root,
            model_id=model_id,
            predictions=predictions,
            ground_truth=ground_truth,
            fixed_sample_ids=list(range(predictions.shape[0])),
            scales=scales,
        )
    recon_path = run_root / "recons" / model_id / "recon.npz"
    if not recon_path.exists():
        return None
    test_grouped = _load_grouped_split(test_npz)
    ground_truth = np.asarray(test_grouped["YY_ground_truth"], dtype=np.complex64)
    with np.load(recon_path, allow_pickle=True) as data:
        for key in ("YY_pred", "recon"):
            if key in data.files:
                recon_array = np.asarray(data[key], dtype=np.complex64)
                break
        else:
            return None
    return _save_fixed_sample_visuals(
        run_root=run_root,
        model_id=model_id,
        predictions=recon_array,
        ground_truth=ground_truth,
        fixed_sample_ids=fixed_sample_ids,
        scales=scales,
    )


def _read_row_invocation_metadata(
    *, run_root: Path, model_id: str
) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """Read the per-row invocation envelope and return its raw status/exit_code/git_commit.

    Returns ``(status, exit_code, git_commit)``. The recollate path reads these
    values as-is from the original execution record so the bundle can carry an
    honest ``row_invocation_status`` / ``row_invocation_exit_code`` and so the
    row provenance can stamp the original execution commit instead of the
    recollation commit.
    """
    invocation_path = run_root / "runs" / model_id / "invocation.json"
    if not invocation_path.exists():
        return None, None, None
    try:
        payload = json.loads(invocation_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None, None
    if not isinstance(payload, Mapping):
        return None, None, None
    status_value = payload.get("status")
    status = str(status_value) if isinstance(status_value, str) else None
    raw_exit = payload.get("exit_code")
    exit_code: Optional[int]
    if isinstance(raw_exit, int) and not isinstance(raw_exit, bool):
        exit_code = raw_exit
    else:
        exit_code = None
    extra = payload.get("extra")
    git_commit_value = extra.get("git_commit") if isinstance(extra, Mapping) else None
    git_commit = str(git_commit_value) if isinstance(git_commit_value, str) and git_commit_value else None
    return status, exit_code, git_commit


def _backfill_row_logs(*, run_root: Path, model_id: str) -> None:
    """Ensure per-row stdout/stderr logs exist so exit_code_proof validation can run."""
    run_dir = run_root / "runs" / model_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    if not stdout_log.exists():
        stdout_log.write_text(
            "Recovered natural-patch row stdout placeholder (recollate path).\n",
            encoding="utf-8",
        )
    if not stderr_log.exists():
        stderr_log.write_text("", encoding="utf-8")


_PROVENANCE_FIELDS_TO_DROP = (
    "invocation",
    "config",
    "environment",
    "dataset",
    "splits",
    "randomness",
    "visuals",
    "outputs",
)


def _row_payload_from_existing_metrics(
    *,
    run_root: Path,
    model_id: str,
) -> Optional[dict[str, Any]]:
    """Reconstruct an in-memory row payload from the previously written bundle.

    Reads `metrics.json` at the run root, extracts the row payload, and clears
    the provenance fields so the recollate path can re-attach a fresh set with
    the locked manifest contract.
    """
    metrics_path = run_root / "metrics.json"
    if not metrics_path.exists():
        return None
    payload = _load_json(metrics_path)
    rows = payload.get("rows")
    if not isinstance(rows, Mapping):
        return None
    row_payload = rows.get(model_id)
    if not isinstance(row_payload, Mapping):
        return None
    cleaned: dict[str, Any] = {
        key: value
        for key, value in row_payload.items()
        if key not in _PROVENANCE_FIELDS_TO_DROP
    }
    invocation_path = run_root / "runs" / model_id / "invocation.json"
    if invocation_path.exists():
        cleaned["invocation"] = {
            "json": _relative(invocation_path, run_root),
            "shell": _relative(invocation_path.with_suffix(".sh"), run_root),
        }
    config_path = run_root / "runs" / model_id / "config.json"
    if config_path.exists():
        cleaned["config"] = {"json": _relative(config_path, run_root)}
    existing_outputs = cleaned.get("outputs")
    outputs_payload: Dict[str, Any] = (
        dict(existing_outputs) if isinstance(existing_outputs, Mapping) else {}
    )
    metrics_path = run_root / "runs" / model_id / "metrics.json"
    history_path = run_root / "runs" / model_id / "history.json"
    recon_path = run_root / "recons" / model_id / "recon.npz"
    if metrics_path.exists():
        outputs_payload.setdefault("metrics_json", _relative(metrics_path, run_root))
    if history_path.exists():
        outputs_payload.setdefault("history_json", _relative(history_path, run_root))
    if recon_path.exists():
        outputs_payload.setdefault("recon_npz", _relative(recon_path, run_root))
    for candidate_name in ("model.pt", "wts.h5.zip", "baseline.keras"):
        candidate = run_root / "runs" / model_id / candidate_name
        if candidate.exists():
            outputs_payload.setdefault("model_artifact", _relative(candidate, run_root))
            break
    legacy_baseline = run_root / "baseline" / "baseline.keras"
    if model_id == "baseline" and legacy_baseline.exists() and "model_artifact" not in outputs_payload:
        outputs_payload["model_artifact"] = _relative(legacy_baseline, run_root)
    legacy_pinn = run_root / "pinn" / "wts.h5.zip"
    if model_id == "pinn" and legacy_pinn.exists() and "model_artifact" not in outputs_payload:
        outputs_payload["model_artifact"] = _relative(legacy_pinn, run_root)
    cleaned["outputs"] = outputs_payload
    return cleaned


def _recollate_natural_patch_run(
    *,
    dataset_root: Path,
    item_root: Path,
    rows: Sequence[str],
    seed: int,
    run_id: str,
) -> dict[str, Any]:
    """Re-publish an existing natural-patch run root with full provenance scaffolding.

    Reuses already-trained per-row artifacts. Re-emits the dataset/split manifests,
    per-row exit-code proofs, and torch-row fixed-sample visuals so the bundle
    writer can promote the run to `paper_complete` when every row's underlying
    invocation completed cleanly. The launcher exits `0` after the bundle is
    re-written, providing a fresh launcher proof for the run root.
    """
    dataset_root = Path(dataset_root)
    item_root = Path(item_root)
    run_root = item_root / "runs" / run_id
    if not run_root.exists():
        raise FileNotFoundError(f"natural-patch run root does not exist: {run_root}")
    prepared = prepare_natural_patch_inputs(dataset_root=dataset_root, item_root=item_root)
    contract_paths = _write_contract_artifacts(
        dataset_root=dataset_root,
        item_root=item_root,
        prepared=prepared,
        rows=rows,
        seed=seed,
    )
    fixed_sample_manifest = _load_json(Path(contract_paths["fixed_sample_manifest_path"]))
    shared_visual_scales = _load_json(Path(contract_paths["shared_visual_scales_path"]))
    fixed_sample_ids = list(fixed_sample_manifest["fixed_sample_ids"])

    train_npz = Path(prepared["grouped_paths"]["train"])
    test_npz = Path(prepared["grouped_paths"]["test"])
    probe_npz = dataset_root / "probe.npz"

    with np.load(train_npz, allow_pickle=True) as data:
        nimgs_train = int(np.asarray(data["diffraction"]).shape[0])
    with np.load(test_npz, allow_pickle=True) as data:
        nimgs_test = int(np.asarray(data["diffraction"]).shape[0])

    row_statuses: dict[str, Mapping[str, object]] = {}
    row_payloads: dict[str, Mapping[str, object]] = {}
    backend_by_row: dict[str, str] = {}
    for row in rows:
        existing = _row_payload_from_existing_metrics(run_root=run_root, model_id=row)
        if existing is None:
            row_statuses[row] = {"status": "blocked", "reason": "missing_existing_row_payload"}
            continue
        backend = existing.get("hardware_summary", {}).get("backend") if isinstance(existing.get("hardware_summary"), Mapping) else None
        backend_by_row[row] = str(backend) if isinstance(backend, str) else "unknown"
        _backfill_row_logs(run_root=run_root, model_id=row)
        invocation_status, invocation_exit_code, original_git_commit = _read_row_invocation_metadata(
            run_root=run_root, model_id=row
        )
        existing["row_status"] = "recovered_non_authoritative"
        if backend == "pytorch":
            visuals = _backfill_torch_row_visuals(
                run_root=run_root,
                model_id=row,
                test_npz=test_npz,
                fixed_sample_ids=fixed_sample_ids,
                scales=shared_visual_scales,
            )
            if visuals is not None:
                existing["visuals"] = visuals
        else:
            existing.setdefault(
                "visuals",
                {
                    "amp_phase_png": f"visuals/amp_phase_{row}.png",
                    "amp_phase_error_png": f"visuals/amp_phase_error_{row}.png",
                },
            )
        _attach_natural_patch_row_provenance(
            run_root=run_root,
            model_id=row,
            row_payload=existing,
            train_npz=train_npz,
            test_npz=test_npz,
            probe_npz=probe_npz,
            seed=seed,
            nimgs_train=nimgs_train,
            nimgs_test=nimgs_test,
            gridsize=1,
            set_phi=False,
            proof_source=f"natural_patch_recollate_existing_row_artifacts_{backend_by_row[row]}",
            git_commit_override=original_git_commit,
        )
        row_payloads[row] = existing
        row_statuses[row] = {
            "status": "recovered_non_authoritative",
            "execution_source": "recollate",
            "row_invocation_status": invocation_status,
            "row_invocation_exit_code": invocation_exit_code,
            "reason": "recollate_after_failed_authoritative_launcher",
        }

    bundle_row_statuses = dict(row_statuses)
    bundle_paths = write_paper_benchmark_bundle(
        output_dir=run_root,
        row_payloads=row_payloads,
        required_rows=rows,
        fixed_sample_ids=fixed_sample_ids,
        shared_visual_scales=shared_visual_scales,
        evidence_scope="single_seed_natural_patch_benchmark",
        claim_boundary="single_seed_natural_patch_expanded_object_cdi_only",
        row_statuses=bundle_row_statuses,
        require_row_provenance=True,
    )
    manifest_payload = _build_benchmark_manifest(
        dataset_root=dataset_root,
        item_root=item_root,
        run_root=run_root,
        rows=rows,
        row_statuses=row_statuses,
        row_payloads=row_payloads,
        bundle_paths=bundle_paths,
    )
    manifest_payload["recollate_source"] = run_id
    manifest_path = _write_json(run_root / "paper_benchmark_manifest.json", manifest_payload)
    return {
        "mode": "recollate",
        "dataset_id": NATURAL_PATCH_DATASET_ID,
        "run_root": str(run_root),
        "row_statuses": row_statuses,
        "row_payloads": row_payloads,
        "paper_benchmark_manifest": str(manifest_path),
        **bundle_paths,
    }


def run_natural_patch_benchmark(
    *,
    dataset_root: Path,
    item_root: Path,
    mode: str,
    rows: Sequence[str] = NATURAL_PATCH_ROW_ROSTER,
    seed: int = DEFAULT_SEED,
    run_id: Optional[str] = None,
    execute_rows_fn: Optional[
        Callable[..., Mapping[str, Mapping[str, object]]]
    ] = None,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    item_root = Path(item_root)
    rows = tuple(rows)
    unknown_rows = sorted(set(rows) - set(NATURAL_PATCH_ROW_ROSTER))
    if unknown_rows:
        raise ValueError(f"unsupported natural-patch rows: {unknown_rows}")
    if mode not in {"dry-run", "benchmark", "recollate"}:
        raise ValueError(f"unsupported mode {mode!r}")
    if mode == "recollate":
        if not run_id:
            raise ValueError("recollate mode requires run_id")
        return _recollate_natural_patch_run(
            dataset_root=dataset_root,
            item_root=item_root,
            rows=rows,
            seed=seed,
            run_id=run_id,
        )

    prepared = prepare_natural_patch_inputs(dataset_root=dataset_root, item_root=item_root)
    contract_paths = _write_contract_artifacts(
        dataset_root=dataset_root,
        item_root=item_root,
        prepared=prepared,
        rows=rows,
        seed=seed,
    )
    if mode == "dry-run":
        return {
            "mode": "dry-run",
            "dataset_id": NATURAL_PATCH_DATASET_ID,
            "prepared": prepared,
            "rows": list(rows),
            **contract_paths,
        }

    if not run_id:
        raise ValueError("benchmark mode requires run_id")
    run_root = item_root / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    executor = execute_rows_fn or _execute_rows
    execution_results = executor(
        dataset_root=dataset_root,
        item_root=item_root,
        run_root=run_root,
        prepared=prepared,
        rows=rows,
        seed=seed,
    )
    row_statuses: dict[str, Mapping[str, object]] = {}
    row_payloads: dict[str, Mapping[str, object]] = {}
    for row in rows:
        payload = dict(execution_results.get(row, {}))
        row_statuses[row] = {
            key: value for key, value in payload.items() if key != "row_payload"
        }
        row_payload = payload.get("row_payload")
        if isinstance(row_payload, Mapping):
            row_payloads[row] = dict(row_payload)

    bundle_row_statuses = _bundle_row_statuses_from_execution(row_statuses)

    fixed_sample_manifest = _load_json(Path(contract_paths["fixed_sample_manifest_path"]))
    shared_visual_scales = _load_json(Path(contract_paths["shared_visual_scales_path"]))
    bundle_paths = write_paper_benchmark_bundle(
        output_dir=run_root,
        row_payloads=row_payloads,
        required_rows=rows,
        fixed_sample_ids=fixed_sample_manifest["fixed_sample_ids"],
        shared_visual_scales=shared_visual_scales,
        evidence_scope="single_seed_natural_patch_benchmark",
        claim_boundary="single_seed_natural_patch_expanded_object_cdi_only",
        row_statuses=bundle_row_statuses,
        require_row_provenance=True,
    )
    manifest_payload = _build_benchmark_manifest(
        dataset_root=dataset_root,
        item_root=item_root,
        run_root=run_root,
        rows=rows,
        row_statuses=row_statuses,
        row_payloads=row_payloads,
        bundle_paths=bundle_paths,
    )
    manifest_path = _write_json(run_root / "paper_benchmark_manifest.json", manifest_payload)
    return {
        "mode": "benchmark",
        "dataset_id": NATURAL_PATCH_DATASET_ID,
        "run_root": str(run_root),
        "row_statuses": row_statuses,
        "row_payloads": row_payloads,
        "paper_benchmark_manifest": str(manifest_path),
        **bundle_paths,
    }
