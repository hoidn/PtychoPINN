"""Natural-patch expanded-object CDI benchmark harness."""

from __future__ import annotations

import json
import hashlib
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
            per_metric.setdefault(metric_name, []).append((float(pair[0]), float(pair[1])))
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
        pred = predictions[sample_id]
        gt = ground_truth[sample_id]
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
    probe_manifest = _load_json(dataset_root / "probe_manifest.json")
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


def _run_torch_row(
    *,
    model_id: str,
    architecture: str,
    training_procedure: str,
    train_npz: Path,
    val_npz: Path,
    test_npz: Path,
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
    row_payload["dataset"]["validation_npz"] = str(val_npz)
    row_payload["dataset"]["test_npz"] = str(test_npz)
    row_payload["visuals"] = _save_fixed_sample_visuals(
        run_root=run_root,
        model_id="pinn",
        predictions=pred_complex,
        ground_truth=gt,
        fixed_sample_ids=fixed_sample_ids,
        scales=scales,
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
    row_payload["dataset"]["validation_npz"] = str(val_npz)
    row_payload["dataset"]["test_npz"] = str(test_npz)
    row_payload["visuals"] = _save_fixed_sample_visuals(
        run_root=run_root,
        model_id="baseline",
        predictions=pred_complex,
        ground_truth=gt,
        fixed_sample_ids=fixed_sample_ids,
        scales=scales,
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
                run_root=run_root,
                seed=seed,
                fixed_sample_ids=fixed_sample_ids,
                scales=scales,
            )
            continue
        results[row] = {"status": "not_protocol_compatible", "reason": f"unsupported row {row}"}
    return results


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
    if mode not in {"dry-run", "benchmark"}:
        raise ValueError(f"unsupported mode {mode!r}")

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
