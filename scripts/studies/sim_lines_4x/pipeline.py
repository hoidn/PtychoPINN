from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from ptycho import params as legacy_params
from ptycho.config.config import update_legacy_dict
from scripts.simulation.synthetic_helpers import (
    make_lines_object,
    make_probe,
    normalize_probe_guess,
    simulate_nongrid_raw_data,
    split_raw_data_by_axis,
)

CUSTOM_PROBE_PATH = Path("ptycho/datasets/Run1084_recon3_postPC_shrunk_3.npz")
PREDICTION_SCALE_CHOICES = ("none", "recorded", "least_squares")


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    gridsize: int
    probe_mode: str  # "idealized" or "custom"
    probe_scale: float
    probe_big: Optional[bool] = None
    probe_mask: Optional[bool] = None


@dataclass(frozen=True)
class RunParams:
    N: int = 64
    object_size: int = 392
    object_seed: int = 42
    sim_seed: int = 42
    buffer: float = 10.0
    split_fraction: float = 0.5
    base_total_images: int = 2000
    group_count: int = 1000
    nphotons: float = 1e9
    neighbor_count: int = 4
    reassemble_M: int = 20


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def compute_least_squares_scalar(
    prediction: np.ndarray, truth: np.ndarray
) -> Tuple[Optional[float], Dict[str, Any]]:
    pred_vals = np.asarray(prediction, dtype=float).ravel()
    truth_vals = np.asarray(truth, dtype=float).ravel()
    mask = np.isfinite(pred_vals) & np.isfinite(truth_vals)
    payload = {
        "finite_count": int(mask.sum()),
        "numerator": None,
        "denominator": None,
    }
    if not mask.any():
        return None, payload
    pred_sel = pred_vals[mask]
    truth_sel = truth_vals[mask]
    numerator = float(np.sum(pred_sel * truth_sel))
    denominator = float(np.sum(pred_sel * pred_sel))
    payload.update({"numerator": numerator, "denominator": denominator})
    if abs(denominator) <= 1e-12:
        return None, payload
    return numerator / denominator, payload


def determine_prediction_scale(
    mode: str,
    recorded_scale: Optional[float],
    prediction: np.ndarray,
    truth: Optional[np.ndarray],
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "mode": mode,
        "value": None,
        "applied": False,
        "source": None,
        "recorded_scale": recorded_scale,
    }
    if mode == "none":
        return info
    if mode == "recorded":
        if recorded_scale is None:
            info["reason"] = "recorded_scale_unavailable"
            return info
        info.update({"value": recorded_scale, "applied": True, "source": "bundle"})
        return info
    if mode == "least_squares":
        if truth is None:
            info["reason"] = "least_squares_requires_ground_truth"
            return info
        scalar, payload = compute_least_squares_scalar(prediction, truth)
        info["least_squares"] = payload
        if scalar is None:
            info["reason"] = "least_squares_undefined"
            return info
        info.update({"value": scalar, "applied": True, "source": "least_squares"})
        return info
    info["reason"] = f"unsupported_mode_{mode}"
    return info


def format_prediction_scale_note(scale_info: Optional[Mapping[str, Any]]) -> str:
    if not scale_info:
        return ""
    if not scale_info.get("applied"):
        return ""
    value = _as_float(scale_info.get("value"))
    mode = scale_info.get("mode") or "unknown"
    if value is None:
        return str(mode)
    return f"{mode}={value:.4g}"


def center_crop_square(array: np.ndarray, size: int) -> np.ndarray:
    data = np.asarray(array)
    if size <= 0:
        raise ValueError("center crop size must be positive")
    height, width = data.shape[:2]
    if size > height or size > width:
        raise ValueError(
            f"center crop size {size} exceeds array dimensions {(height, width)}"
        )
    start_y = (height - size) // 2
    start_x = (width - size) // 2
    end_y = start_y + size
    end_x = start_x + size
    slices = (slice(start_y, end_y), slice(start_x, end_x))
    if data.ndim > 2:
        slices = slices + (slice(None),) * (data.ndim - 2)
    return data[slices]


def derive_counts(
    params: RunParams,
    gridsize: int,
    image_multiplier: int = 1,
) -> Tuple[int, int, int]:
    total_images = params.base_total_images * (gridsize**2) * image_multiplier
    test_count = int(round(total_images * params.split_fraction))
    train_count = total_images - test_count
    return total_images, train_count, test_count


def configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sim_lines_4x")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def build_training_config(
    params: RunParams,
    gridsize: int,
    group_count: int,
    output_dir: Path,
    nepochs: int,
    probe_scale: float,
    probe_big: bool,
    probe_mask: bool,
):
    from ptycho.config.config import ModelConfig, TrainingConfig

    model_config = ModelConfig(
        N=params.N,
        gridsize=gridsize,
        model_type="pinn",
        probe_scale=probe_scale,
        probe_big=probe_big,
        probe_mask=probe_mask,
    )
    return TrainingConfig(
        model=model_config,
        n_groups=group_count,
        nphotons=params.nphotons,
        neighbor_count=params.neighbor_count,
        nepochs=nepochs,
        output_dir=output_dir,
    )


def run_training(train_data, test_data, config) -> Dict[str, object]:
    from ptycho.workflows.backend_selector import train_cdi_model_with_backend

    results = train_cdi_model_with_backend(train_data, test_data, config)
    return results


def save_training_bundle(output_dir: Path) -> None:
    from ptycho import model_manager

    output_dir.mkdir(parents=True, exist_ok=True)
    model_manager.save(str(output_dir))


def run_inference(
    test_data,
    model_dir: Path,
    gridsize: int,
    params: RunParams,
    group_count: int,
    probe_scale: float,
    probe_big: bool,
    probe_mask: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Optional[float]]]:
    from ptycho.config.config import InferenceConfig, ModelConfig
    from ptycho.workflows.backend_selector import load_inference_bundle_with_backend
    from ptycho import loader
    from ptycho import nbutils
    from ptycho import tf_helper

    infer_config = InferenceConfig(
        model=ModelConfig(
            N=params.N,
            gridsize=gridsize,
            probe_scale=probe_scale,
            probe_big=probe_big,
            probe_mask=probe_mask,
        ),
        model_path=model_dir,
        test_data_file=CUSTOM_PROBE_PATH,
        n_groups=group_count,
        neighbor_count=params.neighbor_count,
        backend="tensorflow",
    )
    # CONFIG-001: Sync legacy params.cfg before loader/grouped-data (spec-inference-pipeline.md §1.1)
    update_legacy_dict(legacy_params.cfg, infer_config)
    model, params_dict = load_inference_bundle_with_backend(model_dir, infer_config)
    grouped = test_data.generate_grouped_data(
        params_dict.get("N", params.N),
        K=params.neighbor_count,
        nsamples=group_count,
        gridsize=params_dict.get("gridsize", gridsize),
    )
    container = loader.load(lambda: grouped, test_data.probeGuess, which=None, create_split=False)
    obj_tensor_full, global_offsets = nbutils.reconstruct_image(
        container, diffraction_to_obj=model
    )
    obj_image = tf_helper.reassemble_position(
        obj_tensor_full, global_offsets, M=params.reassemble_M
    )
    amplitude = np.abs(obj_image)
    parameter_scale = params_dict.get("intensity_scale")
    legacy_scale = None
    metadata = container.metadata if hasattr(container, "metadata") else {}
    if isinstance(metadata, dict):
        legacy_scale = metadata.get("intensity_scale")
    scale_meta: Dict[str, Optional[float]] = {
        "bundle": _as_float(parameter_scale),
        "legacy": _as_float(legacy_scale),
    }
    return amplitude, np.angle(obj_image), scale_meta


def save_reconstruction(output_dir: Path, amplitude: np.ndarray, phase: np.ndarray) -> None:
    from ptycho.workflows.components import save_outputs

    save_outputs(amplitude, phase, {}, str(output_dir), crop_mode="square")


def write_run_metadata(path: Path, metadata: Dict[str, object]) -> None:
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def run_scenario(
    scenario: ScenarioSpec,
    output_root: Path,
    nepochs: int,
    buffer: Optional[float] = None,
    image_multiplier: int = 1,
    group_multiplier: int = 1,
    object_seed: Optional[int] = None,
    sim_seed: Optional[int] = None,
    prediction_scale_source: str = "none",
) -> None:
    params = RunParams()
    if object_seed is not None or sim_seed is not None:
        params = replace(
            params,
            object_seed=params.object_seed if object_seed is None else object_seed,
            sim_seed=params.sim_seed if sim_seed is None else sim_seed,
        )
    if scenario.probe_mode == "idealized":
        probe_big = False if scenario.probe_big is None else scenario.probe_big
        probe_mask = True if scenario.probe_mask is None else scenario.probe_mask
    else:
        from ptycho.config.config import ModelConfig

        defaults = ModelConfig()
        probe_big = defaults.probe_big if scenario.probe_big is None else scenario.probe_big
        probe_mask = defaults.probe_mask if scenario.probe_mask is None else scenario.probe_mask
    total_images, train_count, test_count = derive_counts(
        params,
        scenario.gridsize,
        image_multiplier=image_multiplier,
    )
    if prediction_scale_source not in PREDICTION_SCALE_CHOICES:
        raise ValueError(
            f"prediction_scale_source must be one of {PREDICTION_SCALE_CHOICES}"
        )
    group_count = params.group_count * group_multiplier
    scenario_dir = output_root / scenario.name
    train_dir = scenario_dir / "train_outputs"
    inference_dir = scenario_dir / "inference_outputs"

    logger = configure_logging(scenario_dir / "run.log")
    logger.info("Starting scenario: %s", scenario.name)

    if total_images - test_count <= 0:
        raise ValueError("total_images must be greater than test_count")
    expected_test = int(round(total_images * params.split_fraction))
    if expected_test != test_count:
        logger.warning(
            "Split fraction mismatch: expected test_count=%s from split_fraction=%s",
            expected_test,
            params.split_fraction,
        )
    if train_count != test_count:
        raise ValueError("train/test splits must be equal sized")
    if group_count > train_count or group_count > test_count:
        raise ValueError("group_count must be <= train/test image counts")
    logger.info(
        "Run parameters: N=%s object_size=%s object_seed=%s sim_seed=%s split_fraction=%s total_images=%s train_images=%s test_images=%s group_count=%s probe_scale=%s",
        params.N,
        params.object_size,
        params.object_seed,
        params.sim_seed,
        params.split_fraction,
        total_images,
        train_count,
        test_count,
        group_count,
        scenario.probe_scale,
    )

    object_guess = make_lines_object(params.object_size, seed=params.object_seed)
    if scenario.probe_mode == "custom":
        probe_guess = make_probe(params.N, mode="custom", path=CUSTOM_PROBE_PATH)
    elif scenario.probe_mode == "idealized":
        probe_guess = make_probe(params.N, mode="idealized")
    else:
        raise ValueError(f"Unknown probe_mode: {scenario.probe_mode}")
    probe_guess = normalize_probe_guess(
        probe_guess,
        probe_scale=scenario.probe_scale,
        N=params.N,
    )

    if buffer is None:
        buffer = params.buffer

    raw_data = simulate_nongrid_raw_data(
        object_guess,
        probe_guess,
        N=params.N,
        n_images=total_images,
        nphotons=params.nphotons,
        seed=params.sim_seed,
        buffer=buffer,
    )
    train_raw, test_raw = split_raw_data_by_axis(
        raw_data,
        split_fraction=params.split_fraction,
        axis="y",
    )

    train_config = build_training_config(
        params=params,
        gridsize=scenario.gridsize,
        group_count=group_count,
        output_dir=train_dir,
        nepochs=nepochs,
        probe_scale=scenario.probe_scale,
        probe_big=probe_big,
        probe_mask=probe_mask,
    )
    logger.info(
        "Training config: N=%s gridsize=%s n_groups=%s nphotons=%s nepochs=%s probe_scale=%s",
        train_config.model.N,
        train_config.model.gridsize,
        train_config.n_groups,
        train_config.nphotons,
        train_config.nepochs,
        train_config.model.probe_scale,
    )

    # CONFIG-001: Sync legacy params.cfg before training/loader (spec-inference-pipeline.md §1.1)
    update_legacy_dict(legacy_params.cfg, train_config)

    start_time = time.time()
    run_training(train_raw, test_raw, train_config)
    save_training_bundle(train_dir)

    amp, phase, scale_meta = run_inference(
        test_raw,
        model_dir=train_dir,
        gridsize=scenario.gridsize,
        params=params,
        group_count=group_count,
        probe_scale=scenario.probe_scale,
        probe_big=probe_big,
        probe_mask=probe_mask,
    )
    amplitude_unscaled = np.array(amp, copy=True)
    amplitude_unscaled_path = inference_dir / "amplitude_unscaled.npy"
    np.save(amplitude_unscaled_path, amplitude_unscaled.astype(np.float32))

    amp_truth_full = np.abs(object_guess).astype(np.float32, copy=False)
    pair_size = min(amplitude_unscaled.shape[0], amp_truth_full.shape[0])
    amp_for_scale = center_crop_square(amplitude_unscaled, pair_size)
    truth_for_scale = center_crop_square(amp_truth_full, pair_size)
    recorded_scale = scale_meta.get("bundle")
    if recorded_scale is None:
        recorded_scale = scale_meta.get("legacy")
    scale_info = determine_prediction_scale(
        prediction_scale_source,
        recorded_scale,
        amp_for_scale,
        truth_for_scale,
    )
    amp_scaled = amplitude_unscaled
    scale_note = ""
    if scale_info.get("applied") and _as_float(scale_info.get("value")) is not None:
        amp_scaled = amplitude_unscaled * float(scale_info["value"])
        scale_note = format_prediction_scale_note(scale_info)
    save_reconstruction(inference_dir, amp_scaled, phase)
    (inference_dir / "prediction_scale.txt").write_text(
        f"Mode: {scale_info.get('mode')}\n"
        f"Applied: {scale_info.get('applied')}\n"
        f"Value: {scale_info.get('value')}\n"
        f"Recorded scale: {scale_info.get('recorded_scale')}\n"
    )

    metadata = {
        "scenario": scenario.name,
        "probe_mode": scenario.probe_mode,
        "probe_scale": scenario.probe_scale,
        "probe_big": probe_big,
        "probe_mask": probe_mask,
        "gridsize": scenario.gridsize,
        "N": params.N,
        "object_size": params.object_size,
        "object_seed": params.object_seed,
        "sim_seed": params.sim_seed,
        "buffer": buffer,
        "split_fraction": params.split_fraction,
        "base_total_images": params.base_total_images,
        "total_images": total_images,
        "train_count": train_raw.diff3d.shape[0],
        "test_count": test_raw.diff3d.shape[0],
        "group_count": group_count,
        "image_multiplier": image_multiplier,
        "group_multiplier": group_multiplier,
        "nphotons": params.nphotons,
        "neighbor_count": params.neighbor_count,
        "nepochs": nepochs,
        "elapsed_seconds": round(time.time() - start_time, 2),
        "prediction_scale": scale_info,
        "amplitude_unscaled_path": str(amplitude_unscaled_path),
        "prediction_scale_source": prediction_scale_source,
    }
    if scale_note:
        metadata["prediction_scale_note"] = scale_note
    if scale_note:
        metadata["prediction_scale_note"] = scale_note
    write_run_metadata(scenario_dir / "run_metadata.json", metadata)
    logger.info("Completed scenario: %s", scenario.name)
