from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import logging
import sys
import time

import numpy as np

INTEGRATION_PROBE_PATH = Path("ptycho/datasets/Run1084_recon3_postPC_shrunk_3.npz")


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    gridsize: int
    probe_mode: str  # "idealized" or "integration"


@dataclass(frozen=True)
class RunParams:
    N: int = 64
    object_size: int = 392
    split_fraction: float = 0.5
    base_total_images: int = 2000
    group_count: int = 1000
    nphotons: float = 1e9
    neighbor_count: int = 4
    reassemble_M: int = 20


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


def generate_lines_object(object_size: int) -> np.ndarray:
    from ptycho import params
    from ptycho.diffsim import sim_object_image

    previous_source = params.get("data_source")
    params.set("data_source", "lines")
    try:
        obj = sim_object_image(size=object_size, which="train")
    finally:
        params.set("data_source", previous_source)

    if obj.ndim == 3:
        obj = obj[..., 0]
    return obj.astype(np.complex64)


def load_integration_probe(path: Path, N: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Integration probe not found: {path}")
    with np.load(path) as data:
        if "probeGuess" not in data:
            raise KeyError(f"probeGuess missing in {path}")
        probe = data["probeGuess"]
    if probe.shape != (N, N):
        raise ValueError(f"Expected probe shape {(N, N)}, got {probe.shape}")
    return probe.astype(np.complex64)


def generate_ideal_probe(N: int) -> np.ndarray:
    from ptycho import params
    from ptycho import probe

    if params.cfg.get("default_probe_scale") is None:
        params.cfg["default_probe_scale"] = 0.7
    return probe.get_default_probe(N=N, fmt="np").astype(np.complex64)


def simulate_raw_data(
    params: RunParams,
    object_guess: np.ndarray,
    probe_guess: np.ndarray,
    total_images: int,
    buffer: Optional[float] = None,
):
    from ptycho import params as legacy_params
    from ptycho.config.config import ModelConfig, TrainingConfig, update_legacy_dict
    from ptycho.nongrid_simulation import generate_simulated_data

    sim_config = TrainingConfig(
        model=ModelConfig(N=params.N, gridsize=1),
        n_groups=total_images,
        nphotons=params.nphotons,
    )
    update_legacy_dict(legacy_params.cfg, sim_config)
    if buffer is None:
        buffer = float(min(object_guess.shape)) * 0.35
    return generate_simulated_data(
        config=sim_config,
        objectGuess=object_guess,
        probeGuess=probe_guess,
        buffer=buffer,
        return_patches=False,
    )


def _slice_raw_data(raw_data, indices: np.ndarray):
    from ptycho.raw_data import RawData

    scan_index = None
    if raw_data.scan_index is not None:
        scan_index = raw_data.scan_index[indices]
    Y = None
    if raw_data.Y is not None:
        Y = raw_data.Y[indices]
    return RawData(
        xcoords=raw_data.xcoords[indices],
        ycoords=raw_data.ycoords[indices],
        xcoords_start=raw_data.xcoords_start[indices],
        ycoords_start=raw_data.ycoords_start[indices],
        diff3d=raw_data.diff3d[indices],
        probeGuess=raw_data.probeGuess,
        scan_index=scan_index,
        objectGuess=raw_data.objectGuess,
        Y=Y,
        norm_Y_I=raw_data.norm_Y_I,
        metadata=raw_data.metadata,
    )


def split_raw_data_by_y(raw_data, test_count: int) -> Tuple:
    if raw_data.diff3d is None:
        raise ValueError("raw_data.diff3d is required for splitting")
    n_images = raw_data.diff3d.shape[0]
    if test_count <= 0 or test_count >= n_images:
        raise ValueError(f"test_count must be in (0, {n_images})")
    sort_idx = np.argsort(raw_data.ycoords)
    train_idx = sort_idx[: n_images - test_count]
    test_idx = sort_idx[n_images - test_count :]
    return _slice_raw_data(raw_data, train_idx), _slice_raw_data(raw_data, test_idx)


def build_training_config(
    params: RunParams,
    gridsize: int,
    group_count: int,
    output_dir: Path,
    nepochs: int,
):
    from ptycho.config.config import ModelConfig, TrainingConfig

    model_config = ModelConfig(N=params.N, gridsize=gridsize, model_type="pinn")
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
) -> Tuple[np.ndarray, np.ndarray]:
    from ptycho.config.config import InferenceConfig, ModelConfig
    from ptycho.workflows.backend_selector import load_inference_bundle_with_backend
    from ptycho import loader
    from ptycho import nbutils
    from ptycho import tf_helper

    infer_config = InferenceConfig(
        model=ModelConfig(N=params.N, gridsize=gridsize),
        model_path=model_dir,
        test_data_file=INTEGRATION_PROBE_PATH,
        n_groups=group_count,
        neighbor_count=params.neighbor_count,
        backend="tensorflow",
    )
    model, params_dict = load_inference_bundle_with_backend(model_dir, infer_config)
    grouped = test_data.generate_grouped_data(
        params_dict.get("N", params.N),
        K=params.neighbor_count,
        nsamples=group_count,
        gridsize=params_dict.get("gridsize", gridsize),
    )
    container = loader.load(lambda: grouped, test_data.probeGuess, which=None, create_split=False)
    obj_tensor_full, global_offsets = nbutils.reconstruct_image(container, diffraction_to_obj=model)
    obj_image = tf_helper.reassemble_position(obj_tensor_full, global_offsets, M=params.reassemble_M)
    return np.abs(obj_image), np.angle(obj_image)


def save_reconstruction(output_dir: Path, amplitude: np.ndarray, phase: np.ndarray) -> None:
    from ptycho.workflows.components import save_outputs

    save_outputs(amplitude, phase, {}, str(output_dir))


def write_run_metadata(path: Path, metadata: Dict[str, object]) -> None:
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def run_scenario(
    scenario: ScenarioSpec,
    output_root: Path,
    nepochs: int,
    buffer: Optional[float] = None,
    image_multiplier: int = 1,
    group_multiplier: int = 1,
) -> None:
    params = RunParams()
    total_images, train_count, test_count = derive_counts(
        params,
        scenario.gridsize,
        image_multiplier=image_multiplier,
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
        "Run parameters: N=%s object_size=%s split_fraction=%s total_images=%s train_images=%s test_images=%s group_count=%s",
        params.N,
        params.object_size,
        params.split_fraction,
        total_images,
        train_count,
        test_count,
        group_count,
    )

    object_guess = generate_lines_object(params.object_size)
    if scenario.probe_mode == "integration":
        probe_guess = load_integration_probe(INTEGRATION_PROBE_PATH, params.N)
    elif scenario.probe_mode == "idealized":
        probe_guess = generate_ideal_probe(params.N)
    else:
        raise ValueError(f"Unknown probe_mode: {scenario.probe_mode}")

    raw_data = simulate_raw_data(
        params,
        object_guess,
        probe_guess,
        total_images=total_images,
        buffer=buffer,
    )
    train_raw, test_raw = split_raw_data_by_y(raw_data, test_count)

    train_config = build_training_config(
        params=params,
        gridsize=scenario.gridsize,
        group_count=group_count,
        output_dir=train_dir,
        nepochs=nepochs,
    )
    logger.info(
        "Training config: N=%s gridsize=%s n_groups=%s nphotons=%s nepochs=%s",
        train_config.model.N,
        train_config.model.gridsize,
        train_config.n_groups,
        train_config.nphotons,
        train_config.nepochs,
    )

    start_time = time.time()
    run_training(train_raw, test_raw, train_config)
    save_training_bundle(train_dir)

    amp, phase = run_inference(
        test_raw,
        model_dir=train_dir,
        gridsize=scenario.gridsize,
        params=params,
        group_count=group_count,
    )
    save_reconstruction(inference_dir, amp, phase)

    metadata = {
        "scenario": scenario.name,
        "probe_mode": scenario.probe_mode,
        "gridsize": scenario.gridsize,
        "N": params.N,
        "object_size": params.object_size,
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
    }
    write_run_metadata(scenario_dir / "run_metadata.json", metadata)
    logger.info("Completed scenario: %s", scenario.name)
