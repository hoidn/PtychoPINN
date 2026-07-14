"""Deterministic visual-review publication for Task 27 reference arms."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .dataset_provenance import canonical_array_sha256
from .runtime_atomic import atomic_rename_directory_no_replace
from .runtime_errors import RuntimeExecutionError

VISUAL_SCHEMA_VERSION = "grid_lines_reference_visual_manifest_v1"
PANELS = (
    "amplitude_truth",
    "amplitude_reconstruction",
    "amplitude_absolute_error",
    "phase_truth",
    "phase_reconstruction",
    "phase_absolute_error",
)


def _write_visual_bundle_file(path: Path, data: bytes) -> None:
    """Write and fsync one file inside a private visual-bundle staging dir."""
    with Path(path).open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _before_visual_bundle_publish(_destination: Path) -> None:
    """Test hook for injecting a competing publication at the race boundary."""


def _panel_arrays(
    truth: np.ndarray, reconstruction: np.ndarray
) -> tuple[np.ndarray, ...]:
    amp_truth = np.abs(truth)
    amp_reconstruction = np.abs(reconstruction)
    phase_truth = np.angle(truth)
    phase_reconstruction = np.angle(reconstruction)
    phase_error = np.abs(np.angle(np.exp(1j * (phase_reconstruction - phase_truth))))
    return (
        amp_truth,
        amp_reconstruction,
        np.abs(amp_reconstruction - amp_truth),
        phase_truth,
        phase_reconstruction,
        phase_error,
    )


def _positive_limit(*arrays: np.ndarray) -> float:
    finite = np.concatenate(
        [np.asarray(array)[np.isfinite(array)].ravel() for array in arrays]
    )
    maximum = float(np.max(finite))
    return maximum if maximum > 0.0 else 1.0


def publish_reference_visual(
    result: Any,
    *,
    architecture: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Publish the exact gated canvas and truth as a legible six-panel PNG."""
    truth = np.asarray(result.ground_truth)
    reconstruction = np.asarray(result.historical_canvas)
    if truth.shape != reconstruction.shape or truth.ndim != 2:
        raise RuntimeExecutionError(
            "visual_review",
            f"truth/reconstruction must be same-shape 2D arrays, got "
            f"{truth.shape}/{reconstruction.shape}",
        )
    panels = _panel_arrays(truth, reconstruction)
    if any(array.size == 0 or not np.isfinite(array).any() for array in panels):
        raise RuntimeExecutionError(
            "visual_review", "visual review panels must contain finite pixels"
        )
    historical_canvas_sha256 = canonical_array_sha256(reconstruction)
    if historical_canvas_sha256 != result.historical_canvas_sha256:
        raise RuntimeExecutionError(
            "visual_review",
            "rendered reconstruction historical_canvas_sha256 does not match "
            "the gated ReferenceRunResult",
        )
    ground_truth_sha256 = canonical_array_sha256(truth)
    if ground_truth_sha256 != result.ground_truth_sha256:
        raise RuntimeExecutionError(
            "visual_review",
            "rendered ground_truth_sha256 does not match the gated "
            "ReferenceRunResult",
        )
    amplitude_limit = _positive_limit(panels[0], panels[1])
    amplitude_error_limit = _positive_limit(panels[2])
    phase_error_limit = _positive_limit(panels[5])
    panel_limits = {
        "amplitude_truth": [0.0, amplitude_limit],
        "amplitude_reconstruction": [0.0, amplitude_limit],
        "amplitude_absolute_error": [0.0, amplitude_error_limit],
        "phase_truth": [-float(np.pi), float(np.pi)],
        "phase_reconstruction": [-float(np.pi), float(np.pi)],
        "phase_absolute_error": [0.0, phase_error_limit],
    }

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    titles = (
        "Amplitude truth",
        "Amplitude reconstruction",
        "Amplitude absolute error",
        "Phase truth",
        "Phase reconstruction",
        "Phase absolute error",
    )
    cmaps = ("gray", "gray", "magma", "twilight", "twilight", "magma")
    for axis, array, title, cmap, panel_id in zip(
        axes.flat, panels, titles, cmaps, PANELS, strict=True
    ):
        vmin, vmax = panel_limits[panel_id]
        image = axis.imshow(
            array,
            cmap=cmap,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    fig.suptitle(
        f"{result.arm_id} | {architecture} | gain={result.command['amplitude_physics_gain']:.6g}\n"
        f"amp SSIM={result.fixture_amp_ssim:.6f}, phase SSIM={result.fixture_phase_ssim:.6f}; "
        f"amp MAE={result.fixture_amp_mae:.6f}, phase MAE={result.fixture_phase_mae:.6f}",
        fontsize=11,
    )
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=120,
        metadata={"Software": "PtychoPINN Task27 reference visual"},
    )
    plt.close(fig)
    visual_bytes = buffer.getvalue()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = output_dir / "visual_bundle"
    if bundle_dir.exists():
        raise RuntimeExecutionError(
            "visual_bundle",
            f"refusing to overwrite existing visual bundle at {bundle_dir}",
        )
    visual_path = bundle_dir / "visual_review.png"
    manifest = {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "arm_id": result.arm_id,
        "architecture": architecture,
        "visual_path": str(visual_path),
        "visual_sha256": hashlib.sha256(visual_bytes).hexdigest(),
        "historical_canvas_sha256": historical_canvas_sha256,
        "ground_truth_sha256": ground_truth_sha256,
        "historical_canvas_array": {
            "shape": list(reconstruction.shape),
            "dtype": str(reconstruction.dtype),
        },
        "ground_truth_array": {
            "shape": list(truth.shape),
            "dtype": str(truth.dtype),
        },
        "checkpoint_sha256": result.checkpoint_sha256,
        "train_npz_sha256": result.materialized.train_sha256,
        "test_npz_sha256": result.materialized.test_sha256,
        "amplitude_physics_gain": result.command["amplitude_physics_gain"],
        "evaluator": result.gauge_handling,
        "metrics": {
            "fixture_amp_mae": result.fixture_amp_mae,
            "fixture_phase_mae": result.fixture_phase_mae,
            "fixture_amp_ssim": result.fixture_amp_ssim,
            "fixture_phase_ssim": result.fixture_phase_ssim,
        },
        "panels": list(PANELS),
        "panel_limits": panel_limits,
        "normalization_policy": "raw_canvas_shared_truth_reconstruction_limits_v1",
        "no_resize_asserted": result.no_resize_asserted,
        "gauge_normalization_applied_for_visual": False,
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    staging = Path(tempfile.mkdtemp(prefix=".visual_bundle.", dir=output_dir))
    try:
        _write_visual_bundle_file(staging / "visual_review.png", visual_bytes)
        _write_visual_bundle_file(staging / "visual_manifest.json", encoded)
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _before_visual_bundle_publish(bundle_dir)
        atomic_rename_directory_no_replace(
            staging, bundle_dir, stage="visual_bundle"
        )
        parent_fd = os.open(output_dir, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except RuntimeExecutionError:
        raise
    except OSError as error:
        raise RuntimeExecutionError(
            "visual_bundle", f"visual bundle publication interrupted: {error}"
        ) from error
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest
