#!/usr/bin/env python3
"""Render a deterministic visual audit of legacy and boundary-matched probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from skimage.restoration import unwrap_phase

from ptycho.simulation.probe_transform import (
    BOUNDARY_METHOD,
    BOUNDARY_SOLVER,
    apply_probe_transform_pipeline,
    apply_probe_transform_pipeline_with_metadata,
    parse_probe_transform_pipeline,
)


PANEL_LABELS = [
    "prepared amplitude",
    "prepared wrapped phase",
    "legacy global amplitude",
    "legacy global wrapped phase",
    "boundary-matched amplitude",
    "boundary-matched wrapped phase",
    "center absolute difference",
    "across-seam wrapped phase step",
    "outer phase residual",
    "horizontal unwrapped phase profile",
    "vertical unwrapped phase profile",
    "identity and solver annotation",
]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(value)).tobytes(order="C")
    ).hexdigest()


def _load_probe(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        probe = np.load(path, allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as archive:
            if "probeGuess" not in archive:
                raise KeyError(f"probeGuess missing from {path}")
            probe = archive["probeGuess"]
    probe = np.asarray(probe, dtype=np.complex64).squeeze()
    if probe.ndim != 2 or probe.shape[0] != probe.shape[1]:
        raise ValueError(f"source probe must be square 2D, got {probe.shape}")
    return probe


def _wrap_phase(value: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * value))


def _mark_footprint(ax, rows: tuple[int, int], columns: tuple[int, int]) -> None:
    y0, y1 = rows
    x0, x1 = columns
    ax.add_patch(
        Rectangle(
            (x0 - 0.5, y0 - 0.5),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="white",
            linewidth=1.5,
            linestyle="--",
        )
    )


def _mark_outer_boundary(ax, shape: tuple[int, int]) -> None:
    height, width = shape
    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            width,
            height,
            fill=False,
            edgecolor="cyan",
            linewidth=1.5,
            linestyle="-",
        )
    )


def _image_panel(
    fig,
    ax,
    image: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    footprint: tuple[tuple[int, int], tuple[int, int]] | None = None,
    outer_boundary: bool = False,
) -> None:
    rendered = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(title)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    if footprint is not None:
        _mark_footprint(ax, footprint[0], footprint[1])
    if outer_boundary:
        _mark_outer_boundary(ax, image.shape)
    fig.colorbar(rendered, ax=ax, fraction=0.046, pad=0.04)


def render_probe_extension_check(
    *,
    source_probe: str | Path,
    target_size: int,
    smoothing: float,
    output: str | Path,
) -> dict[str, object]:
    """Render the visual contract and return the JSON-native sidecar payload."""
    source_path = Path(source_probe)
    output_path = Path(output)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not isinstance(target_size, int) or target_size <= 0:
        raise ValueError("target_size must be a positive integer")
    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("smoothing must be finite and nonnegative")

    source = _load_probe(source_path)
    if target_size <= source.shape[0]:
        raise ValueError("target_size must exceed the source probe size")
    prepare_steps = (
        parse_probe_transform_pipeline(f"smooth:{smoothing:g}")
        if smoothing > 0
        else []
    )
    prepared = apply_probe_transform_pipeline(source, prepare_steps)
    legacy = apply_probe_transform_pipeline(
        prepared,
        parse_probe_transform_pipeline(f"pad_extrapolate:{target_size}"),
    )
    canonical_pipeline = (
        (f"smooth:{smoothing:g}|" if smoothing > 0 else "")
        + f"pad_extrapolate_boundary_matched:{target_size}"
    )
    boundary_transform = apply_probe_transform_pipeline_with_metadata(
        source,
        parse_probe_transform_pipeline(canonical_pipeline),
    )
    boundary = boundary_transform.probe
    numerical = boundary_transform.boundary_result
    if numerical is None:
        raise RuntimeError("boundary-matched pipeline emitted no solver result")

    rows = numerical.source_rows
    columns = numerical.source_columns
    center = boundary[rows[0] : rows[1], columns[0] : columns[1]]
    center_difference = np.abs(center - prepared)
    center_difference_max = float(np.max(center_difference, initial=0.0))

    prepared_unwrapped = np.asarray(
        unwrap_phase(np.angle(prepared)), dtype=np.float64
    )
    embedded_prepared_phase = np.zeros((target_size, target_size), dtype=np.float64)
    embedded_prepared_phase[rows[0] : rows[1], columns[0] : columns[1]] = (
        prepared_unwrapped
    )
    seam_signed = np.zeros((target_size, target_size), dtype=np.float64)
    seam_signed[numerical.inner_boundary_mask] = _wrap_phase(
        numerical.unwrapped_phase[numerical.inner_boundary_mask]
        - embedded_prepared_phase[numerical.inner_boundary_mask]
    )
    seam_error = np.abs(seam_signed)
    seam_error_max = float(
        np.max(seam_error[numerical.inner_boundary_mask], initial=0.0)
    )
    adjacent_step = np.full(
        (target_size, target_size), np.nan, dtype=np.float64
    )
    y0, y1 = rows
    x0, x1 = columns
    adjacent_step[y0 - 1, x0:x1] = np.abs(
        _wrap_phase(
            numerical.unwrapped_phase[y0 - 1, x0:x1]
            - prepared_unwrapped[0, :]
        )
    )
    adjacent_step[y1, x0:x1] = np.abs(
        _wrap_phase(
            numerical.unwrapped_phase[y1, x0:x1]
            - prepared_unwrapped[-1, :]
        )
    )
    adjacent_step[y0:y1, x0 - 1] = np.abs(
        _wrap_phase(
            numerical.unwrapped_phase[y0:y1, x0 - 1]
            - prepared_unwrapped[:, 0]
        )
    )
    adjacent_step[y0:y1, x1] = np.abs(
        _wrap_phase(
            numerical.unwrapped_phase[y0:y1, x1]
            - prepared_unwrapped[:, -1]
        )
    )
    adjacent_phase_step_max = float(np.nanmax(adjacent_step))
    outer_residual = np.full((target_size, target_size), np.nan, dtype=np.float64)
    outer = ~numerical.source_footprint_mask
    outer_residual[outer] = _wrap_phase(
        numerical.unwrapped_phase[outer] - numerical.quadratic_phase[outer]
    )

    amplitude_min = float(
        min(np.min(np.abs(prepared)), np.min(np.abs(legacy)), np.min(np.abs(boundary)))
    )
    amplitude_max = float(
        max(np.max(np.abs(prepared)), np.max(np.abs(legacy)), np.max(np.abs(boundary)))
    )
    if amplitude_max <= amplitude_min:
        amplitude_max = amplitude_min + 1e-12
    footprint = (rows, columns)

    fig, axes = plt.subplots(4, 3, figsize=(24, 18), dpi=150, constrained_layout=True)
    axes = axes.ravel()
    _image_panel(
        fig,
        axes[0],
        np.abs(prepared),
        title="Prepared source amplitude",
        cmap="gray",
        vmin=amplitude_min,
        vmax=amplitude_max,
    )
    _image_panel(
        fig,
        axes[1],
        np.angle(prepared),
        title="Prepared source wrapped phase",
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    _image_panel(
        fig,
        axes[2],
        np.abs(legacy),
        title="Legacy global-quadratic amplitude",
        cmap="gray",
        vmin=amplitude_min,
        vmax=amplitude_max,
        footprint=footprint,
    )
    _image_panel(
        fig,
        axes[3],
        np.angle(legacy),
        title="Legacy global-quadratic wrapped phase",
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        footprint=footprint,
    )
    _image_panel(
        fig,
        axes[4],
        np.abs(boundary),
        title="Boundary-matched amplitude",
        cmap="gray",
        vmin=amplitude_min,
        vmax=amplitude_max,
        footprint=footprint,
    )
    _image_panel(
        fig,
        axes[5],
        np.angle(boundary),
        title="Boundary-matched wrapped phase",
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        footprint=footprint,
    )
    _image_panel(
        fig,
        axes[6],
        center_difference,
        title=f"Center |new-prepared|; max={center_difference_max:.3e}",
        cmap="magma",
        vmin=0.0,
        vmax=max(center_difference_max, 1e-12),
    )
    _image_panel(
        fig,
        axes[7],
        adjacent_step,
        title=(
            "Across four inner seams wrapped phase step; "
            f"max={adjacent_phase_step_max:.3e} rad"
        ),
        cmap="magma",
        vmin=0.0,
        vmax=np.pi,
        footprint=footprint,
    )
    _image_panel(
        fig,
        axes[8],
        outer_residual,
        title="Outer wrap(phase - fitted quadratic)",
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        footprint=footprint,
        outer_boundary=True,
    )

    target_axis = np.arange(target_size)
    center_index = target_size // 2
    source_center_index = center_index - rows[0]
    prepared_horizontal = np.full(target_size, np.nan)
    prepared_vertical = np.full(target_size, np.nan)
    prepared_horizontal[columns[0] : columns[1]] = prepared_unwrapped[
        source_center_index, :
    ]
    prepared_vertical[rows[0] : rows[1]] = prepared_unwrapped[
        :, source_center_index
    ]
    legacy_unwrapped = np.asarray(unwrap_phase(np.angle(legacy)), dtype=np.float64)
    for ax, prepared_profile, global_profile, matched_profile, quadratic_profile, label in (
        (
            axes[9],
            prepared_horizontal,
            legacy_unwrapped[center_index, :],
            numerical.unwrapped_phase[center_index, :],
            numerical.quadratic_phase[center_index, :],
            "horizontal",
        ),
        (
            axes[10],
            prepared_vertical,
            legacy_unwrapped[:, center_index],
            numerical.unwrapped_phase[:, center_index],
            numerical.quadratic_phase[:, center_index],
            "vertical",
        ),
    ):
        ax.plot(target_axis, prepared_profile, label="prepared source", linewidth=2.2)
        ax.plot(target_axis, global_profile, label="legacy global", linewidth=1.5)
        ax.plot(target_axis, matched_profile, label="boundary matched", linewidth=1.5)
        ax.plot(target_axis, quadratic_profile, label="fitted quadratic", linestyle="--")
        ax.axvline(columns[0] if label == "horizontal" else rows[0], color="k", linestyle=":")
        ax.axvline(columns[1] - 1 if label == "horizontal" else rows[1] - 1, color="k", linestyle=":")
        ax.set_title(f"{label.capitalize()} unwrapped phase profile")
        ax.set_xlabel(f"{('x' if label == 'horizontal' else 'y')} [px]")
        ax.set_ylabel("unwrapped phase [rad]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    hashes = {
        "source_file_sha256": _file_sha256(source_path),
        "source_probe_sha256": _array_sha256(source),
        "prepared_probe_sha256": _array_sha256(prepared),
        "legacy_probe_sha256": _array_sha256(legacy),
        "boundary_matched_probe_sha256": _array_sha256(boundary),
    }
    annotation = (
        f"pipeline: {canonical_pipeline}\n"
        f"source file: {hashes['source_file_sha256']}\n"
        f"source probe: {hashes['source_probe_sha256']}\n"
        f"prepared: {hashes['prepared_probe_sha256']}\n"
        f"legacy output: {hashes['legacy_probe_sha256']}\n"
        f"boundary output: {hashes['boundary_matched_probe_sha256']}\n\n"
        f"method: {BOUNDARY_METHOD}\nsolver: {BOUNDARY_SOLVER}\n"
        f"tolerance: {numerical.solver_tolerance:.3e}\n"
        f"Laplacian residual: {numerical.laplacian_residual:.3e}\n"
        f"inner Dirichlet error: {seam_error_max:.3e} rad\n"
        f"adjacent wrapped phase step: {adjacent_phase_step_max:.3e} rad\n"
        f"center complex difference: {center_difference_max:.3e}"
    )
    axes[11].axis("off")
    axes[11].set_title("Identity and solver annotation")
    axes[11].text(
        0.01,
        0.99,
        annotation,
        va="top",
        ha="left",
        family="monospace",
        fontsize=7.5,
        wrap=True,
        transform=axes[11].transAxes,
    )
    fig.suptitle(
        "Probe extension audit: legacy global quadratic vs C0 boundary matched",
        fontsize=18,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    image_shape = mpimg.imread(output_path).shape

    sidecar: dict[str, object] = {
        "schema_version": "probe_extension_visual_check_v1",
        "source_probe_path": str(source_path),
        "source_shape": list(source.shape),
        "target_shape": list(boundary.shape),
        "smoothing_sigma": float(smoothing),
        "canonical_pipeline": canonical_pipeline,
        "legacy_comparison_pipeline": (
            (f"smooth:{smoothing:g}|" if smoothing > 0 else "")
            + f"pad_extrapolate:{target_size}"
        ),
        **hashes,
        "boundary_method": BOUNDARY_METHOD,
        "solver": BOUNDARY_SOLVER,
        "solver_tolerance": float(numerical.solver_tolerance),
        "laplacian_residual": float(numerical.laplacian_residual),
        "seam_error_max": seam_error_max,
        "adjacent_phase_step_max": adjacent_phase_step_max,
        "center_difference_max": center_difference_max,
        "inner_boundary_overlay": True,
        "outer_boundary_overlay": True,
        "source_rows": list(rows),
        "source_columns": list(columns),
        "quadratic_coefficients": list(numerical.quadratic_coefficients),
        "panel_labels": PANEL_LABELS,
        "png_width_pixels": int(image_shape[1]),
        "png_height_pixels": int(image_shape[0]),
    }
    sidecar_path = output_path.with_suffix(".json")
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return sidecar


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-probe", type=Path, required=True)
    parser.add_argument("--target-size", type=int, required=True)
    parser.add_argument("--smoothing", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    render_probe_extension_check(
        source_probe=args.source_probe,
        target_size=args.target_size,
        smoothing=args.smoothing,
        output=args.output,
    )


if __name__ == "__main__":
    main()
