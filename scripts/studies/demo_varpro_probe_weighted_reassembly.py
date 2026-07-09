"""Visual demo for VarPro scaling and probe-weighted patch reassembly."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptycho_torch.reassembly import (
    VarProScaler,
    VectorizedWeightedAccumulator,
    apply_varpro_canvas_scaling,
)


VARIANTS = (
    ("uniform_no_varpro", "Uniform / no VarPro", "uniform", False),
    ("uniform_varpro", "Uniform / VarPro", "uniform", True),
    ("probe_no_varpro", "Probe / no VarPro", "probe", False),
    ("probe_varpro", "Probe / VarPro", "probe", True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=".artifacts/varpro_probe_weighting_demo")
    parser.add_argument("--input-npz", type=Path, default=None)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def write_invocation(output_root: Path, args: argparse.Namespace) -> None:
    payload = {
        "argv": sys.argv,
        "seed": args.seed,
        "input_npz": None if args.input_npz is None else str(args.input_npz),
        "size": args.size,
        "patch_size": args.patch_size,
        "git_commit": _git_commit(),
    }
    (output_root / "invocation.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def make_truth(size: int = 64) -> np.ndarray:
    y = np.linspace(-1.0, 1.0, size)
    x = np.linspace(-1.0, 1.0, size)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    amplitude = 1.0 + 0.25 * np.exp(-((xx + 0.25) ** 2 + (yy - 0.1) ** 2) / 0.08)
    phase = 0.6 * np.sin(2 * np.pi * xx) + 0.35 * np.cos(2 * np.pi * yy)
    return amplitude * np.exp(1j * phase)


def make_probe_weight(patch_size: int) -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, patch_size)
    py, px = np.meshgrid(grid, grid, indexing="ij")
    probe_weight = np.exp(-3.5 * (px**2 + py**2)).astype(np.float32)
    probe_weight /= probe_weight.max()
    return probe_weight


def _as_2d_complex(array: np.ndarray) -> np.ndarray:
    squeezed = np.squeeze(array)
    if squeezed.ndim != 2:
        raise ValueError(f"Expected a 2D complex image after squeezing, got shape {array.shape}")
    return squeezed.astype(np.complex64)


def _center_crop_2d(array: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}")
    height, width = array.shape
    if size > min(height, width):
        raise ValueError(f"Cannot crop size={size} from array shape {array.shape}")
    y0 = (height - size) // 2
    x0 = (width - size) // 2
    return array[y0 : y0 + size, x0 : x0 + size]


def _resample_probe_weight(probe: np.ndarray, patch_size: int) -> np.ndarray:
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")
    probe_2d = np.squeeze(probe)
    if probe_2d.ndim != 2:
        raise ValueError(f"Expected 2D probe after squeezing, got shape {probe.shape}")
    probe_mag = np.abs(probe_2d.astype(np.complex64)) ** 2
    y_idx = np.linspace(0, probe_mag.shape[0] - 1, patch_size).round().astype(np.int64)
    x_idx = np.linspace(0, probe_mag.shape[1] - 1, patch_size).round().astype(np.int64)
    probe_weight = probe_mag[np.ix_(y_idx, x_idx)].astype(np.float32)
    max_weight = float(probe_weight.max())
    if max_weight <= 0:
        return make_probe_weight(patch_size)
    return probe_weight / max_weight


def load_npz_truth_and_probe_weight(input_npz: Path, *, size: int, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(input_npz, allow_pickle=True) as data:
        for key in ("YY_ground_truth", "YY_full", "objectGuess"):
            if key in data:
                truth = _as_2d_complex(data[key])
                break
        else:
            raise ValueError("Input NPZ must contain one of: YY_ground_truth, YY_full, objectGuess")

        if "probeGuess" in data:
            probe_weight = _resample_probe_weight(data["probeGuess"], patch_size)
        else:
            probe_weight = make_probe_weight(patch_size)

    return _center_crop_2d(truth, size), probe_weight


def _patch_starts(size: int, patch_size: int) -> list[int]:
    if patch_size >= size:
        raise ValueError(f"patch_size must be smaller than size (patch_size={patch_size}, size={size})")
    max_start = size - patch_size - 1
    return sorted({int(round(value)) for value in np.linspace(0, max_start, 5)})


def extract_demo_patches(
    truth: np.ndarray,
    *,
    patch_size: int,
    real_scale: float,
    imag_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = truth.shape[0]
    starts = _patch_starts(size, patch_size)
    patches = []
    centers = []
    coverage = np.zeros_like(truth.real, dtype=np.int32)

    for y0 in starts:
        for x0 in starts:
            patch = truth[y0 : y0 + patch_size, x0 : x0 + patch_size]
            patch_pred = (patch.real / real_scale) + 1j * (patch.imag / imag_scale)
            patch_pred[:, -4:] += -0.30 - 0.20j
            patch_pred[-4:, :] += -0.15 + 0.10j
            patches.append(patch_pred.astype(np.complex64))
            centers.append((x0 + patch_size / 2, y0 + patch_size / 2))
            coverage[y0 : y0 + patch_size, x0 : x0 + patch_size] += 1

    return (
        np.stack(patches),
        np.asarray(centers, dtype=np.float32),
        coverage > 1,
    )


def build_varpro_scaler(device: torch.device, real_scale: float, imag_scale: float) -> VarProScaler:
    scaler = VarProScaler(device)
    y = torch.linspace(-1.0, 1.0, 32, device=device)
    x = torch.linspace(-1.0, 1.0, 32, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    x1 = (1.0 + xx.square()).unsqueeze(0)
    x2 = (0.8 + yy.square()).unsqueeze(0)
    x3 = torch.zeros_like(x1)
    i_raw = (real_scale**2) * x1 + (imag_scale**2) * x2
    scaler.accumulate_batch_from_basis(i_raw, x1, x2, x3)
    return scaler


def reassemble_variant(
    patches: np.ndarray,
    centers: np.ndarray,
    probe_weight: np.ndarray,
    *,
    size: int,
    patch_size: int,
    patch_weighting: str,
    varpro_scaling: bool,
    real_scale: float,
    imag_scale: float,
) -> np.ndarray:
    device = torch.device("cpu")
    canvas_shape = (size + 1, size + 1)
    canvas = torch.zeros(canvas_shape, dtype=torch.complex64, device=device)
    weights = torch.zeros(canvas_shape, dtype=torch.float32, device=device)
    accumulator = VectorizedWeightedAccumulator(canvas_shape, device)

    accumulator.accumulate_batch(
        canvas,
        weights,
        torch.from_numpy(patches).to(device),
        torch.from_numpy(centers).to(device),
        torch.from_numpy(probe_weight).to(device),
        patch_size=patch_size,
        uniform_weighting=(patch_weighting == "uniform"),
    )
    texture_canvas = canvas / (weights + 1e-12)
    scaled_canvas, _, _ = apply_varpro_canvas_scaling(
        texture_canvas,
        build_varpro_scaler(device, real_scale, imag_scale),
        enabled=varpro_scaling,
        verbose=False,
    )
    return scaled_canvas[:size, :size].cpu().numpy()


def compute_metrics(recon: np.ndarray, truth: np.ndarray, seam_mask: np.ndarray) -> dict[str, float]:
    return {
        "complex_mae": float(np.mean(np.abs(recon - truth))),
        "amp_mae": float(np.mean(np.abs(np.abs(recon) - np.abs(truth)))),
        "phase_mae": float(np.mean(np.abs(np.angle(recon * np.conj(truth))))),
        "seam_mae": float(np.mean(np.abs(recon[seam_mask] - truth[seam_mask]))),
    }


def save_reconstruction_grid(output_path: Path, truth: np.ndarray, recons: dict[str, np.ndarray]) -> None:
    names = [("truth", "Truth")] + [(key, title) for key, title, _, _ in VARIANTS]
    amp_vmax = max(np.abs(truth).max(), *(np.abs(recons[key]).max() for key in recons))

    fig, axes = plt.subplots(2, len(names), figsize=(3.0 * len(names), 6.0), constrained_layout=True)
    for col, (key, title) in enumerate(names):
        image = truth if key == "truth" else recons[key]
        axes[0, col].imshow(np.abs(image), cmap="viridis", vmin=0, vmax=amp_vmax)
        axes[0, col].set_title(title)
        axes[0, col].set_axis_off()
        axes[1, col].imshow(np.angle(image), cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[1, col].set_axis_off()
    axes[0, 0].set_ylabel("Amplitude")
    axes[1, 0].set_ylabel("Phase")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_error_grid(output_path: Path, truth: np.ndarray, recons: dict[str, np.ndarray]) -> None:
    errors = {key: np.abs(value - truth) for key, value in recons.items()}
    vmax = max(error.max() for error in errors.values())
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(3.2 * len(VARIANTS), 3.2), constrained_layout=True)
    for ax, (key, title, _, _) in zip(axes, VARIANTS):
        ax.imshow(errors[key], cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_axis_off()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_probe_weight(output_path: Path, probe_weight: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    ax.imshow(probe_weight, cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Probe weight")
    ax.set_axis_off()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_invocation(output_root, args)

    size = args.size
    patch_size = args.patch_size
    real_scale = 1.6
    imag_scale = 0.65

    if args.input_npz is None:
        truth = make_truth(size)
        probe_weight = make_probe_weight(patch_size)
    else:
        truth, probe_weight = load_npz_truth_and_probe_weight(
            args.input_npz,
            size=size,
            patch_size=patch_size,
        )
    patches, centers, seam_mask = extract_demo_patches(
        truth,
        patch_size=patch_size,
        real_scale=real_scale,
        imag_scale=imag_scale,
    )

    recons = {
        key: reassemble_variant(
            patches,
            centers,
            probe_weight,
            size=size,
            patch_size=patch_size,
            patch_weighting=patch_weighting,
            varpro_scaling=varpro_scaling,
            real_scale=real_scale,
            imag_scale=imag_scale,
        )
        for key, _, patch_weighting, varpro_scaling in VARIANTS
    }
    metrics = {key: compute_metrics(value, truth, seam_mask) for key, value in recons.items()}

    (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    save_reconstruction_grid(output_root / "reconstruction_grid.png", truth, recons)
    save_error_grid(output_root / "error_grid.png", truth, recons)
    save_probe_weight(output_root / "probe_weight_map.png", probe_weight)

    if args.show:
        plt.show()

    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
