#!/usr/bin/env python3
"""Prepare cameraman NERSC paired-HDF5 data for hybrid_resnet external-raw studies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from scripts.studies.nersc_pair_adapter import materialize_pair_working_copy, pair_to_external_npz
from scripts.studies.invocation_logging import write_invocation_artifacts

DOWNSAMPLE_POLICY_CHOICES = ("bin-crop", "crop-bin")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _crop_center_2d(array_2d: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    arr = np.asarray(array_2d)
    if arr.ndim != 2:
        raise ValueError(f"Expected rank-2 array, got {arr.shape}")
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid crop target ({target_h}, {target_w}) for shape {arr.shape}")
    if target_h > arr.shape[0] or target_w > arr.shape[1]:
        raise ValueError(f"Crop target ({target_h}, {target_w}) exceeds source shape {arr.shape}")
    y0 = (arr.shape[0] - target_h) // 2
    x0 = (arr.shape[1] - target_w) // 2
    return arr[y0 : y0 + target_h, x0 : x0 + target_w]


def _bin_real_stack(stack: np.ndarray, factor: int) -> np.ndarray:
    arr = np.asarray(stack, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected diffraction stack rank-3 [N,H,W], got {arr.shape}")
    if factor <= 0:
        raise ValueError("factor must be positive")
    if factor == 1:
        return arr.astype(np.float32)
    n_scans, h, w = arr.shape
    bin_h = (h // factor) * factor
    bin_w = (w // factor) * factor
    if bin_h <= 0 or bin_w <= 0:
        raise ValueError(
            f"Diffraction shape {arr.shape} is too small for binning factor {factor}."
        )
    y0 = (h - bin_h) // 2
    x0 = (w - bin_w) // 2
    arr = arr[:, y0 : y0 + bin_h, x0 : x0 + bin_w]
    reshaped = arr.reshape(n_scans, bin_h // factor, factor, bin_w // factor, factor)
    # Diffraction is amplitude; detector binning should aggregate in intensity space.
    summed_intensity = np.square(reshaped).sum(axis=(2, 4), dtype=np.float64)
    return np.sqrt(summed_intensity).astype(np.float32)


def _bin_complex_2d(array_2d: np.ndarray, factor: int) -> np.ndarray:
    arr = np.asarray(array_2d, dtype=np.complex64)
    if arr.ndim != 2:
        raise ValueError(f"Expected rank-2 complex array, got {arr.shape}")
    if factor <= 0:
        raise ValueError("factor must be positive")
    if factor == 1:
        return arr.astype(np.complex64)
    h, w = arr.shape
    bin_h = (h // factor) * factor
    bin_w = (w // factor) * factor
    if bin_h <= 0 or bin_w <= 0:
        raise ValueError(
            f"Real-space shape {arr.shape} is too small for binning factor {factor}."
        )
    arr = _crop_center_2d(arr, bin_h, bin_w)
    reshaped = arr.reshape(bin_h // factor, factor, bin_w // factor, factor)
    return reshaped.mean(axis=(1, 3)).astype(np.complex64)


def _downsample_external_payload(
    data: dict[str, np.ndarray], *, target_n: int, downsample_policy: str = "bin-crop"
) -> dict[str, np.ndarray]:
    if "diff3d" in data and "diffraction" not in data:
        data = {("diffraction" if key == "diff3d" else key): value for key, value in data.items()}

    if "diffraction" not in data:
        raise KeyError("Expected converted payload key 'diffraction' (or source key 'diff3d').")
    if "objectGuess" not in data or "probeGuess" not in data:
        raise KeyError("Expected objectGuess and probeGuess in converted payload.")

    diffraction = np.asarray(data["diffraction"], dtype=np.float32)
    n_scans, src_n, _ = diffraction.shape
    if src_n % target_n != 0:
        raise ValueError(f"Cannot downsample diffraction from N={src_n} to N={target_n}")
    factor = src_n // target_n
    if downsample_policy not in DOWNSAMPLE_POLICY_CHOICES:
        raise ValueError(
            f"Unsupported downsample_policy='{downsample_policy}', "
            f"expected one of {DOWNSAMPLE_POLICY_CHOICES}."
        )

    downsampled: dict[str, np.ndarray] = dict(data)

    object_guess = np.asarray(data["objectGuess"])
    probe_guess = np.asarray(data["probeGuess"])

    if downsample_policy == "bin-crop":
        downsampled["diffraction"] = _bin_real_stack(diffraction, factor)
        downsampled["objectGuess"] = _crop_center_2d(
            object_guess,
            object_guess.shape[0] // factor,
            object_guess.shape[1] // factor,
        ).astype(np.complex64)
        downsampled["probeGuess"] = _crop_center_2d(
            probe_guess,
            probe_guess.shape[0] // factor,
            probe_guess.shape[1] // factor,
        ).astype(np.complex64)
    else:
        downsampled["diffraction"] = np.stack(
            [_crop_center_2d(frame, target_n, target_n) for frame in diffraction], axis=0
        ).astype(np.float32)
        downsampled["objectGuess"] = _bin_complex_2d(object_guess, factor)
        downsampled["probeGuess"] = _bin_complex_2d(probe_guess, factor)

    for key in ("xcoords", "ycoords", "xcoords_start", "ycoords_start"):
        if key in downsampled:
            coords = np.asarray(downsampled[key], dtype=np.float64)
            if downsample_policy == "crop-bin" and factor != 1:
                coords = coords / float(factor)
            downsampled[key] = coords

    if "scan_index" not in downsampled:
        downsampled["scan_index"] = np.arange(n_scans, dtype=np.int64)
    return downsampled


def _split_scanwise_payload(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    n_scans = int(np.asarray(data["xcoords"]).shape[0])
    out: dict[str, np.ndarray] = {}
    for key, value in data.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and int(arr.shape[0]) == n_scans:
            out[key] = arr[mask]
        else:
            out[key] = arr
    return out


def prepare_hybrid_dataset(
    *,
    dp_h5: Path,
    para_h5: Path,
    output_dir: Path,
    half: str = "top",
    target_n: int = 128,
    downsample_policy: str = "bin-crop",
) -> dict[str, Any]:
    """Prepare top/bottom-half train split and full-object test split for hybrid training."""
    if half not in {"top", "bottom"}:
        raise ValueError(f"Unsupported half='{half}', expected 'top' or 'bottom'.")

    dp_h5 = Path(dp_h5)
    para_h5 = Path(para_h5)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    working_dp, working_para = materialize_pair_working_copy(
        dp_h5, para_h5, output_dir / "working_pair"
    )
    canonical_npz = pair_to_external_npz(
        working_dp,
        working_para,
        output_dir / "cameraman256_external_raw.npz",
    )

    with np.load(canonical_npz, allow_pickle=True) as loaded:
        converted = {key: loaded[key] for key in loaded.files}
    downsampled = _downsample_external_payload(
        converted,
        target_n=target_n,
        downsample_policy=downsample_policy,
    )

    ycoords = np.asarray(downsampled["ycoords"], dtype=np.float64)
    split_threshold = float((np.min(ycoords) + np.max(ycoords)) / 2.0)
    train_mask = ycoords >= split_threshold if half == "top" else ycoords < split_threshold
    if not bool(train_mask.any()):
        raise ValueError(f"Train split '{half}' is empty at threshold {split_threshold}.")
    if not bool((~train_mask).any()):
        raise ValueError("Complementary split is empty; cannot create full/evaluative split contract.")

    train_npz = output_dir / f"cameraman256_n{target_n}_{half}_half_train.npz"
    test_npz = output_dir / f"cameraman256_n{target_n}_full_test.npz"
    downsampled_npz = output_dir / f"cameraman256_n{target_n}_full_downsampled.npz"
    manifest_json = output_dir / "manifest.json"

    np.savez_compressed(downsampled_npz, **downsampled)
    np.savez_compressed(train_npz, **_split_scanwise_payload(downsampled, train_mask))
    np.savez_compressed(test_npz, **downsampled)

    manifest = {
        "source_dp": str(dp_h5),
        "source_para": str(para_h5),
        "source_dp_sha256": _sha256(dp_h5),
        "source_para_sha256": _sha256(para_h5),
        "working_dp": str(working_dp),
        "working_para": str(working_para),
        "canonical_external_npz": str(canonical_npz),
        "downsampled_npz": str(downsampled_npz),
        "train_npz": str(train_npz),
        "test_npz": str(test_npz),
        "target_n": int(target_n),
        "downsample_policy": downsample_policy,
        "half": half,
        "split_threshold": split_threshold,
        "n_total": int(ycoords.shape[0]),
        "n_train": int(np.sum(train_mask)),
        "n_test": int(ycoords.shape[0]),
    }
    manifest_json.write_text(json.dumps(manifest, indent=2))

    return {
        "canonical_npz": str(canonical_npz),
        "downsampled_npz": str(downsampled_npz),
        "train_npz": str(train_npz),
        "test_npz": str(test_npz),
        "manifest_json": str(manifest_json),
        "split_threshold": split_threshold,
        "half": half,
        "downsample_policy": downsample_policy,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare NERSC cameraman paired-HDF5 data for hybrid_resnet external studies."
    )
    parser.add_argument("--dp-h5", type=Path, required=True, help="Path to cameraman *_dp.hdf5 file.")
    parser.add_argument(
        "--para-h5",
        type=Path,
        required=True,
        help="Path to cameraman *_para.hdf5 file.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for prepared artifacts.")
    parser.add_argument(
        "--half",
        type=str,
        choices=["top", "bottom"],
        default="top",
        help="Half-space used for the training split (default: top).",
    )
    parser.add_argument("--target-n", type=int, default=128, help="Target diffraction N (default: 128).")
    parser.add_argument(
        "--downsample-policy",
        type=str,
        choices=list(DOWNSAMPLE_POLICY_CHOICES),
        default="bin-crop",
        help=(
            "Downsample policy: 'bin-crop' bins diffraction and crops real-space; "
            "'crop-bin' crops diffraction and bins real-space."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    write_invocation_artifacts(
        output_dir=args.output_dir,
        script_path="scripts/studies/prepare_nersc_hybrid_dataset.py",
        argv=(argv if argv is not None else sys.argv[1:]),
        parsed_args=vars(args),
    )
    result = prepare_hybrid_dataset(
        dp_h5=args.dp_h5,
        para_h5=args.para_h5,
        output_dir=args.output_dir,
        half=args.half,
        target_n=args.target_n,
        downsample_policy=args.downsample_policy,
    )
    print(f"Prepared manifest: {result['manifest_json']}")


if __name__ == "__main__":
    main()
