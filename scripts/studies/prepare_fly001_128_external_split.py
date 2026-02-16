#!/usr/bin/env python3
"""Prepare canonical fly001 N=128 data with top-half train and full-object test outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonicalize(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    canonical: dict[str, np.ndarray] = {}
    for key, value in data.items():
        out_key = "diffraction" if key == "diff3d" else key
        out_value = value.astype(np.float32) if getattr(value, "dtype", None) == np.uint16 else value
        canonical[out_key] = out_value
    return canonical


def _split_payload(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    n_scans = int(data["xcoords"].shape[0])
    split: dict[str, np.ndarray] = {}
    for key, value in data.items():
        if hasattr(value, "shape") and value.shape and value.shape[0] == n_scans:
            split[key] = value[mask]
        else:
            split[key] = value
    return split


def prepare_dataset(*, raw_npz: Path, output_dir: Path) -> dict[str, str | float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(raw_npz, allow_pickle=True) as loaded:
        raw_data = {key: loaded[key] for key in loaded.files}

    canonical = _canonicalize(raw_data)
    if "diffraction" not in canonical:
        raise KeyError("Input NPZ must contain either 'diff3d' or 'diffraction'.")
    if "xcoords" not in canonical or "ycoords" not in canonical:
        raise KeyError("Input NPZ must contain xcoords and ycoords.")

    ycoords = np.asarray(canonical["ycoords"], dtype=np.float64)
    split_threshold = float((ycoords.min() + ycoords.max()) / 2.0)
    train_mask = ycoords >= split_threshold
    bottom_mask = ~train_mask
    if not bool(train_mask.any()) or not bool(bottom_mask.any()):
        raise ValueError(
            "Split produced an empty train or test partition; check ycoords and threshold."
        )

    canonical_npz = output_dir / "fly001_128_train_converted.npz"
    train_npz = output_dir / "fly001_128_top_half_converted.npz"
    test_npz = output_dir / "fly001_128_full_test_converted.npz"
    manifest_json = output_dir / "manifest.json"

    np.savez_compressed(canonical_npz, **canonical)
    np.savez_compressed(train_npz, **_split_payload(canonical, train_mask))
    np.savez_compressed(test_npz, **canonical)

    manifest = {
        "source_file": str(raw_npz),
        "source_sha256": _sha256(raw_npz),
        "canonical_npz": str(canonical_npz),
        "train_npz": str(train_npz),
        "test_npz": str(test_npz),
        "train_split_axis": "ycoords",
        "split_threshold": split_threshold,
        "n_total": int(ycoords.size),
        "n_train": int(train_mask.sum()),
        "n_test": int(ycoords.size),
        "test_policy": "full_object",
    }
    manifest_json.write_text(json.dumps(manifest, indent=2))

    return {
        "canonical_npz": str(canonical_npz),
        "train_npz": str(train_npz),
        "test_npz": str(test_npz),
        "manifest_json": str(manifest_json),
        "split_threshold": split_threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonicalize fly001 N=128 NPZ, write top-half train split, and full-object test set."
    )
    parser.add_argument("--input-npz", type=Path, required=True, help="Path to raw fly001 N=128 NPZ.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for prepared NPZ files.")
    args = parser.parse_args()

    result = prepare_dataset(raw_npz=args.input_npz, output_dir=args.output_dir)
    print(f"Prepared dataset manifest: {result['manifest_json']}")


if __name__ == "__main__":
    main()
