"""Builder for the `natural_patches128_fixedprobe_v1` expanded-object CDI dataset.

This module owns the deterministic dataset construction path used by
``scripts/studies/run_cdi_natural_patch_dataset.py``. It freezes the contract
documented in
``docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-fixedprobe-dataset/execution_plan.md``:

- single-shot CDI forward model (probe x object -> Fraunhofer amplitude)
- N=128 object patches sampled from natural images
- source-image-level split (no parent-image overlap across train/val/test)
- Run1084 probe lineage (smooth_complex(sigma=0.5) -> pad_extrapolate to N=128)

The frozen contract dataclasses live in
``scripts/studies/cdi_natural_patch_dataset_types``. The JSON manifest writer,
contact-sheet renderer, and post-generation audit live in
``scripts/studies/cdi_natural_patch_dataset_io``. This module orchestrates them.

The module never trains models and never writes into the manuscript artifact
root. It only emits durable manifests + canonical NPZ payloads under the
git-ignored dataset root supplied by the caller.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from scripts.studies.cdi_natural_patch_dataset_io import (
    post_audit,
    render_contact_sheet,
    scikit_image_version,
    write_manifests,
)
from scripts.studies.cdi_natural_patch_dataset_types import (
    DEFAULT_CROP_SEED,
    DEFAULT_DATASET_ID,
    DEFAULT_PATCH_SIZE,
    DEFAULT_PROBE_SCALE_MODE,
    DEFAULT_PROBE_SMOOTHING_SIGMA,
    DEFAULT_PROBE_SOURCE,
    DEFAULT_SKIMAGE_SOURCE_NAMES,
    DEFAULT_SPLIT_COUNTS,
    DEFAULT_SPLIT_SEED,
    DEFAULT_SPLIT_SOURCE_COUNTS,
    DEFAULT_TOTAL_CAP,
    SPLIT_NAMES,
    BuildResult,
    NaturalImageRecord,
    ObjectEncodingContract,
    ProbeBundle,
    SimulationContract,
)

__all__ = [
    "BuildResult",
    "DEFAULT_CROP_SEED",
    "DEFAULT_DATASET_ID",
    "DEFAULT_PATCH_SIZE",
    "DEFAULT_PROBE_SCALE_MODE",
    "DEFAULT_PROBE_SMOOTHING_SIGMA",
    "DEFAULT_PROBE_SOURCE",
    "DEFAULT_SKIMAGE_SOURCE_NAMES",
    "DEFAULT_SPLIT_COUNTS",
    "DEFAULT_SPLIT_SEED",
    "DEFAULT_SPLIT_SOURCE_COUNTS",
    "DEFAULT_TOTAL_CAP",
    "NaturalImageRecord",
    "ObjectEncodingContract",
    "ProbeBundle",
    "SimulationContract",
    "SPLIT_NAMES",
    "assign_source_splits",
    "build_dataset",
    "encode_object_patch",
    "forward_amplitude",
    "load_skimage_corpus",
    "post_audit",
    "prepare_probe_from_run1084",
    "render_contact_sheet",
]


# ---------------------------------------------------------------------------
# Object encoding + simulation primitives
# ---------------------------------------------------------------------------


def _to_unit_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert any HxW or HxWxC uint8/float input to float32 in [0, 1]."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[..., :3].astype(np.float32)
        gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    elif arr.ndim == 3 and arr.shape[2] == 1:
        gray = arr[..., 0].astype(np.float32)
    else:
        raise ValueError(f"unsupported source image shape {arr.shape}")
    if arr.dtype == np.uint8:
        gray = gray / 255.0
    elif arr.dtype == np.bool_:
        gray = gray
    elif np.issubdtype(arr.dtype, np.floating):
        if gray.max(initial=0.0) > 1.5:
            gray = gray / 255.0
    else:
        gray = gray.astype(np.float32)
        finite_max = float(np.max(gray)) if gray.size else 1.0
        if finite_max > 1.5:
            gray = gray / max(finite_max, 1.0)
    gray = np.clip(gray, 0.0, 1.0).astype(np.float32, copy=False)
    return gray


def encode_object_patch(
    patch_unit: np.ndarray,
    contract: ObjectEncodingContract = ObjectEncodingContract(),
) -> np.ndarray:
    """Map a [0, 1] grayscale patch to a complex64 object patch."""
    x = np.clip(np.asarray(patch_unit, dtype=np.float32), 0.0, 1.0)
    amp_lo = float(contract.amplitude_min)
    amp_hi = float(contract.amplitude_max)
    phase_lo = float(contract.phase_min_rad)
    phase_hi = float(contract.phase_max_rad)
    amplitude = amp_lo + (amp_hi - amp_lo) * x
    phase = phase_lo + (phase_hi - phase_lo) * x
    obj = amplitude * np.exp(1j * phase)
    return obj.astype(np.complex64, copy=False)


def forward_amplitude(exit_wave: np.ndarray) -> np.ndarray:
    """Compute Fraunhofer-magnitude diffraction for an N x N exit wave."""
    arr = np.asarray(exit_wave)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("exit_wave must be a square 2D array")
    norm = math.sqrt(float(arr.size))
    spectrum = np.fft.fftshift(np.fft.fft2(arr.astype(np.complex128, copy=False))) / norm
    return np.abs(spectrum).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Probe lineage
# ---------------------------------------------------------------------------


def _import_probe_pipeline_helpers():
    from ptycho.workflows.grid_lines_workflow import (
        apply_probe_transform_pipeline,
        normalize_probe_transform_pipeline,
    )
    return apply_probe_transform_pipeline, normalize_probe_transform_pipeline


def prepare_probe_from_run1084(
    probe_npz_path: Path = Path(DEFAULT_PROBE_SOURCE),
    target_N: int = DEFAULT_PATCH_SIZE,
    smoothing_sigma: float = DEFAULT_PROBE_SMOOTHING_SIGMA,
    scale_mode: str = DEFAULT_PROBE_SCALE_MODE,
) -> ProbeBundle:
    """Load Run1084 probeGuess and run the lines128 default preprocessing."""
    apply_pipeline, normalize_pipeline = _import_probe_pipeline_helpers()
    probe_npz_path = Path(probe_npz_path)
    if not probe_npz_path.exists():
        raise FileNotFoundError(f"probe npz missing: {probe_npz_path}")
    with np.load(probe_npz_path) as data:
        if "probeGuess" not in data:
            raise KeyError(f"probeGuess missing from {probe_npz_path}")
        probe_raw = np.asarray(data["probeGuess"])
    if probe_raw.ndim != 2 or probe_raw.shape[0] != probe_raw.shape[1]:
        raise ValueError(f"unsupported probe shape {probe_raw.shape}")
    pipeline_spec, steps = normalize_pipeline(
        target_N=int(target_N),
        probe_shape=(int(probe_raw.shape[0]), int(probe_raw.shape[1])),
        probe_scale_mode=str(scale_mode),
        probe_smoothing_sigma=float(smoothing_sigma),
        probe_transform_pipeline=None,
    )
    probe_transformed = apply_pipeline(probe_raw, steps)
    probe_transformed = np.asarray(probe_transformed, dtype=np.complex64)
    return ProbeBundle(
        probe=probe_transformed,
        source_path=str(probe_npz_path),
        source_shape=tuple(probe_raw.shape),  # type: ignore[arg-type]
        target_N=int(target_N),
        smoothing_sigma=float(smoothing_sigma),
        scale_mode=str(scale_mode),
        pipeline_spec=str(pipeline_spec),
    )


# ---------------------------------------------------------------------------
# Source corpus
# ---------------------------------------------------------------------------


def load_skimage_corpus(
    names: Sequence[str] = DEFAULT_SKIMAGE_SOURCE_NAMES,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> List[NaturalImageRecord]:
    """Load a curated subset of scikit-image bundled natural images."""
    from skimage import data as skdata

    sorted_names = sorted({str(name) for name in names})
    if not sorted_names:
        raise ValueError("source corpus must contain at least one entry")
    records: List[NaturalImageRecord] = []
    for name in sorted_names:
        loader = getattr(skdata, name, None)
        if loader is None:
            raise ValueError(f"skimage.data has no attribute '{name}'")
        gray = _to_unit_grayscale(loader())
        h, w = gray.shape
        if h < patch_size or w < patch_size:
            raise ValueError(
                f"source image '{name}' shape={gray.shape} is smaller than patch_size={patch_size}"
            )
        records.append(NaturalImageRecord(image_id=name, pixels=gray, height=h, width=w))
    return records


# ---------------------------------------------------------------------------
# Splits + payload
# ---------------------------------------------------------------------------


def assign_source_splits(
    records: Sequence[NaturalImageRecord],
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = DEFAULT_SPLIT_SEED,
) -> Dict[str, List[NaturalImageRecord]]:
    """Deterministically partition source images at the source level."""
    total = int(n_train) + int(n_val) + int(n_test)
    if total > len(records):
        raise ValueError(
            f"requested {total} source images but only {len(records)} available"
        )
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError("each split must contain at least one source image")
    sorted_records = sorted(records, key=lambda r: r.image_id)
    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(sorted_records), dtype=np.int64)
    rng.shuffle(indices)
    chosen = indices[:total].tolist()
    train_idx = chosen[:n_train]
    val_idx = chosen[n_train : n_train + n_val]
    test_idx = chosen[n_train + n_val : n_train + n_val + n_test]
    return {
        "train": [sorted_records[i] for i in train_idx],
        "val": [sorted_records[i] for i in val_idx],
        "test": [sorted_records[i] for i in test_idx],
    }


def _allocate_patches_per_image(
    records: Sequence[NaturalImageRecord],
    target_total: int,
) -> List[int]:
    if target_total <= 0:
        raise ValueError("target_total must be positive")
    if not records:
        raise ValueError("split has no source images")
    base, extra = divmod(int(target_total), len(records))
    counts = [base + (1 if i < extra else 0) for i in range(len(records))]
    return counts


def _sample_crops(
    *,
    image_shape: Tuple[int, int],
    count: int,
    patch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = image_shape
    max_y = h - patch_size
    max_x = w - patch_size
    if max_y < 0 or max_x < 0:
        raise ValueError(
            f"image shape {image_shape} too small for patch_size={patch_size}"
        )
    ys = rng.integers(0, max_y + 1, size=count, endpoint=False)
    xs = rng.integers(0, max_x + 1, size=count, endpoint=False)
    return np.stack([ys, xs], axis=1).astype(np.int32, copy=False)


def _build_split_payload(
    *,
    records: Sequence[NaturalImageRecord],
    target_count: int,
    patch_size: int,
    probe: np.ndarray,
    encoding: ObjectEncodingContract,
    crop_seed: int,
) -> Dict[str, np.ndarray]:
    counts = _allocate_patches_per_image(records, target_count)
    objects: List[np.ndarray] = []
    diffractions: List[np.ndarray] = []
    crop_coords: List[np.ndarray] = []
    source_ids: List[str] = []
    patch_ids: List[str] = []
    rng = np.random.default_rng(int(crop_seed))
    for record, count in zip(records, counts):
        if count <= 0:
            continue
        per_image_rng = np.random.default_rng(rng.integers(0, 2**63 - 1, endpoint=False))
        crops = _sample_crops(
            image_shape=record.shape,
            count=int(count),
            patch_size=int(patch_size),
            rng=per_image_rng,
        )
        for idx, (y, x) in enumerate(crops):
            patch = record.pixels[y : y + patch_size, x : x + patch_size]
            obj = encode_object_patch(patch, encoding)
            exit_wave = probe.astype(np.complex64) * obj
            diff = forward_amplitude(exit_wave)
            objects.append(obj)
            diffractions.append(diff)
            crop_coords.append(np.array([y, x, y + patch_size, x + patch_size], dtype=np.int32))
            source_ids.append(record.image_id)
            patch_ids.append(f"{record.image_id}#{idx:05d}")
    if not objects:
        raise ValueError("split produced zero patches")
    return {
        "objects": np.stack(objects, axis=0).astype(np.complex64, copy=False),
        "diffraction": np.stack(diffractions, axis=0).astype(np.float32, copy=False),
        "crop_coords": np.stack(crop_coords, axis=0).astype(np.int32, copy=False),
        "source_ids": np.asarray(source_ids, dtype=object),
        "patch_ids": np.asarray(patch_ids, dtype=object),
    }


def _check_no_source_overlap(splits: Mapping[str, Sequence[NaturalImageRecord]]) -> Dict[str, List[str]]:
    membership: Dict[str, List[str]] = {
        name: sorted([r.image_id for r in records]) for name, records in splits.items()
    }
    seen: set = set()
    for ids in membership.values():
        for image_id in ids:
            if image_id in seen:
                raise ValueError(f"source-image '{image_id}' appears in multiple splits")
            seen.add(image_id)
    return membership


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_dataset(
    *,
    dataset_root: Path,
    records: Sequence[NaturalImageRecord],
    probe_bundle: ProbeBundle,
    split_counts: Mapping[str, int] = DEFAULT_SPLIT_COUNTS,
    split_source_counts: Mapping[str, int] = DEFAULT_SPLIT_SOURCE_COUNTS,
    patch_size: int = DEFAULT_PATCH_SIZE,
    total_cap: int = DEFAULT_TOTAL_CAP,
    split_seed: int = DEFAULT_SPLIT_SEED,
    crop_seed: int = DEFAULT_CROP_SEED,
    dataset_id: str = DEFAULT_DATASET_ID,
    encoding: ObjectEncodingContract = ObjectEncodingContract(),
    simulation: SimulationContract = SimulationContract(),
) -> BuildResult:
    """Generate the natural-patch fixed-probe CDI dataset bundle."""
    dataset_root = Path(dataset_root)
    target_total = int(sum(split_counts.values()))
    if target_total > int(total_cap):
        raise ValueError(
            f"split target {target_total} exceeds total cap {total_cap}; refusing to build"
        )
    if int(probe_bundle.target_N) != int(patch_size):
        raise ValueError(
            f"probe target_N {probe_bundle.target_N} does not match patch_size {patch_size}"
        )
    missing_split_keys = [name for name in SPLIT_NAMES if name not in split_counts]
    if missing_split_keys:
        raise KeyError(f"split_counts missing entries: {missing_split_keys}")
    splits = assign_source_splits(
        records,
        n_train=int(split_source_counts["train"]),
        n_val=int(split_source_counts["val"]),
        n_test=int(split_source_counts["test"]),
        seed=int(split_seed),
    )
    membership = _check_no_source_overlap(splits)

    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "samples").mkdir(parents=True, exist_ok=True)
    payload_paths: Dict[str, Path] = {}
    payload_counts: Dict[str, int] = {}
    artifact_paths: Dict[str, Path] = {}

    crop_rng_master = np.random.default_rng(int(crop_seed))
    for split_name in SPLIT_NAMES:
        per_split_seed = int(crop_rng_master.integers(0, 2**63 - 1, endpoint=False))
        payload = _build_split_payload(
            records=splits[split_name],
            target_count=int(split_counts[split_name]),
            patch_size=int(patch_size),
            probe=probe_bundle.probe,
            encoding=encoding,
            crop_seed=per_split_seed,
        )
        payload_path = dataset_root / f"{split_name}.npz"
        np.savez(payload_path, **payload)
        payload_paths[split_name] = payload_path
        payload_counts[split_name] = int(payload["objects"].shape[0])
        artifact_paths[f"split_{split_name}"] = payload_path

    probe_path = dataset_root / "probe.npz"
    np.savez(probe_path, probeGuess=probe_bundle.probe)
    artifact_paths["probe_npz"] = probe_path

    manifest_paths = write_manifests(
        dataset_root=dataset_root,
        dataset_id=dataset_id,
        patch_size=int(patch_size),
        splits=splits,
        split_payload_paths=payload_paths,
        split_payload_counts=payload_counts,
        probe_bundle=probe_bundle,
        encoding=encoding,
        simulation=simulation,
        split_seed=int(split_seed),
        crop_seed=int(crop_seed),
        skimage_version=scikit_image_version(),
    )
    artifact_paths.update(manifest_paths)

    return BuildResult(
        dataset_root=dataset_root,
        split_counts=payload_counts,
        source_split_membership=membership,
        artifact_paths=artifact_paths,
    )
