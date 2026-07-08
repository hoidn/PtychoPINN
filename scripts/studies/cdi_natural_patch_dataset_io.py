"""I/O helpers for the natural-patch fixed-probe CDI dataset builder.

This module owns the JSON manifests, contact-sheet rendering, and the
post-generation audit. Splitting these out keeps
``scripts/studies/cdi_natural_patch_dataset.py`` under the project's per-module
size limit while preserving the single dataset contract: every output here is
written through the locked builder orchestrator, never as a standalone path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from scripts.studies.cdi_natural_patch_dataset_types import (
    DEFAULT_TOTAL_CAP,
    SPLIT_NAMES,
    NaturalImageRecord,
    ObjectEncodingContract,
    ProbeBundle,
    SimulationContract,
)


def _split_membership(splits: Mapping[str, Sequence[NaturalImageRecord]]) -> Dict[str, List[str]]:
    return {name: sorted([record.image_id for record in records]) for name, records in splits.items()}


def _no_source_overlap(membership: Mapping[str, Sequence[str]]) -> bool:
    seen: set = set()
    for ids in membership.values():
        for image_id in ids:
            if image_id in seen:
                return False
            seen.add(image_id)
    return True


def scikit_image_version() -> str:
    try:
        import skimage  # type: ignore

        return str(getattr(skimage, "__version__", "unknown"))
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def _build_source_records(
    splits: Mapping[str, Sequence[NaturalImageRecord]], skimage_version: str
) -> list:
    records = []
    for split_name, split_records in splits.items():
        for record in split_records:
            records.append(
                {
                    "image_id": record.image_id,
                    "split": split_name,
                    "shape": [int(record.height), int(record.width)],
                    "dtype": "float32_unit",
                    "provenance": {
                        "package": "scikit-image",
                        "version": skimage_version,
                        "loader": f"skimage.data.{record.image_id}",
                    },
                }
            )
    return records


def write_manifests(
    *,
    dataset_root: Path,
    dataset_id: str,
    patch_size: int,
    splits: Mapping[str, Sequence[NaturalImageRecord]],
    split_payload_paths: Mapping[str, Path],
    split_payload_counts: Mapping[str, int],
    probe_bundle: ProbeBundle,
    encoding: ObjectEncodingContract,
    simulation: SimulationContract,
    split_seed: int,
    crop_seed: int,
    skimage_version: str,
) -> Dict[str, Path]:
    """Write the canonical JSON manifests beside the NPZ payloads."""
    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    membership = _split_membership(splits)
    if not _no_source_overlap(membership):
        raise ValueError("source-image overlap detected across splits")
    source_manifest = {
        "dataset_id": dataset_id,
        "corpus": "skimage_bundled_natural_images",
        "package_version": skimage_version,
        "license": "scikit-image data license",
        "source_images": _build_source_records(splits, skimage_version),
    }
    split_manifest = {
        "dataset_id": dataset_id,
        "split_seed": int(split_seed),
        "crop_seed": int(crop_seed),
        "split_membership": membership,
        "split_counts": {name: int(count) for name, count in split_payload_counts.items()},
        "source_image_counts": {name: len(records) for name, records in splits.items()},
        "no_source_overlap": True,
    }
    probe_manifest = {
        "dataset_id": dataset_id,
        "source_path": probe_bundle.source_path,
        "source_shape": list(probe_bundle.source_shape),
        "target_N": int(probe_bundle.target_N),
        "smoothing_sigma": float(probe_bundle.smoothing_sigma),
        "scale_mode": probe_bundle.scale_mode,
        "pipeline_spec": probe_bundle.pipeline_spec,
        "lineage": "lines128_paper_benchmark_default",
    }
    simulation_manifest = {
        "dataset_id": dataset_id,
        "patch_size": int(patch_size),
        "object_encoding": {
            "grayscale": encoding.grayscale,
            "normalization": encoding.normalization,
            "amplitude_min": float(encoding.amplitude_min),
            "amplitude_max": float(encoding.amplitude_max),
            "phase_min_rad": float(encoding.phase_min_rad),
            "phase_max_rad": float(encoding.phase_max_rad),
            "description": encoding.description,
        },
        "simulation": {
            "forward_model": simulation.forward_model,
            "formula": simulation.formula,
            "dtype_object": simulation.dtype_object,
            "dtype_diffraction": simulation.dtype_diffraction,
        },
    }
    dataset_manifest = {
        "dataset_id": dataset_id,
        "schema_version": "natural_patches_v1",
        "claim_boundary": "expanded_object_cdi_dataset_prerequisite",
        "patch_size": int(patch_size),
        "total_patches": int(sum(split_payload_counts.values())),
        "split_counts": {name: int(count) for name, count in split_payload_counts.items()},
        "split_payloads": {name: str(path.name) for name, path in split_payload_paths.items()},
        "manifests": [
            "source_manifest.json",
            "split_manifest.json",
            "probe_manifest.json",
            "simulation_manifest.json",
            "adapter_contract.json",
        ],
        "probe_npz": "probe.npz",
        "contact_sheet": "contact_sheet.png",
        "samples_dir": "samples",
        "scikit_image_version": skimage_version,
    }
    adapter_contract = {
        "dataset_id": dataset_id,
        "schema_version": "natural_patches_v1",
        "claim_boundary": "expanded_object_cdi_dataset_prerequisite",
        "consumer_note": (
            "Each split.npz emits 'objects' (complex64 NxNxN), 'diffraction' "
            "(float32 NxNxN), 'crop_coords' (int32 Nx4: y0,x0,y1,x1), 'source_ids' "
            "(object), and 'patch_ids' (object). Consumers MUST treat each entry as a "
            "single-shot CDI sample with the shared probe in probe.npz['probeGuess']. "
            "To reuse the existing grouped CDI runner contract, expanded-object "
            "benchmarking adapters MUST construct one scan group per object patch with "
            "a single zero-coordinate, set probeGuess to probe.npz['probeGuess'], and "
            "set Y_I/Y_phi from the complex object - they MUST NOT regenerate "
            "diffraction with a different forward operator."
        ),
        "consumer_keys": {
            "objects": {"dtype": "complex64", "shape": ["N_split", "N", "N"]},
            "diffraction": {"dtype": "float32", "shape": ["N_split", "N", "N"]},
            "crop_coords": {"dtype": "int32", "shape": ["N_split", 4]},
            "source_ids": {"dtype": "object", "shape": ["N_split"]},
            "patch_ids": {"dtype": "object", "shape": ["N_split"]},
        },
        "downstream_target": (
            "docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog-gaps/"
            "2026-05-04-cdi-natural-patch-expanded-benchmark.md"
        ),
    }
    paths = {
        "source_manifest": dataset_root / "source_manifest.json",
        "split_manifest": dataset_root / "split_manifest.json",
        "probe_manifest": dataset_root / "probe_manifest.json",
        "simulation_manifest": dataset_root / "simulation_manifest.json",
        "dataset_manifest": dataset_root / "dataset_manifest.json",
        "adapter_contract": dataset_root / "adapter_contract.json",
    }
    paths["source_manifest"].write_text(json.dumps(source_manifest, indent=2))
    paths["split_manifest"].write_text(json.dumps(split_manifest, indent=2))
    paths["probe_manifest"].write_text(json.dumps(probe_manifest, indent=2))
    paths["simulation_manifest"].write_text(json.dumps(simulation_manifest, indent=2))
    paths["dataset_manifest"].write_text(json.dumps(dataset_manifest, indent=2))
    paths["adapter_contract"].write_text(json.dumps(adapter_contract, indent=2))
    return paths


def render_contact_sheet(
    *,
    dataset_root: Path,
    splits_to_show: Sequence[str] = SPLIT_NAMES,
    samples_per_split: int = 3,
) -> Path:
    """Render a small grid showing source patches and diffraction samples."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset_root = Path(dataset_root)
    fig, axes = plt.subplots(
        nrows=len(splits_to_show) * 2,
        ncols=samples_per_split,
        figsize=(2.0 * samples_per_split, 2.0 * len(splits_to_show) * 2),
        squeeze=False,
    )
    samples_dir = dataset_root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for row_split, split_name in enumerate(splits_to_show):
        path = dataset_root / f"{split_name}.npz"
        with np.load(path, allow_pickle=True) as data:
            objects = data["objects"]
            diffractions = data["diffraction"]
            patch_ids = data["patch_ids"]
            count = min(int(samples_per_split), int(objects.shape[0]))
            for col in range(count):
                obj = objects[col]
                diff = diffractions[col]
                ax_obj = axes[row_split * 2][col]
                ax_diff = axes[row_split * 2 + 1][col]
                ax_obj.imshow(np.abs(obj), cmap="gray")
                ax_obj.set_title(f"{split_name} obj |{patch_ids[col]}|", fontsize=7)
                ax_obj.axis("off")
                ax_diff.imshow(np.log1p(diff), cmap="magma")
                ax_diff.set_title(f"{split_name} diff", fontsize=7)
                ax_diff.axis("off")
                sample_path = samples_dir / f"{split_name}_{col:02d}.npz"
                np.savez(sample_path, object=obj, diffraction=diff, patch_id=str(patch_ids[col]))
    contact_sheet_path = dataset_root / "contact_sheet.png"
    fig.tight_layout()
    fig.savefig(contact_sheet_path, dpi=120)
    plt.close(fig)
    return contact_sheet_path


def post_audit(
    *,
    dataset_root: Path,
    expected_split_counts: Mapping[str, int],
    total_cap: int = DEFAULT_TOTAL_CAP,
) -> Dict[str, object]:
    """Run a deterministic audit over a generated dataset root."""
    dataset_root = Path(dataset_root)
    dataset_manifest_path = dataset_root / "dataset_manifest.json"
    if not dataset_manifest_path.exists():
        raise FileNotFoundError(f"dataset_manifest.json missing under {dataset_root}")
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    split_manifest = json.loads((dataset_root / "split_manifest.json").read_text())
    actual_counts: Dict[str, int] = {}
    for split_name in SPLIT_NAMES:
        path = dataset_root / f"{split_name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{split_name}.npz missing under {dataset_root}")
        with np.load(path, allow_pickle=True) as data:
            actual_counts[split_name] = int(data["objects"].shape[0])
    total = int(sum(actual_counts.values()))
    if total > int(total_cap):
        raise ValueError(f"total objects {total} exceeds cap {total_cap}")
    if total != int(dataset_manifest["total_patches"]):
        raise ValueError(
            f"manifest total_patches={dataset_manifest['total_patches']} != actual {total}"
        )
    for split_name, expected in expected_split_counts.items():
        if actual_counts[split_name] != int(expected):
            raise ValueError(
                f"split {split_name}: expected {expected} entries, found {actual_counts[split_name]}"
            )
    membership = split_manifest["split_membership"]
    if not _no_source_overlap(membership):
        raise ValueError("source-image overlap detected in split_manifest membership")
    required_files = [
        "dataset_manifest.json",
        "source_manifest.json",
        "split_manifest.json",
        "probe_manifest.json",
        "simulation_manifest.json",
        "adapter_contract.json",
        "probe.npz",
    ]
    for name in required_files:
        if not (dataset_root / name).exists():
            raise FileNotFoundError(f"required artifact missing: {name}")
    audit = {
        "dataset_root": str(dataset_root),
        "split_counts": actual_counts,
        "total_objects": total,
        "no_source_overlap": True,
        "manifests_present": True,
    }
    return audit
