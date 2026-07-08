"""BRDT smoke-dataset loader and collation.

Consumes the locked dataset/manifest contract emitted by the
``2026-04-29-brdt-dataset-preflight`` item and exposes it through a
``torch.utils.data.Dataset`` plus a small collator.

This module deliberately routes all manifest, normalization, and
operator-input contract decisions through
``scripts.studies.born_rytov_dt.dataset_contract`` so the locked
contract has exactly one source of truth. It does not redefine
geometry, normalization, or the physics-loss rule; it consumes them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.studies.born_rytov_dt import dataset_contract as dc
from scripts.studies.born_rytov_dt.run_config import (
    REJECTED_INPUT_MODES,
    SUPPORTED_INPUT_MODES,
)


SUPPORTED_SPLITS: Tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetAuthority:
    """Resolved manifest-driven authority for a BRDT smoke dataset."""

    manifest_path: Path
    dataset_id: str
    operator_version: str
    split_paths: Dict[str, Path]
    normalization: dc.NormalizationStats
    angles_rad: np.ndarray
    operator_block: Dict[str, Any]
    raw_manifest: Dict[str, Any]


def _resolve_relative(manifest_path: Path, candidate: str) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def load_dataset_authority(manifest_path: Path | str) -> DatasetAuthority:
    """Read the dataset manifest and resolve consumed sub-contracts.

    The returned authority pins:

    - the dataset identity (``dataset_id``),
    - the operator authority pointer (``operator_version``: validation-
      report path, used directly as the durable ``operator_version`` in
      row metadata so geometry/normalize choices are traceable);
    - per-split HDF5 paths;
    - train-only normalization statistics;
    - the locked angle grid;
    - the operator block (forwarded into adapter/operator construction).
    """
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"BRDT dataset manifest not found: {manifest_path}")
    raw = json.loads(manifest_path.read_text())
    missing = [k for k in dc.manifest_required_keys() if k not in raw]
    if missing:
        raise ValueError(
            f"manifest {manifest_path} missing required top-level keys: {missing}"
        )

    artifacts = raw["artifacts"]
    split_paths: Dict[str, Path] = {}
    for split in SUPPORTED_SPLITS:
        if split not in artifacts:
            raise ValueError(f"manifest {manifest_path} missing artifact for split={split!r}")
        split_paths[split] = _resolve_relative(manifest_path, artifacts[split])
        if not split_paths[split].exists():
            raise FileNotFoundError(
                f"BRDT split file missing for split={split!r}: {split_paths[split]}"
            )

    norm_block = raw.get("normalization") or {}
    required_norm_keys = ("q_mean_train", "q_std_train", "q_min_train", "q_max_train")
    missing_norm = [k for k in required_norm_keys if k not in norm_block]
    if missing_norm:
        raise ValueError(
            f"manifest {manifest_path} normalization block missing keys: {missing_norm}"
        )
    stats = dc.NormalizationStats(
        mean=float(norm_block["q_mean_train"]),
        std=float(norm_block["q_std_train"]),
        qmin=float(norm_block["q_min_train"]),
        qmax=float(norm_block["q_max_train"]),
    )

    operator_block = dict(raw["operator"])
    operator_validation_path = operator_block.get("validation_artifact") or operator_block.get(
        "validation_report"
    )
    operator_version = (
        str(operator_validation_path) if operator_validation_path else "unspecified"
    )
    geometry_mismatches = dc.validate_geometry_against_operator_authority(operator_block)
    if geometry_mismatches:
        raise ValueError(
            "manifest operator block does not match the locked smoke geometry: "
            + "; ".join(geometry_mismatches)
        )

    extra = raw.get("extra") or {}
    angles_list = extra.get("angle_grid_rad")
    if angles_list is None:
        angles = dc.locked_angles()
    else:
        angles = np.asarray(angles_list, dtype=np.float64)
    if angles.shape != (dc.LOCKED_ANGLE_COUNT,):
        raise ValueError(
            f"angles_rad has shape {angles.shape}, expected ({dc.LOCKED_ANGLE_COUNT},)"
        )

    dataset_id = str(raw["dataset_identity"]["name"])

    return DatasetAuthority(
        manifest_path=manifest_path,
        dataset_id=dataset_id,
        operator_version=operator_version,
        split_paths=split_paths,
        normalization=stats,
        angles_rad=angles,
        operator_block=operator_block,
        raw_manifest=raw,
    )


def _open_h5(path: Path):
    import h5py  # type: ignore

    return h5py.File(path, "r")


class BRDTSmokeSplit(Dataset):
    """Per-sample loader over one BRDT smoke split.

    Returns a dict per index with keys:

    - ``q_true_physical``: ``(1, 128, 128)`` float32, canonical physics target;
    - ``q_true_norm``: ``(1, 128, 128)`` float32, train-normalized target;
    - ``sinogram``: ``(64, 128, 2)`` float32, observed (noisy) sinogram
      stacked as ``(real, imag)`` channels in the last dim;
    - ``angle_mask``: ``(64,)`` float32 (locked all-ones for full-view);
    - ``sample_seed``: int;
    - ``phantom_family``: str.

    The loader does NOT derive ``born_init_image`` itself; that is the
    responsibility of ``scripts.studies.born_rytov_dt.classical`` so the
    derivation is testable against the locked operator authority.
    """

    def __init__(
        self,
        path: Path,
        *,
        normalization: dc.NormalizationStats,
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.path = Path(path)
        self.normalization = normalization
        with _open_h5(self.path) as fh:
            self._length = int(fh.attrs.get("sample_count", fh["q_true_physical"].shape[0]))
            self._split = str(fh.attrs.get("split", self.path.stem))
        if indices is None:
            self._indices = tuple(range(self._length))
        else:
            self._indices = tuple(int(i) for i in indices)
            if any(i < 0 or i >= self._length for i in self._indices):
                raise IndexError(
                    f"index out of range for split with length {self._length}"
                )

    def __len__(self) -> int:
        return len(self._indices)

    @property
    def split(self) -> str:
        return self._split

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int | str]:
        i = self._indices[int(index)]
        with _open_h5(self.path) as fh:
            q_phys = fh["q_true_physical"][i]
            q_norm = fh["q_true_norm"][i]
            s_real = fh["sinogram_real"][i]
            s_imag = fh["sinogram_imag"][i]
            mask = fh["angle_mask"][:]
            seed = int(fh["sample_seed"][i])
            family_b = bytes(fh["phantom_family"][i])
        if q_phys.ndim == 2:
            q_phys = q_phys[None, ...]
            q_norm = q_norm[None, ...]
        sinogram = np.stack([s_real, s_imag], axis=-1).astype(np.float32)
        return {
            "q_true_physical": torch.from_numpy(np.asarray(q_phys, dtype=np.float32)),
            "q_true_norm": torch.from_numpy(np.asarray(q_norm, dtype=np.float32)),
            "sinogram": torch.from_numpy(sinogram),
            "angle_mask": torch.from_numpy(np.asarray(mask, dtype=np.float32)),
            "sample_seed": seed,
            "phantom_family": family_b.decode("ascii", errors="replace"),
        }


def brdt_collate(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Collate a list of BRDT samples into batched tensors.

    Output shapes:

    - ``q_true_physical``: ``(B, 1, 128, 128)``;
    - ``q_true_norm``: ``(B, 1, 128, 128)``;
    - ``sinogram``: ``(B, 64, 128, 2)``;
    - ``angle_mask``: ``(B, 64)``;
    - ``sample_seed``: ``(B,)`` long tensor;
    - ``phantom_family``: ``list[str]``.
    """
    if not samples:
        raise ValueError("brdt_collate requires at least one sample")
    batch: Dict[str, Any] = {}
    for key in ("q_true_physical", "q_true_norm", "sinogram", "angle_mask"):
        batch[key] = torch.stack([torch.as_tensor(s[key]) for s in samples], dim=0)
    batch["sample_seed"] = torch.as_tensor(
        [int(s["sample_seed"]) for s in samples], dtype=torch.long
    )
    batch["phantom_family"] = [str(s["phantom_family"]) for s in samples]
    return batch


def sinogram_to_channels_first(sinogram: torch.Tensor) -> torch.Tensor:
    """Return complex sinogram as ``(B, 2, angles, detectors)``.

    Dataset batches store sinograms as ``(B, angles, detectors, 2)``. Neural
    adapters use PyTorch's channels-first image convention.
    """
    sino = torch.as_tensor(sinogram)
    if sino.dim() != 4:
        raise ValueError(
            f"sinogram must be 4-D; got shape {tuple(sino.shape)}"
        )
    if sino.shape[-1] == 2:
        return sino.permute(0, 3, 1, 2).contiguous()
    if sino.shape[1] == 2:
        return sino.contiguous()
    raise ValueError(
        "sinogram must have a complex real/imag channel of length 2 in "
        f"either last or channel dimension; got shape {tuple(sino.shape)}"
    )


def assert_input_mode_supported(input_mode: str) -> None:
    """Validate a BRDT input-mode label."""
    if input_mode in REJECTED_INPUT_MODES:
        raise ValueError(
            "BRDT rejects the legacy direct-sinogram alias "
            f"(input_mode={input_mode!r}); use input_mode='sinogram'."
        )
    if input_mode not in SUPPORTED_INPUT_MODES:
        raise ValueError(
            f"unsupported input_mode={input_mode!r}; "
            f"allowed: {SUPPORTED_INPUT_MODES}"
        )


def list_split_indices(split: str, authority: DatasetAuthority) -> List[int]:
    """Return the contiguous index range for ``split`` in the smoke dataset."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unknown split {split!r}; allowed: {SUPPORTED_SPLITS}")
    counts = authority.raw_manifest["split"]["counts"]
    return list(range(int(counts[split])))
