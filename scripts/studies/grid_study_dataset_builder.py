#!/usr/bin/env python3
"""Dataset builder for grid-line studies.

This module provides a single entry point for dataset preparation in study
wrappers:
1. Synthetic grid-lines generation via TF workflow helpers.
2. External raw NPZ ingestion and grouping for Torch-only studies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from ptycho import params
from ptycho.config.config import ModelConfig, TrainingConfig, update_legacy_dict
from ptycho.metadata import MetadataManager
from ptycho.workflows import components as wf_components
from ptycho.workflows import grid_lines_workflow
from ptycho.workflows.grid_lines_workflow import GridLinesConfig


@dataclass
class GridStudyDatasetBundle:
    train_npz: Path
    test_npz: Path
    gt_recon: Path
    tag: str


def _build_metadata_config(
    *,
    n_value: int,
    cfg: GridLinesConfig,
    n_groups: int,
    n_subsample: Optional[int],
    neighbor_count: int,
    subsample_seed: Optional[int],
) -> TrainingConfig:
    return TrainingConfig(
        model=ModelConfig(N=n_value, gridsize=cfg.gridsize),
        nphotons=cfg.nphotons,
        batch_size=cfg.batch_size,
        nepochs=cfg.nepochs,
        n_groups=n_groups,
        n_subsample=n_subsample,
        neighbor_count=neighbor_count,
        subsample_seed=subsample_seed,
        output_dir=cfg.output_dir,
        backend="pytorch",
    )


def _build_split_payload(
    *,
    grouped: Dict[str, np.ndarray],
    probe_guess: np.ndarray,
    yy_full: np.ndarray,
) -> Dict[str, np.ndarray]:
    payload = {
        "diffraction": np.asarray(grouped["X_full"], dtype=np.float32),
        "Y_I": np.asarray(np.abs(grouped["Y"]), dtype=np.float32),
        "Y_phi": np.asarray(np.angle(grouped["Y"]), dtype=np.float32),
        "coords_nominal": np.asarray(grouped["coords_relative"], dtype=np.float32),
        "coords_true": np.asarray(grouped["coords_relative"], dtype=np.float32),
        "coords_offsets": np.asarray(grouped["coords_offsets"], dtype=np.float32),
        "coords_relative": np.asarray(grouped["coords_relative"], dtype=np.float32),
        "YY_full": np.asarray(yy_full, dtype=np.complex64),
        "probeGuess": np.asarray(probe_guess, dtype=np.complex64),
    }
    return payload


def _save_external_split(
    *,
    path: Path,
    payload: Dict[str, np.ndarray],
    metadata_cfg: TrainingConfig,
    cfg: GridLinesConfig,
    n_groups: int,
    n_subsample: Optional[int],
    neighbor_count: int,
    subsample_seed: Optional[int],
) -> None:
    metadata = MetadataManager.create_metadata(
        metadata_cfg,
        script_name="grid_study_dataset_builder",
        coords_type="relative",
        dataset_source="external_raw_npz",
        nimgs_test=cfg.nimgs_test,
        outer_offset_test=cfg.outer_offset_test,
        n_groups=n_groups,
        n_subsample=n_subsample,
        neighbor_count=neighbor_count,
        subsample_seed=subsample_seed,
    )
    MetadataManager.save_with_metadata(str(path), payload, metadata)


def _ensure_canonical_gt(output_dir: Path, yy_full: np.ndarray) -> Path:
    gt_path = output_dir / "recons" / "gt" / "recon.npz"
    yy_full = np.asarray(np.squeeze(yy_full), dtype=np.complex64)
    if gt_path.exists():
        with np.load(gt_path) as existing:
            existing_gt = np.asarray(np.squeeze(existing["YY_pred"]), dtype=np.complex64)
        if existing_gt.shape != yy_full.shape or not np.allclose(existing_gt, yy_full, rtol=1e-6, atol=1e-6):
            raise ValueError("Canonical GT mismatch across dataset bundles.")
        return gt_path
    return grid_lines_workflow.save_recon_artifact(output_dir, "gt", yy_full)


def _build_external_bundle_for_n(
    *,
    cfg: GridLinesConfig,
    n_value: int,
    train_data: Path,
    test_data: Path,
    n_groups: Optional[int],
    n_subsample: Optional[int],
    neighbor_count: int,
    subsample_seed: Optional[int],
) -> GridStudyDatasetBundle:
    raw_train = wf_components.load_data(
        str(train_data),
        n_images=n_groups,
        n_subsample=n_subsample,
        n_samples=(n_groups if n_groups is not None else 1),
        subsample_seed=subsample_seed,
    )
    raw_test = wf_components.load_data(
        str(test_data),
        n_images=n_groups,
        n_subsample=n_subsample,
        n_samples=(n_groups if n_groups is not None else 1),
        subsample_seed=subsample_seed,
    )

    train_group_count = int(n_groups) if n_groups is not None else int(raw_train.xcoords.shape[0])
    test_group_count = int(n_groups) if n_groups is not None else int(raw_test.xcoords.shape[0])
    if train_group_count <= 0 or test_group_count <= 0:
        raise ValueError("Resolved n_groups must be positive for both train and test splits.")

    meta_cfg_train = _build_metadata_config(
        n_value=n_value,
        cfg=cfg,
        n_groups=train_group_count,
        n_subsample=n_subsample,
        neighbor_count=neighbor_count,
        subsample_seed=subsample_seed,
    )
    meta_cfg_test = _build_metadata_config(
        n_value=n_value,
        cfg=cfg,
        n_groups=test_group_count,
        n_subsample=n_subsample,
        neighbor_count=neighbor_count,
        subsample_seed=subsample_seed,
    )
    update_legacy_dict(params.cfg, meta_cfg_train)

    if raw_train.objectGuess is None or raw_test.objectGuess is None:
        raise ValueError("external_raw_npz requires objectGuess for canonical GT reconstruction.")

    grouped_train = raw_train.generate_grouped_data(
        N=n_value,
        K=neighbor_count,
        nsamples=train_group_count,
        dataset_path=str(train_data),
        seed=subsample_seed,
        gridsize=cfg.gridsize,
    )
    grouped_test = raw_test.generate_grouped_data(
        N=n_value,
        K=neighbor_count,
        nsamples=test_group_count,
        dataset_path=str(test_data),
        seed=subsample_seed,
        gridsize=cfg.gridsize,
    )

    if grouped_train.get("Y") is None or grouped_test.get("Y") is None:
        raise ValueError("external_raw_npz requires objectGuess-derived grouped patches (Y).")

    yy_full = np.asarray(np.squeeze(raw_test.objectGuess), dtype=np.complex64)
    gt_recon_path = _ensure_canonical_gt(cfg.output_dir, yy_full)

    dataset_dir = cfg.output_dir / "datasets" / f"N{n_value}" / f"gs{cfg.gridsize}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_npz = dataset_dir / "train.npz"
    test_npz = dataset_dir / "test.npz"

    train_payload = _build_split_payload(
        grouped=grouped_train,
        probe_guess=raw_train.probeGuess,
        yy_full=yy_full,
    )
    test_payload = _build_split_payload(
        grouped=grouped_test,
        probe_guess=raw_test.probeGuess,
        yy_full=yy_full,
    )
    test_payload["YY_ground_truth"] = np.asarray(yy_full, dtype=np.complex64)
    test_payload["norm_Y_I"] = np.asarray(1.0, dtype=np.float32)

    _save_external_split(
        path=train_npz,
        payload=train_payload,
        metadata_cfg=meta_cfg_train,
        cfg=cfg,
        n_groups=train_group_count,
        n_subsample=n_subsample,
        neighbor_count=neighbor_count,
        subsample_seed=subsample_seed,
    )
    _save_external_split(
        path=test_npz,
        payload=test_payload,
        metadata_cfg=meta_cfg_test,
        cfg=cfg,
        n_groups=test_group_count,
        n_subsample=n_subsample,
        neighbor_count=neighbor_count,
        subsample_seed=subsample_seed,
    )

    return GridStudyDatasetBundle(
        train_npz=train_npz,
        test_npz=test_npz,
        gt_recon=gt_recon_path,
        tag=f"N{n_value}",
    )


def build_datasets(
    *,
    dataset_source: str,
    cfg: GridLinesConfig,
    required_ns: Iterable[int],
    train_data: Optional[Path] = None,
    test_data: Optional[Path] = None,
    n_groups: Optional[int] = 512,
    n_subsample: Optional[int] = None,
    neighbor_count: int = 7,
    subsample_seed: Optional[int] = None,
) -> Dict[int, Dict[str, str]]:
    """Build dataset bundles for study wrappers."""
    required_ns = sorted(set(int(n_value) for n_value in required_ns))
    if not required_ns:
        raise ValueError("required_ns must contain at least one N value.")

    if dataset_source == "synthetic_lines":
        return grid_lines_workflow.build_grid_lines_datasets_by_n(cfg, required_ns=required_ns)

    if dataset_source != "external_raw_npz":
        raise ValueError(f"Unknown dataset_source: {dataset_source}")
    if train_data is None or test_data is None:
        raise ValueError("external_raw_npz mode requires both train_data and test_data.")
    if len(required_ns) != 1:
        raise ValueError("external_raw_npz mode currently supports a single N value.")

    n_value = required_ns[0]
    cfg_n = replace(cfg, N=n_value)
    bundle = _build_external_bundle_for_n(
        cfg=cfg_n,
        n_value=n_value,
        train_data=Path(train_data),
        test_data=Path(test_data),
        n_groups=n_groups,
        n_subsample=n_subsample,
        neighbor_count=neighbor_count,
        subsample_seed=subsample_seed,
    )
    return {
        n_value: {
            "train_npz": str(bundle.train_npz),
            "test_npz": str(bundle.test_npz),
            "gt_recon": str(bundle.gt_recon),
            "tag": bundle.tag,
        }
    }
