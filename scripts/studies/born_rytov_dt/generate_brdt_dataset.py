"""Deterministic BRDT smoke/preflight dataset generator.

This script consumes the locked operator authority recorded in
``operator_validation.json`` and emits the small smoke dataset described
in the candidate-lane and dataset-preflight plans. It supports a
``--dry-run-manifest`` mode that validates geometry against the operator
authority without producing any large arrays.

This generator is feasibility-only and intentionally does not generate
the larger downstream decision-support split, classical
backpropagation initializations, or training-side adapters. Those
belong to follow-up backlog items (`brdt-task-adapters` and
`brdt-four-row-preflight`).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ptycho_torch.physics import BornRytovForward2D
from scripts.studies.born_rytov_dt import dataset_contract as dc
from scripts.studies.born_rytov_dt.phantoms import generate_refractive_index


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / ".artifacts"
    / "NEURIPS-HYBRID-RESNET-2026"
    / "backlog"
    / "2026-04-29-brdt-dataset-preflight"
)
DEFAULT_OPERATOR_VALIDATION = (
    REPO_ROOT
    / ".artifacts"
    / "NEURIPS-HYBRID-RESNET-2026"
    / "backlog"
    / "2026-04-29-brdt-operator-validation"
    / "operator_validation.json"
)
GENERATOR_MODULE = "scripts.studies.born_rytov_dt.generate_brdt_dataset"


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------
def _git_revision(repo_root: Path) -> Tuple[str, bool]:
    try:
        rev = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=str(repo_root), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        return rev, bool(status)
    except Exception:
        return "unknown", False


def _env_summary() -> Dict[str, Any]:
    try:
        import h5py  # type: ignore

        h5py_version: Optional[str] = h5py.__version__
    except Exception:
        h5py_version = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "numpy": np.__version__,
        "h5py": h5py_version,
    }


def _generation_command(raw_args: Optional[List[str]]) -> str:
    args = list(sys.argv[1:] if raw_args is None else raw_args)
    return shlex.join(["python", "-m", GENERATOR_MODULE, *map(str, args)])


# ----------------------------------------------------------------------
# Operator authority
# ----------------------------------------------------------------------
def load_operator_authority(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"operator validation artifact not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if "operator" not in payload:
        raise ValueError(f"operator validation JSON missing 'operator' block: {path}")
    if payload.get("verdict") not in ("pass", "pass_with_documented_limits"):
        raise ValueError(
            f"operator validation verdict not pass-equivalent: {payload.get('verdict')!r}"
        )
    return payload["operator"]


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------
def _make_operator(device: torch.device) -> BornRytovForward2D:
    angles = torch.from_numpy(dc.locked_angles()).to(torch.float64)
    op = BornRytovForward2D(
        grid_size=dc.LOCKED_GRID_SIZE,
        detector_size=dc.LOCKED_DETECTOR_SIZE,
        angles=angles,
        wavelength_px=dc.LOCKED_WAVELENGTH_PX,
        medium_ri=dc.LOCKED_MEDIUM_RI,
        mode=dc.LOCKED_OPERATOR_MODE,
        normalize=dc.LOCKED_NORMALIZE,
    )
    return op.to(device)


def _generate_split_qphys(
    object_seeds: List[int],
    families: List[str],
) -> np.ndarray:
    out = np.empty(
        (len(object_seeds), dc.LOCKED_GRID_SIZE, dc.LOCKED_GRID_SIZE),
        dtype=np.float64,
    )
    for idx, (seed, family) in enumerate(zip(object_seeds, families)):
        n_field = generate_refractive_index(
            family=family, seed=int(seed), grid=dc.LOCKED_GRID_SIZE, n_m=dc.LOCKED_MEDIUM_RI
        )
        q = dc.refractive_index_to_q(n_field, n_m=dc.LOCKED_MEDIUM_RI)
        out[idx] = q
    return out


def _operator_forward_batch(
    op: BornRytovForward2D, q_batch: np.ndarray, device: torch.device
) -> np.ndarray:
    """Apply BRDT operator to a numpy batch and return (B, A, D, 2) float32."""
    q_t = torch.from_numpy(q_batch).to(device=device, dtype=torch.float64).unsqueeze(1)
    with torch.no_grad():
        out_t = op(q_t)
    return out_t.cpu().to(torch.float32).numpy()


def _add_complex_gaussian_noise(
    sino: np.ndarray, noise_sigma: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 31337)
    noise = rng.standard_normal(sino.shape).astype(sino.dtype) * float(noise_sigma)
    return sino + noise


def _measured_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    s = np.linalg.norm(clean.astype(np.float64))
    n = np.linalg.norm((noisy - clean).astype(np.float64))
    if n < 1e-30 or s < 1e-30:
        return float("inf") if n < 1e-30 else 0.0
    return float(20.0 * math.log10(s / n))


def _write_split_h5(
    path: Path,
    split: str,
    q_phys: np.ndarray,
    q_norm: np.ndarray,
    sino_clean: np.ndarray,
    sino_noisy: np.ndarray,
    angles: np.ndarray,
    object_seeds: List[int],
    families: List[str],
) -> None:
    import h5py  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["split"] = split
        f.attrs["dataset_name"] = dc.DATASET_NAME
        f.attrs["sample_count"] = q_phys.shape[0]
        f.attrs["forward_input_is_physical_q"] = True
        f.attrs["model_output_space"] = "normalized_q"
        f.attrs["physics_loss_rule"] = dc.PHYSICS_LOSS_RULE
        # arrays
        f.create_dataset(
            "q_true_physical",
            data=q_phys.astype(np.float32),
            compression="gzip",
            compression_opts=4,
            chunks=(1, dc.LOCKED_GRID_SIZE, dc.LOCKED_GRID_SIZE),
        )
        f.create_dataset(
            "q_true_norm",
            data=q_norm.astype(np.float32),
            compression="gzip",
            compression_opts=4,
            chunks=(1, dc.LOCKED_GRID_SIZE, dc.LOCKED_GRID_SIZE),
        )
        f.create_dataset(
            "sinogram_real",
            data=sino_noisy[..., 0].astype(np.float32),
            compression="gzip",
            compression_opts=4,
            chunks=(1, dc.LOCKED_ANGLE_COUNT, dc.LOCKED_DETECTOR_SIZE),
        )
        f.create_dataset(
            "sinogram_imag",
            data=sino_noisy[..., 1].astype(np.float32),
            compression="gzip",
            compression_opts=4,
            chunks=(1, dc.LOCKED_ANGLE_COUNT, dc.LOCKED_DETECTOR_SIZE),
        )
        f.create_dataset(
            "sinogram_clean_real",
            data=sino_clean[..., 0].astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "sinogram_clean_imag",
            data=sino_clean[..., 1].astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "angle_mask",
            data=np.ones((dc.LOCKED_ANGLE_COUNT,), dtype=np.float32),
        )
        f.create_dataset(
            "angles_rad",
            data=angles.astype(np.float64),
        )
        f.create_dataset(
            "sample_seed",
            data=np.asarray(object_seeds, dtype=np.int64),
        )
        ascii_families = np.asarray(families, dtype="S32")
        f.create_dataset("phantom_family", data=ascii_families)


# ----------------------------------------------------------------------
# Dry-run
# ----------------------------------------------------------------------
def write_dry_run(
    *,
    output_root: Path,
    operator_authority: Optional[Dict[str, Any]],
    split_seed: int,
    counts: dc.SplitCounts,
    noise_sigma: float,
    operator_validation_path: Path,
    generation_command: str,
    blocking_issues: Optional[List[str]] = None,
) -> Dict[str, Any]:
    blocking_issues = [] if blocking_issues is None else list(blocking_issues)
    mismatches = (
        dc.validate_geometry_against_operator_authority(operator_authority)
        if operator_authority is not None
        else []
    )
    seeds = dc.deterministic_object_seeds(counts, split_seed=split_seed)
    families = dc.assign_phantom_families(counts, split_seed=split_seed)
    estimated_paths = {
        s: str(output_root / "dataset" / f"{dc.DATASET_NAME}_{s}.h5")
        for s in ("train", "val", "test")
    }
    sha, dirty = _git_revision(REPO_ROOT)
    verdict = "ready_for_smoke_generation" if not mismatches and not blocking_issues else "not_ready"
    manifest_skeleton_path = output_root / dc.DRY_RUN_MANIFEST_NAME
    summary_path = output_root / "dry_run_summary.json"
    manifest = dc.build_manifest(
        output_root=str(output_root),
        operator_validation_path=str(operator_validation_path),
        counts=counts,
        split_seed=split_seed,
        object_seeds=seeds,
        families=families,
        normalization=None,
        noise_sigma=noise_sigma,
        measured_snr=None,
        git_sha=sha,
        git_dirty=dirty,
        generation_command=generation_command,
        environment=_env_summary(),
        artifact_paths=estimated_paths,
        extra={
            "array_generation_skipped": True,
            "estimated_artifacts_only": True,
            "generation_mode": "dry_run_manifest",
            "verdict": verdict,
            "blocking_issues": blocking_issues,
            "dry_run_summary_path": str(summary_path),
        },
    )
    summary = {
        "schema_version": "1.0",
        "verdict": verdict,
        "mode": "dry_run_manifest",
        "operator_authority_path": str(operator_validation_path),
        "operator_authority_block": operator_authority,
        "blocking_issues": blocking_issues,
        "geometry_mismatches": mismatches,
        "requested_geometry": {
            "grid_size": dc.LOCKED_GRID_SIZE,
            "detector_size": dc.LOCKED_DETECTOR_SIZE,
            "angle_count": dc.LOCKED_ANGLE_COUNT,
            "wavelength_px": dc.LOCKED_WAVELENGTH_PX,
            "medium_ri": dc.LOCKED_MEDIUM_RI,
            "mode": dc.LOCKED_OPERATOR_MODE,
            "normalize": dc.LOCKED_NORMALIZE,
        },
        "split": {
            "split_seed": split_seed,
            "counts": counts.as_dict(),
            "object_seeds": seeds,
            "phantom_family_assignment": families,
        },
        "estimated_artifact_paths": estimated_paths,
        "manifest_skeleton_path": str(manifest_skeleton_path),
        "noise_sigma_physical_units": float(noise_sigma),
        "generation_command": generation_command,
        "git_sha": sha,
        "git_dirty": dirty,
        "environment": _env_summary(),
        "claim_boundary": (
            "Feasibility-only smoke dataset. Adapters and four-row preflight "
            "remain out of scope for this item."
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    dc.write_manifest(manifest, str(manifest_skeleton_path))
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    return summary


# ----------------------------------------------------------------------
# Live generation
# ----------------------------------------------------------------------
def run_live_generation(
    *,
    output_root: Path,
    operator_authority: Dict[str, Any],
    operator_validation_path: Path,
    split_seed: int,
    counts: dc.SplitCounts,
    noise_sigma: float,
    generation_command: str,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    mismatches = dc.validate_geometry_against_operator_authority(operator_authority)
    if mismatches:
        raise ValueError(
            "Operator authority does not match locked smoke geometry: "
            + "; ".join(mismatches)
        )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    seeds = dc.deterministic_object_seeds(counts, split_seed=split_seed)
    families = dc.assign_phantom_families(counts, split_seed=split_seed)
    angles = dc.locked_angles()

    op = _make_operator(torch_device)

    # Phase 1: synth physical q for each split.
    q_phys = {
        split: _generate_split_qphys(seeds[split], families[split])
        for split in ("train", "val", "test")
    }

    # Phase 2: train-only normalization stats.
    stats = dc.compute_train_normalization(q_phys["train"])

    # Phase 3: clean + noisy sinograms; we propagate physical q through the
    # operator. Per the contract, A always sees physical q.
    sino_clean: Dict[str, np.ndarray] = {}
    sino_noisy: Dict[str, np.ndarray] = {}
    snr_per_split: Dict[str, float] = {}
    for split, q_arr in q_phys.items():
        clean = _operator_forward_batch(op, q_arr, torch_device)
        sino_clean[split] = clean
        noisy = _add_complex_gaussian_noise(
            clean,
            noise_sigma,
            seed=dc.deterministic_noise_seed(split_seed, split),
        )
        sino_noisy[split] = noisy
        snr_per_split[f"{split}_db"] = _measured_snr_db(clean, noisy)

    # Phase 4: write HDF5 files.
    artifact_paths: Dict[str, str] = {}
    dataset_dir = output_root / "dataset"
    for split in ("train", "val", "test"):
        out_path = dataset_dir / f"{dc.DATASET_NAME}_{split}.h5"
        q_norm = dc.normalize_q(q_phys[split], stats)
        _write_split_h5(
            path=out_path,
            split=split,
            q_phys=q_phys[split],
            q_norm=q_norm,
            sino_clean=sino_clean[split],
            sino_noisy=sino_noisy[split],
            angles=angles,
            object_seeds=seeds[split],
            families=families[split],
        )
        artifact_paths[split] = str(out_path)

    sha, dirty = _git_revision(REPO_ROOT)
    manifest = dc.build_manifest(
        output_root=str(output_root),
        operator_validation_path=str(operator_validation_path),
        counts=counts,
        split_seed=split_seed,
        object_seeds=seeds,
        families=families,
        normalization=stats,
        noise_sigma=noise_sigma,
        measured_snr=snr_per_split,
        git_sha=sha,
        git_dirty=dirty,
        generation_command=generation_command,
        environment=_env_summary(),
        artifact_paths=artifact_paths,
        extra={
            "device": str(torch_device),
            "angle_grid_rad": angles.tolist(),
        },
    )
    manifest_path = output_root / "dataset_manifest.json"
    dc.write_manifest(manifest, str(manifest_path))
    return {
        "manifest_path": str(manifest_path),
        "artifact_paths": artifact_paths,
        "snr": snr_per_split,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Artifact root for the smoke dataset (default: item artifact root).",
    )
    p.add_argument(
        "--operator-validation",
        type=Path,
        default=DEFAULT_OPERATOR_VALIDATION,
        help="Path to operator_validation.json that locks the operator contract.",
    )
    p.add_argument("--split-seed", type=int, default=42, help="Deterministic split seed.")
    p.add_argument("--train-count", type=int, default=16)
    p.add_argument("--val-count", type=int, default=4)
    p.add_argument("--test-count", type=int, default=4)
    p.add_argument(
        "--noise-sigma",
        type=float,
        default=1e-3,
        help="Noise sigma in physical sinogram units.",
    )
    p.add_argument(
        "--dry-run-manifest",
        action="store_true",
        help="Write only dry_run_summary.json and a manifest skeleton; do not generate arrays.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device override; defaults to cuda if available.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    output_root: Path = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    counts = dc.SplitCounts(train=args.train_count, val=args.val_count, test=args.test_count)
    generation_command = _generation_command(argv)

    if args.dry_run_manifest:
        operator_authority: Optional[Dict[str, Any]] = None
        blocking_issues: List[str] = []
        try:
            operator_authority = load_operator_authority(args.operator_validation)
        except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError) as exc:
            blocking_issues.append(str(exc))
        summary = write_dry_run(
            output_root=output_root,
            operator_authority=operator_authority,
            split_seed=args.split_seed,
            counts=counts,
            noise_sigma=args.noise_sigma,
            operator_validation_path=args.operator_validation,
            generation_command=generation_command,
            blocking_issues=blocking_issues,
        )
        print(f"verdict: {summary['verdict']}", file=sys.stderr)
        print(f"wrote {output_root / 'dry_run_summary.json'}", file=sys.stderr)
        return 0 if summary["verdict"] == "ready_for_smoke_generation" else 2

    operator_authority = load_operator_authority(args.operator_validation)
    started = time.perf_counter()
    print(f"[{datetime.now(timezone.utc).isoformat()}] generating BRDT smoke dataset", file=sys.stderr)
    result = run_live_generation(
        output_root=output_root,
        operator_authority=operator_authority,
        operator_validation_path=args.operator_validation,
        split_seed=args.split_seed,
        counts=counts,
        noise_sigma=args.noise_sigma,
        generation_command=generation_command,
        device=args.device,
    )
    elapsed = time.perf_counter() - started
    print(f"manifest: {result['manifest_path']}", file=sys.stderr)
    print(f"snr_db: {result['snr']}", file=sys.stderr)
    print(f"elapsed_s: {elapsed:.2f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
